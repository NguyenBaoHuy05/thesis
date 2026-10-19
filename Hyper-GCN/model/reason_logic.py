"""Baseline E1b (plan §7): the logic head of REASON on our concept decoder.

Ported from REASON (Ilyas, Mehta, Ge, arXiv 2605.07140, MIT License, D:/Thesis/REASON/model/crl/components.py,
itself based on RRL / CRL): binarized concepts with NOT [b; 1 - b] -> two Union layers (soft AND + soft OR
with learned switchboards, gradient grafting between the binarized and the continuous pass, skip connection)
-> linear head over the rules. Only the training path is ported (no rule extraction). Changes: the concept
input is logit(c) of our decoder (REASON feeds its concept logits), RMSNorm is a short local version.

As in REASON's main.py: weights of the logic layers are clipped to [0, 1] after every optimizer step
(after_step), an L2 penalty (l2_weight) is added to the loss (regularizer), the logic layers have their own
learning rate (logic_lr, REASON 1e-4) and gradient norms are clipped per group (decoder 1.0, logic 0.5).
Run 1 without this recipe and with the decoder output at zero collapsed: train top1 ~4% (E1b, 2026-10-19).
"""

import torch
import torch.nn as nn

from model.attribute_decoder import AttributeDecoder

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON = 1e-10
EPS = 1e-6


class GradGraft(torch.autograd.Function):
    """Forward the binarized result X, backward through the continuous result Y."""

    @staticmethod
    def forward(ctx, X, Y):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class Binarizer(torch.autograd.Function):
    @staticmethod
    def forward(_, x):
        return (x.detach() > 0.0).float()

    @staticmethod
    def backward(_, grad_output):
        return grad_output.clone()


class Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        y = -1.0 / (-1.0 + torch.sum(torch.log(X), dim=1))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y = ctx.saved_tensors
        return grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = torch.finfo(x.dtype).eps
        return x / x.pow(2).mean(-1, keepdim=True).add(eps).sqrt() * self.weight


class ScaledSoftsign(nn.Module):
    def __init__(self, scale=3.0, beta=1.0):
        super().__init__()
        self.scale, self.beta = scale, beta

    def forward(self, x):
        return self.scale * x / (self.beta + torch.abs(x))


class BinarizeLayer(nn.Module):
    def __init__(self, n, use_not):
        super().__init__()
        self.use_not = use_not
        self.output_dim = 2 * n if use_not else n

    def forward(self, x):
        x = Binarizer.apply(x)
        return torch.cat((x, 1 - x), dim=1) if self.use_not else x


class _Logic(nn.Module):
    def __init__(self, input_dim, output_dim, use_not):
        super().__init__()
        self.use_not = use_not
        self.W = nn.Parameter(INIT_RANGE * torch.rand(input_dim * 2 if use_not else input_dim, output_dim))

    def _inputs(self, x):
        return torch.cat((x, 1 - x), dim=1) if self.use_not else x

    def forward(self, x):
        x = self._inputs(x)
        with torch.no_grad():
            hard = self.binarized(x, Binarizer.apply(self.W - THRESHOLD))
        return GradGraft.apply(hard, self.continuous(x))

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class ConjunctionLayer(_Logic):
    def continuous(self, x):
        return Product.apply(1 - (1 - x).unsqueeze(-1) * self.W)

    def binarized(self, x, Wb):
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb, dim=1)


class DisjunctionLayer(_Logic):
    def continuous(self, x):
        return 1 - Product.apply(1 - x.unsqueeze(-1) * self.W)

    def binarized(self, x, Wb):
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb, dim=1)


class UnionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.output_dim = output_dim * 2
        self.con_layer = ConjunctionLayer(input_dim, output_dim, use_not)
        self.dis_layer = DisjunctionLayer(input_dim, output_dim, use_not)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()

    def l2_norm(self):
        return torch.sum(self.con_layer.W ** 2) + torch.sum(self.dis_layer.W ** 2)


class ConceptLogicLayers(nn.Module):
    """binarize(+NOT) -> Union(l1) -> Union(l2, NOT) -> Linear over [union2 ; union1] (skip, as REASON), / exp(t)."""

    def __init__(self, n_concepts, n_actions, l1_dim=128, l2_dim=128, use_not=True, use_skip=True, temperature=1.0):
        super().__init__()
        self.use_skip = use_skip
        self.norm = RMSNorm(n_concepts)
        self.softsign = ScaledSoftsign()
        self.binarize = BinarizeLayer(n_concepts, use_not)
        self.union1 = UnionLayer(self.binarize.output_dim, l1_dim, use_not=False)
        self.union2 = UnionLayer(self.union1.output_dim, l2_dim, use_not=True)
        lr_in = self.union2.output_dim + (self.union1.output_dim if use_skip else 0)
        self.lr = nn.Linear(lr_in, n_actions)
        self.t = nn.Parameter(torch.log(torch.tensor([float(temperature)])))

    def forward(self, concept_logits):
        x1 = self.union1(self.binarize(self.softsign(self.norm(concept_logits))))
        x = self.union2(x1)
        if self.use_skip:
            x = torch.cat((x, x1), dim=1)
        return self.lr(x) / torch.exp(self.t)

    def l2_penalty(self):
        return self.union1.l2_norm() + self.union2.l2_norm() + torch.sum(self.lr.weight ** 2)

    def clip_weights(self):
        self.union1.clip()
        self.union2.clip()


class ReasonLogicHead(nn.Module):
    def __init__(self, in_channels, M, joint_mask, decoder_args=None, logic_args=None, l2_weight=1e-6,
                 logic_lr=1e-4, clip_decoder=1.0, clip_logic=0.5):
        super().__init__()
        self.register_buffer('M', M.float())
        self.l2_weight, self.logic_lr = float(l2_weight), float(logic_lr)
        self.clip_decoder, self.clip_logic = float(clip_decoder), float(clip_logic)
        self.decoder = AttributeDecoder(in_channels, torch.eye(M.shape[1]), joint_mask, **(decoder_args or {}))
        # AttributeDecoder starts its output layer at zero (c = 0.5, logit 0). RMSNorm of an all-zero logit
        # vector divides by sqrt(eps) and blows the first gradients up (~3000x); REASON's concept fc has the
        # default init, so use it here too.
        self.decoder.out.reset_parameters()
        self.logic = ConceptLogicLayers(M.shape[1], M.shape[0], **(logic_args or {}))

    @property
    def tau(self):
        return torch.exp(self.logic.t).squeeze()

    def logits(self, c):
        c = c.clamp(EPS, 1 - EPS)
        return self.logic(torch.log(c) - torch.log1p(-c))

    def regularizer(self):
        return self.l2_weight * self.logic.l2_penalty()

    def param_groups(self, lr, weight_decay=0.0):
        return [dict(params=list(self.decoder.parameters()), lr=lr, weight_decay=weight_decay),
                dict(params=list(self.logic.parameters()), lr=self.logic_lr, weight_decay=0.0)]

    def clip_grads(self):
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=self.clip_decoder)
        torch.nn.utils.clip_grad_norm_(self.logic.parameters(), max_norm=self.clip_logic)

    def after_step(self):
        self.logic.clip_weights()

    def forward(self, features, valid):
        c = self.decoder(features, valid)
        return dict(c=c, logits=self.logits(c))

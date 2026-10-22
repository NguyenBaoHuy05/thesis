"""A3: concept-conditioned hyperedges inside Hyper-GCN large (drawio/architecture_a3_concept_hyperedge.drawio,
plan: KG_RETRIEVAL_PLAN/A3_CONCEPT_HYPEREDGE/A3_PLAN.md).

    F6   = l1 .. l6 (K0, frozen)                          computed once, read by both passes
    z    = pool(l7 .. l9 (K0, frozen)(F6))                 pass 1: exactly K0's pooled feature
    c    = sigmoid(W_c z)                                  102 concepts, BCE against M[y]
    H_KG[v, k] = B[k, v] * c_k                             one hyperedge per spatial concept (74); B: concept -> joints
    G_KG = hyper_norm(H_KG, softplus(w_KG))                as HYPERGC.hyper_norm; rows of the virtual joints are 0
    y    = fc(pool(l7' .. l9'(F6, A + beta_l G_KG)))       pass 2: l7'..l9' start as copies of K0's l7..l9, beta_l = 0

At initialization pass 2 equals pass 1, so the output is K0's.
stage 'concept' (S1): the output is K0's logits, only the concept head learns (BCE).
stage 'a3' (S2): the output is pass 2; c enters G_KG detached (G_KG reads concept evidence, CE does not reshape it).
Controls: use_kg False = K0-ft (same pass 2, no G_KG); mask 'random' = B replaced by random joint sets of the same
sizes (fixed by mask_seed, rebuilt identically at every construction, not stored in the checkpoint).

The copies a3.layers.l7..l9 are filled from l7..l9 when a state_dict without them is loaded (an official checkpoint):
load_model_weights still lists them as "initialized", but they hold K0's weights.
eval_outputs: concept_match (class from the concepts alone, no learned class weight) and, in stage a3 with use_kg,
no_kg (pass 2 of the same checkpoint with beta = 0). One GPU only (the last forward is kept on the module).
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.concept_logic import EPS, concept_joint_mask, load_concepts
from model.concept_schema import NTU_ACTION_NAMES
from model.hypergcn_large import Model as HyperGCN

PASS2 = ('l7', 'l8', 'l9')


def hyper_norm(H, W):
    """HYPERGC.hyper_norm as a function: H [N, S, V, E] incidence, W [N, S, E] hyperedge weights -> A [N, S, V, V]."""
    w = torch.diag_embed(W)
    w_ = w / (torch.norm(H, 1, dim=2, keepdim=True) + 1e-8)
    H_w = H @ w
    h_ = H_w / (torch.norm(H_w, 1, dim=3, keepdim=True) + 1e-8)
    return h_ @ w_ @ H.transpose(3, 2)


def random_joint_sets(mask, seed):
    """[K, V] bool with the row sizes of `mask` and random joints (control for the knowledge in B)."""
    generator = torch.Generator().manual_seed(int(seed))
    out = torch.zeros_like(mask)
    for k, size in enumerate(mask.sum(1).tolist()):
        out[k, torch.randperm(mask.shape[1], generator=generator)[:size]] = True
    return out


class A3Head(nn.Module):
    """Concept head, concept hyperedges and the pass-2 layers (all under the prefix a3.)."""

    def __init__(self, in_channels, layers, matrix, metadata, virtual_num, mask='concept', mask_seed=0):
        super().__init__()
        M, names, meta = load_concepts(matrix, metadata, NTU_ACTION_NAMES)
        spatial = [k for k, n in enumerate(names) if meta[n]['scope'] == 'spatial']
        B = concept_joint_mask(names, meta)[spatial]
        if mask == 'random':
            B = random_joint_sets(B, mask_seed)
        elif mask != 'concept':
            raise ValueError("mask must be 'concept' or 'random'")
        self.register_buffer('M', M, persistent=False)
        self.register_buffer('spatial', torch.tensor(spatial), persistent=False)
        self.register_buffer('B', B.float(), persistent=False)
        self.virtual_num = int(virtual_num)
        self.concept = nn.Linear(in_channels, M.shape[1])
        self.w_kg = nn.Parameter(torch.full((len(spatial),), math.log(math.e - 1)))   # softplus = 1
        self.beta = nn.Parameter(torch.zeros(len(layers)))
        self.layers = nn.ModuleDict(layers)

    def adjacency(self, c):
        """c [N, 102] -> G_KG [N, 1, V + virtual, V + virtual] (shared by the 8 subsets)."""
        H = self.B.t().unsqueeze(0) * c[:, self.spatial].unsqueeze(1)             # [N, V, K]
        H = F.pad(H, (0, 0, 0, self.virtual_num))                                 # virtual joints: no concept
        W = F.softplus(self.w_kg).expand(c.shape[0], -1)
        return hyper_norm(H.unsqueeze(1), W.unsqueeze(1))

    def concept_match(self, c):
        """[N, A] log-likelihood of c under each class row of M (as Stage1Decoders.concept_match)."""
        log_c, log_not = torch.log(c.clamp(EPS, 1 - EPS)), torch.log((1 - c).clamp(EPS, 1 - EPS))
        M = self.M
        return log_c @ M.t() / M.sum(1).clamp_min(1) + log_not @ (1 - M).t() / (1 - M).sum(1).clamp_min(1)


def run_top(layers, x, extra=None):
    """l7, l8, l9 on [N*M, C, T, V] with the residual of l9 from l7 (hypergcn_large.RESIDUAL_FROM).
    extra: None, or one extra adjacency per layer."""
    (l7, l8, l9), (e7, e8, e9) = layers, extra or (None,) * 3
    x7, h7 = l7(x, e7)
    x8, h8 = l8(x7, e8)
    x9, h9 = l9(x8 + x7, e9)
    return x9, (h7, h8, h9)


class Model(HyperGCN):
    def __init__(self, a3_args=None, **kwargs):
        super().__init__(**kwargs)
        args = dict(a3_args or {})
        self.stage = args.pop('stage', 'a3')
        if self.stage not in ('concept', 'a3'):
            raise ValueError("stage must be 'concept' or 'a3'")
        self.use_kg = bool(args.pop('use_kg', True))
        self.detach_concepts = bool(args.pop('detach_concepts', True))
        self.concept_weight = float(args.pop('concept_weight', 0.5))
        layers = {name: copy.deepcopy(getattr(self, name)) for name in PASS2}
        self.a3 = A3Head(self.embedding_channels * 4, layers, virtual_num=kwargs.get('hyper_joints', 0), **args)
        self._last = None

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Fill a3.layers.l7..l9 from l7..l9 when the state_dict has no pass-2 layers (official checkpoint)."""
        state = dict(state_dict)
        for name in PASS2:
            prefix = f'a3.layers.{name}.'
            if not any(k.startswith(prefix) for k in state):
                state.update({prefix + k[len(name) + 1:]: v for k, v in state_dict.items()
                              if k.startswith(name + '.')})
        return super().load_state_dict(state, strict, **kwargs)

    def pool(self, x, N):
        """[N*M, C, T, V] -> [N, C]: Model.forward's pooling (mean over T, V, then actors)."""
        return x.view(N, -1, *x.shape[1:]).flatten(3).mean(3).mean(1)

    def forward(self, x):
        N = x.shape[0]
        f6, hyper_outputs = self.features(x, until='l6')
        M = f6.shape[1]
        f6 = f6.flatten(0, 1)
        p1, h1 = run_top((self.l7, self.l8, self.l9), f6)
        z = self.pool(p1, N)
        c = torch.sigmoid(self.a3.concept(z))
        self._last = dict(c=c, no_kg=None)
        if self.stage == 'concept':
            return self.fc(self.drop_out(z)), hyper_outputs + h1

        layers = tuple(self.a3.layers[name] for name in PASS2)
        extra = None
        if self.use_kg:
            G = self.a3.adjacency(c.detach() if self.detach_concepts else c).repeat_interleave(M, 0)
            extra = tuple(b * G for b in self.a3.beta)
        p2, h2 = run_top(layers, f6, extra)
        output = self.fc(self.drop_out(self.pool(p2, N)))
        if self.use_kg and not self.training:
            self._last['no_kg'] = self.fc(self.pool(run_top(layers, f6)[0], N))
        return output, hyper_outputs + h2

    def extra_loss(self, labels):
        """concept_weight * BCE(c, M[y]) of the last forward (main.py adds it to CE + divergence)."""
        c = self._last['c']
        bce = F.binary_cross_entropy(c.clamp(EPS, 1 - EPS), self.a3.M[labels])
        terms = dict(concept_bce=bce.item())
        if self.stage == 'a3' and self.use_kg:
            terms.update({f'beta_{n}': b.item() for n, b in zip(PASS2, self.a3.beta)})
            terms['w_kg'] = F.softplus(self.a3.w_kg).mean().item()
        return self.concept_weight * bce, terms

    def eval_outputs(self):
        out = dict(concept_match=self.a3.concept_match(self._last['c'].detach()))
        if self._last['no_kg'] is not None:
            out['no_kg'] = self._last['no_kg'].detach()
        return out

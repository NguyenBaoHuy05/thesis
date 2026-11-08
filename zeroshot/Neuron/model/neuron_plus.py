"""Neuron++: Neuron's head with a stronger text side, description pooling and loss. Stage 1 of the upgrade.

skeleton -> Shift-GCN (frozen) -> cube (n, 256, T=16, V=25) -> Neuron visual head (unchanged, ModelMatch.visual_vectors)
         -> 3 spatial + 3 temporal unit vectors v (n, 768)
text   : Neuron's CLIP features of the LLM descriptions, 3 granularities per branch, (C, 10, 768) each
  (a) text adapter      : residual low-rank map on every description embedding (adapter_rank > 0), zero-initialised
  (b) class graph       : nodes = classes, edges = cosine between the mean fine-grained embeddings (spatial ||
                          temporal), top-k softmax. prop_alpha > 0 mixes the class mean of the neighbours into
                          every description embedding:  E'_c,j = norm((1-a) E_c,j + a * sum_v A_cv m_v)
                          The same graph gives the hard negatives of the losses (neg_mode graph).
  (c) description pooling: agg = mean (Neuron: its top-10 over 10 descriptions is the plain mean) | max | lse
                          (learned-temperature log-sum-exp, soft max over descriptions)
score  : cosine (n, C) for each of the 6 outputs. Prediction = (spa[2] + tem[2]) / 2 (one prediction per sample).

Losses (every one is applied to each of the 6 outputs, then averaged, as Neuron does with its CE)
  ce    : cross-entropy of  scale * cos  (Neuron: scale 0.1 fixed, which keeps its softmax near uniform, CE ~ ln C;
          ce_scale_learn=True with ce_scale ~ 10 is the CLIP-style temperature)
  xnce  : InfoCPL eq. 5 (X-sampling InfoNCE): the text of y_i against all samples of the batch, same-class samples
          masked out; with `ce` this makes the symmetric InfoNCE of DynaPURLS
  cal   : FS-VAE eq. 6 calibrated alignment, sigma((d_pos - d_neg) / tau), both directions
          (skeleton anchor vs text negatives, text anchor vs skeleton negatives of the batch), d = ||a - b||
  tri   : TDSM eq. 10 triplet margin on the cosines, max(cos_neg - cos_pos + margin, 0)
  negatives of cal / tri: neg_mode = graph (the graph_k nearest classes of the true class) | random (graph_k
  random other classes) | all (every other class).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shiftgcn_match import ModelMatch


def _as3(t):
    """(C, 768) -> (C, 1, 768); (C, K, 768) unchanged."""
    return t if t.dim() == 3 else t[:, None]


def class_graph(spatial, temporal, k, temp):
    """Class graph from the fine-grained semantics (index 2 of each list) of the classes given.
    Returns A (C, C) row-stochastic with k non-zero entries per row (no self edge) and nbr (C, k) the neighbour ids."""
    with torch.no_grad():
        mean = lambda t: F.normalize(F.normalize(_as3(t), dim=-1).mean(1), dim=-1)
        m = F.normalize(torch.cat([mean(spatial[2]), mean(temporal[2])], 1), dim=-1)
        S = m @ m.T
        S.fill_diagonal_(-2.0)                                    # no self edge
        k = max(1, min(k, S.shape[0] - 1))
        val, nbr = S.topk(k, dim=1)
        A = torch.zeros_like(S).scatter(1, nbr, F.softmax(val / temp, dim=1))
    return A, nbr


def propagate(sem, A, alpha):
    """sem: list of (C, K, 768) unit vectors; mixes the neighbours' class mean into every description."""
    out = []
    for t in sem:
        m = F.normalize(t.mean(1), dim=-1)                        # (C, 768)
        nb = A @ m
        out.append(F.normalize((1 - alpha) * t + alpha * nb[:, None], dim=-1))
    return out


class LowRank(nn.Module):
    """x + up(down(x)), up zero-initialised: starts as the identity."""

    def __init__(self, d, r):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, d, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))


def neg_mask(y, nbr, n_cls, mode, k):
    """(n, C) bool: the classes that count as negatives of each sample's class y."""
    n = len(y)
    m = torch.zeros(n, n_cls, dtype=torch.bool, device=y.device)
    if mode == 'graph':
        idx = nbr[y]; m.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
    elif mode == 'random':
        r = torch.rand(n, n_cls, device=y.device)
        r.scatter_(1, y[:, None], -1.0)
        idx = r.topk(min(k, n_cls - 1), dim=1).indices; m.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
    elif mode == 'all':
        m[:] = True
    else:
        raise ValueError(f'neg_mode must be graph, random or all, got {mode}')
    m.scatter_(1, y[:, None], torch.zeros_like(y[:, None], dtype=torch.bool))
    return m


def xnce_loss(cos, y, scale):
    """InfoCPL eq. 5. logits[i, j] = scale * cos(sample j, class y_i); positive j = i; other samples of the
    same class are masked out of the denominator."""
    n = len(y)
    logits = scale * cos[:, y].T
    idx = torch.arange(n, device=y.device)
    same = (y[:, None] == y[None, :]) & ~torch.eye(n, dtype=torch.bool, device=y.device)
    return F.cross_entropy(logits.masked_fill(same, -1e4), idx)


def cal_loss(cos, y, mask, tau):
    """FS-VAE eq. 6, both directions. d = ||a - b|| of unit vectors = sqrt(2 - 2 cos)."""
    d = (2 - 2 * cos).clamp(min=1e-6).sqrt()                      # (n, C)
    dpos = d.gather(1, y[:, None])                                # (n, 1)
    t1 = torch.sigmoid((dpos - d) / tau)                          # skeleton anchor, text negatives
    l1 = (t1 * mask).sum() / mask.sum().clamp(min=1)
    pair = mask[:, y]                                             # [i, j]: class of sample j is a negative of i's
    dtext = d[:, y].T                                             # [i, j] = d(sample j, text y_i)
    t2 = torch.sigmoid((dpos - dtext) / tau)                      # text anchor, skeleton negatives in the batch
    l2 = (t2 * pair).sum() / pair.sum().clamp(min=1)
    return (l1 + l2) / 2


def tri_loss(cos, y, mask, margin):
    """TDSM eq. 10 on the cosines: max(cos_neg - cos_pos + margin, 0) over the negatives."""
    l = F.relu(cos - cos.gather(1, y[:, None]) + margin) * mask
    return l.sum() / mask.sum().clamp(min=1)


class NeuronPlus(ModelMatch):
    def __init__(self, num_class=55, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 n_cls_total=60, freeze_bn=True,
                 agg='mean', agg_temp=0.1,
                 ce_scale=0.1, ce_scale_learn=False,
                 adapter_rank=0,
                 prop_alpha=0.0, graph_k=5, graph_temp=0.1,
                 neg_mode='graph', cal_tau=0.1, tri_margin=0.2):
        super().__init__(num_class, num_point, num_person, graph, graph_args, in_channels)   # encoder frozen here
        self.freeze_bn = freeze_bn
        assert agg in ('mean', 'max', 'lse'), agg
        self.agg = agg
        self.log_agg_temp = nn.Parameter(torch.tensor(math.log(agg_temp)), requires_grad=agg == 'lse')
        ls = torch.tensor(math.log(ce_scale))
        if ce_scale_learn:
            self.log_ce_scale = nn.Parameter(ls)
        else:
            self.register_buffer('log_ce_scale', ls)
        self.adapters = nn.ModuleDict()
        if adapter_rank > 0:
            for b in ('s', 't'):
                for r in range(3):
                    self.adapters[f'{b}{r}'] = LowRank(768, adapter_rank)
        self.prop_alpha, self.graph_k, self.graph_temp = prop_alpha, graph_k, graph_temp
        self.neg_mode, self.cal_tau, self.tri_margin = neg_mode, cal_tau, tri_margin

    # ---------------------------------------------------------------- parameters
    def plus_parameters(self):
        """New parameters (AdamW); the Neuron head keeps Neuron's SGD."""
        ps = list(self.adapters.parameters())
        for p in (self.log_agg_temp, self.log_ce_scale):
            if isinstance(p, nn.Parameter) and p.requires_grad:
                ps.append(p)
        return ps

    def neuron_parameters(self):
        new = {id(p) for p in self.plus_parameters()}
        return [p for p in self.parameters() if p.requires_grad and id(p) not in new]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:                       # frozen encoder: keep its BatchNorm statistics fixed too
            self.feature_extractor.eval()
        return self

    # ---------------------------------------------------------------- text side
    def prepare(self, spatial_round, temporal_round, alpha=None):
        """Neuron semantics of the classes given (lists of 3 tensors (C, 10, 768)) -> adapted / propagated unit
        vectors (two lists of 3 tensors) and the neighbour ids (C, k) of the class graph."""
        alpha = self.prop_alpha if alpha is None else alpha
        sem = {}
        for b, rounds in (('s', spatial_round), ('t', temporal_round)):
            sem[b] = [F.normalize(_as3(t), dim=-1) for t in rounds]
        A, nbr = class_graph(sem['s'], sem['t'], self.graph_k, self.graph_temp)    # graph from the raw CLIP text
        for b in ('s', 't'):
            if self.adapters:
                sem[b] = [F.normalize(self.adapters[f'{b}{r}'](t), dim=-1) for r, t in enumerate(sem[b])]
            if alpha > 0:
                sem[b] = propagate(sem[b], A, alpha)
        return sem['s'], sem['t'], nbr

    def pool(self, cos):
        """(n, C, K) cosines to the K descriptions of each class -> (n, C)."""
        if self.agg == 'mean':
            return cos.mean(2)
        if self.agg == 'max':
            return cos.max(2).values
        t = self.log_agg_temp.exp().clamp(min=1e-3)
        return t * (torch.logsumexp(cos / t, dim=2) - math.log(cos.shape[2]))

    def score(self, vis, sem):
        return [self.pool(torch.einsum('nd,ckd->nck', v, s)) for v, s in zip(vis, sem)]

    # ---------------------------------------------------------------- forward
    def forward(self, x, spatial_round, temporal_round, classes=None):
        """x: (n, 3, T, V, M); *_round: Neuron class semantics of the classes to score."""
        _, _, cube = self.feature_extractor(x)                                        # (n, 256, 16, 25)
        return self.forward_cube(cube, spatial_round, temporal_round)

    def forward_cube(self, cube, spatial_round, temporal_round, classes=None, extra_alphas=()):
        """From a (cached) encoder cube. out['spa'] / out['tem']: 3 cosine matrices (n, C) each (coarse -> fine).
        extra_alphas: also score the fine-grained round with these propagation strengths (diagnosis; the visual
        head runs once)."""
        sp_v, tp_v = self.visual_vectors(cube)
        sem_s, sem_t, nbr = self.prepare(spatial_round, temporal_round)
        out = dict(spa=self.score(sp_v, sem_s), tem=self.score(tp_v, sem_t), nbr=nbr)
        if extra_alphas:
            out['extra'] = {}
            for a in extra_alphas:
                s, t, _ = self.prepare(spatial_round, temporal_round, alpha=a)
                out['extra'][a] = (self.score(sp_v[2:], s[2:])[0] + self.score(tp_v[2:], t[2:])[0]) / 2
        return out

    @staticmethod
    def s_neu(out):
        """Fine-grained spatial + temporal, one prediction per sample."""
        return (out['spa'][2] + out['tem'][2]) / 2

    # ---------------------------------------------------------------- losses
    def losses(self, out, y, lambda_ce=1.0, lambda_xnce=0.0, lambda_cal=0.0, lambda_tri=0.0):
        """y: labels re-indexed to the columns of the scored classes. Every loss: mean over the 6 outputs."""
        scale = self.log_ce_scale.exp().clamp(max=100)
        cos_list = list(out['spa']) + list(out['tem'])
        n_cls = cos_list[0].shape[1]
        mask = None
        if lambda_cal > 0 or lambda_tri > 0:
            mask = neg_mask(y, out['nbr'], n_cls, self.neg_mode, self.graph_k)
        L, k = {}, len(cos_list)
        if lambda_ce > 0:
            L['ce'] = lambda_ce * sum(F.cross_entropy(c * scale, y) for c in cos_list) / k
        if lambda_xnce > 0:
            L['xnce'] = lambda_xnce * sum(xnce_loss(c, y, scale) for c in cos_list) / k
        if lambda_cal > 0:
            L['cal'] = lambda_cal * sum(cal_loss(c, y, mask, self.cal_tau) for c in cos_list) / k
        if lambda_tri > 0:
            L['tri'] = lambda_tri * sum(tri_loss(c, y, mask, self.tri_margin) for c in cos_list) / k
        return L

    @torch.no_grad()
    def second_person(self, x):
        """(n,) bool from the skeleton x (n, 3, T, V, M): person 2 has non-zero joints in > half of the frames.
        Kept because the cube cache stores it (cache files are reused)."""
        return (x[..., 1].abs().sum((1, 3)) > 0).float().mean(1) > 0.5

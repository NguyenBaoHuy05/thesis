"""A3-G: predicate-defined, per-frame hyperedges inside Hyper-GCN large
(plan: KG_RETRIEVAL_PLAN/A3G_PREDICATE_HYPEREDGE/A3G_PLAN.md).

    F6   = l1 .. l6 (K0, frozen)
    p, B = GeoPredicates(x)                       p [N·M, E=21, 64] in [0, 1], B [N·M, E, 25]; no gradient
    H[n, t, v, e] = B[n, e, v] · p_t[n, e]         p_t = p avg-pooled to the T the layer's GCN sees (l7: 32, l8/l9: 16)
    G_geo[n, t] = hyper_norm(H[n, t], softplus(w_geo))      [N·M, T, 28, 28], virtual joints 0, shared by the 8 subsets
    y    = fc(pool(l7' .. l9'(F6, temporal_adjacency = beta_l · G_geo)))
l7'..l9' start as copies of K0's l7..l9 and beta = 0, so the output at initialization is K0's.

mode (a3g_args): geo (main model) | shuffle (p of sample n + 1 in the batch) | random (B = random joint sets of the
same sizes) | input (no G_geo; F6 += Linear(E -> C)(p_32), zero-initialized, same at every joint) | off (K0-ft).
eval_outputs: no_geo (same checkpoint, beta = 0) for geo / shuffle / random. One GPU only (the last forward is kept).
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.geo_predicates import HYPEREDGES, GeoPredicates
from model.hypergcn_a3 import PASS2, hyper_norm
from model.hypergcn_large import Model as HyperGCN

MODES = ('geo', 'shuffle', 'random', 'input', 'off')
GRAPH_MODES = ('geo', 'shuffle', 'random')


def run_top(layers, x, temporal=(None, None, None)):
    """l7, l8, l9 on [N*M, C, T, V] with the residual of l9 from l7; temporal: one adjacency (or None) per layer."""
    (l7, l8, l9), (g7, g8, g9) = layers, temporal
    x7, h7 = l7(x, temporal_adjacency=g7)
    x8, h8 = l8(x7, temporal_adjacency=g8)
    x9, h9 = l9(x8 + x7, temporal_adjacency=g9)
    return x9, (h7, h8, h9)


class A3GHead(nn.Module):
    """Predicates, hyperedge weights, beta and the pass-2 layers (all under the prefix a3g.)."""

    def __init__(self, in_channels, layers, virtual_num, mode='geo', stats=None, seed=0):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.mode, self.seed, self.virtual_num = mode, int(seed), int(virtual_num)
        self.geo = GeoPredicates(stats if mode != 'off' else None)
        self.w_geo = nn.Parameter(torch.full((len(HYPEREDGES),), float(torch.log(torch.expm1(torch.tensor(1.0))))))
        self.beta = nn.Parameter(torch.zeros(len(PASS2)))
        self.layers = nn.ModuleDict(layers)
        self.input_proj = None
        if mode == 'input':
            self.input_proj = nn.Linear(len(HYPEREDGES), in_channels)
            nn.init.zeros_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

    def hyperedges(self, x):
        """x [N, 3, T, 25, M] -> (p [N·M, E, T], B [N·M, E, 25]) after the control of this mode."""
        p, B = self.geo(x, random=self.mode == 'random', seed=self.seed)
        if self.mode == 'shuffle':
            p = p.roll(-1, dims=0)                                       # p of sample n + 1, B of sample n
        return p.flatten(0, 1), B.flatten(0, 1)

    def adjacency(self, p, B, T):
        """G_geo [N·M, T, V + virtual, V + virtual] on T frames."""
        p_t = F.adaptive_avg_pool1d(p, T)                                # [N·M, E, T]
        H = B.transpose(1, 2).unsqueeze(1) * p_t.transpose(1, 2).unsqueeze(2)     # [N·M, T, V, E]
        H = F.pad(H, (0, 0, 0, self.virtual_num))                        # virtual joints: no hyperedge
        W = F.softplus(self.w_geo).expand(*H.shape[:2], -1)
        return hyper_norm(H, W)


class Model(HyperGCN):
    def __init__(self, a3g_args=None, **kwargs):
        super().__init__(**kwargs)
        args = dict(a3g_args or {})
        mode = args.pop('mode', 'geo')
        layers = {name: copy.deepcopy(getattr(self, name)) for name in PASS2}
        self.a3g = A3GHead(self.embedding_channels * 2, layers, virtual_num=kwargs.get('hyper_joints', 0),
                           mode=mode, stats=args.pop('stats', None), seed=args.pop('seed', 0))
        if args:
            raise ValueError(f'unknown a3g_args: {sorted(args)}')
        self.mode = mode
        self._last = {}

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Fill a3g.layers.l7..l9 from l7..l9 when the state_dict has no pass-2 layers (official checkpoint)."""
        state = dict(state_dict)
        for name in PASS2:
            prefix = f'a3g.layers.{name}.'
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
        f6 = f6.flatten(0, 1)
        layers = tuple(self.a3g.layers[name] for name in PASS2)
        temporal = (None, None, None)
        if self.mode != 'off':
            p, B = self.a3g.hyperedges(x)
        if self.mode == 'input':
            p6 = F.adaptive_avg_pool1d(p, f6.shape[2])                   # [N·M, E, T6]
            f6 = f6 + self.a3g.input_proj(p6.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        elif self.mode in GRAPH_MODES:
            t7 = f6.shape[2]                                             # l7's GCN runs before its stride-2 TCN
            t8 = (t7 - 1) // 2 + 1                                       # l8 and l9 see the strided length
            g7, g8 = self.a3g.adjacency(p, B, t7), self.a3g.adjacency(p, B, t8)
            temporal = tuple(b * g for b, g in zip(self.a3g.beta, (g7, g8, g8)))
        p2, h2 = run_top(layers, f6, temporal)
        output = self.fc(self.drop_out(self.pool(p2, N)))
        self._last = {}
        if self.mode in GRAPH_MODES and not self.training:
            self._last['no_geo'] = self.fc(self.pool(run_top(layers, f6)[0], N)).detach()
        return output, hyper_outputs + h2

    def extra_loss(self, labels):
        """No extra loss; logs beta per layer and the mean hyperedge weight."""
        terms = {}
        if self.mode in GRAPH_MODES:
            terms = {f'beta_{n}': b.item() for n, b in zip(PASS2, self.a3g.beta)}
            terms['w_geo'] = F.softplus(self.a3g.w_geo).mean().item()
        return self.a3g.beta.new_zeros(()), terms

    def eval_outputs(self):
        return dict(self._last)

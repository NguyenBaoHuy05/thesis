"""Stratified R-GCN over the concept–action graph (plan §6.2, drawio ⑤).

One R-GCN, trained end-to-end. Node states are per sample: [N, nodes of a type, d]. The relation weights use a
basis decomposition W_r = sum_b a_rb V_b (Schlichtkrull et al. 2018) and each relation's adjacency is
row-normalized (1 / c_{v,r}). Messages follow the order of the logic, so there is no cycle:

    B1: concept -> AND (r1), OR (r2), NOT(j|i) (r8)
    B2: action (initial state) -> NOT(j|i) (r6, auxiliary; use_r6=False drops it)
    B3: concept (r3), AND (r4), OR (r5), NOT(j|i) (r7) -> action

Each step updates only its target types: h_t <- LN(h_t + GELU(W0_t h_t + sum_r A_r h_src W_r)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

RELATIONS = ('r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8')
# relation -> (source type, target type)
ENDS = dict(r1=('concept', 'motif'), r2=('concept', 'orr'), r3=('concept', 'action'), r4=('motif', 'action'),
            r5=('orr', 'action'), r6=('action', 'notn'), r7=('notn', 'action'), r8=('concept', 'notn'))
STEPS = ((('r1', 'r2', 'r8'), ('motif', 'orr', 'notn')),
         (('r6',), ('notn',)),
         (('r3', 'r4', 'r5', 'r7'), ('action',)))


def node_counts(graph):
    return dict(concept=len(graph.concepts), action=len(graph.actions), motif=len(graph.motifs),
                orr=len(graph.ors), notn=len(graph.nots))


def relation_adjacency(graph):
    """{relation: [targets, sources] row-normalized float tensor}."""
    counts = node_counts(graph)
    out = {}
    for r, pairs in graph.edges().items():
        src, tgt = ENDS[r]
        A = torch.zeros(counts[tgt], counts[src])
        for s, t in pairs:
            A[t, s] = 1.0
        out[r] = A / A.sum(1, keepdim=True).clamp_min(1.0)
    return out


class StratifiedRGCN(nn.Module):
    def __init__(self, graph, d=256, bases=4, use_r6=True, dropout=0.1):
        super().__init__()
        self.relations = tuple(r for r in RELATIONS if use_r6 or r != 'r6')
        for r, A in relation_adjacency(graph).items():
            self.register_buffer(f'adj_{r}', A)
        self.bases = nn.Parameter(torch.randn(bases, d, d) * d ** -0.5)
        self.coef = nn.Parameter(torch.randn(len(RELATIONS), bases) / bases ** 0.5)
        self.self_loop = nn.ModuleDict({t: nn.Linear(d, d) for _, targets in STEPS for t in targets})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(d) for _, targets in STEPS for t in targets})
        self.drop = nn.Dropout(dropout)

    def weight(self, r):
        return torch.einsum('b,bij->ij', self.coef[RELATIONS.index(r)], self.bases)

    def forward(self, h):
        """h: {type: [N, n_type, d]} initial states (concept, action, motif, orr, notn) -> updated dict.
        The action state that B2 reads is the initial one (B3 runs after B2)."""
        h = dict(h)
        for rels, targets in STEPS:
            msgs = {t: 0 for t in targets}
            for r in rels:
                if r not in self.relations:
                    continue
                src, tgt = ENDS[r]
                A = getattr(self, f'adj_{r}')
                if A.numel() == 0 or h[src].shape[1] == 0:
                    continue
                msgs[tgt] = msgs[tgt] + torch.einsum('ts,nsd->ntd', A, h[src]) @ self.weight(r)
            for t in targets:
                if h[t].shape[1] == 0:
                    continue
                h[t] = self.norm[t](h[t] + self.drop(F.gelu(self.self_loop[t](h[t]) + msgs[t])))
        return h

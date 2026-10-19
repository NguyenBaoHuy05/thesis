"""Fuzzy-logic executor of the concept–action graph (plan §6.1).

Everything is computed in log space on c in (0, 1)^K (clamped to [EPS, 1 - EPS]):

    AND_g     = sum_k a_gk log c_k / sum_k a_gk                        weighted geometric mean (motif)
    OR_o      = log(1 - prod_k (1 - c_k)^a_ok)                         noisy-OR
    NOT(j|i)  = log(1 - exp(AND over M_j minus M_i))
    pos_i     = weighted mean of {AND_g, OR_o, log c_k (direct)} entering action i
    neg_i     = weighted mean of {NOT(j|i)}                            (0 if i has no partner)
    glob_i    = mean over concepts outside row i (and outside its OR groups) of log(1 - c_k)   (closed world, CL v2)
    log s_i   = (pos_i + nu [i has a partner] neg_i + omega glob_i) / (1 + nu [i has a partner] + omega)

With no motif, no OR gate and no NOT node this is exactly model/concept_logic.py: logic_scores(balance=True).
Edge weights a = softplus(theta), theta initialized so that a = 1; learned only with learn_alpha=True.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6
ALPHA_INIT = math.log(math.e - 1)          # softplus(ALPHA_INIT) = 1


def log1mexp(x):
    """log(1 - exp(x)) for x < 0, stable on both sides of log 2."""
    x = x.clamp(max=-EPS)
    return torch.where(x > -math.log(2), torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def graph_masks(graph):
    """Dense 0/1 masks of the graph: dict of float tensors."""
    A, K = len(graph.actions), len(graph.concepts)
    G, O, N = len(graph.motifs), len(graph.ors), len(graph.nots)
    m = dict(motif=torch.zeros(G, K), orr=torch.zeros(O, K), notc=torch.zeros(N, K),
             direct=torch.zeros(A, K), act_motif=torch.zeros(A, G), act_or=torch.zeros(A, O),
             act_not=torch.zeros(A, N), row=torch.zeros(A, K), absent=torch.ones(A, K))
    for g, members in enumerate(graph.motifs):
        m['motif'][g, members] = 1
    for o, members in enumerate(graph.ors):
        m['orr'][o, members] = 1
    for n, (i, _j, diff) in enumerate(graph.nots):
        m['notc'][n, diff] = 1
        m['act_not'][i, n] = 1
    for a in range(A):
        m['direct'][a, graph.action_direct[a]] = 1
        m['act_motif'][a, graph.action_motifs[a]] = 1
        m['act_or'][a, graph.action_ors[a]] = 1
        m['row'][a, graph.rows[a]] = 1
        m['absent'][a, graph.rows[a]] = 0
        for o in graph.action_ors[a]:
            m['absent'][a, graph.ors[o]] = 0       # an OR alternative is not "absent"
    return m


class FuzzyGraphExecutor(nn.Module):
    WEIGHTED = ('motif', 'orr', 'notc', 'direct', 'act_motif', 'act_or', 'act_not')

    def __init__(self, graph, omega=1.0, nu=1.0, learn_alpha=False):
        super().__init__()
        self.omega, self.nu, self.learn_alpha = float(omega), float(nu), bool(learn_alpha)
        for name, mask in graph_masks(graph).items():
            self.register_buffer(f'mask_{name}', mask)
        self.theta = nn.ParameterDict({
            name: nn.Parameter(torch.full_like(getattr(self, f'mask_{name}'), ALPHA_INIT), requires_grad=learn_alpha)
            for name in self.WEIGHTED})

    def weight(self, name):
        mask = getattr(self, f'mask_{name}')
        return F.softplus(self.theta[name]) * mask if self.learn_alpha else mask

    def alpha_penalty(self):
        """sum |a - 1| over the existing edges (plan §6.3, eta)."""
        if not self.learn_alpha:
            return self.mask_row.new_zeros(())
        return sum(((F.softplus(self.theta[n]) - 1).abs() * getattr(self, f'mask_{n}')).sum() for n in self.WEIGHTED)

    @staticmethod
    def _mean(values, weights):
        """values [N, X], weights [Y, X] -> [N, Y] weighted mean (0 where a row has no weight)."""
        return values @ weights.t() / weights.sum(1).clamp_min(EPS)

    def nodes(self, c):
        """Log truth values of every node: dict(motif [N, G], orr [N, O], notv [N, Nn])."""
        c = c.clamp(EPS, 1 - EPS)
        log_c, log_1mc = torch.log(c), torch.log1p(-c)
        motif = self._mean(log_c, self.weight('motif'))
        orr = log1mexp(log_1mc @ self.weight('orr').t())
        notv = log1mexp(self._mean(log_c, self.weight('notc')))
        return dict(log_c=log_c, log_1mc=log_1mc, motif=motif, orr=orr, notv=notv)

    def forward(self, c):
        """c [N, K] -> log s [N, A] (<= 0)."""
        v = self.nodes(c)
        wd, wm, wo, wn = (self.weight(n) for n in ('direct', 'act_motif', 'act_or', 'act_not'))
        pos = (v['log_c'] @ wd.t() + v['motif'] @ wm.t() + v['orr'] @ wo.t()) \
            / (wd.sum(1) + wm.sum(1) + wo.sum(1)).clamp_min(EPS)
        has = (self.mask_act_not.sum(1) > 0).float()
        neg = self._mean(v['notv'], wn)
        glob = self._mean(v['log_1mc'], self.mask_absent)
        return (pos + self.nu * has * neg + self.omega * glob) / (1 + self.nu * has + self.omega)

    @torch.no_grad()
    def trace(self, c, action):
        """Proof trace of one sample (c [K]) for one action: truth value of every input of that action."""
        v = {k: t[0] for k, t in self.nodes(c[None]).items()}
        out = dict(direct={}, motif={}, orr={}, nots={})
        for k in self.mask_direct[action].nonzero().flatten().tolist():
            out['direct'][k] = v['log_c'][k].exp().item()
        for g in self.mask_act_motif[action].nonzero().flatten().tolist():
            out['motif'][g] = v['motif'][g].exp().item()
        for o in self.mask_act_or[action].nonzero().flatten().tolist():
            out['orr'][o] = v['orr'][o].exp().item()
        for n in self.mask_act_not[action].nonzero().flatten().tolist():
            out['nots'][n] = v['notv'][n].exp().item()
        return out

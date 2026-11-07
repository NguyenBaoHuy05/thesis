"""Phase 2 rule branch: primitives grounded on body part x time features (P=8, T=16, 256).

Per-time primitive logits
  h_{p,t}     = TemporalConv(Linear(x_{p,t}) + part_emb_p + time_emb_t)          per part token
  l_k(t)      = max over the parts allowed for primitive k of (W h_{p,t})_k        side-invariant
                ("any hand" = max over left / right hand), parts from primitive.json 'part'
Rule truth of class y
  require k   : occurs at some time  -> l_k^E = tau * logmeanexp_t(l_k(t) / tau), truth sigmoid(l_k^E)
  forbid k    : never occurs         -> truth 1 - sigmoid(l_k^E)
  sequence    : k_1 then k_2 ... (strictly later times); best ordered path by dynamic programming,
                S_seq = max_{t1<...<tm} mean_j log sigmoid(l_{k_j}(t_j))
  s_rule(y)   = [sum_k w_k (R+ log p^E + R- log(1 - p^E)) + w_seq * m_y * S_seq(y)] / [sum_k w_k |R_yk| + w_seq * m_y]
Losses: L_cls (CE on beta * s_rule) + L_prim (BCE on l^E, KG targets) + L_cons (implication a -> b at
every time) + L_excl (exclusive groups: at most one primitive true at a time, penalty p_i(t) p_j(t)).
Exclusive groups made only of hand / arm primitives are skipped: the primitives are side-invariant
("any hand"), so the two hands may be in different members of the group at the same time.
"""
import json
import math
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import PART_NAMES
from phase1 import Normed, prim_bce

PRIM_PART_TOKENS = {
    'body': PART_NAMES, 'interaction': PART_NAMES,
    'torso': ['torso'], 'head': ['head'],
    'arm': ['l_arm', 'r_arm'], 'hand': ['l_hand', 'r_hand', 'l_arm', 'r_arm'],
    'leg': ['l_leg', 'r_leg'],
}
SIDE_INVARIANT_PARTS = ('hand', 'arm')


def load_kg_structure(rules_path, prim_path):
    """Part mask (K, P), exclusive pairs (used / skipped groups), and sequence rules [(class0, [k...])]."""
    prims = json.load(open(prim_path, encoding='utf-8'))
    pidx = {p['id']: i for i, p in enumerate(prims)}
    part_mask = np.zeros((len(prims), len(PART_NAMES)), bool)
    for i, p in enumerate(prims):
        for name in PRIM_PART_TOKENS[p['part']]:
            part_mask[i, PART_NAMES.index(name)] = True
    groups = {}
    for i, p in enumerate(prims):
        if p.get('exclusive_group'):
            groups.setdefault(p['exclusive_group'], []).append(i)
    used = {g: m for g, m in groups.items() if not all(prims[k]['part'] in SIDE_INVARIANT_PARTS for k in m)}
    skipped = sorted(set(groups) - set(used))
    pairs = [pr for m in used.values() for pr in combinations(m, 2)]
    kg = json.load(open(rules_path, encoding='utf-8'))
    seqs = [(r['id'] - 1, [pidx[k] for k in r['sequence']]) for r in kg['rules'] if len(r.get('sequence', [])) >= 2]
    return dict(part_mask=part_mask, excl_pairs=np.array(pairs, np.int64).reshape(-1, 2), seqs=seqs,
                excl_used=sorted(used), excl_skipped=skipped)


class TempBlock(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 3, padding=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):                                  # h: (N, T, d)
        return h + self.drop(F.gelu(self.conv(self.ln(h).transpose(1, 2)).transpose(1, 2)))


class PartTimeRule(Normed):
    def __init__(self, R, imp, part_mask, excl_pairs, seqs, d_in=256, n_parts=8, n_time=16, d=256,
                 n_blocks=2, dropout=0.2, tau=1.0, beta_init=5.0, lam_prim=1.0, lam_cons=1.0, lam_excl=1.0,
                 use_seq=True, **_):
        super().__init__(d_in)
        R = torch.as_tensor(R)
        K = R.shape[1]
        self.register_buffer('Rp', (R > 0).float())
        self.register_buffer('Rn', (R < 0).float())
        self.register_buffer('imp', torch.as_tensor(imp))
        self.register_buffer('allow', torch.as_tensor(part_mask).T.contiguous())     # (P, K)
        self.register_buffer('excl', torch.as_tensor(excl_pairs))
        self.seqs = seqs if use_seq else []
        n_cls = R.shape[0]
        m = torch.zeros(n_cls)
        for c, ks in self.seqs:
            m[c] = len(ks)
        self.register_buffer('seq_len', m)                                            # (n_cls,)
        self.tau, self.lam_prim, self.lam_cons, self.lam_excl = tau, lam_prim, lam_cons, lam_excl
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout))
        self.part_emb = nn.Parameter(0.02 * torch.randn(n_parts, d))
        self.time_emb = nn.Parameter(0.02 * torch.randn(n_time, d))
        self.blocks = nn.ModuleList([TempBlock(d, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(d, K)
        self.w_raw = nn.Parameter(torch.full((K,), math.log(math.e - 1)))           # softplus -> 1
        self.w_seq_raw = nn.Parameter(torch.tensor(math.log(math.e - 1)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))

    def time_logits(self, x):
        """x: (B, P, T, C) raw -> per-time primitive logits (B, T, K)."""
        h = self.proj(self.norm(x)) + self.part_emb[None, :, None] + self.time_emb[None, None]
        B, P, T, d = h.shape
        h = h.reshape(B * P, T, d)
        for blk in self.blocks:
            h = blk(h)
        lg = self.head(h.reshape(B, P, T, d))                                         # (B, P, T, K)
        lg = lg.masked_fill(~self.allow[None, :, None, :], -1e4)
        return lg.max(1).values

    def seq_scores(self, lp, classes):
        """lp: (B, T, K) per-time log-truth. Returns S (B, C) and m (C,) for the scored classes."""
        B, C = lp.shape[0], len(classes)
        S = lp.new_zeros(B, C)
        if not self.seqs:
            return S, self.seq_len[classes]
        col = {int(c): j for j, c in enumerate(classes.tolist())}
        for c, ks in self.seqs:
            if c not in col:
                continue
            dp = lp[:, :, ks[0]]
            for k in ks[1:]:
                prev = torch.cummax(dp, dim=1).values                                 # best path ending <= t
                prev = torch.cat([prev.new_full((B, 1), -1e4), prev[:, :-1]], 1)      # strictly earlier
                dp = prev + lp[:, :, k]
            S[:, col[c]] = dp.max(1).values / len(ks)
        return S, self.seq_len[classes]

    def forward(self, x, classes):
        lt = self.time_logits(x)                                                      # (B, T, K)
        T = lt.shape[1]
        le = self.tau * (torch.logsumexp(lt / self.tau, 1) - math.log(T))           # (B, K) "at some time"
        w = F.softplus(self.w_raw)
        Rp, Rn = self.Rp[classes], self.Rn[classes]
        num = F.logsigmoid(le) @ (Rp * w).T + F.logsigmoid(-le) @ (Rn * w).T
        den = ((Rp + Rn) * w).sum(1)
        S, m = self.seq_scores(F.logsigmoid(lt), classes)
        ws = F.softplus(self.w_seq_raw)
        s_rule = (num + ws * m * S) / (den + ws * m).clamp(min=1e-6)
        return dict(lt=lt, le=le, s_rule=s_rule, z=self.log_beta.exp() * s_rule)

    def losses(self, out, y_local, y_global, stage):
        L = {}
        if stage == 2:
            L['cls'] = F.cross_entropy(out['z'], y_local)
        L['prim'] = self.lam_prim * prim_bce(out['le'], self.Rp[y_global], self.Rn[y_global])
        p = torch.sigmoid(out['lt'])                                                  # (B, T, K)
        L['cons'] = self.lam_cons * (p[..., self.imp[:, 0]] * (1 - p[..., self.imp[:, 1]])).mean()
        if len(self.excl):
            L['excl'] = self.lam_excl * (p[..., self.excl[:, 0]] * p[..., self.excl[:, 1]]).mean()
        return L


def make_parttime(R, imp, struct, device='cuda', **kw):
    return lambda: PartTimeRule(R, imp, struct['part_mask'], struct['excl_pairs'], struct['seqs'], **kw).to(device)

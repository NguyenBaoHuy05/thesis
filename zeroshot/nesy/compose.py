"""Rule-composed class prototypes for zero-shot skeleton action recognition.

Video, class text and primitives live in one CLIP text space (D = 768).

Video side (input: frozen Shift-GCN cube, P=8 body parts x T=16 times x 256):
  tokens  = Linear(x_{p,t}) + part_emb_p + time_emb_t  -> Transformer over the P*T tokens
  v       = normalize(W_g mean_{p,t} h)                 whole-video embedding
  v_t     = normalize(W_t mean_p h_{.,t})               per-time embeddings (for sequence rules)

Class side (a rule is a recipe for the class prototype):
  t_y     = normalize( c_y + a * mean_{k in require(y)} e_k - b * mean_{k in forbid(y)} e_k )
  c_y     = CLIP text of the class (fixed); e_k = primitive embedding, initialised from the CLIP text of
            its description in primitive.json and fine-tuned on seen classes with ||e_k - e_k^0||^2 kept small.
  Unseen classes are composed from the same e_k through their own rules.

Score:
  s(y)    = v . t_y + lam_seq * O_y
  O_y     = path(k1 -> ... -> km) - path(km -> ... -> k1)      only for classes with a sequence rule
  path    = max over t1 < ... < tm of mean_j v_{tj} . e_{kj}     (dynamic programming over time)
  O_y measures order only (presence is already in the require part), so it is ~0 when order is unclear.
  Since v . t_y is linear in the literals, every prediction splits into per-literal contributions.

Training: CE over seen classes on scale * s + lam_reg * mean_k ||e_k - e_k^0||^2.
Rule dropout: during training every literal of every class rule is dropped with probability p_drop.
use_rules=False gives the text-only baseline (t_y = c_y) with the same video encoder.
"""
import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from protocol import Normed, predict, zsl_acc


def path_score(sim, ks):
    """sim: (B, T, K). Best strictly-increasing time path through primitives ks; mean similarity, (B,)."""
    dp = sim[:, :, ks[0]]
    for k in ks[1:]:
        prev = torch.cummax(dp, dim=1).values                                    # best path ending <= t
        prev = torch.cat([prev.new_full((prev.shape[0], 1), -1e4), prev[:, :-1]], 1)   # strictly earlier
        dp = prev + sim[:, :, k]
    return dp.max(1).values / len(ks)


class RuleComposer(Normed):
    def __init__(self, class_text, prim_text, R, seqs, d_in=256, n_parts=8, n_time=16, d=256, n_layers=2,
                 n_heads=4, dropout=0.1, use_rules=True, use_seq=True, p_drop=0.2, lam_reg=1.0):
        super().__init__(d_in)
        self.use_rules, self.use_seq, self.p_drop, self.lam_reg = use_rules, use_seq, p_drop, lam_reg
        self.register_buffer('C', F.normalize(torch.as_tensor(class_text, dtype=torch.float32), dim=-1))
        self.register_buffer('E0', F.normalize(torch.as_tensor(prim_text, dtype=torch.float32), dim=-1))
        self.E = nn.Parameter(self.E0.clone())
        R = torch.as_tensor(R)
        self.register_buffer('Rp', (R > 0).float())
        self.register_buffer('Rn', (R < 0).float())
        self.seqs = dict(seqs)
        D = self.C.shape[1]

        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.GELU())
        self.part_emb = nn.Parameter(0.02 * torch.randn(n_parts, d))
        self.time_emb = nn.Parameter(0.02 * torch.randn(n_time, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.out_g = nn.Linear(d, D)
        self.out_t = nn.Linear(d, D)
        self.a = nn.Parameter(torch.tensor(1.0))           # weight of the require part
        self.b = nn.Parameter(torch.tensor(1.0))           # weight of the forbid part
        self.lam_seq = nn.Parameter(torch.tensor(1.0))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    # ------------------------------------------------------------------ class side
    def prim_emb(self):
        return F.normalize(self.E, dim=-1)

    def prototypes(self, classes):
        c = self.C[classes]
        if not self.use_rules:
            return F.normalize(c, dim=-1)
        Rp, Rn = self.Rp[classes], self.Rn[classes]
        if self.training and self.p_drop > 0:                                       # rule dropout
            Rp = Rp * (torch.rand_like(Rp) > self.p_drop)
            Rn = Rn * (torch.rand_like(Rn) > self.p_drop)
        E = self.prim_emb()
        req = (Rp @ E) / Rp.sum(1, keepdim=True).clamp(min=1)
        forb = (Rn @ E) / Rn.sum(1, keepdim=True).clamp(min=1)
        return F.normalize(c + self.a * req - self.b * forb, dim=-1)

    # ------------------------------------------------------------------ video side
    def encode(self, x):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):     # L4: bf16 for the encoder only
            h = self.proj(self.norm(x)) + self.part_emb[None, :, None] + self.time_emb[None, None]
            B, P, T, d = h.shape
            h = self.enc(h.reshape(B, P * T, d))
        h = self.ln(h.float()).reshape(B, P, T, d)
        v = F.normalize(self.out_g(h.mean((1, 2))), dim=-1)                          # (B, D)
        vt = F.normalize(self.out_t(h.mean(1)), dim=-1)                              # (B, T, D)
        return v, vt

    def order_scores(self, vt, classes):
        O = vt.new_zeros(vt.shape[0], len(classes))
        if not (self.use_rules and self.use_seq and self.seqs):
            return O
        sim = vt @ self.prim_emb().T                                                 # (B, T, K)
        col = {int(c): j for j, c in enumerate(classes.tolist())}
        for c, ks in self.seqs.items():
            if c in col:
                O[:, col[c]] = path_score(sim, ks) - path_score(sim, ks[::-1])
        return O

    def forward(self, x, classes):
        v, vt = self.encode(x)
        s = v @ self.prototypes(classes).T + self.lam_seq * self.order_scores(vt, classes)
        return dict(z=self.logit_scale.exp().clamp(max=100) * s)

    def losses(self, out, y_local, y_global, stage):
        L = dict(cls=F.cross_entropy(out['z'], y_local))
        if self.use_rules and self.lam_reg > 0:
            L['reg'] = self.lam_reg * ((self.prim_emb() - self.E0) ** 2).sum(1).mean()
        return L

    # ------------------------------------------------------------------ interventions
    @contextmanager
    def rules_override(self, R, seqs):
        """Temporarily replace the rule matrix and sequence rules (for intervention tests)."""
        old = (self.Rp.clone(), self.Rn.clone(), self.seqs)
        R = torch.as_tensor(R, device=self.Rp.device)
        self.Rp.copy_((R > 0).float())
        self.Rn.copy_((R < 0).float())
        self.seqs = dict(seqs)
        try:
            yield
        finally:
            self.Rp.copy_(old[0])
            self.Rn.copy_(old[1])
            self.seqs = old[2]


def make_composer(class_text, prim_text, R, seqs, device='cuda', **kw):
    return lambda: RuleComposer(class_text, prim_text, R, seqs, **kw).to(device)


def rule_interventions(R, seqs, n_perm=5, seed=0):
    """Returns extra_eval(model, ctx) for protocol.run_split. Test ZSL on unseen classes when
      zsl_no_rules  : the unseen rules are emptied (prototype = class text only, no sequence);
      zsl_shuffled  : the unseen rules (and sequences) are permuted among the unseen classes,
                      mean over n_perm derangements.
    A drop under both means the model really uses the rules of the unseen classes."""
    def extra_eval(model, ctx):
        if not model.use_rules:
            return {}
        X, y, idx, cls = ctx['X'], ctx['y'], ctx['idx_u'], ctx['cls_u']
        y_u, u = y[idx], cls.tolist()
        acc = lambda: zsl_acc(predict(model, X, idx, cls), y_u, cls)
        R0 = np.asarray(R)
        Rn = R0.copy()
        Rn[u] = 0
        with model.rules_override(Rn, {c: k for c, k in seqs.items() if c not in u}):
            no_rules = acc()
        rng, shuf = np.random.RandomState(seed), []
        for _ in range(n_perm):
            while True:
                perm = rng.permutation(len(u))
                if not (perm == np.arange(len(u))).any():
                    break
            Rs, Ss = R0.copy(), {c: k for c, k in seqs.items() if c not in u}
            for i, j in enumerate(perm):          # class u[i] gets the rule of u[j]
                Rs[u[i]] = R0[u[j]]
                if u[j] in seqs:
                    Ss[u[i]] = seqs[u[j]]
            with model.rules_override(Rs, Ss):
                shuf.append(acc())
        return dict(zsl_no_rules=no_rules, zsl_shuffled=float(np.mean(shuf)))
    return extra_eval

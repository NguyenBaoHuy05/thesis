"""Causal Concept Program (pipeline v4, docs/architecture_v4.html).

    x --HyperGCN--> logits_net, feat_l6
    x --geo_torch--> per-frame signals --ConceptLayer--> z (N, n_concepts) in [0, 1]
    feat_l6 --latent head--> latent concepts (part of z)
    per-frame concept logits --BeforeLayer--> before(a, b) truth values
    z, before --LogicProgram--> logits_logic
    fused = logits_logic + alpha * logits_net

LogicProgram: one weighted conjunction per class in the log domain (product t-norm):
    score_k = sum_c w+_kc log z_c + sum_c w-_kc log(1 - z_c) + sum_p u_kp log before_p + b_k
    logits_logic = gamma * score
w+, w-, u >= 0 (softplus). w+ starts at 1 for the concepts the CSV lists for class k and
at 0.02 elsewhere; L1 sparsity lets training drop CSV literals that do not hold and grow
new ones (knowledge revision, eval_ccp.py). alpha >= 0 is penalised so that the decision
goes through the logic layer.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sip.backbone import NUM_LAYERS, head, init_state, run_layers
from sip.concept_bank import ConceptBank
from sip.geo_torch import NUM_SIGNALS, SIGNAL_UNITS, SIGNALS, geo_signals

__all__ = ["EMPTY_Q", "masked_softmax_pool", "masked_mean", "ConceptLayer", "BeforeLayer",
           "LogicProgram", "CCP", "build_pairs", "backbone_pass", "calibrate", "export_rules"]

EMPTY_Q = -3.0   # quantified value of a concept with no defined frame (standardised units)


def _inv_softplus(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    return torch.log(torch.expm1(v))


def masked_softmax_pool(u, m, kappa):
    """Soft maximum over the last axis restricted to mask m:
    max + log(mean_t exp(kappa (u_t - max))) / kappa, EMPTY_Q where m is empty."""
    um = u.masked_fill(~m, -1e4)
    mx = um.amax(-1, keepdim=True).detach().clamp(min=-1e3)
    e = torch.exp(kappa * (um - mx)) * m
    cnt = m.sum(-1).clamp_min(1)
    q = mx.squeeze(-1) + torch.log(e.sum(-1).clamp_min(1e-30) / cnt) / kappa
    return torch.where(m.any(-1), q, torch.full_like(q, EMPTY_Q))


def masked_mean(u, m):
    cnt = m.sum(-1)
    q = (u * m).sum(-1) / cnt.clamp_min(1)
    return torch.where(cnt > 0, q, torch.full_like(q, EMPTY_Q))


class ConceptLayer(nn.Module):
    def __init__(self, bank: ConceptBank, c6: int, kappa: float = 4.0, latent_hidden: int = 128):
        super().__init__()
        sig, dirs, mostly = bank.geo_tables()
        gi, li = bank.geo_index, bank.latent_index
        self.n_concepts = len(bank.specs)
        self.kappa = kappa
        self.register_buffer("geo_index", torch.as_tensor(gi))
        self.register_buffer("latent_index", torch.as_tensor(li))
        order = np.concatenate([gi, li])
        self.register_buffer("inv_order", torch.as_tensor(np.argsort(order)))
        self.register_buffer("sig_index", torch.as_tensor(sig))
        self.register_buffer("dirs", torch.as_tensor(dirs))
        self.register_buffer("mostly", torch.as_tensor(mostly))
        self.register_buffer("sig_mu", torch.zeros(NUM_SIGNALS))
        self.register_buffer("sig_sd", torch.ones(NUM_SIGNALS))
        G, Lt = len(gi), len(li)
        self.tau = nn.Parameter(torch.zeros(G))
        self.t_raw = nn.Parameter(torch.full((G,), float(_inv_softplus(0.45))))
        self.latent = nn.Sequential(
            nn.Conv1d(c6, latent_hidden, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Conv1d(latent_hidden, Lt, 1))

    def temperature(self):
        return F.softplus(self.t_raw) + 0.05

    def geo_u(self, S, Mk):
        """Standardised, direction-signed per-frame signal of every geo concept: (N, G, T)."""
        u = S[:, self.sig_index]
        mu = self.sig_mu[self.sig_index][:, None]
        sd = self.sig_sd[self.sig_index][:, None]
        return self.dirs[:, None] * (u - mu) / sd, Mk[:, self.sig_index]

    def geo_q(self, S, Mk):
        u, m = self.geo_u(S, Mk)
        q = torch.where(self.mostly[None], masked_mean(u, m), masked_softmax_pool(u, m, self.kappa))
        return q, u, m

    def geo_logit(self, S, Mk):
        """(N, G) concept logits of the geo concepts only (used by the counterfactual generator)."""
        q, _, _ = self.geo_q(S, Mk)
        return (q - self.tau) / self.temperature()

    def forward(self, S, Mk, feat6, present, valid):
        N, _, T = S.shape
        q, u, m = self.geo_q(S, Mk)
        temp = self.temperature()
        g_logit = (q - self.tau) / temp
        g_frame = (u - self.tau[:, None]) / temp[:, None]

        # latent concepts from layer-6 features: mean over joints, then over present persons
        nm, C, T6, V = feat6.shape
        M = nm // N
        f = feat6.mean(-1).view(N, M, C, T6)
        w = present.to(f.dtype)
        w = w / w.sum(1, keepdim=True).clamp_min(1.0)
        f = (f * w[:, :, None, None]).sum(1)
        l_frame = F.interpolate(self.latent(f), size=T, mode="linear", align_corners=False)
        lm = valid[:, None, :].expand_as(l_frame)
        l_logit = masked_softmax_pool(l_frame, lm, 1.0)

        logit = torch.cat([g_logit, l_logit], 1)[:, self.inv_order]
        frame_logit = torch.cat([g_frame, l_frame], 1)[:, self.inv_order]
        frame_mask = torch.cat([m, lm], 1)[:, self.inv_order]
        return dict(logit=logit, z=torch.sigmoid(logit), frame_logit=frame_logit, frame_mask=frame_mask)


def build_pairs(bank: ConceptBank):
    """Ordered concept pairs (a, b) for before(a, b): both listed by the CSV for some class,
    both with a meaningful time of occurrence (geo "exists" or a non-temporal latent).
    Returns pairs (P, 2) int64 and pair_mask (K, P) float32 (pair p belongs to class k)."""
    def timed(i):
        s = bank.specs[i]
        if s.kind == "geo":
            return s.quant == "exists"
        return not s.name.startswith("temporal_")

    per_class = []
    for k in range(bank.csv_matrix.shape[0]):
        cs = [i for i in np.flatnonzero(bank.csv_matrix[k] > 0) if timed(i)]
        per_class.append([(a, b) for a in cs for b in cs if a != b])
    uniq = sorted({p for ps in per_class for p in ps})
    index = {p: j for j, p in enumerate(uniq)}
    mask = np.zeros((len(per_class), len(uniq)), dtype=np.float32)
    for k, ps in enumerate(per_class):
        for p in ps:
            mask[k, index[p]] = 1.0
    pairs = np.asarray(uniq, dtype=np.int64).reshape(-1, 2)
    return pairs, mask


class BeforeLayer(nn.Module):
    """before(a, b) = sigmoid((t_b - t_a) / tau_t) * z_a * z_b, t_c = softmax-weighted time
    (in [0, 1]) of concept c's per-frame logit."""

    def __init__(self, pairs, tau_t: float = 0.05):
        super().__init__()
        pairs = torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)
        self.register_buffer("a", pairs[:, 0].contiguous())
        self.register_buffer("b", pairs[:, 1].contiguous())
        self.tau_t = tau_t

    def _time(self, fl, fm):
        T = fl.shape[-1]
        t = torch.linspace(0.0, 1.0, T, device=fl.device, dtype=fl.dtype)
        w = torch.softmax(fl.masked_fill(~fm, -1e4), dim=-1)
        return (w * t).sum(-1)

    def forward(self, frame_logit, frame_mask, z):
        if self.a.numel() == 0:
            return z.new_zeros(z.shape[0], 0)
        ta = self._time(frame_logit[:, self.a], frame_mask[:, self.a])
        tb = self._time(frame_logit[:, self.b], frame_mask[:, self.b])
        return torch.sigmoid((tb - ta) / self.tau_t) * z[:, self.a] * z[:, self.b]


class LogicProgram(nn.Module):
    def __init__(self, csv_matrix, pair_mask, w_on: float = 1.0, w_off: float = 0.02):
        super().__init__()
        csv_t = torch.as_tensor(csv_matrix, dtype=torch.float32)
        K, C = csv_t.shape
        self.a_pos = nn.Parameter(_inv_softplus(torch.where(csv_t > 0, torch.tensor(w_on), torch.tensor(w_off))))
        self.a_neg = nn.Parameter(torch.full((K, C), float(_inv_softplus(w_off))))
        pm = torch.as_tensor(pair_mask, dtype=torch.float32)
        assert pm.dim() == 2 and pm.shape[0] == K, pm.shape
        self.register_buffer("pair_mask", pm)
        self.a_pair = nn.Parameter(torch.full(pm.shape, float(_inv_softplus(w_off))))
        self.bias = nn.Parameter(torch.zeros(K))
        self.log_gamma = nn.Parameter(torch.zeros(()))

    def weights(self):
        return F.softplus(self.a_pos), F.softplus(self.a_neg), F.softplus(self.a_pair) * self.pair_mask

    def forward(self, z, before):
        wp, wn, wb = self.weights()
        s = torch.log(z.clamp(1e-4, 1.0)) @ wp.t() + torch.log((1.0 - z).clamp(1e-4, 1.0)) @ wn.t()
        if before.shape[1] > 0:
            s = s + torch.log(before.clamp(1e-4, 1.0)) @ wb.t()
        return (s + self.bias) * torch.exp(self.log_gamma)

    def sparsity(self):
        wp, wn, wb = self.weights()
        return (wp.sum() + wn.sum() + wb.sum()) / wp.shape[0]


class CCP(nn.Module):
    def __init__(self, bank: ConceptBank, c6: int, use_before: bool = True, kappa: float = 4.0):
        super().__init__()
        self.concept_names = list(bank.names)
        self.class_names = list(bank.class_names)
        self.specs = bank.specs
        self.concepts = ConceptLayer(bank, c6, kappa=kappa)
        if use_before:
            pairs, pair_mask = build_pairs(bank)
        else:
            pairs = np.zeros((0, 2), dtype=np.int64)
            pair_mask = np.zeros((bank.csv_matrix.shape[0], 0), dtype=np.float32)
        self.before = BeforeLayer(pairs)
        self.logic = LogicProgram(bank.csv_matrix, pair_mask)
        self.a_gate = nn.Parameter(_inv_softplus(0.5))
        self.register_buffer("csv_matrix", torch.as_tensor(bank.csv_matrix))
        self.register_buffer("supervised", torch.as_tensor(bank.supervised))
        self.register_buffer("pairs", torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2))

    def alpha(self):
        return F.softplus(self.a_gate)

    def forward(self, x, feat6, logits_net=None, geo=None):
        S, Mk, info = geo if geo is not None else geo_signals(x)
        present = info["valid_all"].any(-1)
        c = self.concepts(S, Mk, feat6, present, info["valid"])
        b = self.before(c["frame_logit"], c["frame_mask"], c["z"])
        logic = self.logic(c["z"], b)
        alpha = self.alpha()
        fused = logic if logits_net is None else logic + alpha * logits_net
        return dict(z=c["z"], logit_z=c["logit"], before=b, logic=logic, fused=fused,
                    alpha=alpha, info=info)

    def concept_loss(self, logit_z, y, smooth: float = 0.05):
        """MIL loss: the clip-level (quantified) truth value of every CSV concept should
        match the CSV row of the true class (label smoothing for class-level noise)."""
        sup = self.supervised
        target = self.csv_matrix[y][:, sup]
        target = target * (1.0 - 2.0 * smooth) + smooth
        return F.binary_cross_entropy_with_logits(logit_z[:, sup], target)


def backbone_pass(model, x):
    """HyperGCN forward returning (logits_net, layer-6 output (N*M, C6, T6, V))."""
    st = run_layers(model, init_state(model, x), 6)
    feat6 = st["x"]
    st = run_layers(model, st, NUM_LAYERS)
    logits, _ = head(model, st)
    return logits, feat6


@torch.no_grad()
def calibrate(ccp: CCP, loader, device, n_batches: int = 50):
    """Signal standardisation (masked mean / std over valid frames) and threshold init
    tau_c = median of q_c, both on the given (TRAIN-FIT) loader."""
    s1 = torch.zeros(NUM_SIGNALS, dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    cnt = torch.zeros_like(s1)
    for b, (data, _, _) in enumerate(loader):
        if b >= n_batches:
            break
        S, Mk, _ = geo_signals(data.float().to(device))
        m = Mk.to(torch.float64)
        Sd = S.to(torch.float64) * m
        s1 += Sd.sum((0, 2))
        s2 += (Sd * S.to(torch.float64)).sum((0, 2))
        cnt += m.sum((0, 2))
    mu = s1 / cnt.clamp_min(1)
    var = s2 / cnt.clamp_min(1) - mu ** 2
    sd = var.clamp_min(0).sqrt()
    ccp.concepts.sig_mu.copy_(mu.float())
    ccp.concepts.sig_sd.copy_(sd.clamp_min(1e-4).float())
    qs = []
    for b, (data, _, _) in enumerate(loader):
        if b >= n_batches:
            break
        S, Mk, _ = geo_signals(data.float().to(device))
        q, _, _ = ccp.concepts.geo_q(S, Mk)
        qs.append(q)
    q = torch.cat(qs)
    ccp.concepts.tau.copy_(q.median(0).values)
    print("[calibrate] {} signals over {:.0f} frames; tau init = median q".format(
        NUM_SIGNALS, float(cnt.max())))
    for i, s in enumerate(SIGNALS):
        print("  {:20s} mu={:+.4f} sd={:.4f} [{}]".format(s, float(mu[i]), float(sd[i]), SIGNAL_UNITS[s]))


def _literal_text(ccp: CCP, c: int) -> dict:
    spec = ccp.specs[c]
    if spec.kind == "latent":
        return dict(text="latent:{}".format(spec.name), signal=None, threshold=None)
    gpos = int((ccp.concepts.geo_index == c).nonzero()[0, 0])
    si = int(ccp.concepts.sig_index[gpos])
    mu = float(ccp.concepts.sig_mu[si])
    sd = float(ccp.concepts.sig_sd[si])
    tau = float(ccp.concepts.tau[gpos])
    d = spec.direction
    thr = mu + d * sd * tau
    quant = "max_t" if spec.quant == "exists" else "mean_t"
    unit = SIGNAL_UNITS[spec.signal]
    text = "{}[{}] {} {:.3g} {}  ({})".format(quant, spec.signal, ">" if d > 0 else "<", thr, unit, spec.name)
    return dict(text=text, signal=spec.signal, threshold=thr, unit=unit, quant=quant,
                op=">" if d > 0 else "<")


@torch.no_grad()
def export_rules(ccp: CCP, min_w: float = 0.3) -> list[dict]:
    """One rule per class: literals with weight >= min_w, strongest first. Thresholds of
    geo literals are given back in signal units (soft max_t ~ max over valid frames,
    mean_t = mean over valid frames)."""
    wp, wn, wb = ccp.logic.weights()
    csv_m = ccp.csv_matrix
    rules = []
    for k, cname in enumerate(ccp.class_names):
        lits = []
        for c in torch.nonzero(wp[k] >= min_w).flatten().tolist():
            lits.append(dict(kind="pos", concept=ccp.concept_names[c], index=c, weight=float(wp[k, c]),
                             in_csv=bool(csv_m[k, c] > 0), **_literal_text(ccp, c)))
        for c in torch.nonzero(wn[k] >= min_w).flatten().tolist():
            t = _literal_text(ccp, c)
            t["text"] = "NOT " + t["text"]
            lits.append(dict(kind="neg", concept=ccp.concept_names[c], index=c, weight=float(wn[k, c]),
                             in_csv=bool(csv_m[k, c] > 0), **t))
        for p in torch.nonzero(wb[k] >= min_w).flatten().tolist():
            a, b = int(ccp.pairs[p, 0]), int(ccp.pairs[p, 1])
            lits.append(dict(kind="before", concept="before({}, {})".format(ccp.concept_names[a], ccp.concept_names[b]),
                             index=p, weight=float(wb[k, p]), in_csv=False,
                             text="before({}, {})".format(ccp.concept_names[a], ccp.concept_names[b])))
        lits.sort(key=lambda d: -d["weight"])
        rules.append(dict(class_idx=k, class_name=cname, literals=lits, bias=float(ccp.logic.bias[k])))
    return rules

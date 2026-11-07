"""Phase 1 of the thesis NeSy zero-shot model + the evaluation protocol shared with phase 2.

Phase 1 works on pooled frozen Shift-GCN features (N, 256). Two branches score every class y:
  s_emb(y)  = scale * cos(phi(x), t_y)       t_y = CLIP text prototype of class y
  s_rule(y) = sum_k w_k [R+_yk log p_k + R-_yk log(1 - p_k)] / sum_k w_k |R_yk|
              p = sigmoid(psi(x)) are primitive probabilities, R+/R- = require/forbid from the KG.
              In log space this is the length-normalised product t-norm; w_k is shared by all classes.
  z(y)      = s_emb(y) + beta * s_rule(y)          (mode 'fused'; 'emb' / 'rule' keep one branch)

Losses: L_cls (CE on z over the training classes) + L_emb (CE on s_emb alone) + lam_prim * L_prim
(BCE on primitives, targets = KG row of the sample's class, don't-care masked) + lam_cons * L_cons
(implication a -> b, penalty p_a * (1 - p_b)). Stage 1 (warm-up) has no L_cls.

Protocol (run_split), shared with phase 2. Any model works if it has set_norm(mu, sd),
forward(x, classes) -> dict with 'z' (and optionally 's_emb' / 's_rule'), and losses(out, y_local, y_global, stage).
  1. Validation: pseudo-unseen classes taken from the seen classes (same count as the real unseen set,
     fixed seed) + 10% of the other seen samples. Train on the rest, evaluate every epoch.
     The epoch with the best val ZSL and its GZSL gamma (best val H) are kept.
  2. Final: train on all seen classes for the same number of epochs, evaluating the test set every epoch.
     Two numbers are reported:
       clean       : test at the epoch chosen on validation, GZSL with the validation gamma.
       best-on-test: max test ZSL over epochs; max test H over epochs with gamma searched on test.
                     This is what most papers report (e.g. Neuron keeps the best test epoch and
                     grid-searches the GZSL calibration on test). It is optimistic.
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNSEEN = {  # 0-based class ids, same as feeders/feeder_ntu.py
    'ntu60_xsub_seen55_unseen5':    [10, 11, 19, 26, 56],
    'ntu60_xsub_seen48_unseen12':   [3, 5, 9, 12, 15, 40, 42, 47, 51, 56, 58, 59],
    'ntu120_xsub_seen110_unseen10': [4, 13, 37, 43, 49, 65, 88, 95, 99, 106],
    'ntu120_xsub_seen96_unseen24':  [5, 9, 11, 16, 18, 20, 22, 29, 35, 39, 45, 49, 59, 68, 70, 81, 84, 87, 93, 94, 104, 113, 114, 119],
}


def n_classes(task):
    return 60 if task.startswith('ntu60') else 120


# ============================================================================= knowledge graph
def load_kg(rules_path, prim_path, n_cls=120):
    """Returns primitive ids, class names, R (n_cls, K) with +1 require / -1 forbid / 0 don't care,
    and the implication constraints as an (M, 2) index array."""
    prims = [p['id'] for p in json.load(open(prim_path, encoding='utf-8'))]
    pidx = {p: i for i, p in enumerate(prims)}
    kg = json.load(open(rules_path, encoding='utf-8'))
    R = np.zeros((n_cls, len(prims)), np.float32)
    names = [None] * n_cls
    for r in kg['rules']:
        c = r['id'] - 1
        names[c] = r['class']
        assert not set(r['require']) & set(r['forbid']), r['class']
        for k in r['require']:
            R[c, pidx[k]] = 1
        for k in r['forbid']:
            R[c, pidx[k]] = -1
    assert all(n is not None for n in names), 'KG must cover every class'
    imp = np.array([[pidx[a], pidx[b]] for a, b in kg['constraints']['implication']], np.int64)
    return prims, names, R, imp


def coverage(R, seen, unseen, prims):
    """For each unseen class: required primitives that no training class requires. Their detector
    never gets a positive target, so the rule branch cannot confirm them."""
    seen_pos = (R[seen] > 0).any(0)
    rows = []
    for c in unseen:
        req = np.where(R[c] > 0)[0]
        miss = [prims[k] for k in req if not seen_pos[k]]
        rows.append(dict(cls=c, n_require=len(req), n_uncovered=len(miss), uncovered=', '.join(miss)))
    return rows


def kg_ceiling(R, classes, dontcare=0.5, eps=1e-3):
    """Upper bound of the rule branch with PERFECT primitives. Each class c gets an oracle primitive
    vector: 1-eps where c requires, eps where c forbids, `dontcare` elsewhere (0.5 = unknown,
    0 = assume absent). Scores are the same mean log-truth as s_rule (w_k = 1).
    Returns (accuracy with ties split evenly, per-class rows listing the classes tied with or above it)."""
    classes = list(classes)
    Rc = R[classes]
    Rp, Rn = (Rc > 0).astype(np.float64), (Rc < 0).astype(np.float64)
    P = np.where(Rc > 0, 1 - eps, np.where(Rc < 0, eps, dontcare)).clip(eps, 1 - eps)
    S = (np.log(P) @ Rp.T + np.log(1 - P) @ Rn.T) / (Rp + Rn).sum(1).clip(min=1)
    rows, acc = [], 0.0
    for i, c in enumerate(classes):
        tie = S[i] >= S[i, i] - 1e-9
        acc += 1.0 / tie.sum() if S[i].max() <= S[i, i] + 1e-9 else 0.0
        rows.append(dict(cls=c, n_tied_or_above=int(tie.sum()) - 1,
                         confused_with=[classes[j] for j in np.where(tie)[0] if j != i]))
    return acc / len(classes), rows


# ============================================================================= models
@torch.no_grad()
def channel_stats(X, idx, chunk=8192):
    """Per-channel mean / std over the samples idx (all leading dims pooled), computed on X's device."""
    s, s2, n = 0.0, 0.0, 0
    for i in range(0, len(idx), chunk):
        x = X[idx[i:i + chunk]].float().reshape(-1, X.shape[-1])
        s, s2, n = s + x.sum(0), s2 + (x * x).sum(0), n + x.shape[0]
    mu = s / n
    return mu, (s2 / n - mu * mu).clamp(min=0).sqrt() + 1e-6


class Normed(nn.Module):
    """Holds per-channel input normalisation as buffers, so the data stay raw (fp16) on the GPU."""

    def __init__(self, d_in):
        super().__init__()
        self.register_buffer('mu', torch.zeros(d_in))
        self.register_buffer('sd', torch.ones(d_in))

    def set_norm(self, mu, sd):
        self.mu.copy_(mu)
        self.sd.copy_(sd)

    def norm(self, x):
        return (x.float() - self.mu) / self.sd


class NeSyZSL(Normed):
    """Phase-1 model on pooled features (B, d_in)."""

    def __init__(self, text_proto, R, imp, d_in=256, hidden=1024, dropout=0.3,
                 use_emb=True, use_rule=True, beta_init=5.0, lam_prim=1.0, lam_cons=1.0):
        super().__init__(d_in)
        assert use_emb or use_rule
        self.use_emb, self.use_rule = use_emb, use_rule
        self.lam_prim, self.lam_cons = lam_prim, lam_cons
        self.register_buffer('T', F.normalize(torch.as_tensor(text_proto, dtype=torch.float32), dim=-1))
        R = torch.as_tensor(R)
        self.register_buffer('Rp', (R > 0).float())
        self.register_buffer('Rn', (R < 0).float())
        self.register_buffer('imp', torch.as_tensor(imp))
        K, d_txt = R.shape[1], self.T.shape[1]
        if use_emb:
            self.emb = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_txt))
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        if use_rule:
            self.prim = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, K))
            self.w_raw = nn.Parameter(torch.full((K,), math.log(math.e - 1)))   # softplus -> w_k = 1
            self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))

    def forward(self, x, classes):
        """x: (B, d_in) raw; classes: (C,) global class ids to score. Returns dict of (B, C) scores."""
        x = self.norm(x)
        out, z = {}, 0
        if self.use_emb:
            e = F.normalize(self.emb(x), dim=-1)
            out['s_emb'] = self.logit_scale.exp().clamp(max=100) * e @ self.T[classes].T
            z = z + out['s_emb']
        if self.use_rule:
            logit = self.prim(x)
            out['prim_logit'] = logit
            out['s_rule'] = rule_score(logit, self.Rp[classes], self.Rn[classes], F.softplus(self.w_raw))
            z = z + self.log_beta.exp() * out['s_rule']
        out['z'] = z
        return out

    def losses(self, out, y_local, y_global, stage):
        L = {}
        if stage == 2:
            L['cls'] = F.cross_entropy(out['z'], y_local)
        if self.use_emb and (self.use_rule or stage == 1):
            L['emb'] = F.cross_entropy(out['s_emb'], y_local)   # keeps the embedding branch usable on its own
        if self.use_rule:
            L['prim'] = self.lam_prim * prim_bce(out['prim_logit'], self.Rp[y_global], self.Rn[y_global])
            p = torch.sigmoid(out['prim_logit'])
            L['cons'] = self.lam_cons * (p[:, self.imp[:, 0]] * (1 - p[:, self.imp[:, 1]])).mean()
        return L


def rule_score(logit, Rp, Rn, w):
    """Mean log-truth of each class rule. logit: (B, K) primitive logits; Rp/Rn: (C, K); w: (K,)."""
    num = F.logsigmoid(logit) @ (Rp * w).T + F.logsigmoid(-logit) @ (Rn * w).T
    return num / ((Rp + Rn) * w).sum(1).clamp(min=1e-6)


def prim_bce(logit, Rp_y, Rn_y):
    """BCE on primitives against the KG row of each sample's class; don't-care entries are masked."""
    mask = Rp_y + Rn_y
    bce = F.binary_cross_entropy_with_logits(logit, Rp_y, reduction='none')
    return (bce * mask).sum() / mask.sum().clamp(min=1)


def make_pooled(mode, text_proto, R, imp, device='cuda', **kw):
    """Factory for run_split: mode 'emb' | 'rule' | 'fused'."""
    return lambda: NeSyZSL(text_proto, R, imp, use_emb=mode in ('emb', 'fused'),
                           use_rule=mode in ('rule', 'fused'), **kw).to(device)


# ============================================================================= train / eval
KEEP = ('z', 's_emb', 's_rule')


@torch.no_grad()
def predict(model, X, idx, classes, bs=2048):
    model.eval()
    outs = [model(X[idx[i:i + bs]], classes) for i in range(0, len(idx), bs)]
    return {k: torch.cat([o[k] for o in outs]) for k in KEEP if k in outs[0]}


def zsl_acc(z, y, classes):
    return float((classes[z.argmax(1)] == y).float().mean())


def gzsl_eval(z, y, classes, seen_t, gamma):
    """Calibrated stacking: subtract gamma from the scores of seen classes. Per-sample top-1 accuracy."""
    pred = classes[(z - gamma * torch.isin(classes, seen_t).float()).argmax(1)]
    s_mask = torch.isin(y, seen_t)
    S = float((pred[s_mask] == y[s_mask]).float().mean())
    U = float((pred[~s_mask] == y[~s_mask]).float().mean())
    return S, U, (2 * S * U / (S + U) if S + U > 0 else 0.0)


def best_gamma(z, y, classes, seen_t, n=101):
    span = float(z.max() - z.min())
    res = [(g, *gzsl_eval(z, y, classes, seen_t, g)) for g in np.linspace(-0.2 * span, span, n)]
    return max(res, key=lambda r: r[3])          # (gamma, S, U, H)


def train(model, X, y, idx, classes, epochs=30, warmup=5, lr=1e-3, wd=1e-4, bs=256, eval_fn=None, seed=0):
    """Trains on the samples idx of (X, y). classes = training class ids (global).
    eval_fn(model, epoch) runs after every epoch; its dict is merged into the history row."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    remap = torch.full((int(classes.max()) + 1,), -1, dtype=torch.long, device=X.device)
    remap[classes] = torch.arange(len(classes), device=X.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for ep in range(epochs):
        model.train()
        stage = 1 if ep < warmup else 2
        perm = idx[torch.randperm(len(idx), generator=g).to(X.device)]
        tot = {}
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            yb = y[b]
            L = model.losses(model(X[b], classes), remap[yb], yb, stage)
            opt.zero_grad()
            sum(L.values()).backward()
            opt.step()
            for k, v in L.items():
                tot[k] = tot.get(k, 0.0) + float(v.detach()) * len(b)
        sched.step()
        rec = dict(epoch=ep + 1, stage=stage, **{f'L_{k}': v / len(perm) for k, v in tot.items()})
        if eval_fn is not None:
            rec.update(eval_fn(model, ep + 1))
        history.append(rec)
    return history


def _cpu(d):
    return {k: v.detach().cpu() for k, v in d.items()}


def run_split(data, task, make_model, name='', seed=0, val_frac=0.1, train_kw=None, log=print):
    """data: dict of tensors on one device: Xtr (N, ...), ytr (N,) = seen-class training set;
    Xte, yte = the whole test set (GZSL); ZSL uses the test samples of unseen classes.
    Returns (result dict, history dict, scores dict for posthoc_fusion)."""
    train_kw = dict(train_kw or {})
    epochs = train_kw.pop('epochs', 30)
    Xtr, ytr, Xte, yte = data['Xtr'], data['ytr'], data['Xte'], data['yte']
    dev = Xtr.device
    T = lambda a: torch.as_tensor(np.asarray(a), device=dev)
    ytr_np, yte_np = ytr.cpu().numpy(), yte.cpu().numpy()
    unseen = sorted(UNSEEN[task])
    seen = sorted(np.unique(ytr_np).tolist())
    assert not set(seen) & set(unseen)

    # ---------------------------------------------------------------- 1. validation (pseudo-unseen)
    rng = np.random.RandomState(seed)
    v_unseen = sorted(rng.choice(seen, len(unseen), replace=False).tolist())
    v_seen = [c for c in seen if c not in v_unseen]
    pool = np.where(np.isin(ytr_np, v_seen))[0]
    hold = rng.rand(len(pool)) < val_frac
    fit_i, vu_i = T(pool[~hold]), T(np.where(np.isin(ytr_np, v_unseen))[0])
    va_i = T(np.concatenate([pool[hold], np.where(np.isin(ytr_np, v_unseen))[0]]))
    cls_vu, cls_va, vseen_t = T(v_unseen), T(sorted(v_seen + v_unseen)), T(v_seen)
    y_vu, y_va = ytr[vu_i], ytr[va_i]
    vbest = dict(val_zsl=-1.0)

    def val_fn(model, ep):
        o_u, o_a = predict(model, Xtr, vu_i, cls_vu), predict(model, Xtr, va_i, cls_va)
        vz = zsl_acc(o_u['z'], y_vu, cls_vu)
        gam, _, _, H = best_gamma(o_a['z'], y_va, cls_va, vseen_t)
        if vz > vbest['val_zsl']:
            vbest.update(val_zsl=vz, val_H=H, gamma=gam, epoch=ep, u=_cpu(o_u), a=_cpu(o_a))
        return dict(val_zsl=vz, val_H=H)

    torch.manual_seed(seed)
    model = make_model()
    model.set_norm(*channel_stats(Xtr, fit_i))
    h_val = train(model, Xtr, ytr, fit_i, T(v_seen), epochs=epochs, eval_fn=val_fn, seed=seed, **train_kw)
    e_star, gamma_val = vbest['epoch'], vbest['gamma']
    log(f'[{task} {name} seed{seed}] val: best ZSL {vbest["val_zsl"]:.4f} @ep{e_star}, H {vbest["val_H"]:.4f}, '
        f'last ZSL {h_val[-1]["val_zsl"]:.4f}')

    # ---------------------------------------------------------------- 2. final (all seen classes)
    all_i = T(np.arange(len(ytr_np)))
    te_u_i, te_a_i = T(np.where(np.isin(yte_np, unseen))[0]), T(np.arange(len(yte_np)))
    cls_u, cls_all, seen_t = T(unseen), T(np.arange(n_classes(task))), T(seen)
    y_u, y_a = yte[te_u_i], yte
    st = dict(best_zsl=-1.0, best_H=-1.0)

    def test_fn(model, ep):
        o_u, o_a = predict(model, Xte, te_u_i, cls_u), predict(model, Xte, te_a_i, cls_all)
        zs = zsl_acc(o_u['z'], y_u, cls_u)
        g_t, S_t, U_t, H_t = best_gamma(o_a['z'], y_a, cls_all, seen_t)
        if ep == e_star:
            S, U, H = gzsl_eval(o_a['z'], y_a, cls_all, seen_t, gamma_val)
            st.update(zsl=zs, S=S, U=U, H=H, clean_u=_cpu(o_u), clean_a=_cpu(o_a))
        if zs > st['best_zsl']:
            st.update(best_zsl=zs, best_zsl_ep=ep, best_u=_cpu(o_u))
        if H_t > st['best_H']:
            st.update(best_H=H_t, best_S=S_t, best_U=U_t, best_H_ep=ep, best_a=_cpu(o_a))
        return dict(test_zsl=zs, test_H_tuned=H_t)

    torch.manual_seed(seed)
    model = make_model()
    model.set_norm(*channel_stats(Xtr, all_i))
    h_test = train(model, Xtr, ytr, all_i, seen_t, epochs=epochs, eval_fn=test_fn, seed=seed, **train_kw)

    res = dict(task=task, model=name, seed=seed, epoch_val=e_star, val_zsl=vbest['val_zsl'],
               zsl=st['zsl'], gzsl_S=st['S'], gzsl_U=st['U'], gzsl_H=st['H'],
               zsl_best_test=st['best_zsl'], ep_best_zsl=st['best_zsl_ep'],
               H_best_test=st['best_H'], S_best_test=st['best_S'], U_best_test=st['best_U'], ep_best_H=st['best_H_ep'])
    if hasattr(model, 'log_beta'):
        res['beta'] = float(model.log_beta.exp())
    log(f'[{task} {name} seed{seed}] test clean: ZSL {res["zsl"]:.4f} H {res["gzsl_H"]:.4f} | '
        f'best-on-test: ZSL {res["zsl_best_test"]:.4f} H {res["H_best_test"]:.4f}')
    cpu = lambda t: t.detach().cpu()
    scores = dict(val_u=vbest['u'], val_a=vbest['a'], clean_u=st['clean_u'], clean_a=st['clean_a'],
                  best_u=st['best_u'], best_a=st['best_a'],
                  y_vu=cpu(y_vu), y_va=cpu(y_va), y_u=cpu(y_u), y_a=cpu(y_a),
                  cls_vu=cpu(cls_vu), cls_va=cpu(cls_va), cls_u=cpu(cls_u), cls_all=cpu(cls_all),
                  vseen=cpu(vseen_t), seen=cpu(seen_t))
    return res, dict(val=h_val, test=h_test), scores


def posthoc_fusion(sc_emb, sc_rule, betas=(0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)):
    """z = s_emb + beta * s_rule from two separately trained models (same seed -> same validation split).
    beta is chosen on validation ZSL (beta = 0 is the embedding branch alone).
      clean        : both branches at their validation epoch, gamma chosen on validation.
      best-on-test : each branch at its best test epoch, gamma searched on test (beta still from validation).
    Returns (chosen row, table over all betas)."""
    assert torch.equal(sc_emb['cls_vu'], sc_rule['cls_vu']), 'the two runs must share the validation split'
    e, r = sc_emb, sc_rule
    table = []
    for b in betas:
        f = lambda key: e[key]['s_emb'] + b * r[key]['s_rule']
        vz = zsl_acc(f('val_u'), e['y_vu'], e['cls_vu'])
        gam, _, _, vH = best_gamma(f('val_a'), e['y_va'], e['cls_va'], e['vseen'])
        S, U, H = gzsl_eval(f('clean_a'), e['y_a'], e['cls_all'], e['seen'], gam)
        _, S_t, U_t, H_t = best_gamma(f('best_a'), e['y_a'], e['cls_all'], e['seen'])
        table.append(dict(beta=b, val_zsl=vz, val_H=vH,
                          zsl=zsl_acc(f('clean_u'), e['y_u'], e['cls_u']), gzsl_S=S, gzsl_U=U, gzsl_H=H,
                          zsl_best_test=zsl_acc(f('best_u'), e['y_u'], e['cls_u']),
                          H_best_test=H_t, S_best_test=S_t, U_best_test=U_t))
    chosen = max(table, key=lambda row: (row['val_zsl'], -row['beta']))   # tie -> smaller beta
    return chosen, table

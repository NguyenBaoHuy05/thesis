"""Phase 1 of the thesis NeSy zero-shot model, on pooled frozen Shift-GCN features (N, 256).

Two branches score every class y:
  s_emb(y)  = scale * cos(phi(x), t_y)       t_y = CLIP text prototype of class y
  s_rule(y) = sum_k w_k [R+_yk log p_k + R-_yk log(1 - p_k)] / sum_k w_k |R_yk|
              p = sigmoid(psi(x)) are primitive probabilities, R+/R- = require/forbid from the KG.
              In log space this is the length-normalised product t-norm; w_k is shared by all classes.
  z(y)      = s_emb(y) + beta * s_rule(y)

Losses: L_cls (CE on z over the training classes) + L_emb (CE on s_emb alone) + lam_prim * L_prim
(BCE on primitives, targets = KG row of the sample's class, don't-care masked) + lam_cons * L_cons
(implication a -> b, penalty p_a * (1 - p_b)). Stage 1 (warm-up) uses L_emb + L_prim + L_cons; stage 2 uses all.

Protocol (no test-set selection):
  1. Validation: hold out pseudo-unseen classes taken from the seen classes (same count as the real
     unseen set, fixed seed) and 10% of the remaining seen samples. Train on the rest and pick the
     number of epochs (best val ZSL) and the GZSL calibration gamma (best val H).
  2. Retrain on all seen classes for that many epochs, then evaluate the test set once.
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


# ----------------------------------------------------------------------------- knowledge graph
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


# ----------------------------------------------------------------------------- model
class NeSyZSL(nn.Module):
    def __init__(self, text_proto, R, imp, d_in=256, hidden=1024, dropout=0.3,
                 use_emb=True, use_rule=True, beta_init=5.0):
        super().__init__()
        assert use_emb or use_rule
        self.use_emb, self.use_rule = use_emb, use_rule
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
        """x: (B, d_in); classes: (C,) global class ids to score. Returns dict of (B, C) scores."""
        out, z = {}, 0
        if self.use_emb:
            e = F.normalize(self.emb(x), dim=-1)
            out['s_emb'] = self.logit_scale.exp().clamp(max=100) * e @ self.T[classes].T
            z = z + out['s_emb']
        if self.use_rule:
            logit = self.prim(x)
            out['prim_logit'] = logit
            w = F.softplus(self.w_raw)
            Rp, Rn = self.Rp[classes], self.Rn[classes]
            num = F.logsigmoid(logit) @ (Rp * w).T + F.logsigmoid(-logit) @ (Rn * w).T
            den = ((Rp + Rn) * w).sum(1).clamp(min=1e-6)
            out['s_rule'] = num / den                       # mean log-truth of the rule, <= 0
            z = z + self.log_beta.exp() * out['s_rule']
        out['z'] = z
        return out


def compute_losses(model, out, y_local, y_global, lam_prim=1.0, lam_cons=1.0, stage=2):
    L = {}
    if stage == 2:
        L['cls'] = F.cross_entropy(out['z'], y_local)
    if model.use_emb and (model.use_rule or stage == 1):
        L['emb'] = F.cross_entropy(out['s_emb'], y_local)   # keeps the embedding branch usable on its own
    if model.use_rule:
        tgt, mask = model.Rp[y_global], model.Rp[y_global] + model.Rn[y_global]
        bce = F.binary_cross_entropy_with_logits(out['prim_logit'], tgt, reduction='none')
        L['prim'] = lam_prim * (bce * mask).sum() / mask.sum().clamp(min=1)
        p = torch.sigmoid(out['prim_logit'])
        a, b = model.imp[:, 0], model.imp[:, 1]
        L['cons'] = lam_cons * (p[:, a] * (1 - p[:, b])).mean()
    return L


# ----------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, X, classes, bs=4096):
    model.eval()
    outs = [model(X[i:i + bs], classes) for i in range(0, len(X), bs)]
    return {k: torch.cat([o[k] for o in outs]).cpu() for k in outs[0]}


def zsl_acc(z, y, classes):
    return float((classes.cpu()[z.argmax(1)] == y.cpu()).float().mean())


def gzsl_eval(z, y, classes, seen_set, gamma):
    """Calibrated stacking: subtract gamma from the scores of seen classes. Per-sample top-1 accuracy."""
    classes, y = classes.cpu(), y.cpu()
    seen_t = torch.tensor(sorted(seen_set))
    pred = classes[(z - gamma * torch.isin(classes, seen_t).float()).argmax(1)]
    s_mask = torch.isin(y, seen_t)
    S = float((pred[s_mask] == y[s_mask]).float().mean())
    U = float((pred[~s_mask] == y[~s_mask]).float().mean())
    H = 2 * S * U / (S + U) if S + U > 0 else 0.0
    return S, U, H


def best_gamma(z, y, classes, seen_set, n=201):
    span = float(z.max() - z.min())
    grid = np.linspace(-0.2 * span, span, n)
    res = [(g, *gzsl_eval(z, y, classes, seen_set, g)) for g in grid]
    return max(res, key=lambda r: r[3])          # (gamma, S, U, H)


def train(model, X, y, classes, epochs=30, warmup=5, lr=1e-3, wd=1e-4, bs=256,
          lam_prim=1.0, lam_cons=1.0, eval_fn=None, seed=0):
    """X, y on device; classes = training class ids (global). eval_fn(model, epoch) is called after
    every epoch and its return values are collected (used for validation curves)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    remap = torch.full((int(classes.max()) + 1,), -1, dtype=torch.long, device=X.device)
    remap[classes] = torch.arange(len(classes), device=X.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for ep in range(epochs):
        model.train()
        stage = 1 if ep < warmup else 2
        perm = torch.randperm(len(X), generator=g).to(X.device)
        tot = {}
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            out = model(X[idx], classes)
            L = compute_losses(model, out, remap[y[idx]], y[idx], lam_prim, lam_cons, stage)
            loss = sum(L.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in L.items():
                tot[k] = tot.get(k, 0.0) + float(v.detach()) * len(idx)
        sched.step()
        rec = dict(epoch=ep + 1, stage=stage, **{f'L_{k}': v / len(X) for k, v in tot.items()})
        if eval_fn is not None:
            rec.update(eval_fn(model, ep + 1))
        history.append(rec)
    return history


# ----------------------------------------------------------------------------- full protocol
class Standardize:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6

    def __call__(self, X):
        return (X - self.mu) / self.sd


def run_split(npz, task, text_proto, R, imp, mode='fused', seed=0, device='cuda', val_frac=0.1,
              model_kw=None, train_kw=None, log=print):
    """mode: 'emb' | 'rule' | 'fused'. Returns (result dict, validation history, scores).
    scores keeps the branch scores (s_emb / s_rule / z) on validation (at the chosen epoch) and on the
    test set, for post-hoc fusion of separately trained branches (see posthoc_fusion)."""
    model_kw, train_kw = dict(model_kw or {}), dict(train_kw or {})
    model_kw.update(use_emb=mode in ('emb', 'fused'), use_rule=mode in ('rule', 'fused'))
    unseen = sorted(UNSEEN[task])
    tr_y = npz['train_label']
    seen = sorted(np.unique(tr_y).tolist())
    rng = np.random.RandomState(seed)

    # --- 1. validation on pseudo-unseen classes
    v_unseen = sorted(rng.choice(seen, len(unseen), replace=False).tolist())
    v_seen = [c for c in seen if c not in v_unseen]
    pool = np.where(np.isin(tr_y, v_seen))[0]
    hold = rng.rand(len(pool)) < val_frac
    fit_idx, vs_idx = pool[~hold], pool[hold]
    vu_idx = np.where(np.isin(tr_y, v_unseen))[0]

    tf = lambda a: torch.as_tensor(a, device=device)
    norm = Standardize(npz['train_data'][fit_idx])
    Xf, yf = tf(norm(npz['train_data'][fit_idx])), tf(tr_y[fit_idx])
    Xvu, yvu = tf(norm(npz['train_data'][vu_idx])), tf(tr_y[vu_idx])
    Xv = tf(norm(npz['train_data'][np.concatenate([vs_idx, vu_idx])]))
    yv = tf(tr_y[np.concatenate([vs_idx, vu_idx])])
    cls_fit, cls_vu = tf(np.array(v_seen)), tf(np.array(v_unseen))
    cls_vall = tf(np.array(sorted(v_seen + v_unseen)))

    keep = ('z', 's_emb', 's_rule')
    val_scores = []

    def val_fn(model, ep):
        o_u, o_a = predict(model, Xvu, cls_vu), predict(model, Xv, cls_vall)
        val_scores.append(dict(u={k: o_u[k] for k in keep if k in o_u}, a={k: o_a[k] for k in keep if k in o_a}))
        gam, S, U, H = best_gamma(o_a['z'], yv, cls_vall, set(v_seen))
        return dict(val_zsl=zsl_acc(o_u['z'], yvu, cls_vu), val_H=H, val_gamma=gam)

    torch.manual_seed(seed)
    model = NeSyZSL(text_proto, R, imp, **model_kw).to(device)
    hist = train(model, Xf, yf, cls_fit, eval_fn=val_fn, seed=seed, **train_kw)
    best = max(hist, key=lambda r: r['val_zsl'])
    epochs = best['epoch']
    log(f'[{task} {mode} seed{seed}] val: best ZSL {best["val_zsl"]:.4f} @ep{epochs}, '
        f'H {best["val_H"]:.4f}, last ZSL {hist[-1]["val_zsl"]:.4f}')

    # --- 2. retrain on all seen classes, evaluate the test set once
    norm = Standardize(npz['train_data'])
    Xtr, ytr = tf(norm(npz['train_data'])), tf(tr_y)
    torch.manual_seed(seed)
    model = NeSyZSL(text_proto, R, imp, **model_kw).to(device)
    train(model, Xtr, ytr, tf(np.array(seen)), seed=seed, **{**train_kw, 'epochs': epochs,
                                                            'warmup': min(train_kw.get('warmup', 5), epochs)})
    cls_u, cls_all = tf(np.array(unseen)), tf(np.arange(len(seen) + len(unseen)))
    zsl_out = predict(model, tf(norm(npz['zsl_data'])), cls_u)
    gz_out = predict(model, tf(norm(npz['gzsl_data'])), cls_all)
    gz = gz_out['z']
    S, U, H = gzsl_eval(gz, tf(npz['gzsl_label']), cls_all, set(seen), best['val_gamma'])
    res = dict(task=task, mode=mode, seed=seed, epochs=epochs,
               zsl=zsl_acc(zsl_out['z'], tf(npz['zsl_label']), cls_u),
               gzsl_S=S, gzsl_U=U, gzsl_H=H, gamma=best['val_gamma'], val_zsl=best['val_zsl'])
    for k in ('s_emb', 's_rule'):   # each branch alone, from the same fused model
        if k in zsl_out and mode == 'fused':
            res[f'zsl_{k}_only'] = zsl_acc(zsl_out[k], tf(npz['zsl_label']), cls_u)
    if model.use_rule:
        res['beta'] = float(model.log_beta.exp())
    log(f'[{task} {mode} seed{seed}] test: ZSL {res["zsl"]:.4f} | GZSL S {S:.4f} U {U:.4f} H {H:.4f}')
    scores = dict(val_u=val_scores[epochs - 1]['u'], val_a=val_scores[epochs - 1]['a'],
                  test_u={k: zsl_out[k] for k in keep if k in zsl_out},
                  test_a={k: gz_out[k] for k in keep if k in gz_out},
                  y_vu=yvu.cpu(), y_v=yv.cpu(), y_zsl=torch.as_tensor(npz['zsl_label']),
                  y_gzsl=torch.as_tensor(npz['gzsl_label']), cls_vu=cls_vu.cpu(), cls_vall=cls_vall.cpu(),
                  cls_u=cls_u.cpu(), cls_all=cls_all.cpu(), v_seen=set(v_seen), seen=set(seen))
    return res, hist, scores


def posthoc_fusion(sc_emb, sc_rule, betas=(0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)):
    """z = s_emb + beta * s_rule from two separately trained models (same seed, hence same validation
    split). beta is chosen on validation ZSL (beta = 0 is the embedding branch alone), then gamma on
    validation H for that beta. Returns (chosen row, table over all betas)."""
    assert torch.equal(sc_emb['cls_vu'], sc_rule['cls_vu']), 'the two runs must share the validation split'
    table = []
    for b in betas:
        fuse = lambda part: sc_emb[part]['s_emb'] + b * sc_rule[part]['s_rule']
        vz = zsl_acc(fuse('val_u'), sc_emb['y_vu'], sc_emb['cls_vu'])
        gam, *_ , vH = best_gamma(fuse('val_a'), sc_emb['y_v'], sc_emb['cls_vall'], sc_emb['v_seen'])
        tz = zsl_acc(fuse('test_u'), sc_emb['y_zsl'], sc_emb['cls_u'])
        S, U, H = gzsl_eval(fuse('test_a'), sc_emb['y_gzsl'], sc_emb['cls_all'], sc_emb['seen'], gam)
        table.append(dict(beta=b, val_zsl=vz, val_H=vH, zsl=tz, gzsl_S=S, gzsl_U=U, gzsl_H=H, gamma=gam))
    chosen = max(table, key=lambda r: (r['val_zsl'], -r['beta']))   # tie -> smaller beta
    return chosen, table

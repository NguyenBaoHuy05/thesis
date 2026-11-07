"""KG loading + the zero-shot evaluation protocol (no test-set selection, plus best-on-test columns).

A model works with run_split if it has
  set_norm(mu, sd)                    per-channel input normalisation (buffers)
  forward(x, classes) -> dict         with 'z': (B, C) class scores for the global class ids `classes`
  losses(out, y_local, y_global, stage) -> dict of loss tensors

Protocol (run_split):
  1. Validation: pseudo-unseen classes taken from the seen classes (same count as the real unseen set,
     fixed seed) + 10% of the other seen samples. Train on the rest, evaluate every epoch; keep the
     epoch with the best val ZSL and its GZSL calibration gamma (best val H).
  2. Final: train on all seen classes for the same number of epochs, evaluating the test set every epoch.
       clean        : test at the epoch chosen on validation, GZSL with the validation gamma.
       best-on-test : max test ZSL over epochs; max test H over epochs with gamma searched on test.
                      This is what most papers report (e.g. Neuron keeps the best test epoch and
                      grid-searches the GZSL calibration on test). It is optimistic.
"""
import json

import numpy as np
import torch
import torch.nn as nn

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
    """Returns primitives (list of dicts from primitive.json), class names, R (n_cls, K) with
    +1 require / -1 forbid / 0 don't care, and seqs {class0: [k1, k2, ...]} for rules with >= 2 phases."""
    prims = json.load(open(prim_path, encoding='utf-8'))
    pidx = {p['id']: i for i, p in enumerate(prims)}
    kg = json.load(open(rules_path, encoding='utf-8'))
    R = np.zeros((n_cls, len(prims)), np.float32)
    names, seqs = [None] * n_cls, {}
    for r in kg['rules']:
        c = r['id'] - 1
        names[c] = r['class']
        assert not set(r['require']) & set(r['forbid']), r['class']
        for k in r['require']:
            R[c, pidx[k]] = 1
        for k in r['forbid']:
            R[c, pidx[k]] = -1
        if len(r.get('sequence', [])) >= 2:
            seqs[c] = [pidx[k] for k in r['sequence']]
    assert all(n is not None for n in names), 'KG must cover every class'
    return prims, names, R, seqs


def coverage(R, seen, unseen, prim_ids):
    """For each unseen class: primitives in its rule (require or forbid) that no seen class uses.
    Their embedding is never trained, so it keeps its CLIP-text initialisation."""
    seen_used = (R[seen] != 0).any(0)
    rows = []
    for c in unseen:
        lits = np.where(R[c] != 0)[0]
        miss = [prim_ids[k] for k in lits if not seen_used[k]]
        rows.append(dict(cls=c, n_literals=len(lits), n_unused_in_seen=len(miss), primitives=', '.join(miss)))
    return rows


# ============================================================================= normalisation
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


# ============================================================================= train / eval
@torch.no_grad()
def predict(model, X, idx, classes, bs=1024):
    model.eval()
    return torch.cat([model(X[idx[i:i + bs]], classes)['z'] for i in range(0, len(idx), bs)])


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


def train(model, X, y, idx, classes, epochs=30, warmup=0, lr=1e-3, wd=1e-4, bs=256, eval_fn=None, seed=0):
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


def run_split(data, task, make_model, name='', seed=0, val_frac=0.1, train_kw=None, extra_eval=None, log=print):
    """data: dict of tensors on one device: Xtr (N, ...), ytr (N,) = seen-class training set;
    Xte, yte = the whole test set (GZSL); ZSL uses the test samples of unseen classes.
    extra_eval(model, ctx) -> dict is called on the final model at the clean epoch (ctx has the test
    indices / classes), e.g. for rule-intervention tests. Returns (result dict, history dict)."""
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
    vu_np = np.where(np.isin(ytr_np, v_unseen))[0]
    fit_i, vu_i, va_i = T(pool[~hold]), T(vu_np), T(np.concatenate([pool[hold], vu_np]))
    cls_vu, cls_va, vseen_t = T(v_unseen), T(sorted(v_seen + v_unseen)), T(v_seen)
    y_vu, y_va = ytr[vu_i], ytr[va_i]
    vbest = dict(val_zsl=-1.0)

    def val_fn(model, ep):
        vz = zsl_acc(predict(model, Xtr, vu_i, cls_vu), y_vu, cls_vu)
        gam, _, _, H = best_gamma(predict(model, Xtr, va_i, cls_va), y_va, cls_va, vseen_t)
        if vz > vbest['val_zsl']:
            vbest.update(val_zsl=vz, val_H=H, gamma=gam, epoch=ep)
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
    y_u = yte[te_u_i]
    ctx = dict(X=Xte, y=yte, idx_u=te_u_i, idx_a=te_a_i, cls_u=cls_u, cls_all=cls_all, seen_t=seen_t, gamma=gamma_val)
    st = dict(best_zsl=-1.0, best_H=-1.0, extra={})

    def test_fn(model, ep):
        z_u, z_a = predict(model, Xte, te_u_i, cls_u), predict(model, Xte, te_a_i, cls_all)
        zs = zsl_acc(z_u, y_u, cls_u)
        g_t, S_t, U_t, H_t = best_gamma(z_a, yte, cls_all, seen_t)
        if ep == e_star:
            S, U, H = gzsl_eval(z_a, yte, cls_all, seen_t, gamma_val)
            st.update(zsl=zs, S=S, U=U, H=H)
            if extra_eval is not None:
                st['extra'] = extra_eval(model, ctx)
        if zs > st['best_zsl']:
            st.update(best_zsl=zs, best_zsl_ep=ep)
        if H_t > st['best_H']:
            st.update(best_H=H_t, best_S=S_t, best_U=U_t, best_H_ep=ep)
        return dict(test_zsl=zs, test_H_tuned=H_t)

    torch.manual_seed(seed)
    model = make_model()
    model.set_norm(*channel_stats(Xtr, all_i))
    h_test = train(model, Xtr, ytr, all_i, seen_t, epochs=epochs, eval_fn=test_fn, seed=seed, **train_kw)

    res = dict(task=task, model=name, seed=seed, epoch_val=e_star, val_zsl=vbest['val_zsl'],
               zsl=st['zsl'], gzsl_S=st['S'], gzsl_U=st['U'], gzsl_H=st['H'],
               zsl_best_test=st['best_zsl'], ep_best_zsl=st['best_zsl_ep'],
               H_best_test=st['best_H'], S_best_test=st['best_S'], U_best_test=st['best_U'], ep_best_H=st['best_H_ep'],
               **st['extra'])
    log(f'[{task} {name} seed{seed}] test clean: ZSL {res["zsl"]:.4f} H {res["gzsl_H"]:.4f} | '
        f'best-on-test: ZSL {res["zsl_best_test"]:.4f} H {res["H_best_test"]:.4f}'
        + (f' | {st["extra"]}' if st['extra'] else ''))
    return res, dict(val=h_val, test=h_test)

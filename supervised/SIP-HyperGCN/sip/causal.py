"""Causal dependency of classes on intervened units (measures or body regions), v3.

Works on any intervention file of docs/PIPELINE_V3.md: `intervention_{val,test}.npz`
(units = measures, LEACE with continuous Z) or `region_intervention_{val,test}.npz`
(units = body regions, mean-ablation of joints). The last R units are the random
controls `__random_0__ ..`. Notation, for the evaluated samples i with true class y_i:

    c0_i        base correctness (no intervention)
    c_{l,u,i}   correctness after intervening on unit u at layer l
    cr_{l,i}    mean over the (non-skipped) random controls r of c_{l,r,i}

    acc_k(.)    mean of the correctness over the samples of class k
    dacc[l,u,k]   = acc_k(c0) - acc_k(c_{l,u})
    drand[l,k]    = mean_r (acc_k(c0) - acc_k(c_{l,r}))        drand_sd = std over r
    excess[l,u,k] = dacc - drand = mean_{i in k} d_{l,u,i},   d_{l,u,i} = cr_{l,i} - c_{l,u,i}

p_class[l,u,k] (one-sided, H1: intervening on u hurts class k more than the random controls)
    Paired bootstrap over the samples of class k: resample the n_k per-sample paired
    differences d_{l,u,i} (i in class k) with replacement B times, and
        p = (1 + #{b : mean_b(d) <= 0}) / (1 + B)
    (percentile bootstrap p-value of H0: E[d] <= 0). The same resampling indices are
    shared by all (l, u) of a class (common random numbers). A class whose samples are all
    unaffected (d = 0) gets p = 1.

t1 (unit-level, per (l, u)): sign-flip permutation test over classes on excess.
    obs = mean_k excess[l,u,k]; null: excess_k -> s_k * excess_k with s_k iid +-1;
        t1_p = (1 + #{perm : null_mean >= obs}) / (1 + n_perm),   t1_mean = obs.
    Only classes with at least one evaluated sample are used.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "RANDOM_PREFIX",
    "is_random_name",
    "split_names",
    "per_class_rate",
    "causal_effects",
    "bootstrap_p_class",
    "signflip_test",
    "class_stats",
    "confusion_matrix",
    "centroid_shift_matrix",
    "load_intervention",
    "analyze_intervention",
]

RANDOM_PREFIX = "__random"


def is_random_name(name: str) -> bool:
    """True for the random-control entries ("__random_0__", ...)."""
    return str(name).startswith(RANDOM_PREFIX)


def split_names(names):
    """-> (unit_idx, rand_idx): positions of the real units and of the random controls."""
    names = [str(n) for n in names]
    unit_idx = [i for i, n in enumerate(names) if not is_random_name(n)]
    rand_idx = [i for i, n in enumerate(names) if is_random_name(n)]
    return unit_idx, rand_idx


def _onehot(labels: np.ndarray, K: int) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int64)
    oh = np.zeros((len(labels), K), dtype=np.float64)
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


def _scalar(x, cast=str, default=None):
    try:
        return cast(np.asarray(x).item())
    except Exception:
        return default


def per_class_rate(M: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """(..., N) values -> (..., K) per-class mean over the samples (NaN for empty classes)."""
    M = np.asarray(M, dtype=np.float64)
    oh = _onehot(labels, K)
    cnt = oh.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (M.reshape(-1, M.shape[-1]) @ oh) / cnt
    out[:, cnt == 0] = np.nan
    return out.reshape(M.shape[:-1] + (K,))


def causal_effects(correct, base_correct, labels, K: int, unit_idx, rand_idx,
                   skipped=None) -> dict:
    """Per-class accuracy effects of the interventions.

    correct: (L, U+R, N) bool; base_correct: (N,) bool; labels: (N,) int.
    skipped: optional (L, U+R) bool -> entries set to NaN / excluded from the random mean.

    Returns dict with dacc (L,U,K), drand (L,K), drand_sd (L,K), excess (L,U,K),
    base_acc (K,), count (K,), d (L,U,N) float32 per-sample paired differences
    d = mean_r c_r - c_u (NaN rows for skipped (l, u)), n_random_used (L,).
    """
    correct = np.asarray(correct)
    L, _, N = correct.shape
    labels = np.asarray(labels).astype(np.int64)
    base_acc = per_class_rate(np.asarray(base_correct, dtype=np.float64), labels, K)
    count = np.bincount(labels, minlength=K)[:K].astype(np.int64)
    if skipped is None or np.shape(skipped) != correct.shape[:2]:
        skipped = np.zeros(correct.shape[:2], dtype=bool)
    skipped = np.asarray(skipped, dtype=bool)
    unit_idx = list(unit_idx)
    rand_idx = list(rand_idx)
    U = len(unit_idx)

    acc_u = per_class_rate(correct[:, unit_idx, :].astype(np.float32), labels, K)  # (L,U,K)
    dacc = base_acc[None, None, :] - acc_u
    dacc[skipped[:, unit_idx]] = np.nan

    drand = np.full((L, K), np.nan)
    drand_sd = np.full((L, K), np.nan)
    cr = np.full((L, N), np.nan, dtype=np.float32)
    n_used = np.zeros(L, dtype=np.int64)
    for li in range(L):
        ok = [r for r in rand_idx if not skipped[li, r]]
        n_used[li] = len(ok)
        if not ok:
            continue
        acc_r = per_class_rate(correct[li, ok, :].astype(np.float32), labels, K)  # (R',K)
        dr = base_acc[None, :] - acc_r
        drand[li] = dr.mean(0)
        drand_sd[li] = dr.std(0, ddof=1) if len(ok) > 1 else 0.0
        cr[li] = correct[li, ok, :].astype(np.float32).mean(0)

    excess = dacc - drand[:, None, :]
    d = cr[:, None, :] - correct[:, unit_idx, :].astype(np.float32)          # (L,U,N)
    d[skipped[:, unit_idx]] = np.nan
    assert d.shape == (L, U, N)
    return {"dacc": dacc, "drand": drand, "drand_sd": drand_sd, "excess": excess,
            "base_acc": base_acc, "count": count, "d": d, "n_random_used": n_used}


def bootstrap_p_class(d: np.ndarray, labels: np.ndarray, K: int, n_boot: int = 1000,
                      rng: np.random.Generator | None = None, tol: float = 1e-9) -> np.ndarray:
    """One-sided paired-bootstrap p-values (see module doc). d: (L, U, N) -> (L, U, K)."""
    rng = np.random.default_rng(0) if rng is None else rng
    d = np.asarray(d, dtype=np.float64)
    L, U, N = d.shape
    labels = np.asarray(labels).astype(np.int64)
    flat = d.reshape(L * U, N)
    finite_row = np.isfinite(flat).all(1)
    flat0 = np.where(np.isfinite(flat), flat, 0.0)
    out = np.full((L * U, K), np.nan)
    for k in range(K):
        idx = np.flatnonzero(labels == k)
        n = len(idx)
        if n == 0:
            continue
        dk = flat0[:, idx].T                                             # (n, LU)
        draw = rng.integers(0, n, size=(n_boot, n))
        W = np.bincount((draw + n * np.arange(n_boot)[:, None]).ravel(),
                        minlength=n_boot * n).reshape(n_boot, n).astype(np.float64)
        means = (W @ dk) / n                                             # (B, LU)
        p = (1.0 + np.sum(means <= tol, axis=0)) / (1.0 + n_boot)
        p[~finite_row] = np.nan
        out[:, k] = p
    return out.reshape(L, U, K)


def signflip_test(X: np.ndarray, n_perm: int = 2000, rng: np.random.Generator | None = None,
                  valid: np.ndarray | None = None, tol: float = 1e-12):
    """Sign-flip permutation test over the last axis (H1: mean > 0).

    X: (..., K). valid: optional (K,) bool mask of the classes to use.
    Returns (mean (...,), p (...,)); rows containing NaN on the valid classes -> NaN.
    One set of sign flips (n_perm, K') is shared by all rows.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    X = np.asarray(X, dtype=np.float64)
    lead = X.shape[:-1]
    X2 = X.reshape(-1, X.shape[-1])
    if valid is not None:
        X2 = X2[:, np.asarray(valid, dtype=bool)]
    Kv = X2.shape[1]
    mean = np.full(X2.shape[0], np.nan)
    p = np.full(X2.shape[0], np.nan)
    ok = np.isfinite(X2).all(1)
    if Kv == 0 or not ok.any():
        return mean.reshape(lead), p.reshape(lead)
    Xo = X2[ok]
    obs = Xo.mean(1)                                                     # (M,)
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, Kv))
    null = (flips @ Xo.T) / Kv                                           # (n_perm, M)
    mean[ok] = obs
    p[ok] = (1.0 + np.sum(null >= obs[None, :] - tol, axis=0)) / (1.0 + n_perm)
    return mean.reshape(lead), p.reshape(lead)


def class_stats(values: np.ndarray, labels: np.ndarray, K: int):
    """(N, D) continuous values -> (mean (D, K), std (D, K)) per class (ddof=0;
    NaN for empty classes)."""
    V = np.asarray(values, dtype=np.float64)
    mean = per_class_rate(V.T, labels, K)                                # (D, K)
    sq = per_class_rate((V ** 2).T, labels, K)
    std = np.sqrt(np.maximum(sq - mean ** 2, 0.0))
    return mean, std


def confusion_matrix(labels: np.ndarray, pred: np.ndarray, K: int) -> np.ndarray:
    """(K, K) int64 counts, rows = true class, cols = predicted class."""
    labels = np.asarray(labels).astype(np.int64)
    pred = np.asarray(pred).astype(np.int64)
    return np.bincount(labels * K + pred, minlength=K * K)[:K * K].reshape(K, K)


def centroid_shift_matrix(final_feat, base_final_feat, labels, K: int, unit_idx,
                          skipped=None) -> np.ndarray:
    """(L, U, K) class-centroid displacement of the pooled final feature, normalised by
    the base within-class RMS radius (sip.probe.centroid_shift)."""
    from sip.probe import centroid_shift
    L = final_feat.shape[0]
    Fb = np.asarray(base_final_feat, dtype=np.float32)
    out = np.full((L, len(unit_idx), K), np.nan)
    for li in range(L):
        for j, uj in enumerate(unit_idx):
            if skipped is not None and skipped[li, uj]:
                continue
            Xi = np.asarray(final_feat[li, uj], dtype=np.float32)
            out[li, j] = centroid_shift(Fb, Xi, labels, K)["centroid_shift"]
    return out


# ------------------------------------------------------------- whole-file analysis
def load_intervention(path: str) -> dict:
    """Read one intervention npz (measures or regions) into plain arrays.

    Returns dict: names (all units incl. randoms), unit_idx, rand_idx, layers (L,),
    labels (N,), correct (L, U+R, N) bool, base_pred (N,), skipped (L, U+R) or None,
    K, eval_index, val_index, eval_split, npz (the open NpzFile, for final_feat).
    """
    z = np.load(path, allow_pickle=False)
    files = set(z.files)
    names = [str(n) for n in z["unit_names"]]
    unit_idx, rand_idx = split_names(names)
    if not rand_idx:
        raise ValueError(f"{path}: no __random_r__ controls in unit_names")
    layers = np.asarray(z["layers"]).astype(np.int64)
    labels = np.asarray(z["labels"]).astype(np.int64)
    N = len(labels)
    if "class_count" in files:
        K = int(np.asarray(z["class_count"]).shape[0])
    elif "base_logits" in files:
        K = int(z["base_logits"].shape[1])
    else:
        K = int(np.asarray(z["base_acc_per_class"]).shape[0])
    if "n_random" in files and _scalar(z["n_random"], int, len(rand_idx)) != len(rand_idx):
        print(f"[warn] {path}: n_random={_scalar(z['n_random'], int)} but "
              f"{len(rand_idx)} random names")
    if "correct" in files:
        correct = np.asarray(z["correct"]).astype(bool)
    elif "pred_label" in files:
        correct = np.asarray(z["pred_label"]).astype(np.int64) == labels[None, None, :]
    else:
        raise ValueError(f"{path}: needs `correct` or `pred_label`")
    if correct.shape != (len(layers), len(names), N):
        raise ValueError(f"{path}: correct has shape {correct.shape}, expected "
                         f"{(len(layers), len(names), N)}")
    if "base_pred" in files:
        base_pred = np.asarray(z["base_pred"]).astype(np.int64)
    else:
        base_pred = np.asarray(z["base_logits"]).argmax(1).astype(np.int64)
    skipped = np.asarray(z["skipped"]).astype(bool) if "skipped" in files else None
    if "eval_index" in files:
        eval_index = np.asarray(z["eval_index"]).astype(np.int64)
    else:
        print(f"[warn] {path}: no eval_index; assuming the first N feats rows")
        eval_index = np.arange(N, dtype=np.int64)
    val_index = (np.asarray(z["val_index"]).astype(np.int64) if "val_index" in files
                 else np.zeros(0, dtype=np.int64))
    eval_split = _scalar(z["eval_split"], str, "") if "eval_split" in files else ""
    return {"names": names, "unit_idx": unit_idx, "rand_idx": rand_idx, "layers": layers,
            "labels": labels, "correct": correct, "base_pred": base_pred,
            "skipped": skipped, "K": K, "eval_index": eval_index, "val_index": val_index,
            "eval_split": eval_split, "npz": z}


def analyze_intervention(inter: dict, n_boot: int = 1000, n_perm: int = 2000,
                         rng: np.random.Generator | None = None,
                         centroid: bool = False) -> dict:
    """Causal matrix of one loaded intervention (load_intervention).

    Returns the spec keys of causal_{measures,regions}.npz except class_names:
    unit_names, layers, dacc, drand, drand_sd, excess, p_class, t1_p, t1_mean, confusion,
    class_count, base_acc, val_index, eval_index, eval_split; plus (not saved by default)
    `n_random_used` and, if centroid=True and final_feat is present, `centroid_shift`.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    K = inter["K"]
    labels = inter["labels"]
    unit_idx, rand_idx = inter["unit_idx"], inter["rand_idx"]
    eff = causal_effects(inter["correct"], inter["base_pred"] == labels, labels, K,
                         unit_idx, rand_idx, inter["skipped"])
    p_class = bootstrap_p_class(eff["d"], labels, K, n_boot=n_boot, rng=rng)
    valid = eff["count"] > 0
    t1_mean, t1_p = signflip_test(eff["excess"], n_perm=n_perm, rng=rng, valid=valid)
    out = {
        "unit_names": np.array([inter["names"][i] for i in unit_idx]),
        "layers": inter["layers"],
        "dacc": eff["dacc"], "drand": eff["drand"], "drand_sd": eff["drand_sd"],
        "excess": eff["excess"], "p_class": p_class, "t1_p": t1_p, "t1_mean": t1_mean,
        "confusion": confusion_matrix(labels, inter["base_pred"], K),
        "class_count": eff["count"], "base_acc": eff["base_acc"],
        "val_index": inter["val_index"], "eval_index": inter["eval_index"],
        "eval_split": np.array(inter["eval_split"]),
        "n_random_used": eff["n_random_used"],
    }
    z = inter.get("npz")
    if centroid and z is not None and "final_feat" in z.files and "base_final_feat" in z.files:
        out["centroid_shift"] = centroid_shift_matrix(
            z["final_feat"], z["base_final_feat"], labels, K, unit_idx, inter["skipped"])
    return out

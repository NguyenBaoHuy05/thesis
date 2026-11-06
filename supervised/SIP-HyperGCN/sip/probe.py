"""Layer-wise continuous probes (ridge regression) and class-cluster metrics (v3).

A measure m (sip.measures) is a small vector of temporal statistics, Z_m (N, d_m),
d_m = number of its columns. A probe predicts the STANDARDISED Z_m (mean / sd of the fit
rows) from the pooled feature of one layer, X = feat_l{l} (N, C), with ridge regression.

Ridge (closed form, no sklearn):
    X is standardised with the fit-row mean / sd (sd < 1e-8 -> 1), Z is centred.
    With the thin SVD Xs = U S V^T,  W(a) = V diag(s / (s^2 + a)) U^T Zc.
    The penalty a is chosen per measure (shared by its columns) from `alphas` by
    generalised cross-validation on the fit rows:
        GCV(a) = sum_cols ||(I - H_a) Zc||^2 / (n (1 - tr(H_a) / n)^2),
        H_a = U diag(s^2 / (s^2 + a)) U^T.
    One SVD per layer serves every measure and every a.

Metrics on the held-out rows (all pooled over the measure's standardised columns,
SS = sum over rows and columns):
    probe_r2        = 1 - SS(Z - Zhat) / SS(Z - mean(Z))
    class_r2        = 1 - SS(Z - mu_fit[y]) / SS(Z - mean(Z))
                      class-mean baseline: predict Z by the FIT-row mean of the TRUE class
                      (classes absent from the fit rows -> global fit mean). It measures how
                      much of Z is just a copy of the class label.
    gain_r2         = probe_r2 - class_r2
    within_class_r2 = 1 - SS_w(Z - Zhat) / SS_w(Z)
                      SS_w(A) = sum_i ||A_i - mean_{j: y_j = y_i} A_j||^2 is the within-class
                      sum of squares, class means taken on the EVALUATED rows. Equivalently,
                      the R^2 of the class-centred residual Z - Zbar_y predicted by the
                      class-centred probe output Zhat - Zhatbar_y. It is invariant to any
                      per-class offset of the probe, so it is > 0 only if the feature
                      encodes sample-level variation of Z beyond the class identity.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

__all__ = [
    "DEFAULT_ALPHAS",
    "group_columns",
    "standardize_fit",
    "ridge_fit",
    "ridge_predict",
    "class_mean_table",
    "within_ss",
    "total_ss",
    "within_class_var_frac",
    "regression_metrics",
    "probe_layer",
    "layerwise_probe",
    "class_centroids",
    "within_class_spread",
    "fisher_ratio",
    "centroid_shift",
    "knn_class_accuracy",
    "linear_class_accuracy",
    "per_class_accuracy",
]

DEFAULT_ALPHAS = tuple(float(a) for a in 10.0 ** np.arange(-1.0, 6.01, 0.5))


# ----------------------------------------------------------------------------- utils
def _subsample(n: int, max_n: int | None, seed: int) -> np.ndarray:
    idx = np.arange(n)
    if max_n is not None and n > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(n, max_n, replace=False))
    return idx


def group_columns(cols) -> "OrderedDict[str, np.ndarray]":
    """Flat "measure:stat" column names -> ordered {measure: column indices}
    (order of first appearance, i.e. sip.measures.MEASURE_NAMES order)."""
    out: "OrderedDict[str, list]" = OrderedDict()
    for i, c in enumerate(cols):
        out.setdefault(str(c).split(":", 1)[0], []).append(i)
    return OrderedDict((k, np.asarray(v, dtype=np.int64)) for k, v in out.items())


def _as_groups(groups, D: int) -> "OrderedDict[str, np.ndarray]":
    if groups is None:
        return OrderedDict([("all", np.arange(D))])
    items = groups.items() if isinstance(groups, dict) else enumerate(groups)
    out = OrderedDict()
    for name, g in items:
        idx = np.arange(D)[g] if isinstance(g, slice) else np.asarray(g, dtype=np.int64)
        out[str(name)] = idx
    return out


def standardize_fit(A, eps: float = 1e-8):
    """Column mean / sd of A (sd <= eps -> 1, so constant columns map to 0)."""
    A = np.asarray(A, dtype=np.float64)
    mu = A.mean(0)
    sd = A.std(0)
    sd = np.where(sd > eps, sd, 1.0)
    return mu, sd


# ----------------------------------------------------------------------------- ridge
def ridge_fit(X, Z, groups=None, alphas=DEFAULT_ALPHAS) -> dict:
    """Multi-output ridge X -> Z with a GCV-chosen penalty per column group (module doc).

    X: (n, p); Z: (n, D) (pass it already standardised); groups: {name: column idx or
    slice} or None (one group). Returns a model dict for ridge_predict.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    n, p = X.shape
    D = Z.shape[1]
    x_mu, x_sd = standardize_fit(X)
    Xs = (X - x_mu) / x_sd
    z_mu = Z.mean(0)
    Zc = Z - z_mu
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)                   # U (n,r), Vt (r,p)
    del Xs
    UtZ = U.T @ Zc                                                       # (r, D)
    del U
    zz = np.sum(Zc ** 2, axis=0)                                         # (D,)
    s2 = s ** 2
    G = _as_groups(groups, D)
    W = np.zeros((p, D))
    alpha_of, gcv_of = {}, {}
    for name, idx in G.items():
        if idx.size == 0:
            continue
        best_g, best_a = np.inf, float(alphas[0])
        for a in alphas:
            f = s2 / (s2 + a)
            rss = zz[idx] - np.sum((2.0 * f - f * f)[:, None] * UtZ[:, idx] ** 2, axis=0)
            dof = 1.0 - (f.sum() + 1.0) / n  # +1 for the (centred-out) intercept
            g = float(np.sum(np.maximum(rss, 0.0)) / (n * dof * dof)) if dof > 0 else np.inf
            if g < best_g:
                best_g, best_a = g, float(a)
        W[:, idx] = Vt.T @ ((s / (s2 + best_a))[:, None] * UtZ[:, idx])
        alpha_of[name] = best_a
        gcv_of[name] = best_g
    return {"x_mu": x_mu, "x_sd": x_sd, "z_mu": z_mu, "W": W, "alpha": alpha_of,
            "gcv": gcv_of, "groups": G}


def ridge_predict(model: dict, X, chunk: int = 8192) -> np.ndarray:
    X = np.asarray(X)
    out = np.empty((X.shape[0], model["W"].shape[1]), dtype=np.float64)
    for a in range(0, X.shape[0], chunk):
        Xs = (np.asarray(X[a:a + chunk], dtype=np.float64) - model["x_mu"]) / model["x_sd"]
        out[a:a + chunk] = Xs @ model["W"] + model["z_mu"]
    return out


# --------------------------------------------------------------- R^2 and class stats
def class_mean_table(Z, labels, K: int) -> np.ndarray:
    """(K, d) per-class mean of Z; classes without rows get the global mean."""
    Z = np.asarray(Z, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    cnt = np.bincount(labels, minlength=K)[:K].astype(np.float64)
    S = np.zeros((K, Z.shape[1]))
    np.add.at(S, labels, Z)
    mu = np.tile(Z.mean(0), (K, 1))
    ok = cnt > 0
    mu[ok] = S[ok] / cnt[ok, None]
    return mu


def total_ss(A) -> np.ndarray:
    """(n, d) -> (d,) sum of squares around the column mean."""
    A = np.asarray(A, dtype=np.float64)
    return np.sum((A - A.mean(0)) ** 2, axis=0)


def within_ss(A, labels, K: int) -> np.ndarray:
    """(n, d) -> (d,) within-class sum of squares sum_i ||A_i - mean_{class(i)} A||^2."""
    A = np.asarray(A, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    cnt = np.bincount(labels, minlength=K)[:K].astype(np.float64)
    S = np.zeros((K, A.shape[1]))
    np.add.at(S, labels, A)
    between = np.sum(S ** 2 / np.maximum(cnt, 1.0)[:, None], axis=0)
    return np.maximum(np.sum(A ** 2, axis=0) - between, 0.0)


def within_class_var_frac(Z, labels, K: int) -> np.ndarray:
    """(n, d) -> (d,) within-class SS / total SS per column (NaN for constant columns)."""
    w = within_ss(Z, labels, K)
    t = total_ss(Z)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(t > 0, w / np.where(t > 0, t, 1.0), np.nan)


def _r2(sse: float, sst: float) -> float:
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def regression_metrics(Z, Zhat, labels, class_mean, K: int) -> dict:
    """Pooled R^2 metrics of one measure (module doc). Z, Zhat: (n, d) standardised;
    class_mean: (K, d) FIT-row class means of the standardised Z."""
    Z = np.asarray(Z, dtype=np.float64)
    Zhat = np.asarray(Zhat, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    sst = float(total_ss(Z).sum())
    probe = _r2(float(np.sum((Z - Zhat) ** 2)), sst)
    cls = _r2(float(np.sum((Z - np.asarray(class_mean)[labels]) ** 2)), sst)
    wc = _r2(float(within_ss(Z - Zhat, labels, K).sum()), float(within_ss(Z, labels, K).sum()))
    return {"probe_r2": probe, "class_r2": cls, "gain_r2": probe - cls, "within_class_r2": wc}


# ----------------------------------------------------------------------- the probes
def probe_layer(X_fit, Z_fit, lab_fit, X_ev, Z_ev, lab_ev, groups, K: int,
                alphas=DEFAULT_ALPHAS) -> "OrderedDict[str, dict]":
    """Fit ridge probes of every measure group at ONE layer (fit rows -> evaluated rows).

    Z_fit / Z_ev: raw (N, D) measure columns; standardised here with the fit-row stats.
    Returns {measure: {"metrics": dict(+alpha, n_cols), "Z": (n_ev, d), "Zhat": (n_ev, d),
    "Zcls": (n_ev, d)}} (standardised; kept for bootstraps over the evaluated rows).
    """
    mu, sd = standardize_fit(Z_fit)
    Zf = (np.asarray(Z_fit, dtype=np.float64) - mu) / sd
    Ze = (np.asarray(Z_ev, dtype=np.float64) - mu) / sd
    G = _as_groups(groups, Zf.shape[1])
    model = ridge_fit(X_fit, Zf, G, alphas)
    Zhat = ridge_predict(model, X_ev)
    cm = class_mean_table(Zf, lab_fit, K)
    lab_ev = np.asarray(lab_ev).astype(np.int64)
    out = OrderedDict()
    for name, idx in G.items():
        Zcls = cm[lab_ev][:, idx]
        m = regression_metrics(Ze[:, idx], Zhat[:, idx], lab_ev, cm[:, idx], K)
        m["alpha"] = float(model["alpha"].get(name, np.nan))
        m["n_cols"] = int(idx.size)
        out[name] = {"metrics": m, "Z": Ze[:, idx], "Zhat": Zhat[:, idx], "Zcls": Zcls}
    return out


def layerwise_probe(train: dict, test: dict, layers, groups, num_class: int,
                    alphas=DEFAULT_ALPHAS, max_train: int | None = None, seed: int = 0,
                    verbose: bool = True, meas_key: str = "meas") -> list[dict]:
    """Probe every (layer, measure): fit on `train` rows, score on `test` rows.

    train/test: dict-like with feat_l{l}, labels, `meas_key` (N, D).
    groups: {measure: column indices into meas}. Returns one row dict per (layer, measure).
    """
    lab_tr = np.asarray(train["labels"]).astype(np.int64)
    lab_te = np.asarray(test["labels"]).astype(np.int64)
    idx = _subsample(len(lab_tr), max_train, seed)
    Ztr = np.asarray(train[meas_key], dtype=np.float64)[idx]
    Zte = np.asarray(test[meas_key], dtype=np.float64)
    rows = []
    for l in layers:
        Xtr = np.asarray(train[f"feat_l{l}"], dtype=np.float32)[idx]
        Xte = np.asarray(test[f"feat_l{l}"], dtype=np.float32)
        res = probe_layer(Xtr, Ztr, lab_tr[idx], Xte, Zte, lab_te, groups, num_class, alphas)
        for name, r in res.items():
            m = r["metrics"]
            rows.append({"layer": int(l), "measure": name, **m})
            if verbose:
                print(f"[probe] l{l} {name:<22s} R2={m['probe_r2']:.3f} "
                      f"class={m['class_r2']:.3f} gain={m['gain_r2']:+.3f} "
                      f"wc={m['within_class_r2']:.3f} a={m['alpha']:.3g}", flush=True)
    return rows


# ------------------------------------------------------------------ cluster metrics
def class_centroids(X, y, num_class: int):
    """Returns (centroids (K, D), counts (K,)). Empty classes get NaN centroids."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    cnt = np.bincount(y, minlength=num_class).astype(float)
    S = np.zeros((num_class, X.shape[1]))
    np.add.at(S, y, X)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = S / cnt[:, None]
    return mu, cnt


def within_class_spread(X, y, num_class: int, centroids=None) -> np.ndarray:
    """(K,) RMS distance of class-k samples to their centroid: sqrt(mean ||x - mu_k||^2)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if centroids is None:
        centroids, _ = class_centroids(X, y, num_class)
    d2 = np.sum((X - centroids[y]) ** 2, axis=1)
    cnt = np.bincount(y, minlength=num_class).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.bincount(y, weights=d2, minlength=num_class) / cnt)


def fisher_ratio(X, y, num_class: int) -> float:
    """tr(S_B) / tr(S_W), S_B = sum_k n_k (mu_k - mu)(mu_k - mu)^T / N,
    S_W = sum_i (x_i - mu_{y_i})(x_i - mu_{y_i})^T / N."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    mu_k, cnt = class_centroids(X, y, num_class)
    present = cnt > 0
    mu = X.mean(0)
    sb = np.sum(cnt[present] * np.sum((mu_k[present] - mu) ** 2, axis=1)) / len(y)
    sw = np.sum((X - mu_k[y]) ** 2) / len(y)
    return float(sb / sw) if sw > 0 else float("nan")


def centroid_shift(X_base, X_int, y, num_class: int) -> dict:
    """Per-class displacement caused by an intervention, normalised by the base
    within-class spread (RMS radius) of that class.

    centroid_shift[k]    = ||mu_k(int) - mu_k(base)|| / spread_k(base)
    sample_shift[k]      = mean_{i in k} ||x_i(int) - x_i(base)|| / spread_k(base)
    nearest_centroid_flip[k] = fraction of class-k samples whose nearest BASE
                           centroid differs before vs after the intervention.
    """
    Xb = np.asarray(X_base, dtype=np.float64)
    Xi = np.asarray(X_int, dtype=np.float64)
    y = np.asarray(y).astype(int)
    mu_b, cnt = class_centroids(Xb, y, num_class)
    mu_i, _ = class_centroids(Xi, y, num_class)
    spread = within_class_spread(Xb, y, num_class, mu_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        cshift = np.linalg.norm(mu_i - mu_b, axis=1) / spread
        sshift = np.bincount(y, weights=np.linalg.norm(Xi - Xb, axis=1),
                             minlength=num_class) / cnt / spread
    present = cnt > 0
    M = np.where(present[:, None], mu_b, 0.0)               # (K, D), absent -> 0
    m2 = np.where(present, np.sum(M ** 2, axis=1), np.inf)  # absent classes never win

    def _nearest(Z):
        # argmin_k ||z - m_k||^2 = argmin_k (|m_k|^2 - 2 z.m_k)
        return np.argmin(m2[None, :] - 2.0 * (Z @ M.T), axis=1)

    flip = (_nearest(Xb) != _nearest(Xi)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        flip_k = np.bincount(y, weights=flip, minlength=num_class) / cnt
    return {"centroid_shift": cshift, "sample_shift": sshift,
            "nearest_centroid_flip": flip_k, "spread": spread}


def knn_class_accuracy(X_tr, y_tr, X_te, y_te, k: int = 5, max_train: int | None = None,
                       seed: int = 0) -> float:
    """kNN (cosine, via L2-normalised features) class accuracy."""
    from sklearn.neighbors import KNeighborsClassifier
    idx = _subsample(len(y_tr), max_train, seed)

    def _n(A):
        A = np.asarray(A, dtype=np.float32)
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(_n(X_tr)[idx], np.asarray(y_tr)[idx])
    return float(np.mean(clf.predict(_n(X_te)) == np.asarray(y_te)))


def linear_class_accuracy(X_tr, y_tr, X_te, y_te, C: float = 1.0, max_iter: int = 500,
                          max_train: int | None = None, seed: int = 0) -> float:
    """Multinomial logistic-regression (standardised) class accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    idx = _subsample(len(y_tr), max_train, seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))
    clf.fit(np.asarray(X_tr, dtype=np.float32)[idx], np.asarray(y_tr)[idx])
    return float(np.mean(clf.predict(np.asarray(X_te, dtype=np.float32)) == np.asarray(y_te)))


def per_class_accuracy(pred, y, num_class: int) -> np.ndarray:
    """(K,) accuracy per true class (NaN for absent classes)."""
    y = np.asarray(y).astype(int)
    hit = (np.asarray(pred).astype(int) == y).astype(float)
    cnt = np.bincount(y, minlength=num_class).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.bincount(y, weights=hit, minlength=num_class) / cnt

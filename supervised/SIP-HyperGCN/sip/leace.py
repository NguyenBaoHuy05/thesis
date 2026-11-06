"""Closed-form LEACE (Belrose et al., 2023, "LEACE: Perfect linear concept erasure
in closed form").

Given features X (N, D) and concept labels Z (N, K), LEACE finds the affine map

    r(x) = x - W^+ P W (x - mu)

with
    W   = Sigma_XX^{-1/2}                      (whitening, on the support of Sigma_XX)
    W^+ = Sigma_XX^{+1/2}                      (its pseudo-inverse)
    P   = orthogonal projector onto colspace(W Sigma_XZ)
    E   = W^+ P W                              (oblique projection, rank <= K)

After erasure Cov(r(X), Z) = 0 (so no linear classifier can predict Z better than a
constant), and among all such maps r minimises E||r(x) - x||^2_M for every
PSD M (in particular the mean squared edit).

Proof sketch of the zero cross-covariance (used by `verify_erasure`):
    Cov(r(X), Z) = Sigma_XZ - E Sigma_XZ = Sigma_XZ - W^+ P W Sigma_XZ
                 = Sigma_XZ - W^+ W Sigma_XZ        (P fixes colspace(W Sigma_XZ))
                 = Sigma_XZ - U_r U_r^T Sigma_XZ = 0
because Sigma_XZ = Xc^T Zc / N lies in colspace(Xc^T) = colspace(Sigma_XX) = span(U_r)
(exactly, up to the eigenvalues we truncate for numerical stability).

Shared interface (README):
    fit_leace(X, Z) -> {"mean": (D,), "E": (D, D)}
    erased = X - (X - mean) @ E.T
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "fit_leace",
    "apply_leace_np",
    "apply_leace_torch",
    "verify_erasure",
    "random_eraser",
    "save_eraser",
    "load_eraser",
]


def _as_2d(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.ndim != 2:
        raise ValueError(f"Z must be (N,) or (N, K), got shape {Z.shape}")
    return Z.astype(np.float64)


def fit_leace(
    X: np.ndarray,
    Z: np.ndarray,
    eps: float = 1e-8,
    rel_tol: float = 1e-6,
    rank_tol: float = 1e-6,
) -> dict:
    """Fit the closed-form LEACE eraser.

    Args:
        X: (N, D) features.
        Z: (N,) or (N, K) concept labels / scores (binary, one-hot or continuous).
        eps: absolute floor on retained eigenvalues of Sigma_XX.
        rel_tol: eigenvalues of Sigma_XX below rel_tol * max_eig are treated as zero
            (rank truncation; W and W^+ live on the retained eigenspace).
        rank_tol: singular values of W Sigma_XZ below rank_tol * max_sv are dropped
            when building the projector P (handles collinear / constant concepts).

    Returns:
        dict with
            "mean": (D,) float32 feature mean,
            "E":    (D, D) float32 oblique projection, erased = X - (X - mean) @ E.T,
            "rank": int, rank of the erased subspace (<= K),
            "whiten_rank": int, number of retained eigenvalues of Sigma_XX.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be (N, D), got shape {X.shape}")
    Z = _as_2d(Z)
    N, D = X.shape
    if Z.shape[0] != N:
        raise ValueError(f"X has {N} rows but Z has {Z.shape[0]}")

    mu = X.mean(axis=0)
    Xc = X - mu
    Zc = Z - Z.mean(axis=0)

    sigma_xx = (Xc.T @ Xc) / N                       # (D, D)
    sigma_xz = (Xc.T @ Zc) / N                       # (D, K)
    sigma_xx = 0.5 * (sigma_xx + sigma_xx.T)         # enforce symmetry

    # --- whitening W = Sigma^{-1/2}, W^+ = Sigma^{1/2} on the retained eigenspace
    evals, evecs = np.linalg.eigh(sigma_xx)          # ascending
    max_ev = float(evals.max()) if evals.size else 0.0
    thresh = max(eps, rel_tol * max_ev)
    keep = evals > thresh
    U = evecs[:, keep]                               # (D, r)
    lam = evals[keep]                                # (r,)
    W = (U / np.sqrt(lam)) @ U.T                     # (D, D)
    W_pinv = (U * np.sqrt(lam)) @ U.T                # (D, D)

    # --- P = orthogonal projector onto colspace(W Sigma_XZ)
    M = W @ sigma_xz                                 # (D, K)
    if M.size == 0 or not np.any(np.abs(M) > 0):
        E = np.zeros((D, D))
        rank = 0
    else:
        Um, s, _ = np.linalg.svd(M, full_matrices=False)
        r = int(np.sum(s > rank_tol * s.max()))
        Q = Um[:, :r]                                # (D, r) orthonormal basis
        P = Q @ Q.T
        E = W_pinv @ P @ W
        rank = r

    return {
        "mean": mu.astype(np.float32),
        "E": E.astype(np.float32),
        "rank": int(rank),
        "whiten_rank": int(keep.sum()),
    }


def apply_leace_np(X: np.ndarray, eraser: dict) -> np.ndarray:
    """Apply the eraser to X of shape (..., D) (channel axis last)."""
    X = np.asarray(X)
    out_dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float32
    mean = np.asarray(eraser["mean"], dtype=np.float64)
    E = np.asarray(eraser["E"], dtype=np.float64)
    Xf = X.astype(np.float64)
    return (Xf - (Xf - mean) @ E.T).astype(out_dtype)


def apply_leace_torch(h, eraser: dict, channel_dim: int = 1):
    """Apply the same affine eraser along `channel_dim` of a torch tensor.

    Works for pooled (N, C) features (channel_dim=1 or -1) and for feature maps such
    as (N*M, C, T, V) (channel_dim=1): the channel axis is moved last, the map
    x - (x - mean) @ E^T is applied per position, and the axis is moved back.
    Because the map is affine, applying it per (t, v) position and then average
    pooling equals erasing the pooled feature on which the eraser was fitted.
    """
    import torch  # lazy: torch is not available on every machine

    cache = eraser.setdefault("_torch_cache", {})
    key = (str(h.device), str(h.dtype))
    if key not in cache:
        cache[key] = (
            torch.as_tensor(np.asarray(eraser["mean"]), dtype=h.dtype, device=h.device),
            torch.as_tensor(np.asarray(eraser["E"]), dtype=h.dtype, device=h.device),
        )
    mean, E = cache[key]
    C = h.shape[channel_dim]
    if C != E.shape[0]:
        raise ValueError(f"channel size {C} != eraser dim {E.shape[0]}")
    hm = torch.movedim(h, channel_dim, -1)
    out = hm - torch.matmul(hm - mean, E.transpose(0, 1))
    return torch.movedim(out, -1, channel_dim).contiguous()


def verify_erasure(X: np.ndarray, Z: np.ndarray, eraser: dict) -> dict:
    """Self-check: cross-covariance between erased X and Z should be ~0.

    Returns:
        cross_cov_norm_before: ||Cov(X, Z)||_F
        cross_cov_norm:        ||Cov(X', Z)||_F  (should be ~0, float32 round-off)
        cross_cov_rel:         ratio after / before
        mean_sq_edit:          mean_i ||x_i - x'_i||^2
        rel_sq_edit:           mean_sq_edit / mean_i ||x_i - mu||^2
    """
    X = np.asarray(X, dtype=np.float64)
    Z = _as_2d(Z)
    Xe = apply_leace_np(X, eraser).astype(np.float64)
    N = X.shape[0]
    Zc = Z - Z.mean(axis=0)
    before = np.linalg.norm((X - X.mean(0)).T @ Zc / N)
    after = np.linalg.norm((Xe - Xe.mean(0)).T @ Zc / N)
    edit = np.sum((X - Xe) ** 2, axis=1).mean()
    total = np.sum((X - X.mean(0)) ** 2, axis=1).mean()
    return {
        "cross_cov_norm_before": float(before),
        "cross_cov_norm": float(after),
        "cross_cov_rel": float(after / before) if before > 0 else 0.0,
        "mean_sq_edit": float(edit),
        "rel_sq_edit": float(edit / total) if total > 0 else 0.0,
    }


def random_eraser(X: np.ndarray, rank: int = 1, seed: int = 0) -> dict:
    """Control eraser: orthogonal projection removing `rank` random directions.

    Same output format as `fit_leace` (erased = X - (X - mean) @ E.T).
    """
    X = np.asarray(X, dtype=np.float64)
    D = X.shape[1]
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((D, max(int(rank), 0)))
    if G.shape[1] == 0:
        E = np.zeros((D, D))
    else:
        Q, _ = np.linalg.qr(G)
        E = Q @ Q.T
    return {"mean": X.mean(0).astype(np.float32), "E": E.astype(np.float32),
            "rank": int(rank), "whiten_rank": D}


def save_eraser(path: str, eraser: dict) -> None:
    np.savez(path, mean=eraser["mean"], E=eraser["E"],
             rank=np.int64(eraser.get("rank", -1)))


def load_eraser(path: str) -> dict:
    d = np.load(path)
    return {"mean": d["mean"], "E": d["E"], "rank": int(d["rank"])}

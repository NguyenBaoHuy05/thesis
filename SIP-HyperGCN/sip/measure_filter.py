"""Strong-measure filter (v3): five tiers + region cross-check, decided on VAL only.

Data split (docs/PIPELINE_V3.md): VAL = `val_index` rows of train_feats.npz (the rows the
VAL interventions were run on); TRAIN-FIT = all other train rows. Nothing from TEST.

A measure m has columns Z_m = meas[:, cols(m)] ("m:stat", sip.measures.COLUMNS) and body
regions sip.measures.MEASURE_REGIONS[m]. Tiers (m is selected iff all six hold):

1. grounded      m is in sip.measures.MEASURE_NAMES (closed-form geometric function of the
                 joint coordinates, no learned component), it has >= 1 column in
                 train_feats `meas_cols`, and all its values are finite.
2. within_class  within_frac = mean over m's columns of (within-class SS / total SS) on
                 TRAIN-FIT (columns constant on TRAIN-FIT are ignored); holds iff
                 within_frac >= min_within_frac. A measure that is (almost) constant
                 inside each class is just a copy of the label.
3. encoded       For every measure layer l of causal_measures: ridge probe (sip.probe,
                 GCV penalty) feat_l -> standardised Z_m fit on TRAIN-FIT, scored on VAL:
                 probe_r2, class_r2 (TRAIN-FIT class mean of the true class),
                 gain_r2 = probe_r2 - class_r2, within_class_r2 (sip.probe doc).
                 Chosen layer = argmax gain_r2 (fixed here, BEFORE any causal result is
                 read). Holds iff gain_r2 >= min_gain_r2 and within_class_r2 >= min_wc_r2
                 at the chosen layer.
4. causal        At the chosen layer, from causal_measures.npz (VAL): t1_p < alpha and
                 t1_mean > min_t1_mean.
5. stable        (a) enc_agreement: bootstrap the VAL rows n_boot_enc times (probe fixed),
                 recompute gain_r2 and within_class_r2 and the encoded decision;
                 agreement = fraction of resamples whose decision equals the original one.
                 (b) t1_agreement: resample the classes (with replacement) n_boot_t1 times,
                 recompute the sign-flip t1 test on the resampled excess vector at the
                 chosen layer and the causal decision; agreement with the original decision.
                 stable iff both agreements >= min_agreement; stability = min of the two.
6. region_check  Some region r of m has region t1_p < region_alpha in causal_regions.npz
                 (VAL) at one of region_layers (default 0, 3: the layers where joint
                 mean-ablation still measures the information source). best_region /
                 region_layer / region_t1_p = the (r, l) with the smallest t1_p (ties ->
                 larger t1_mean).
"""
from __future__ import annotations

import numpy as np

from sip.causal import signflip_test
from sip.probe import total_ss, within_class_var_frac, within_ss

__all__ = [
    "TIERS",
    "DEFAULT_CRITERIA",
    "grounded_tier",
    "within_frac",
    "choose_layer",
    "encoded_ok",
    "encoded_bootstrap_agreement",
    "causal_decision",
    "t1_bootstrap_agreement",
    "region_check",
]

TIERS = ["grounded", "within_class", "encoded", "causal", "stable", "region_check"]

DEFAULT_CRITERIA = {
    "min_within_frac": 0.2,
    "min_gain_r2": 0.02,
    "min_wc_r2": 0.05,
    "alpha": 0.05,
    "min_t1_mean": 0.005,
    "n_boot_enc": 200,
    "n_boot_t1": 200,
    "n_perm_t1": 1000,
    "min_agreement": 0.8,
    "region_alpha": 0.05,
    "region_layers": [0, 3],
}


def grounded_tier(name: str, Z) -> bool:
    """Tier 1 for one measure; Z = its (N, d) columns (may have d = 0)."""
    try:
        from sip.measures import MEASURE_NAMES
        known = str(name) in set(MEASURE_NAMES)
    except Exception:  # sip.measures unavailable -> cannot vouch for the formula
        known = False
    Z = np.asarray(Z)
    return bool(known and Z.ndim == 2 and Z.shape[1] > 0 and np.isfinite(Z).all())


def within_frac(Z_fit, lab_fit, K: int) -> float:
    """Tier-2 statistic: mean over columns of within-class SS / total SS (TRAIN-FIT)."""
    r = within_class_var_frac(Z_fit, lab_fit, K)
    r = r[np.isfinite(r)]
    return float(r.mean()) if r.size else float("nan")


def choose_layer(gain: np.ndarray) -> np.ndarray:
    """gain (L, M) -> (M,) index of the layer with the largest finite gain (0 if none)."""
    g = np.where(np.isfinite(gain), gain, -np.inf)
    return np.argmax(g, axis=0)


def encoded_ok(gain_r2, within_class_r2, crit: dict) -> bool:
    g, w = float(gain_r2), float(within_class_r2)
    return bool(np.isfinite(g) and np.isfinite(w) and g >= crit["min_gain_r2"]
                and w >= crit["min_wc_r2"])


def _r2(sse, sst):
    return 1.0 - sse / sst if sst > 0 else float("nan")


def encoded_bootstrap_agreement(Z, Zhat, Zcls, labels, K: int, decision: bool, crit: dict,
                                rng: np.random.Generator) -> tuple[float, float]:
    """Bootstrap of the VAL rows with the probe fixed. Z, Zhat, Zcls: (n, d) standardised
    target / probe output / TRAIN-FIT class-mean prediction. Returns (agreement with
    `decision`, fraction of resamples where encoded holds)."""
    Z = np.asarray(Z, dtype=np.float64)
    Zhat = np.asarray(Zhat, dtype=np.float64)
    Zcls = np.asarray(Zcls, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    n = len(labels)
    B = int(crit["n_boot_enc"])
    if n == 0 or B <= 0:
        return float("nan"), float("nan")
    dec = np.zeros(B, dtype=bool)
    for b in range(B):
        i = rng.integers(0, n, size=n)
        z, e, lab = Z[i], Z[i] - Zhat[i], labels[i]
        sst = float(total_ss(z).sum())
        probe = _r2(float(np.sum(e ** 2)), sst)
        cls = _r2(float(np.sum((z - Zcls[i]) ** 2)), sst)
        wc = _r2(float(within_ss(e, lab, K).sum()), float(within_ss(z, lab, K).sum()))
        dec[b] = encoded_ok(probe - cls, wc, crit)
    return float(np.mean(dec == bool(decision))), float(np.mean(dec))


def causal_decision(t1_p, t1_mean, crit: dict):
    t1_p = np.asarray(t1_p, dtype=np.float64)
    t1_mean = np.asarray(t1_mean, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (np.isfinite(t1_p) & np.isfinite(t1_mean) & (t1_p < crit["alpha"])
                & (t1_mean > crit["min_t1_mean"]))


def t1_bootstrap_agreement(excess_k: np.ndarray, decision: bool, crit: dict,
                           rng: np.random.Generator) -> tuple[float, float]:
    """excess_k: (K',) excess over the valid classes at the chosen layer.
    Returns (agreement with `decision`, fraction of resamples where causal holds)."""
    x = np.asarray(excess_k, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, x.size, size=(int(crit["n_boot_t1"]), x.size))
    mean, p = signflip_test(x[idx], n_perm=int(crit["n_perm_t1"]), rng=rng)    # (B,)
    dec = causal_decision(p, mean, crit)
    return float(np.mean(dec == bool(decision))), float(np.mean(dec))


def region_check(regions, reg_names, reg_layers, reg_t1_p, reg_t1_mean, crit: dict) -> dict:
    """Tier 6 for one measure.

    regions: the measure's region names; reg_names (R,), reg_layers (L_r,), reg_t1_p /
    reg_t1_mean (L_r, R) from causal_regions.npz. Returns {"ok", "best_region",
    "region_layer", "region_t1_p", "region_t1_mean"} (None / NaN if no candidate).
    """
    reg_names = [str(r) for r in reg_names]
    reg_layers = [int(l) for l in reg_layers]
    best = None                                     # (p, -mean, region, layer, mean)
    for r in regions:
        if str(r) not in reg_names:
            continue
        ri = reg_names.index(str(r))
        for l in crit["region_layers"]:
            if int(l) not in reg_layers:
                continue
            li = reg_layers.index(int(l))
            p, m = float(reg_t1_p[li, ri]), float(reg_t1_mean[li, ri])
            if not np.isfinite(p):
                continue
            key = (p, -m if np.isfinite(m) else np.inf)
            if best is None or key < best[:2]:
                best = (key[0], key[1], str(r), int(l), m)
    if best is None:
        return {"ok": False, "best_region": None, "region_layer": None,
                "region_t1_p": float("nan"), "region_t1_mean": float("nan")}
    return {"ok": bool(best[0] < crit["region_alpha"]), "best_region": best[2],
            "region_layer": best[3], "region_t1_p": best[0], "region_t1_mean": best[4]}

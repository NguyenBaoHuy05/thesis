"""Step 4 (v3): strong-measure filter (tiers 1-5 + region check, VAL only;
see sip/measure_filter.py).

Usage:
    python scripts/select_measures.py --train_feats work_dir/sip/train_feats.npz \
        --causal_dir work_dir/sip/causal_val --out_dir work_dir/sip/measures

Inputs: train_feats.npz (feat_l*, labels, meas, meas_cols) and, in --causal_dir,
causal_measures.npz + causal_regions.npz of the VAL run (analyze_intervention.py).
VAL = `val_index` stored in causal_measures.npz (rows of train_feats.npz);
TRAIN-FIT = the other train rows. Probes are fit on TRAIN-FIT and scored on VAL; the
layer of each measure (largest gain_r2 among the measure layers of causal_measures) is
fixed before any causal result is read.

Outputs (in --out_dir):
  strong_measures.json  {"measures": [{"name", "layer", "regions", "tiers": {grounded,
                        within_class, encoded, causal, stable, region_check},
                        "metrics": {within_frac, probe_r2, class_r2, gain_r2,
                        within_class_r2, t1_p, t1_mean, stability, region_t1_p,
                        best_region, region_layer}}], "selected": [...],
                        "layer_of": {name: layer}, "criteria": {...}}
                        (strict JSON: NaN / inf are written as null)
  measure_tiers.csv     one row per measure (tiers + metrics + bootstrap details)
  probe_val.csv         probe metrics per (layer, measure) on VAL
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from sip.measure_filter import (DEFAULT_CRITERIA, TIERS, causal_decision,  # noqa: E402
                                choose_layer, encoded_bootstrap_agreement, encoded_ok,
                                grounded_tier, region_check, t1_bootstrap_agreement,
                                within_frac)
from sip.probe import DEFAULT_ALPHAS, group_columns, probe_layer  # noqa: E402

PROBE_KEYS = ["probe_r2", "class_r2", "gain_r2", "within_class_r2", "alpha"]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                        for k, v in r.items()})


def json_safe(x):
    """Recursively convert numpy types to Python and non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if np.isfinite(x) else None
    return x


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    D = DEFAULT_CRITERIA
    ap = argparse.ArgumentParser(description="Select strong measures (VAL only)")
    ap.add_argument("--train_feats", required=True)
    ap.add_argument("--causal_dir", required=True,
                    help="dir with causal_measures.npz and causal_regions.npz (VAL run)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_within_frac", type=float, default=D["min_within_frac"])
    ap.add_argument("--min_gain_r2", type=float, default=D["min_gain_r2"])
    ap.add_argument("--min_wc_r2", type=float, default=D["min_wc_r2"])
    ap.add_argument("--alpha", type=float, default=D["alpha"], help="t1_p threshold")
    ap.add_argument("--min_t1_mean", type=float, default=D["min_t1_mean"])
    ap.add_argument("--n_boot_enc", type=int, default=D["n_boot_enc"])
    ap.add_argument("--n_boot_t1", type=int, default=D["n_boot_t1"])
    ap.add_argument("--n_perm_t1", type=int, default=D["n_perm_t1"])
    ap.add_argument("--min_agreement", type=float, default=D["min_agreement"])
    ap.add_argument("--region_alpha", type=float, default=D["region_alpha"])
    ap.add_argument("--region_layers", type=int, nargs="+", default=D["region_layers"])
    ap.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
                    help="ridge penalty grid (GCV per measure)")
    ap.add_argument("--max_fit", type=int, default=None,
                    help="subsample TRAIN-FIT rows for the probes (speed)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    crit = {k: getattr(args, k) for k in DEFAULT_CRITERIA}
    crit["region_layers"] = [int(l) for l in args.region_layers]

    # ---- causal (VAL): measures + regions
    cm_path = os.path.join(args.causal_dir, "causal_measures.npz")
    cr_path = os.path.join(args.causal_dir, "causal_regions.npz")
    for p in (cm_path, cr_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} missing (run analyze_intervention.py with "
                                    f"--measures and --regions)")
    cz = np.load(cm_path, allow_pickle=False)
    if "val_index" not in cz.files or len(cz["val_index"]) == 0:
        raise ValueError("causal_measures.npz has no val_index (VAL run of the v3 pipeline?)")
    if str(np.asarray(cz["eval_split"]).item()) not in ("train", ""):
        print("[warn] --causal_dir was not computed on the VAL run (eval_split != train)")
    names = [str(n) for n in cz["unit_names"]]
    layers = [int(l) for l in cz["layers"]]
    t1_p = np.asarray(cz["t1_p"], dtype=np.float64)          # (L,M)
    t1_mean = np.asarray(cz["t1_mean"], dtype=np.float64)    # (L,M)
    excess = np.asarray(cz["excess"], dtype=np.float64)      # (L,M,K)
    count = np.asarray(cz["class_count"])
    K = excess.shape[2]
    valid_k = count > 0
    val_idx = np.asarray(cz["val_index"]).astype(np.int64)
    L, M = len(layers), len(names)

    rz = np.load(cr_path, allow_pickle=False)
    reg_names = [str(n) for n in rz["unit_names"]]
    reg_layers = [int(l) for l in rz["layers"]]
    reg_t1_p = np.asarray(rz["t1_p"], dtype=np.float64)
    reg_t1_mean = np.asarray(rz["t1_mean"], dtype=np.float64)
    if "eval_index" in rz.files and not np.array_equal(np.asarray(rz["eval_index"]),
                                                       np.asarray(cz["eval_index"])):
        print("[warn] causal_regions and causal_measures were computed on different rows")
    missing_rl = [l for l in crit["region_layers"] if l not in reg_layers]
    if missing_rl:
        print(f"[warn] region layers {missing_rl} not in causal_regions layers {reg_layers}")
    try:
        from sip.measures import MEASURE_REGIONS
    except Exception as e:
        print(f"[warn] sip.measures.MEASURE_REGIONS unavailable ({e}); region_check fails")
        MEASURE_REGIONS = {}

    # ---- train features: VAL / TRAIN-FIT
    tf = np.load(args.train_feats, allow_pickle=False)
    labels = np.asarray(tf["labels"]).astype(np.int64)
    meas = np.asarray(tf["meas"], dtype=np.float64)
    groups_all = group_columns([str(c) for c in tf["meas_cols"]])
    missing = [n for n in names if n not in groups_all]
    if missing:
        print(f"[warn] measures without columns in train_feats meas_cols: {missing}")
    groups = {n: groups_all[n] for n in names if n in groups_all}
    fit_mask = np.ones(len(labels), dtype=bool)
    fit_mask[val_idx] = False
    fit_idx = np.flatnonzero(fit_mask)
    if args.max_fit is not None and len(fit_idx) > args.max_fit:
        fit_idx = np.sort(rng.choice(fit_idx, args.max_fit, replace=False))
    lab_fit, lab_val = labels[fit_idx], labels[val_idx]
    print(f"[split] TRAIN-FIT={len(fit_idx)} VAL={len(val_idx)} L={L} M={M} K={K}")

    # finite-safe copies for probing (grounded tier already flags non-finite measures)
    meas_safe = np.where(np.isfinite(meas), meas, 0.0)
    Z_fit, Z_val = meas_safe[fit_idx], meas_safe[val_idx]

    # ---- tiers 1, 2
    grounded, wfrac = {}, {}
    meas_used = meas[np.concatenate([fit_idx, val_idx])]
    for n in names:
        cols = groups.get(n, np.zeros(0, dtype=np.int64))
        grounded[n] = grounded_tier(n, meas_used[:, cols])
        wfrac[n] = within_frac(Z_fit[:, cols], lab_fit, K) if cols.size else float("nan")
    del meas_used, meas, meas_safe
    within = {n: bool(np.isfinite(wfrac[n]) and wfrac[n] >= args.min_within_frac)
              for n in names}

    # ---- tier 3: probes at every measure layer, then choose the layer (pre-causal)
    E = {k: np.full((L, M), np.nan) for k in PROBE_KEYS}
    keep = {}                      # (li, name) -> (Z, Zhat, Zcls) standardised VAL arrays
    probe_rows = []
    for li, l in enumerate(layers):
        X = np.asarray(tf[f"feat_l{l}"], dtype=np.float32)
        res = probe_layer(X[fit_idx], Z_fit, lab_fit, X[val_idx], Z_val, lab_val, groups, K,
                          alphas=args.alphas)
        del X
        for j, n in enumerate(names):
            if n not in res:
                continue
            m = res[n]["metrics"]
            for k in PROBE_KEYS:
                E[k][li, j] = m[k]
            keep[(li, n)] = (res[n]["Z"], res[n]["Zhat"], res[n]["Zcls"])
            probe_rows.append({"layer": l, "measure": n, "n_cols": m["n_cols"],
                               **{k: _f(m[k]) for k in PROBE_KEYS}})
            print(f"[probe] l{l} {n:<22s} R2={m['probe_r2']:.3f} class={m['class_r2']:.3f} "
                  f"gain={m['gain_r2']:+.3f} wc={m['within_class_r2']:.3f}", flush=True)
        del res
    del tf
    write_csv(os.path.join(args.out_dir, "probe_val.csv"), probe_rows)
    best = choose_layer(E["gain_r2"])                         # (M,) layer index (pre-causal)
    enc = np.array([encoded_ok(E["gain_r2"][best[j], j], E["within_class_r2"][best[j], j],
                               crit) for j in range(M)])

    # ---- tier 4 (causal results are read only from here on)
    t1p_c = t1_p[best, np.arange(M)]
    t1m_c = t1_mean[best, np.arange(M)]
    causal = causal_decision(t1p_c, t1m_c, crit)

    # ---- tiers 5, 6
    out_measures, rows, selected, layer_of = [], [], [], {}
    for j, n in enumerate(names):
        li = int(best[j])
        if (li, n) in keep:
            Zv, Zh, Zc = keep[(li, n)]
            enc_agree, enc_frac = encoded_bootstrap_agreement(Zv, Zh, Zc, lab_val, K,
                                                              bool(enc[j]), crit, rng)
        else:
            enc_agree, enc_frac = float("nan"), float("nan")
        t1_agree, t1_frac = t1_bootstrap_agreement(excess[li, j][valid_k], bool(causal[j]),
                                                   crit, rng)
        stable = bool(np.isfinite(enc_agree) and np.isfinite(t1_agree)
                      and enc_agree >= args.min_agreement and t1_agree >= args.min_agreement)
        stability = (float(min(enc_agree, t1_agree))
                     if np.isfinite(enc_agree) and np.isfinite(t1_agree) else float("nan"))
        regions = [str(r) for r in MEASURE_REGIONS.get(n, [])]
        rc = region_check(regions, reg_names, reg_layers, reg_t1_p, reg_t1_mean, crit)

        tiers = {"grounded": bool(grounded[n]), "within_class": within[n],
                 "encoded": bool(enc[j]), "causal": bool(causal[j]), "stable": stable,
                 "region_check": bool(rc["ok"])}
        metrics = {
            "within_frac": _f(wfrac[n]),
            "probe_r2": _f(E["probe_r2"][li, j]), "class_r2": _f(E["class_r2"][li, j]),
            "gain_r2": _f(E["gain_r2"][li, j]),
            "within_class_r2": _f(E["within_class_r2"][li, j]),
            "t1_p": _f(t1p_c[j]), "t1_mean": _f(t1m_c[j]), "stability": stability,
            "region_t1_p": _f(rc["region_t1_p"]), "best_region": rc["best_region"],
            "region_layer": rc["region_layer"],
        }
        out_measures.append({"name": n, "layer": layers[li], "regions": regions,
                             "tiers": tiers, "metrics": metrics})
        layer_of[n] = layers[li]
        if all(tiers[t] for t in TIERS):
            selected.append(n)
        rows.append({"measure": n, "layer": layers[li], "regions": ";".join(regions),
                     **tiers, "selected": n in selected, **metrics,
                     "ridge_alpha": _f(E["alpha"][li, j]),
                     "n_cols": int(groups[n].size) if n in groups else 0,
                     "enc_agreement": enc_agree, "enc_boot_frac": enc_frac,
                     "t1_agreement": t1_agree, "t1_boot_frac": t1_frac,
                     "region_t1_mean": _f(rc["region_t1_mean"])})

    criteria = dict(crit)
    criteria.update({
        "layer_choice": "argmax probe gain_r2 (TRAIN-FIT -> VAL) over the measure layers, "
                        "chosen before reading causal results",
        "grounded": "measure in sip.measures.MEASURE_NAMES (closed-form geometric function "
                    "of joint coordinates) with finite columns in meas",
        "within_class": "mean over columns of within-class SS / total SS on TRAIN-FIT "
                        ">= min_within_frac",
        "encoded": "ridge probe feat_l -> standardised Z (TRAIN-FIT fit, VAL score): "
                   "gain_r2 = probe_r2 - class_r2 >= min_gain_r2 and within_class_r2 >= "
                   "min_wc_r2 at the chosen layer",
        "causal": "t1_p < alpha and t1_mean > min_t1_mean at the chosen layer "
                  "(causal_measures.npz, VAL)",
        "stable": "bootstrap over VAL rows (n_boot_enc) of the encoded decision and over "
                  "classes (n_boot_t1) of the causal decision both agree >= min_agreement; "
                  "stability = min of the two agreements",
        "region_check": "some region of the measure has region t1_p < region_alpha at one "
                        "of region_layers (causal_regions.npz, VAL)",
        "layers": layers, "region_file_layers": reg_layers,
        "n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)),
        "ridge_alphas": [float(a) for a in args.alphas], "seed": args.seed,
        "causal_dir": os.path.abspath(args.causal_dir),
        "train_feats": os.path.abspath(args.train_feats),
    })
    out = {"measures": out_measures, "selected": selected, "layer_of": layer_of,
           "criteria": criteria}
    with open(os.path.join(args.out_dir, "strong_measures.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe(out), f, indent=2, allow_nan=False)
    write_csv(os.path.join(args.out_dir, "measure_tiers.csv"), rows)

    print(f"\n{'measure':22s} {'layer':>5s} " + " ".join(f"{t[:6]:>6s}" for t in TIERS))
    for p in out_measures:
        print(f"{p['name']:22s} {p['layer']:5d} "
              + " ".join(f"{('Y' if p['tiers'][t] else '.'):>6s}" for t in TIERS))
    print(f"[done] selected {len(selected)}/{M}: {selected}")


if __name__ == "__main__":
    main()

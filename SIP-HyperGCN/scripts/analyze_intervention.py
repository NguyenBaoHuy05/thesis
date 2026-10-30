"""Step 3 (v3): causal class x unit matrices for measures and body regions.

Usage (docs/PIPELINE_V3.md):
    python scripts/analyze_intervention.py \
        --measures work_dir/sip/intervention_val.npz \
        --regions  work_dir/sip/region_intervention_val.npz \
        --feats work_dir/sip/train_feats.npz --out_dir work_dir/sip/causal_val
    python scripts/analyze_intervention.py \
        --measures work_dir/sip/intervention_test.npz \
        --regions  work_dir/sip/region_intervention_test.npz \
        --feats work_dir/sip/test_feats.npz --out_dir work_dir/sip/causal_test

At least one of --measures / --regions is needed. --feats is the feature file of the
evaluated split (train_feats.npz for the VAL run, test_feats.npz for the TEST run); its
`meas` rows `eval_index` give the per-class statistics of the measure columns.
--concept_csv is only used to read class names (first column, NTU label order).

Statistics: see sip/causal.py (dacc / drand / excess, paired-bootstrap p_class,
class-sign-flip t1).

Outputs (in --out_dir):
  causal_measures.npz / causal_regions.npz, keys:
      unit_names (U,) (no randoms), class_names (K,), layers (L,), dacc (L,U,K),
      drand (L,K), drand_sd (L,K), excess (L,U,K), p_class (L,U,K), t1_p (L,U),
      t1_mean (L,U), confusion (K,K), class_count (K,), base_acc (K,), val_index,
      eval_index, eval_split (str)
      causal_measures.npz also: class_mean (D,K), class_std (D,K) of the `meas` columns on
      the eval rows, meas_cols (D,)
      (+ centroid_shift (L,U,K) only with --centroid and final_feat in the input)
  causal_summary_{measures,regions}.csv   one row per (layer, unit)
  excess_best_{measures,regions}.png      U x K excess at each unit's best layer
                                          (max t1_mean); dots mark p_class < alpha
  excess_layers_{measures,regions}.png    U x K excess, one panel per layer
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from sip.causal import analyze_intervention, class_stats, load_intervention  # noqa: E402

SPEC_KEYS = ["unit_names", "class_names", "layers", "dacc", "drand", "drand_sd", "excess",
             "p_class", "t1_p", "t1_mean", "confusion", "class_count", "base_acc",
             "val_index", "eval_index", "eval_split"]


def load_class_names(csv_path, K):
    try:
        from sip.concept_map import GCCR_CONCEPT_CSV, load_concept_csv
        names = load_concept_csv(csv_path or GCCR_CONCEPT_CSV)[0]
        names = [str(n) for n in list(names)[:K]]
    except Exception as e:  # missing file / bad format
        print(f"[warn] class names unavailable ({e}); using class_k")
        names = []
    return names + [f"class_{k}" for k in range(len(names), K)]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                        for k, v in r.items()})


def _span(M):
    s = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    return s if s > 0 else 1.0


def plot_best_layer(path, res, class_names, alpha, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    excess, p_class, t1_mean = res["excess"], res["p_class"], res["t1_mean"]
    layers, names = res["layers"], [str(n) for n in res["unit_names"]]
    L, U, K = excess.shape
    best = np.argmax(np.where(np.isfinite(t1_mean), t1_mean, -np.inf), axis=0)    # (U,)
    M = excess[best, np.arange(U), :]
    S = p_class[best, np.arange(U), :]
    span = _span(M)
    fig, ax = plt.subplots(figsize=(4 + 0.2 * K, 1.5 + 0.3 * U))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-span, vmax=span)
    yy, xx = np.nonzero(np.nan_to_num(S, nan=1.0) < alpha)
    ax.scatter(xx, yy, s=4, c="k", marker="o", linewidths=0)
    ax.set_xticks(range(K))
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticks(range(U))
    ax.set_yticklabels([f"{n} (l{layers[b]})" for n, b in zip(names, best)], fontsize=6)
    ax.set_title(f"{kind}: excess accuracy drop over random controls at the best layer "
                 f"(max t1_mean); dot = p_class < {alpha}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_layers(path, res, class_names, alpha, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    excess, p_class = res["excess"], res["p_class"]
    layers, names = res["layers"], [str(n) for n in res["unit_names"]]
    L, U, K = excess.shape
    span = _span(excess)
    fig, axes = plt.subplots(L, 1, figsize=(4 + 0.2 * K, L * (1.2 + 0.28 * U)), squeeze=False)
    for li in range(L):
        ax = axes[li, 0]
        im = ax.imshow(excess[li], aspect="auto", cmap="RdBu_r", vmin=-span, vmax=span)
        yy, xx = np.nonzero(np.nan_to_num(p_class[li], nan=1.0) < alpha)
        ax.scatter(xx, yy, s=4, c="k", marker="o", linewidths=0)
        ax.set_yticks(range(U))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xticks(range(K))
        ax.set_xticklabels(class_names if li == L - 1 else [""] * K, rotation=90, fontsize=5)
        ax.set_title(f"{kind} @ l{int(layers[li])} (dot = p_class < {alpha})", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def summary_rows(res, class_names, alpha, unit_label):
    excess, dacc, drand = res["excess"], res["dacc"], res["drand"]
    p_class, t1_mean, t1_p = res["p_class"], res["t1_mean"], res["t1_p"]
    count = res["class_count"]
    valid = count > 0
    w = count / max(count.sum(), 1)
    rows = []
    for li, l in enumerate(res["layers"]):
        for j, name in enumerate(res["unit_names"]):
            ex = excess[li, j]
            if not np.isfinite(ex[valid]).all():
                continue
            order = np.argsort(-np.where(valid, ex, -np.inf))[:3]
            row = {
                "layer": int(l), unit_label: str(name),
                "t1_mean": float(t1_mean[li, j]), "t1_p": float(t1_p[li, j]),
                "overall_dacc": float(np.nansum(w * dacc[li, j])),
                "overall_drand": float(np.nansum(w * drand[li])),
                "max_excess": float(ex[order[0]]),
                "n_class_sig": int(np.sum(p_class[li, j][valid] < alpha)),
                "top_classes": ";".join(f"{class_names[k]}:{ex[k]:.3f}" for k in order),
            }
            if "centroid_shift" in res:
                row["mean_centroid_shift"] = float(np.nanmean(res["centroid_shift"][li, j][valid]))
            rows.append(row)
    return rows


def measure_class_stats(feats_path, inter, K):
    """class_mean / class_std (D, K) of feats `meas` on the eval rows, and meas_cols."""
    feats = np.load(feats_path, allow_pickle=False)
    if "meas" not in feats.files:
        raise ValueError(f"{feats_path} has no `meas` (re-run extract_features.py, v3)")
    meas_cols = np.array([str(c) for c in feats["meas_cols"]]) if "meas_cols" in feats.files \
        else np.array([f"col_{i}" for i in range(feats["meas"].shape[1])])
    D = len(meas_cols)
    nan = np.full((D, K), np.nan)
    f_labels = np.asarray(feats["labels"]).astype(np.int64)
    ev = inter["eval_index"]
    if ev.size and ev.max() >= len(f_labels):
        print("[warn] eval_index out of range of --feats rows; class_mean/std = NaN")
        return nan, nan.copy(), meas_cols
    if not np.array_equal(f_labels[ev], inter["labels"]):
        print("[warn] --feats labels[eval_index] != intervention labels (wrong feats file?); "
              "class_mean/std = NaN")
        return nan, nan.copy(), meas_cols
    meas = np.asarray(feats["meas"], dtype=np.float64)[ev]
    if not np.isfinite(meas).all():
        print(f"[warn] {int((~np.isfinite(meas)).sum())} non-finite meas values on eval rows")
    mean, std = class_stats(meas, inter["labels"], K)
    return mean, std, meas_cols


def run_one(kind, path, args, rng, class_names_cache):
    inter = load_intervention(path)
    K = inter["K"]
    if "names" not in class_names_cache:
        class_names_cache["names"] = load_class_names(args.concept_csv, K)
    class_names = class_names_cache["names"]
    if len(class_names) != K:
        class_names = load_class_names(args.concept_csv, K)
    res = analyze_intervention(inter, n_boot=args.n_boot, n_perm=args.n_perm, rng=rng,
                               centroid=args.centroid)
    print(f"[{kind}] {path}: L={len(res['layers'])} U={len(res['unit_names'])} "
          f"R={len(inter['rand_idx'])} N={len(inter['labels'])} K={K}; random controls used "
          f"per layer: {res['n_random_used'].tolist()}")
    res["class_names"] = np.array(class_names)
    out = {k: res[k] for k in SPEC_KEYS}
    if "centroid_shift" in res:
        out["centroid_shift"] = res["centroid_shift"]
    if kind == "measures":
        cm, cs, cols = measure_class_stats(args.feats, inter, K)
        out.update({"class_mean": cm, "class_std": cs, "meas_cols": cols})
    np.savez(os.path.join(args.out_dir, f"causal_{kind}.npz"), **out)

    unit_label = "measure" if kind == "measures" else "region"
    rows = summary_rows(res, class_names, args.alpha, unit_label)
    write_csv(os.path.join(args.out_dir, f"causal_summary_{kind}.csv"), rows)
    for fn, name in ((plot_best_layer, f"excess_best_{kind}.png"),
                     (plot_layers, f"excess_layers_{kind}.png")):
        try:
            fn(os.path.join(args.out_dir, name), res, class_names, args.alpha, kind)
        except Exception as e:
            print(f"[warn] {name} failed: {e}")
    print(f"{'layer':>5s} {unit_label:24s} {'t1_mean':>8s} {'t1_p':>7s} {'#sig':>5s}")
    for r in rows:
        print(f"{r['layer']:5d} {r[unit_label]:24s} {r['t1_mean']:8.4f} {r['t1_p']:7.4f} "
              f"{r['n_class_sig']:5d}")
    return inter


def main():
    ap = argparse.ArgumentParser(description="Causal class x measure / region matrices")
    ap.add_argument("--measures", default=None, help="intervention_{val,test}.npz (LEACE)")
    ap.add_argument("--regions", default=None, help="region_intervention_{val,test}.npz")
    ap.add_argument("--feats", required=True,
                    help="<eval_split>_feats.npz (rows eval_index -> class stats of meas)")
    ap.add_argument("--concept_csv", default=None,
                    help="only for class names (default: sip.concept_map.GCCR_CONCEPT_CSV)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=1000, help="bootstrap resamples for p_class")
    ap.add_argument("--n_perm", type=int, default=2000, help="sign-flip permutations for t1")
    ap.add_argument("--alpha", type=float, default=0.05, help="only for summary counts / plots")
    ap.add_argument("--centroid", action="store_true",
                    help="also save centroid_shift (L,U,K) when final_feat is present")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.measures is None and args.regions is None:
        ap.error("give --measures and/or --regions")
    os.makedirs(args.out_dir, exist_ok=True)
    cache, done = {}, {}
    for i, (kind, path) in enumerate((("measures", args.measures), ("regions", args.regions))):
        if path is not None:
            # one RNG per kind: region p-values do not depend on whether measures ran too
            rng = np.random.default_rng([args.seed, i])
            done[kind] = run_one(kind, path, args, rng, cache)
    if len(done) == 2:
        a, b = done["measures"], done["regions"]
        if not (np.array_equal(a["eval_index"], b["eval_index"])
                and np.array_equal(a["labels"], b["labels"])):
            print("[warn] measure and region interventions were run on different rows")
        if a["eval_split"] != b["eval_split"]:
            print(f"[warn] eval_split differs: measures={a['eval_split']!r} "
                  f"regions={b['eval_split']!r}")
    print(f"[done] {args.out_dir}: " + ", ".join(f"causal_{k}.npz" for k in done))


if __name__ == "__main__":
    main()

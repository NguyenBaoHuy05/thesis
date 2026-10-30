"""Layer-wise ridge probing of the continuous measures (v3, descriptive; train -> test).

Usage:
    python scripts/run_probe.py --train work_dir/sip/train_feats.npz \
        --test work_dir/sip/test_feats.npz --out_dir work_dir/sip/probe

For every layer l and measure m: ridge probe feat_l -> standardised Z_m = meas[:, cols(m)]
fit on all train rows, scored on test (sip.probe: GCV penalty per measure, pooled R^2,
class-mean baseline from the train class means, within-class R^2). The filter
(select_measures.py) uses the TRAIN-FIT -> VAL version of the same probe instead.

Outputs (in out_dir):
    probe.csv          layer x measure: probe_r2, class_r2, gain_r2, within_class_r2,
                       alpha, n_cols
    layer_metrics.csv  per layer: Fisher ratio (+ kNN / linear class accuracy with
                       --class_metrics)
    probe_heatmaps.png probe R^2 | gain over the class-mean baseline | within-class R^2
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

from sip.probe import (DEFAULT_ALPHAS, fisher_ratio, group_columns,  # noqa: E402
                       knn_class_accuracy, layerwise_probe, linear_class_accuracy)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                        for k, v in r.items()})


def heatmaps(rows, layers, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, P = len(layers), len(names)
    lidx = {l: i for i, l in enumerate(layers)}
    pidx = {n: j for j, n in enumerate(names)}
    panels = [("probe_r2", "Probe R^2 (test)", "Blues", 0.0, 1.0),
              ("gain_r2", "R^2 gain over class-mean baseline", "RdBu_r", None, None),
              ("within_class_r2", "Within-class R^2 (0 = class info only)", "RdBu_r",
               None, None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 + 0.35 * L * len(panels),
                                                      2 + 0.32 * P), squeeze=False)
    for ax, (key, title, cmap, vmin, vmax) in zip(axes[0], panels):
        M = np.full((P, L), np.nan)
        for r in rows:
            M[pidx[r["measure"]], lidx[r["layer"]]] = r[key]
        if cmap == "RdBu_r":
            span = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
            span = span if span > 0 else 1.0
            vmin, vmax = -span, span
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(L))
        ax.set_xticklabels([f"l{l}" for l in layers])
        ax.set_yticks(range(P))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(title, fontsize=9)
        if L * P <= 300:
            for i in range(P):
                for j in range(L):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5,
                                color="#222222")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Layer-wise measure probing (ridge, R^2)")
    ap.add_argument("--train", required=True, help="train_feats.npz")
    ap.add_argument("--test", required=True, help="test_feats.npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(1, 10)))
    ap.add_argument("--measures", nargs="+", default=None, help="subset (default all)")
    ap.add_argument("--num_class", type=int, default=60)
    ap.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
                    help="ridge penalty grid (GCV per measure)")
    ap.add_argument("--max_train", type=int, default=None, help="subsample train for speed")
    ap.add_argument("--class_metrics", action="store_true",
                    help="also compute kNN and linear class accuracy per layer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train = np.load(args.train, allow_pickle=False)
    test = np.load(args.test, allow_pickle=False)
    cols = [str(c) for c in train["meas_cols"]]
    assert cols == [str(c) for c in test["meas_cols"]], "train/test meas_cols differ"
    groups = group_columns(cols)
    if args.measures is not None:
        unknown = [m for m in args.measures if m not in groups]
        if unknown:
            raise ValueError(f"unknown measures: {unknown}")
        groups = {m: groups[m] for m in args.measures}
    names = list(groups)
    layers = [l for l in args.layers if f"feat_l{l}" in train.files and f"feat_l{l}" in test.files]

    rows = layerwise_probe(train, test, layers, groups, args.num_class, alphas=args.alphas,
                           max_train=args.max_train, seed=args.seed)
    write_csv(os.path.join(args.out_dir, "probe.csv"), rows)
    try:
        heatmaps(rows, layers, names, os.path.join(args.out_dir, "probe_heatmaps.png"))
    except Exception as e:
        print(f"[warn] heatmap failed: {e}")

    lab_tr = np.asarray(train["labels"])
    lab_te = np.asarray(test["labels"])
    lrows = []
    for l in layers:
        Xte = np.asarray(test[f"feat_l{l}"], dtype=np.float32)
        r = {"layer": l, "fisher_ratio_test": fisher_ratio(Xte, lab_te, args.num_class)}
        if args.class_metrics:
            Xtr = np.asarray(train[f"feat_l{l}"], dtype=np.float32)
            r["knn5_acc"] = knn_class_accuracy(Xtr, lab_tr, Xte, lab_te,
                                               max_train=args.max_train, seed=args.seed)
            r["linear_acc"] = linear_class_accuracy(Xtr, lab_tr, Xte, lab_te,
                                                    max_train=args.max_train, seed=args.seed)
        print("[layer]", r, flush=True)
        lrows.append(r)
    write_csv(os.path.join(args.out_dir, "layer_metrics.csv"), lrows)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()

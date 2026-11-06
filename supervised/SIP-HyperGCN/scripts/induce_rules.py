"""Step 5 (pipeline v3): induce one conjunctive threshold rule per class.

Usage:
    python scripts/induce_rules.py --causal_dir $W/causal_val \
        --strong $W/measures/strong_measures.json \
        --train_feats $W/train_feats.npz --out_dir $W/rules

Reads <causal_dir>/causal_measures.npz and <causal_dir>/causal_regions.npz (VAL). VAL rows
(excluded from threshold learning / data support) come from ``val_index`` in
causal_measures.npz, or from ``--val_index_from`` (any npz with a ``val_index`` key).
Outputs: <out_dir>/rules.json, <out_dir>/rules.md. See sip/rule_induction.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from sip.rule_induction import (DEFAULT_PARAMS, induce_rules, load_npz,  # noqa: E402
                                load_strong, rule_to_string, train_fit_mask, write_rules_md)


def get_parser():
    D = DEFAULT_PARAMS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--causal_dir", required=True,
                   help="VAL causal dir with causal_measures.npz and causal_regions.npz")
    p.add_argument("--strong", required=True, help="strong_measures.json")
    p.add_argument("--train_feats", required=True, help="train_feats.npz (meas, meas_cols, labels)")
    p.add_argument("--val_index_from", default=None,
                   help="npz with val_index (used if causal_measures.npz has none)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_excess", type=float, default=D["min_excess"])
    p.add_argument("--alpha", type=float, default=D["alpha"], help="BH-FDR q threshold")
    p.add_argument("--no_fdr", action="store_true", help="use raw p_class instead of BH-FDR q")
    p.add_argument("--region_layers", type=int, nargs="+", default=D["region_layers"])
    p.add_argument("--region_min_excess", type=float, default=D["region_min_excess"])
    p.add_argument("--region_alpha", type=float, default=D["region_alpha"])
    p.add_argument("--no_region_check", action="store_true",
                   help="skip the per-class region cross-check (ablation only)")
    p.add_argument("--top_m", type=int, default=D["top_m_confusable"])
    p.add_argument("--n_quantiles", type=int, default=D["n_quantiles"])
    p.add_argument("--theta_decimals", type=int, default=D["theta_decimals"])
    p.add_argument("--stump_min_coverage", type=float, default=D["stump_min_coverage"])
    p.add_argument("--min_coverage", type=float, default=D["min_coverage"])
    p.add_argument("--max_len", type=int, default=D["max_len"])
    p.add_argument("--min_support", type=float, default=D["min_support"])
    p.add_argument("--consistency_jaccard", type=float, default=D["consistency_jaccard"])
    return p


def main():
    arg = get_parser().parse_args()
    try:  # rule strings contain ∧
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass
    cm_path = os.path.join(arg.causal_dir, "causal_measures.npz")
    cr_path = os.path.join(arg.causal_dir, "causal_regions.npz")
    causal_m = load_npz(cm_path)
    causal_r = None
    if os.path.isfile(cr_path):
        causal_r = load_npz(cr_path)
    elif not arg.no_region_check:
        raise SystemExit(f"{cr_path} not found (needed for the region check; "
                         "--no_region_check to skip)")
    for name, c in (("causal_measures", causal_m), ("causal_regions", causal_r)):
        if c is not None and "eval_split" in c and str(c["eval_split"]) == "test":
            print(f"WARNING: {name} eval_split=test; rules must be induced from the VAL causal dir")
    strong = load_strong(arg.strong)
    tf = load_npz(arg.train_feats)
    for key in ("meas", "meas_cols", "labels"):
        if key not in tf:
            raise SystemExit(f"{arg.train_feats} has no '{key}' (v3 extract_features output needed)")

    if "val_index" in causal_m:
        val_index, src = causal_m["val_index"], cm_path
    elif arg.val_index_from:
        vz = load_npz(arg.val_index_from)
        if "val_index" not in vz:
            raise SystemExit(f"{arg.val_index_from} has no val_index")
        val_index, src = vz["val_index"], arg.val_index_from
    else:
        raise SystemExit("val_index missing from causal_measures.npz; pass --val_index_from "
                         "intervention_val.npz (anti-leakage: VAL rows must be excluded)")
    labels = np.asarray(tf["labels"]).reshape(-1)
    n_train = len(labels)
    fit = train_fit_mask(n_train, val_index)
    if causal_r is not None and "val_index" in causal_r:
        vr = np.sort(np.asarray(causal_r["val_index"]).reshape(-1))
        if not np.array_equal(vr, np.sort(np.asarray(val_index).reshape(-1))):
            print("WARNING: causal_regions val_index differs from causal_measures val_index")
    if "eval_index" in causal_m:  # sanity: VAL causal matrix must be measured on val rows
        ei = np.asarray(causal_m["eval_index"]).reshape(-1)
        if ei.size and ei.max() < n_train and fit[ei].any():
            print(f"WARNING: {int(fit[ei].sum())} eval_index rows of causal_measures are not "
                  "in val_index (is this the VAL causal dir?)")
    print(f"train rows {n_train}, VAL {n_train - int(fit.sum())} (from {src}), "
          f"TRAIN-FIT {int(fit.sum())}")

    net_pred = np.asarray(tf["logits"]).argmax(axis=1) if "logits" in tf else None
    params = {
        "min_excess": arg.min_excess, "alpha": arg.alpha, "fdr": not arg.no_fdr,
        "region_check": not arg.no_region_check, "region_layers": list(arg.region_layers),
        "region_min_excess": arg.region_min_excess, "region_alpha": arg.region_alpha,
        "top_m_confusable": arg.top_m, "n_quantiles": arg.n_quantiles,
        "theta_decimals": arg.theta_decimals, "stump_min_coverage": arg.stump_min_coverage,
        "min_coverage": arg.min_coverage, "max_len": arg.max_len,
        "min_support": arg.min_support, "consistency_jaccard": arg.consistency_jaccard,
    }
    out = induce_rules(causal_m, causal_r, strong, tf["meas"], tf["meas_cols"], labels, fit,
                       net_pred=net_pred, params=params)
    out["params"].update({"causal_dir": os.path.abspath(arg.causal_dir),
                          "strong": os.path.abspath(arg.strong),
                          "train_feats": os.path.abspath(arg.train_feats)})

    os.makedirs(arg.out_dir, exist_ok=True)
    with open(os.path.join(arg.out_dir, "rules.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    write_rules_md(out, os.path.join(arg.out_dir, "rules.md"))

    for w in out["warnings"]:
        print("WARNING:", w)
    n_rule = sum(1 for r in out["rules"] if r["literals"])
    n_learn = sum(1 for r in out["rules"] if r["learnable"]["learnable"])
    print(f"usable measures: {len(out['measures'])}; rules: {n_rule}/{len(out['rules'])}; "
          f"learnable: {n_learn}")
    for r in out["rules"]:
        if r["literals"]:
            print(f"  [{r['class_idx']:2d}] {r['class_name']}: {rule_to_string(r['literals'])}")
    print("wrote", os.path.join(arg.out_dir, "rules.json"))


if __name__ == "__main__":
    main()

"""Step 6 (pipeline v3): evaluate threshold rules on TEST and check them against the concepts CSV.

Usage:
    python scripts/evaluate_rules.py --rules $W/rules/rules.json --test_feats $W/test_feats.npz \
        --causal_dir $W/causal_test --strong $W/measures/strong_measures.json \
        --out_dir $W/rules_eval [--concept_csv PATH | --no_concepts] [--no_region_confirm]

Reads <causal_dir>/causal_measures.npz and <causal_dir>/causal_regions.npz (TEST).
Outputs: <out_dir>/rule_report.csv, rule_report.md, learnable_rules.json. See sip/rule_eval.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sip.rule_eval import evaluate_rules, write_reports  # noqa: E402
from sip.rule_induction import load_npz, load_strong  # noqa: E402


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rules", required=True, help="rules.json from induce_rules.py")
    p.add_argument("--test_feats", required=True, help="test_feats.npz (meas, meas_cols, labels, logits)")
    p.add_argument("--causal_dir", default=None,
                   help="TEST causal dir (causal_measures.npz, causal_regions.npz) for necessity")
    p.add_argument("--strong", default=None, help="strong_measures.json (layer_of)")
    p.add_argument("--concept_csv", default=None,
                   help="concepts CSV (default: sip.concept_map.GCCR_CONCEPT_CSV)")
    p.add_argument("--no_concepts", action="store_true", help="skip the knowledge check")
    p.add_argument("--no_region_confirm", action="store_true",
                   help="causal confirmation from measure necessity only")
    p.add_argument("--alpha", type=float, default=0.05, help="necessity p threshold on TEST")
    p.add_argument("--top_m", type=int, default=3,
                   help="confusable classes if rules.json has none stored")
    p.add_argument("--out_dir", required=True)
    return p


def main():
    arg = get_parser().parse_args()
    try:  # rule strings contain ∧ / θ
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass
    with open(arg.rules, "r", encoding="utf-8") as fh:
        rules_js = json.load(fh)
    tf = load_npz(arg.test_feats)
    for key in ("meas", "meas_cols", "labels", "logits"):
        if key not in tf:
            raise SystemExit(f"{arg.test_feats} has no '{key}' (v3 extract_features output needed)")

    causal_m = causal_r = None
    if arg.causal_dir:
        cm_path = os.path.join(arg.causal_dir, "causal_measures.npz")
        cr_path = os.path.join(arg.causal_dir, "causal_regions.npz")
        if os.path.isfile(cm_path):
            causal_m = load_npz(cm_path)
        else:
            print(f"WARNING: {cm_path} not found")
        if os.path.isfile(cr_path):
            causal_r = load_npz(cr_path)
        else:
            print(f"WARNING: {cr_path} not found")
        for name, c in (("causal_measures", causal_m), ("causal_regions", causal_r)):
            if c is not None and "eval_split" in c and str(c["eval_split"]) != "test":
                print(f"WARNING: {name} eval_split={c['eval_split']} (expected test)")
    strong = load_strong(arg.strong) if arg.strong else None
    if strong is None:
        print("WARNING: no --strong; measure layers taken from rules.json evidence")
    if causal_m is None:
        print("WARNING: no TEST causal_measures; no rule can be causally confirmed")

    concept_csv = None
    if not arg.no_concepts:
        if arg.concept_csv:
            concept_csv = arg.concept_csv
        else:
            from sip.concept_map import GCCR_CONCEPT_CSV
            concept_csv = GCCR_CONCEPT_CSV

    res = evaluate_rules(rules_js, tf["meas"], tf["meas_cols"], tf["labels"], tf["logits"],
                         causal_m=causal_m, causal_r=causal_r, strong=strong,
                         concept_csv=concept_csv, alpha=arg.alpha, top_m=arg.top_m,
                         require_region=not arg.no_region_confirm)
    write_reports(res, rules_js, arg.out_dir)

    sm = res["summary"]
    for w in res["warnings"]:
        print("WARNING:", w)
    print(f"rules {sm['n_rules']}/{sm['n_classes']}, learnable {sm['n_learnable']}, "
          f"measure-confirmed {sm['n_measure_confirmed']}, region-confirmed "
          f"{sm['n_region_confirmed']}, causally confirmed {sm['n_causally_confirmed']}, "
          f"step-5 {sm['n_step5']}")
    print(f"mean coverage {sm['mean_coverage']:.3f} precision {sm['mean_precision']:.3f} "
          f"fidelity {sm['mean_fidelity']:.3f} discrimination {sm['mean_discrimination']:.3f}")
    c = sm["classifier_all_rules"]
    print(f"rule classifier: acc {c['accuracy']:.3f} (covered {c['covered_frac']:.3f}), "
          f"global fidelity {c['global_fidelity']:.3f}; network acc {c['network_accuracy']:.3f}")
    print(f"findings (CSV opposite-direction disagreements): {len(res['findings'])}")
    print("wrote", os.path.join(arg.out_dir, "rule_report.md"))


if __name__ == "__main__":
    main()

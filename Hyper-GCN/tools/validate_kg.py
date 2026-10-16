"""Check annotations/action_kg.json (KG_CONSTRUCTION_LLM_PLAN.md, S6/S8).

  python tools/validate_kg.py [--kg annotations/action_kg.json] [--pairs-out kg_work/s6_pairs.json]

Errors: unknown nodes, asymmetric similar, excludes violations, attribute degree out of
[2, 60], actions with < 3 has or no lacks. Report: pairs of actions with identical or
near-identical (Hamming < 2) signed profiles; these feed the S6 repair loop.
"""

import argparse
import itertools
import json
import os
import sys

MIN_DEGREE, MAX_DEGREE = 2, 60


def profile(action, attributes):
    """Signed profile: +1 has, -1 lacks, 0 unknown."""
    return tuple(1 if a in action["has"] else -1 if a in action["lacks"] else 0 for a in attributes)


def validate(kg):
    attributes = sorted(kg["attributes"])
    actions = kg["actions"]
    errors, warnings = [], []
    for name, a in actions.items():
        for kind in ("has", "lacks"):
            for attr in a[kind]:
                if attr not in kg["attributes"]:
                    errors.append(f"{name}: {kind} unknown attribute {attr}")
        for other in a["similar"] + a["opposite"]:
            if other not in actions:
                errors.append(f"{name}: unknown related action {other}")
        for other in a["similar"]:
            if other in actions and name not in actions[other]["similar"]:
                errors.append(f"similar not symmetric: {name} -> {other}")
        if len(a["has"]) < 3:
            warnings.append(f"{name}: only {len(a['has'])} has")
        if not a["lacks"]:
            warnings.append(f"{name}: no lacks")
    for x, y in kg["attribute_relations"].get("excludes", []):
        for name, a in actions.items():
            if x in a["has"] and y in a["has"]:
                errors.append(f"{name}: has both {x} and {y} (excludes)")
    degree = {attr: sum(attr in a["has"] for a in actions.values()) for attr in attributes}
    for attr, d in degree.items():
        if not MIN_DEGREE <= d <= MAX_DEGREE:
            errors.append(f"attribute {attr}: degree {d} outside [{MIN_DEGREE}, {MAX_DEGREE}]")

    opposite = {frozenset((n, o)) for n, a in actions.items() for o in a["opposite"]}
    profiles = {n: profile(a, attributes) for n, a in actions.items()}
    pairs = []
    for c1, c2 in itertools.combinations(sorted(actions, key=lambda n: actions[n]["id"]), 2):
        hamming = sum(p != q for p, q in zip(profiles[c1], profiles[c2]))
        if hamming < 2:
            pairs.append({"c1": c1, "c2": c2, "hamming": hamming,
                          "opposite": frozenset((c1, c2)) in opposite})
    return errors, warnings, degree, pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", default="annotations/action_kg.json")
    parser.add_argument("--pairs-out", default="annotations/kg_work/s6_pairs.json")
    args = parser.parse_args()
    with open(args.kg, encoding="utf-8") as handle:
        kg = json.load(handle)
    errors, warnings, degree, pairs = validate(kg)
    has_counts = [len(a["has"]) for a in kg["actions"].values()]
    print(f"{len(kg['actions'])} actions, {len(kg['attributes'])} attributes; "
          f"has/action min {min(has_counts)} mean {sum(has_counts) / len(has_counts):.1f} max {max(has_counts)}")
    print(f"attribute degree min {min(degree.values())} max {max(degree.values())}")
    print(f"{len(errors)} errors, {len(warnings)} warnings, {len(pairs)} indistinct pairs "
          f"({sum(p['opposite'] for p in pairs)} opposite)")
    for line in errors + warnings:
        print("  " + line)
    os.makedirs(os.path.dirname(args.pairs_out), exist_ok=True)
    with open(args.pairs_out, "w", encoding="utf-8") as handle:
        json.dump({"pairs": pairs}, handle, indent=1)
    sys.exit(1 if errors else 0)

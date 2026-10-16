"""S7 hand review of annotations/action_kg.json (KG_CONSTRUCTION_LLM_PLAN.md).

  python tools/kg_review.py export [--out annotations/kg_review/edges.csv]
  python tools/kg_review.py import [--csv annotations/kg_review/edges.csv]

export: one row per has/lacks edge, plus the edges dropped by excludes, sorted by review
priority (1 = check first). Fill only the column human_label with one of
has2, has1, unknown, lacks1, lacks2, or leave it empty to accept the LLM label.
import: applies the non-empty human labels, sets source="human" on changed edges and
reviewed=true on every action with at least one reviewed row, and recomputes the hash.
Note: tools/kg_build.py build regenerates the KG from the LLM answers and discards these
edits, so run the review after the last build.
"""

import argparse
import csv
import hashlib
import json
import os

KG = "annotations/action_kg.json"
FIELDS = ["priority", "action", "attribute", "llm_label", "agree", "evidence", "note", "human_label"]
LABELS = {"has2": ("has", 2), "has1": ("has", 1), "lacks1": ("lacks", 1), "lacks2": ("lacks", 2)}


def label_of(kind, weight):
    return f"{kind}{weight}"


def export(kg, out):
    rows = []
    for name, a in kg["actions"].items():
        for kind in ("has", "lacks"):
            for attr, edge in a[kind].items():
                agree = edge.get("agree")
                priority = 2 if agree is not None and agree < 1 else 3
                rows.append(dict(priority=priority, action=name, attribute=attr,
                                 llm_label=label_of(kind, edge["w"]), agree=agree,
                                 evidence=edge.get("evidence", ""), note=edge.get("source", ""), human_label=""))
    for item in kg.get("dropped_edges", []):
        rows.append(dict(priority=1, action=item["action"], attribute=item["attribute"],
                         llm_label="dropped", agree=item["edge"].get("agree"),
                         evidence=item["edge"].get("evidence", ""), note=item["reason"], human_label=""))
    rows.sort(key=lambda r: (r["priority"], kg["actions"][r["action"]]["id"], r["attribute"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


def apply(kg, path):
    changed = 0
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["human_label"].strip()
            if not label:
                continue
            a = kg["actions"][row["action"]]
            a["reviewed"] = True
            a["has"].pop(row["attribute"], None)
            a["lacks"].pop(row["attribute"], None)
            if label in LABELS:
                kind, weight = LABELS[label]
                a[kind][row["attribute"]] = {"w": weight, "agree": None, "evidence": row["evidence"],
                                             "source": "human"}
            elif label != "unknown":
                raise ValueError(f"bad human_label {label!r} for {row['action']} / {row['attribute']}")
            changed += 1
    kg.pop("hash", None)
    payload = json.dumps(kg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    kg["hash"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    with open(KG, "w", encoding="utf-8") as handle:
        json.dump(kg, handle, indent=1, ensure_ascii=False)
    print(f"{changed} reviewed rows applied; run tools/validate_kg.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["export", "import"])
    parser.add_argument("--csv", "--out", dest="csv", default="annotations/kg_review/edges.csv")
    args = parser.parse_args()
    with open(KG, encoding="utf-8") as handle:
        kg = json.load(handle)
    export(kg, args.csv) if args.mode == "export" else apply(kg, args.csv)

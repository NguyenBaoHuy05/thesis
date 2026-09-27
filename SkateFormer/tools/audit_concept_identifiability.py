"""Static identifiability audit for unary HCG-NS concept labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.concept_ontology import load_action_concept_matrix_v2, load_ontology_metadata, ontology_summary


def audit(concept_csv: Path, metadata_csv: Path, num_classes: int, num_concepts: int) -> dict:
    support, action_names, concept_names = load_action_concept_matrix_v2(
        concept_csv, num_classes, num_concepts
    )
    metadata = load_ontology_metadata(metadata_csv, num_concepts)
    usage = support.sum(dim=0)
    rows = []
    for concept_id, (name, count, row) in enumerate(zip(concept_names, usage.tolist(), metadata)):
        rows.append({
            "concept_id": concept_id,
            "concept_name": name,
            "positive_action_count": int(count),
            "parent_name": row["parent_name"],
            "status": "orphan" if count == 0 else "rare" if count <= 2 else "usable",
        })
    return {
        "num_actions": num_classes,
        "num_concepts": num_concepts,
        "ontology": ontology_summary(metadata),
        "orphan_concepts": [row["concept_id"] for row in rows if row["status"] == "orphan"],
        "rare_concepts": [row["concept_id"] for row in rows if row["status"] == "rare"],
        "concepts": rows,
        "action_names": action_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-csv", default="concepts/ntu_action_concept_v2.csv")
    parser.add_argument("--metadata-csv", default="concepts/ntu_concept_metadata_v2.csv")
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--num-concepts", type=int, default=78)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = audit(Path(args.concept_csv), Path(args.metadata_csv), args.num_classes, args.num_concepts)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

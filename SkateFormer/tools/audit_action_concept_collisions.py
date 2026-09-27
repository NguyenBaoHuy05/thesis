"""Find action pairs that remain hard under unary concept support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.concept_ontology import load_action_concept_matrix_v2


def audit(concept_csv: Path, num_classes: int, num_concepts: int, max_distance: int = 2) -> dict:
    support, action_names, _ = load_action_concept_matrix_v2(
        concept_csv, num_classes, num_concepts
    )
    collisions = []
    for first in range(num_classes):
        for second in range(first + 1, num_classes):
            distance = int((support[first] != support[second]).sum().item())
            if distance <= int(max_distance):
                differing = torch_nonzero(support[first] != support[second])
                collisions.append({
                    "action_a": first,
                    "action_b": second,
                    "name_a": action_names[first],
                    "name_b": action_names[second],
                    "hamming_distance": distance,
                    "differing_concepts": differing,
                })
    collisions.sort(key=lambda row: (row["hamming_distance"], row["action_a"], row["action_b"]))
    return {
        "max_distance": int(max_distance),
        "num_collisions": len(collisions),
        "collisions": collisions,
    }


def torch_nonzero(mask):
    return [int(value) for value in mask.nonzero(as_tuple=False).flatten().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-csv", default="concepts/ntu_action_concept_v2.csv")
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--num-concepts", type=int, default=78)
    parser.add_argument("--max-distance", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = audit(Path(args.concept_csv), args.num_classes, args.num_concepts, args.max_distance)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

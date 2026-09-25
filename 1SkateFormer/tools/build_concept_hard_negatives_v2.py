"""Build action-level hard-negative pairs from the v2 concept signatures.

The generated pairs remain action pairs because the training loss operates on
action logits.  Concept annotations are used only to select semantically close
actions with an explicit context/temporal discriminator; no test confusion
matrix is used.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _positive_sets(path: Path) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"concept matrix is empty: {path}")
    concepts = [name for name in rows[0] if name != "action_class"]
    return rows, {
        row["action_class"]: {name for name in concepts if row.get(name) == "1"}
        for row in rows
    }


def build(
    action_path: Path,
    output_path: Path,
    min_shared: int = 2,
    min_jaccard: float = 0.30,
) -> None:
    rows, signatures = _positive_sets(action_path)
    actions = [row["action_class"] for row in rows]
    action_id = {name: index for index, name in enumerate(actions)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "left_action_id", "left_action", "right_action_id", "right_action",
        "shared_concepts", "discriminating_concepts", "jaccard",
        "selection_rule", "source",
    ]

    # These are the concept contrasts that can make two otherwise similar
    # action signatures distinct.  This is a semantic prior, not test tuning.
    def is_discriminator(name: str) -> bool:
        return (
            name.startswith("context_")
            or name.startswith("temporal_")
            or name == "interaction_two_person"
            or name in {
                "hand_contact_mouth_region",
                "hand_contact_head_region",
                "hand_contact_chest_region",
                "hand_contact_neck_region",
                "hand_contact_back_region",
                "hand_fine_repetitive_motion",
                "hand_bimanual_hold",
                "hand_repetitive_tapping",
                "hand_repetitive_pressing",
                "hand_repetitive_slicing",
                "hand_repetitive_pull_apart",
                "hand_repetitive_opening",
                "hand_contact_face_region",
                "hand_contact_nose_region",
                "arm_cross_chest",
                "hand_fold_motion",
                "hand_rotational_manipulation",
            }
        )

    candidates: list[dict[str, object]] = []
    for left_index, left in enumerate(actions):
        for right in actions[left_index + 1:]:
            left_set, right_set = signatures[left], signatures[right]
            shared = sorted(left_set & right_set)
            different = sorted((left_set ^ right_set))
            discriminators = [name for name in different if is_discriminator(name)]
            union = left_set | right_set
            jaccard = len(shared) / len(union) if union else 0.0
            if (
                len(shared) < min_shared
                or jaccard < min_jaccard
                or not discriminators
            ):
                continue
            candidates.append({
                "left_action_id": action_id[left],
                "left_action": left,
                "right_action_id": action_id[right],
                "right_action": right,
                "shared_concepts": "|".join(shared),
                "discriminating_concepts": "|".join(discriminators),
                "jaccard": f"{jaccard:.6f}",
                "selection_rule": "shared unary concepts + context/temporal discriminator",
                "source": f"{action_path.name}; no test confusion matrix",
            })

    # Keep every semantically defensible candidate, in deterministic order.
    # The config can choose a reviewed subset without regenerating semantics.
    candidates.sort(key=lambda row: (
        -float(row["jaccard"]), int(row["left_action_id"]),
        int(row["right_action_id"]),
    ))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Wrote {output_path} ({len(candidates)} candidates)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="concepts/ntu_action_concept_v2_3.csv")
    parser.add_argument(
        "--output",
        default="concepts/ntu_hard_negative_pairs_v2_3_candidates.csv",
    )
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--min-jaccard", type=float, default=0.30)
    args = parser.parse_args()
    build(
        Path(args.actions), Path(args.output),
        min_shared=args.min_shared,
        min_jaccard=args.min_jaccard,
    )


if __name__ == "__main__":
    main()

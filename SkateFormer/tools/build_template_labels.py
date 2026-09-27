"""
build_template_labels.py
------------------------
Parse concepts/ntu_rgbd_60_120.csv and produce:
  - template_labels.npz  (per-split NPZ for each protocol)

The NPZ contains:
  concept_targets [N, 74]  — float32; values: 1.0 = positive, 0.0 = negative.
  concept_mask    [N, 74]  — bool;    True  = position included in BCE loss.
                              Currently all known positions are included.
  action_labels   [N]      — int64;   original action label (0-indexed).
  sample_names    [N]      — str;     e.g. "train_0", "train_1", ...

The CSV has 74 concept columns (after "action_class") for 120 classes.
NTU60 uses rows 0-59; NTU120 uses all 120 rows.

Usage:
    python tools/build_template_labels.py \\
        --data_path data/ntu/NTU60_CS.npz \\
        --split train \\
        --num_classes 60 \\
        --out_dir work_dir/final_pipeline/ntu60_cs/pseudo_labels \\
        --protocol ntu60_cs

    # For NTU120:
    python tools/build_template_labels.py \\
        --data_path data/ntu120/NTU120_CSub.npz \\
        --split train \\
        --num_classes 120 \\
        --out_dir work_dir/final_pipeline/ntu120_csub/pseudo_labels \\
        --protocol ntu120_csub
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Allow running from repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NUM_CONCEPTS = 74
CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "concepts", "ntu_rgbd_60_120.csv"
)


def load_concept_csv(csv_path: str, num_classes: int) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Read the concept CSV and return:
      template_matrix: [num_classes, NUM_CONCEPTS] float32 with 0.0/1.0
      class_names:     list of action class name strings (length num_classes)
      concept_names:   list of concept name strings (length NUM_CONCEPTS)
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    # First line is header
    header = lines[0].split(",")
    concept_names = header[1:]   # drop "action_class"

    assert len(concept_names) == NUM_CONCEPTS, (
        f"Expected {NUM_CONCEPTS} concept columns, got {len(concept_names)}"
    )

    class_names = []
    template_rows = []
    for line in lines[1:]:
        parts = line.split(",")
        class_names.append(parts[0])
        row = [float(v) for v in parts[1:]]
        assert len(row) == NUM_CONCEPTS, f"Row {parts[0]}: expected {NUM_CONCEPTS} values, got {len(row)}"
        template_rows.append(row)

    template_matrix = np.array(template_rows, dtype=np.float32)  # [120, 74]

    if num_classes == 60:
        # NTU60 actions are the first 60 rows
        assert len(template_rows) >= 60, f"CSV has only {len(template_rows)} rows"
        template_matrix = template_matrix[:60]
        class_names = class_names[:60]
    elif num_classes == 120:
        assert len(template_rows) >= 120, f"CSV has only {len(template_rows)} rows"
        template_matrix = template_matrix[:120]
        class_names = class_names[:120]
    else:
        raise ValueError(f"num_classes must be 60 or 120, got {num_classes}")

    return template_matrix, class_names, concept_names


def build_template_labels(
    data_path: str,
    split: str,
    num_classes: int,
    out_dir: str,
    protocol: str,
    csv_path: str = CSV_PATH,
):
    """Build and save template labels for one data split."""
    os.makedirs(out_dir, exist_ok=True)

    # Load action labels from NPZ
    npz = np.load(data_path)
    if split == "train":
        labels = np.where(npz["y_train"] > 0)[1]      # [N] zero-indexed action labels
        sample_names = np.array([f"train_{i}" for i in range(len(labels))])
    elif split == "test":
        labels = np.where(npz["y_test"] > 0)[1]
        sample_names = np.array([f"test_{i}" for i in range(len(labels))])
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split}")

    N = len(labels)
    assert labels.max() < num_classes, (
        f"Action label {labels.max()} >= num_classes {num_classes}. "
        f"Check protocol / data."
    )

    # Load template
    template_matrix, class_names, concept_names = load_concept_csv(csv_path, num_classes)

    # Build per-sample concept targets
    concept_targets = template_matrix[labels]   # [N, 74], values 0.0 or 1.0

    # Mask: all known positions are included (no unknowns in fixed template)
    concept_mask = np.ones((N, NUM_CONCEPTS), dtype=bool)

    # Save
    out_path = os.path.join(out_dir, f"template_labels_{split}.npz")
    np.savez_compressed(
        out_path,
        concept_targets=concept_targets,
        concept_mask=concept_mask,
        action_labels=labels.astype(np.int64),
        sample_names=sample_names,
    )

    # Save metadata
    meta = {
        "protocol": protocol,
        "split": split,
        "data_path": data_path,
        "num_samples": int(N),
        "num_classes": num_classes,
        "num_concepts": NUM_CONCEPTS,
        "class_names": class_names,
        "concept_names": concept_names,
        "template_stats": {
            "mean_positives_per_class": float(template_matrix.sum(axis=1).mean()),
            "mean_positives_per_concept": float(template_matrix.sum(axis=0).mean()),
        },
        "out_path": out_path,
    }
    meta_path = os.path.join(out_dir, f"template_meta_{split}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[build_template_labels] Saved {N} samples → {out_path}")
    print(f"  Template positives/class: {meta['template_stats']['mean_positives_per_class']:.2f}")
    print(f"  Template positives/concept: {meta['template_stats']['mean_positives_per_concept']:.2f}")
    return out_path, meta


def get_template_matrix(
    num_classes: int, csv_path: str = CSV_PATH
) -> np.ndarray:
    """Convenience: just return the template matrix [num_classes, NUM_CONCEPTS]."""
    template_matrix, _, _ = load_concept_csv(csv_path, num_classes)
    return template_matrix


def main():
    parser = argparse.ArgumentParser(description="Build template concept labels from CSV")
    parser.add_argument("--data_path", required=True, help="Path to .npz data file")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--num_classes", type=int, required=True, choices=[60, 120])
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--protocol", required=True, help="Protocol name (e.g. ntu60_cs)")
    parser.add_argument("--csv_path", default=CSV_PATH, help="Path to concept CSV")
    args = parser.parse_args()

    build_template_labels(
        data_path=args.data_path,
        split=args.split,
        num_classes=args.num_classes,
        out_dir=args.out_dir,
        protocol=args.protocol,
        csv_path=args.csv_path,
    )


if __name__ == "__main__":
    main()

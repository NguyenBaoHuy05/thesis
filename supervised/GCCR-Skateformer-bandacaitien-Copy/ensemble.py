"""
Universal Ensemble Late-Fusion Script
=====================================
Author: Antigravity AI & Thesis Research Team

Late-fusion (weighted summation of logits) across multiple modalities (Joint, Bone,
Joint-Motion, Bone-Motion) and/or across different neural backbones
(e.g., SkateFormer + CTR-GCN + ST-GCN + Hyper-GCN).

Usage
-----
  # 2-Stream (Joint + Bone):
  python ensemble.py \\
      --data   data/ntu/NTU60_CS.npz \\
      --scores work_dir/ntu_cs/skateformer_j/j_test_score.pkl 1.0 \\
               work_dir/ntu_cs/skateformer_b/b_test_score.pkl 1.0

  # Multi-Backbone Ensemble (SkateFormer + CTR-GCN + Hyper-GCN):
  python ensemble.py \\
      --data   data/ntu/NTU60_CS.npz \\
      --scores work_dir/skateformer_j_score.pkl 1.0 \\
               work_dir/ctrgcn_b_score.pkl      1.0 \\
               work_dir/hypergcn_j_score.pkl    0.8
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle
from typing import List, Tuple, Dict, Optional
import numpy as np


def load_score_pkl(path: str) -> np.ndarray:
    """Load a score .pkl file saved by train_joint.py or test_backbone.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Score file not found: {path}")

    with open(path, "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        keys = sorted(raw.keys())
        scores = np.stack([raw[k] for k in keys], axis=0)
    elif isinstance(raw, (list, np.ndarray)):
        scores = np.asarray(raw)
    else:
        raise ValueError(f"Unrecognised score file format in {path}: {type(raw)}")

    if scores.ndim != 2:
        raise ValueError(f"Expected 2-D score array (N, C), got shape {scores.shape}")
    return scores


def load_labels(data_path: str) -> np.ndarray:
    """Load ground-truth test labels from NTU .npz dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    npz = np.load(data_path, allow_pickle=True)

    for key in ("test_label", "y_test"):
        if key in npz:
            arr = npz[key]
            if arr.ndim > 1 and arr.shape[1] > 1:
                return np.where(arr > 0)[1]
            return arr.squeeze().astype(int)

    raise KeyError(f"No label key found in {data_path}. Available keys: {list(npz.keys())}")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Universal Late-Fusion Ensemble for Skeleton Action Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, help="Path to NTU60_CS.npz / NTU120_CSet.npz")

    # Flexible multi-stream / multi-backbone input
    p.add_argument("--scores", nargs="+", default=None,
                   help="List of pairs: <score_pkl_path> <weight>. Example: --scores s1.pkl 1.0 s2.pkl 0.6")

    # Standard named modality arguments (backward compatible)
    p.add_argument("--j-pkl", default=None, help="Joint score file")
    p.add_argument("--b-pkl", default=None, help="Bone score file")
    p.add_argument("--jm-pkl", default=None, help="Joint-Motion score file")
    p.add_argument("--bm-pkl", default=None, help="Bone-Motion score file")

    p.add_argument("--w-j", type=float, default=1.0, help="Weight for Joint modality")
    p.add_argument("--w-b", type=float, default=1.0, help="Weight for Bone modality")
    p.add_argument("--w-jm", type=float, default=0.6, help="Weight for Joint-Motion modality")
    p.add_argument("--w-bm", type=float, default=0.6, help="Weight for Bone-Motion modality")

    p.add_argument("--out", default=None, help="Optional path to save fused logits as .pkl")
    return p


def evaluate(fused: np.ndarray, labels: np.ndarray) -> dict:
    n = len(labels)
    assert fused.shape[0] == n, f"Fused score has {fused.shape[0]} samples but labels have {n}"
    top1 = int((fused.argmax(axis=1) == labels).sum())
    top5 = sum(int(labels[i] in fused[i].argsort()[-5:]) for i in range(n))
    return {"top1": top1 / n, "top5": top5 / n, "n": n}


def main():
    args = get_parser().parse_args()

    # 1. Load labels
    print(f"[Ensemble] Loading labels from : {args.data}")
    labels = load_labels(args.data)
    print(f"[Ensemble] Test set size       : {len(labels)}")

    modalities: List[Tuple[str, float, np.ndarray]] = []

    # 2. Parse scores
    if args.scores is not None and len(args.scores) > 0:
        if len(args.scores) % 2 != 0:
            raise ValueError("--scores must be provided as pairs of <path> <weight>")
        for i in range(0, len(args.scores), 2):
            path = args.scores[i]
            w = float(args.scores[i + 1])
            name = os.path.basename(path)
            print(f"[Ensemble] Loading {name} (w={w}) <- {path}")
            modalities.append((name, w, load_score_pkl(path)))
    else:
        if args.j_pkl:
            print(f"[Ensemble] Loading Joint (w={args.w_j}) <- {args.j_pkl}")
            modalities.append(("joint", args.w_j, load_score_pkl(args.j_pkl)))
        if args.b_pkl:
            print(f"[Ensemble] Loading Bone (w={args.w_b}) <- {args.b_pkl}")
            modalities.append(("bone", args.w_b, load_score_pkl(args.b_pkl)))
        if args.jm_pkl:
            print(f"[Ensemble] Loading Joint-Motion (w={args.w_jm}) <- {args.jm_pkl}")
            modalities.append(("joint-motion", args.w_jm, load_score_pkl(args.jm_pkl)))
        if args.bm_pkl:
            print(f"[Ensemble] Loading Bone-Motion (w={args.w_bm}) <- {args.bm_pkl}")
            modalities.append(("bone-motion", args.w_bm, load_score_pkl(args.bm_pkl)))

    if not modalities:
        raise ValueError("No score files provided. Specify --scores <path> <w> or --j-pkl / --b-pkl.")

    # 3. Fuse scores
    fused = None
    for name, w, scores in modalities:
        weighted = w * scores
        if fused is None:
            fused = weighted
        else:
            if fused.shape != scores.shape:
                raise ValueError(f"Shape mismatch: expected {fused.shape}, got {scores.shape} for '{name}'")
            fused = fused + weighted

    # 4. Evaluate
    metrics = evaluate(fused, labels)
    stream_desc = " + ".join(f"{name} (w={w})" for name, w, _ in modalities)

    print("\n" + "=" * 64)
    print("  Ensemble Late-Fusion Results")
    print("=" * 64)
    print(f"  Streams       : {stream_desc}")
    print(f"  Total Samples : {metrics['n']}")
    print(f"  Top-1 Accuracy: {metrics['top1'] * 100:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['top5'] * 100:.2f}%")
    print("=" * 64 + "\n")

    if args.out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "wb") as f:
            pickle.dump(fused, f)
        print(f"[Ensemble] Fused logits saved -> {args.out}")


if __name__ == "__main__":
    main()

"""
refine_pseudolabels.py
----------------------
Refine concept pseudo-labels by combining the fixed template with OOF teacher predictions.

Refinement logic (3-state: positive / negative / unknown):
  - Template=1 AND teacher agrees (prob ≥ pos_threshold):    → positive (1.0)
  - Template=1 AND teacher strongly disagrees (prob ≤ flip_threshold): → unknown (-1.0)
  - Template=0 AND teacher is very confident (prob ≥ new_pos_threshold): → candidate positive (soft)
  - All other: keep template value (mask unknown if borderline).

Output:
    refined_labels_v1.npz
        concept_targets [N, 74]  float32: 1.0, 0.0, or -1.0 (unknown)
        concept_mask    [N, 74]  bool
        action_labels   [N]      int64
        soft_targets    [N, 74]  float32: blended template + teacher
    refinement_report.json      statistics per concept and overall

Usage:
    python tools/refine_pseudolabels.py \\
        --template_npz work_dir/final_pipeline/ntu60_cs/pseudo_labels/template_labels_train.npz \\
        --oof_npz work_dir/final_pipeline/ntu60_cs/pseudo_labels/oof_probs.npz \\
        --out_dir work_dir/final_pipeline/ntu60_cs/pseudo_labels \\
        --round 1 \\
        --pos_threshold 0.6 \\
        --flip_threshold 0.25 \\
        --new_pos_threshold 0.8 \\
        --min_confidence 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

NUM_CONCEPTS = 74


# ---------------------------------------------------------------------------
# Core refinement
# ---------------------------------------------------------------------------

def refine_labels(
    template_targets: np.ndarray,   # [N, C]  0.0 / 1.0
    template_mask: np.ndarray,      # [N, C]  bool
    oof_probs: np.ndarray,          # [N, C]  float in [0, 1]
    oof_confidence: np.ndarray,     # [N]     float in [0, 1]
    action_labels: np.ndarray,      # [N]     int
    pos_threshold: float = 0.6,
    flip_threshold: float = 0.25,
    new_pos_threshold: float = 0.80,
    min_confidence: float = 0.50,
    soft_blend: float = 0.3,
    round_id: int = 1,
) -> tuple:
    """
    Refine pseudo-labels using OOF teacher predictions.

    Returns:
        refined_targets: [N, C]  float32  (-1.0 = unknown, 0.0 = neg, 1.0 = pos)
        refined_mask:    [N, C]  bool
        soft_targets:    [N, C]  float32  blended labels for soft BCE
        stats:           dict    per-concept and global statistics
    """
    N, C = template_targets.shape
    refined = template_targets.copy().astype(np.float32)
    refined_mask = template_mask.copy()
    soft_targets = template_targets.copy().astype(np.float32)

    stats = {
        "n_samples": int(N),
        "n_concepts": int(C),
        "round": round_id,
        "pos_threshold": pos_threshold,
        "flip_threshold": flip_threshold,
        "new_pos_threshold": new_pos_threshold,
        "min_confidence": min_confidence,
        "changes": {
            "kept_positive": 0,
            "flipped_to_unknown": 0,
            "new_positive": 0,
            "kept_negative": 0,
            "kept_unknown": 0,
        },
        "per_concept": [],
    }

    # Low-confidence samples: keep template, include in mask for known
    low_conf_mask = oof_confidence < min_confidence   # [N]

    for c in range(C):
        t_pos  = template_targets[:, c] == 1.0
        t_neg  = template_targets[:, c] == 0.0
        t_unk  = template_targets[:, c] < 0

        p = oof_probs[:, c]   # [N]

        new_concept_targets = refined[:, c].copy()
        new_concept_mask    = refined_mask[:, c].copy()

        # 1) Template positive positions
        pos_agree     = t_pos & (p >= pos_threshold)  & ~low_conf_mask
        pos_disagree  = t_pos & (p <= flip_threshold) & ~low_conf_mask

        new_concept_targets[pos_agree]    = 1.0
        new_concept_mask[pos_agree]       = True
        new_concept_targets[pos_disagree] = -1.0   # downgrade to unknown
        new_concept_mask[pos_disagree]    = False   # exclude from loss

        # Low confidence + template positive: keep as positive but include
        pos_low_conf = t_pos & low_conf_mask
        new_concept_targets[pos_low_conf] = 1.0
        new_concept_mask[pos_low_conf]    = True

        # 2) Template negative positions: check if teacher very confident positive
        new_from_neg = t_neg & (p >= new_pos_threshold) & ~low_conf_mask
        new_concept_targets[new_from_neg] = 1.0
        new_concept_mask[new_from_neg]    = True

        # Keep remaining negatives as-is
        neg_keep = t_neg & ~new_from_neg
        new_concept_targets[neg_keep] = 0.0
        new_concept_mask[neg_keep]    = True

        # 3) Template unknown positions
        unk_to_pos = t_unk & (p >= new_pos_threshold) & ~low_conf_mask
        new_concept_targets[unk_to_pos] = 1.0
        new_concept_mask[unk_to_pos]    = True

        unk_remain = t_unk & ~unk_to_pos
        new_concept_targets[unk_remain] = -1.0
        new_concept_mask[unk_remain]    = False

        refined[:, c] = new_concept_targets
        refined_mask[:, c] = new_concept_mask

        # Soft targets: blend template with teacher probs
        # For known positions: blend = (1-alpha)*template + alpha*teacher
        in_mask = new_concept_mask
        soft_targets[in_mask, c] = (
            (1.0 - soft_blend) * new_concept_targets[in_mask].clip(0, 1)
            + soft_blend * p[in_mask]
        )

        # Per-concept stats
        stats["changes"]["kept_positive"]    += int(pos_agree.sum())
        stats["changes"]["flipped_to_unknown"] += int(pos_disagree.sum())
        stats["changes"]["new_positive"]     += int(new_from_neg.sum() + unk_to_pos.sum())
        stats["changes"]["kept_negative"]    += int(neg_keep.sum())
        stats["changes"]["kept_unknown"]     += int(unk_remain.sum())

        stats["per_concept"].append({
            "concept_id": c,
            "template_positives": int(t_pos.sum()),
            "kept_positive": int(pos_agree.sum()),
            "flipped_to_unknown": int(pos_disagree.sum()),
            "new_positive": int((new_from_neg | unk_to_pos).sum()),
            "refined_positives": int((refined[:, c] == 1.0).sum()),
            "mean_teacher_prob": float(p.mean()),
        })

    return refined, refined_mask, soft_targets, stats


# ---------------------------------------------------------------------------
# Action accuracy check
# ---------------------------------------------------------------------------

def compute_action_accuracy(
    targets: np.ndarray,
    action_labels: np.ndarray,
    num_classes: int,
) -> float:
    """
    Estimate action 'accuracy' from concept labels using majority vote
    based on template dot-product similarity.  This is approximate.
    """
    # We can't compute action accuracy without template here, so return -1
    return -1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def refine_pseudolabels(
    template_npz: str,
    oof_npz: str,
    out_dir: str,
    round_id: int = 1,
    pos_threshold: float = 0.6,
    flip_threshold: float = 0.25,
    new_pos_threshold: float = 0.80,
    min_confidence: float = 0.50,
    soft_blend: float = 0.3,
):
    os.makedirs(out_dir, exist_ok=True)

    # Load template labels
    tpl = np.load(template_npz)
    template_targets = tpl["concept_targets"].astype(np.float32)   # [N, C]
    template_mask    = tpl["concept_mask"].astype(bool)             # [N, C]
    action_labels    = tpl["action_labels"].astype(np.int64)        # [N]
    sample_names     = tpl.get("sample_names", np.array([]))

    # Load OOF predictions
    oof = np.load(oof_npz)
    oof_probs      = oof["oof_probs"].astype(np.float32)    # [N, C]
    oof_confidence = oof["oof_confidence"].astype(np.float32)  # [N]

    assert len(template_targets) == len(oof_probs), (
        f"Template {len(template_targets)} samples vs OOF {len(oof_probs)} samples"
    )

    N, C = template_targets.shape
    print(f"[Refine] {N} samples, {C} concepts")
    print(f"  Thresholds: pos={pos_threshold} flip={flip_threshold} "
          f"new_pos={new_pos_threshold} min_conf={min_confidence}")

    # Run refinement
    refined, refined_mask, soft_targets, stats = refine_labels(
        template_targets=template_targets,
        template_mask=template_mask,
        oof_probs=oof_probs,
        oof_confidence=oof_confidence,
        action_labels=action_labels,
        pos_threshold=pos_threshold,
        flip_threshold=flip_threshold,
        new_pos_threshold=new_pos_threshold,
        min_confidence=min_confidence,
        soft_blend=soft_blend,
        round_id=round_id,
    )

    # Summary
    ch = stats["changes"]
    total_positions = N * C
    print(f"\n[Refine] Round {round_id} summary:")
    print(f"  Kept positive:        {ch['kept_positive']:>8} ({ch['kept_positive']/total_positions*100:.2f}%)")
    print(f"  Flipped → unknown:    {ch['flipped_to_unknown']:>8} ({ch['flipped_to_unknown']/total_positions*100:.2f}%)")
    print(f"  New positives:        {ch['new_positive']:>8} ({ch['new_positive']/total_positions*100:.2f}%)")
    print(f"  Kept negative:        {ch['kept_negative']:>8} ({ch['kept_negative']/total_positions*100:.2f}%)")
    print(f"  Kept unknown:         {ch['kept_unknown']:>8} ({ch['kept_unknown']/total_positions*100:.2f}%)")

    # Save refined labels
    out_path = os.path.join(out_dir, f"refined_labels_v{round_id}.npz")
    np.savez_compressed(
        out_path,
        concept_targets=refined,
        concept_mask=refined_mask,
        soft_targets=soft_targets,
        action_labels=action_labels,
        sample_names=sample_names,
    )

    # Save report
    report_path = os.path.join(out_dir, f"refinement_report_v{round_id}.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[Refine] Saved to {out_path}")
    print(f"[Refine] Report: {report_path}")
    return out_path, stats


def main():
    parser = argparse.ArgumentParser(description="Refine pseudo-labels from OOF predictions")
    parser.add_argument("--template_npz",       required=True)
    parser.add_argument("--oof_npz",            required=True)
    parser.add_argument("--out_dir",            required=True)
    parser.add_argument("--round",              type=int,   default=1)
    parser.add_argument("--pos_threshold",      type=float, default=0.6)
    parser.add_argument("--flip_threshold",     type=float, default=0.25)
    parser.add_argument("--new_pos_threshold",  type=float, default=0.80)
    parser.add_argument("--min_confidence",     type=float, default=0.50)
    parser.add_argument("--soft_blend",         type=float, default=0.3)
    args = parser.parse_args()

    refine_pseudolabels(
        template_npz=args.template_npz,
        oof_npz=args.oof_npz,
        out_dir=args.out_dir,
        round_id=args.round,
        pos_threshold=args.pos_threshold,
        flip_threshold=args.flip_threshold,
        new_pos_threshold=args.new_pos_threshold,
        min_confidence=args.min_confidence,
        soft_blend=args.soft_blend,
    )


if __name__ == "__main__":
    main()

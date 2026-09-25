"""
evaluate_concepts.py
--------------------
Comprehensive concept evaluation: computes per-concept and aggregate metrics
from stored prediction NPZ files or from model inference.

Metrics reported:
  - Macro-F1, Micro-F1, Precision, Recall  (at threshold 0.5)
  - Per-concept: F1, Precision, Recall, TP/FP/FN, support
  - Coverage (fraction of non-unknown positions in mask)
  - Mean entropy (calibration proxy)
  - Action Top-1 accuracy from symbolic and learned logits
  - Template vs refined label disagreement analysis

Usage (from stored predictions):
    python tools/evaluate_concepts.py \\
        --pred_npz work_dir/final_pipeline/ntu60_cs/eval_epoch20.npz \\
        --concept_csv concepts/ntu_rgbd_60_120.csv \\
        --num_classes 60 \\
        --out work_dir/final_pipeline/ntu60_cs/metrics/concept_eval.json

Usage (from OOF predictions against template):
    python tools/evaluate_concepts.py \\
        --oof_npz work_dir/final_pipeline/ntu60_cs/pseudo_labels/oof_probs.npz \\
        --template_npz work_dir/final_pipeline/ntu60_cs/pseudo_labels/template_labels_train.npz \\
        --concept_csv concepts/ntu_rgbd_60_120.csv \\
        --num_classes 60 \\
        --out work_dir/final_pipeline/ntu60_cs/metrics/oof_concept_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.build_template_labels import load_concept_csv


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def calibration_metrics(probs: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute Brier score and expected calibration error on known labels."""
    if probs.size == 0:
        return {"brier": -1.0, "ece": -1.0}
    brier = float(np.mean((probs - targets) ** 2))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        in_bin = (probs >= left) & (
            (probs < right) if right < 1.0 else (probs <= right)
        )
        if not np.any(in_bin):
            continue
        confidence = float(probs[in_bin].mean())
        accuracy = float(targets[in_bin].mean())
        ece += float(in_bin.mean()) * abs(confidence - accuracy)
    return {"brier": brier, "ece": float(ece)}

def compute_concept_metrics(
    probs: np.ndarray,        # [N, C]
    targets: np.ndarray,      # [N, C]  0, 1, or -1 (unknown)
    concept_names: list,
    threshold: float = 0.5,
) -> dict:
    """Full concept evaluation with per-concept breakdown."""
    N, C = probs.shape
    mask = targets >= 0    # [N, C] known positions
    preds = (probs >= threshold).astype(np.int32)

    # Global metrics (flatten all known positions)
    flat_t = targets[mask].astype(np.int32)
    flat_p = preds[mask]
    flat_prob = probs[mask]

    # Guard: need both classes present for some metrics
    has_both = len(np.unique(flat_t)) > 1

    result = {
        "threshold":  threshold,
        "n_samples":  int(N),
        "n_concepts": int(C),
        "coverage":   float(mask.mean()),
        "global": {
            "macro_f1":  float(f1_score(flat_t, flat_p, average="macro",  zero_division=0)),
            "micro_f1":  float(f1_score(flat_t, flat_p, average="micro",  zero_division=0)),
            "precision": float(precision_score(flat_t, flat_p, average="micro", zero_division=0)),
            "recall":    float(recall_score(flat_t, flat_p, average="micro",    zero_division=0)),
            "mean_prob":    float(probs.mean()),
            "mean_entropy": float(
                (-probs * np.log(probs.clip(1e-7))
                 - (1 - probs) * np.log((1 - probs).clip(1e-7))).mean()
            ),
            **calibration_metrics(flat_prob, flat_t),
        },
        "per_concept": [],
    }

    if has_both:
        try:
            result["global"]["micro_ap"]  = float(average_precision_score(flat_t, flat_prob, average="micro"))
        except Exception:
            result["global"]["micro_ap"] = -1.0

    for c in range(C):
        cm = mask[:, c]
        tc = targets[cm, c].astype(np.int32)
        pc = preds[cm, c]
        probc = probs[cm, c]

        n_pos = int((tc == 1).sum())
        n_neg = int((tc == 0).sum())

        entry = {
            "concept_id":   c,
            "concept_name": concept_names[c] if c < len(concept_names) else f"c{c}",
            "n_known":      int(cm.sum()),
            "n_positive":   n_pos,
            "n_negative":   n_neg,
            "mean_prob":    float(probc.mean()) if len(probc) > 0 else -1.0,
            "f1":           float(f1_score(tc, pc, zero_division=0)),
            "precision":    float(precision_score(tc, pc, zero_division=0)),
            "recall":       float(recall_score(tc, pc, zero_division=0)),
        }
        entry.update(calibration_metrics(probc, tc))
        result["per_concept"].append(entry)

    return result


def compute_action_metrics(
    action_scores: np.ndarray,  # [N, num_classes]
    action_labels: np.ndarray,  # [N]
) -> dict:
    if action_scores.size == 0:
        return {"top1": -1.0, "top5": -1.0}

    preds = action_scores.argmax(1)
    top1 = float((preds == action_labels).mean())

    # Top-5
    top5_k = min(5, action_scores.shape[1])
    top5_preds = np.argsort(action_scores, axis=1)[:, -top5_k:]
    top5 = float(
        np.array([action_labels[i] in top5_preds[i] for i in range(len(action_labels))]).mean()
    )
    return {"top1": top1, "top5": top5}


# ---------------------------------------------------------------------------
# Compare template vs refined labels
# ---------------------------------------------------------------------------

def compare_template_vs_refined(
    template_targets: np.ndarray,   # [N, C]
    refined_targets: np.ndarray,    # [N, C]
    concept_names: list,
) -> dict:
    """Report label changes from template to refined."""
    N, C = template_targets.shape
    total = N * C

    tpl_pos = (template_targets == 1.0).sum()
    tpl_neg = (template_targets == 0.0).sum()

    ref_pos = (refined_targets == 1.0).sum()
    ref_neg = (refined_targets == 0.0).sum()
    ref_unk = (refined_targets < 0).sum()

    flipped = (template_targets == 1.0) & (refined_targets < 0)  # pos → unknown
    new_pos = (template_targets == 0.0) & (refined_targets == 1.0)  # neg → pos

    return {
        "total_positions": int(total),
        "template_positives": int(tpl_pos),
        "template_negatives": int(tpl_neg),
        "refined_positives": int(ref_pos),
        "refined_negatives": int(ref_neg),
        "refined_unknown": int(ref_unk),
        "pos_flipped_to_unknown": int(flipped.sum()),
        "neg_flipped_to_positive": int(new_pos.sum()),
        "change_rate": float((flipped | new_pos).sum() / total),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_concepts(
    probs: np.ndarray,
    targets: np.ndarray,
    action_scores: np.ndarray,
    action_labels: np.ndarray,
    concept_names: list,
    out_path: str,
    template_targets: np.ndarray = None,
    refined_targets: np.ndarray = None,
):
    result = compute_concept_metrics(probs, targets, concept_names)
    result["action"] = compute_action_metrics(action_scores, action_labels)

    if template_targets is not None and refined_targets is not None:
        result["label_changes"] = compare_template_vs_refined(
            template_targets, refined_targets, concept_names
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[evaluate_concepts] Results saved to {out_path}")
    print(f"  Macro-F1:  {result['global']['macro_f1']:.4f}")
    print(f"  Micro-F1:  {result['global']['micro_f1']:.4f}")
    print(f"  Precision: {result['global']['precision']:.4f}")
    print(f"  Recall:    {result['global']['recall']:.4f}")
    print(f"  Coverage:  {result['coverage']:.4f}")
    print(f"  Entropy:   {result['global']['mean_entropy']:.4f}")
    print(f"  Action Top-1: {result['action']['top1'] * 100:.2f}%")
    print(f"  Action Top-5: {result['action']['top5'] * 100:.2f}%")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate concept decoder predictions")
    parser.add_argument("--pred_npz",     default=None, help="Prediction NPZ from eval phase")
    parser.add_argument("--oof_npz",      default=None, help="OOF prediction NPZ")
    parser.add_argument("--template_npz", default=None, help="Template labels NPZ (targets)")
    parser.add_argument("--refined_npz",  default=None, help="Refined labels NPZ for comparison")
    parser.add_argument("--concept_csv",  required=True, help="Path to concept CSV")
    parser.add_argument("--num_classes",  type=int, required=True, choices=[60, 120])
    parser.add_argument("--out",          required=True, help="Output JSON path")
    parser.add_argument("--threshold",    type=float, default=0.5)
    args = parser.parse_args()

    # Load concept names
    _, _, concept_names = load_concept_csv(args.concept_csv, args.num_classes)

    # Determine source of predictions and targets
    if args.pred_npz:
        data = np.load(args.pred_npz)
        probs         = data["concept_probs"]
        targets       = data["concept_targets"]
        action_scores = data.get("action_scores", np.array([]))
        action_labels = data["action_labels"]
    elif args.oof_npz and args.template_npz:
        oof  = np.load(args.oof_npz)
        tpl  = np.load(args.template_npz)
        probs         = oof["oof_probs"]
        targets       = tpl["concept_targets"]
        action_scores = oof.get("oof_action", np.array([]))
        action_labels = tpl["action_labels"]
    else:
        raise ValueError("Must provide either --pred_npz or both --oof_npz and --template_npz")

    # Optional: load refined labels for comparison
    template_targets = None
    refined_targets  = None
    if args.template_npz and args.refined_npz:
        template_targets = np.load(args.template_npz)["concept_targets"]
        refined_targets  = np.load(args.refined_npz)["concept_targets"]

    evaluate_concepts(
        probs=probs,
        targets=targets,
        action_scores=action_scores if hasattr(action_scores, '__len__') else np.array([]),
        action_labels=action_labels,
        concept_names=concept_names,
        out_path=args.out,
        template_targets=template_targets,
        refined_targets=refined_targets,
    )


if __name__ == "__main__":
    main()

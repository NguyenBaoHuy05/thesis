"""Evaluate saved v2 concept and parent predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def multilabel(probabilities: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = probabilities >= threshold
    target = targets.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    f1 = float((2 * tp) / max(2 * tp + fp + fn, 1))
    return {
        "micro_f1": f1,
        "exact_match": float(np.all(pred == target, axis=1).mean()),
        "sparsity": float(pred.mean()),
        "brier": float(np.mean((probabilities - targets) ** 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="path to test_predictions.npz")
    parser.add_argument("--rulebook", default=None, help="optional final_rules.json for branch contribution")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    with np.load(args.predictions) as payload:
        result = {
            "concept": multilabel(payload["concept_probs"], payload["concept_targets"]),
            "parent": multilabel(payload["parent_probs"], payload["parent_targets"]),
        }
        if "concept_negative_targets" in payload:
            negative = payload["concept_negative_targets"].astype(bool)
            negative_probs = payload["concept_probs"][negative]
            result["known_negative"] = {
                "count": int(negative.sum()),
                "violation_rate": float(
                    (negative_probs >= 0.5).mean() if negative_probs.size else 0.0
                ),
                "mean_probability": float(
                    negative_probs.mean() if negative_probs.size else 0.0
                ),
            }
        if "raw_concept_probs" in payload and "parent_for_child" in payload:
            delta = payload["raw_concept_probs"] - payload["parent_for_child"]
            result["hierarchy"] = {
                "violation_rate": float((delta > 0.0).mean()),
                "violation_mean": float(np.maximum(delta, 0.0).mean()),
            }
        if args.rulebook:
            rulebook = json.loads(Path(args.rulebook).read_text(encoding="utf-8"))
            dense = rulebook["dense_weights"]
            concept_weights = np.asarray(dense["concept"], dtype=np.float64)
            parent_weights = np.asarray(dense["parent"], dtype=np.float64)
            bias = np.asarray(dense["bias"], dtype=np.float64)
            concept_score = payload["concept_probs"] @ concept_weights.T
            concept_score /= np.maximum(concept_weights.sum(axis=1, keepdims=True).T, 1e-6)
            parent_score = payload["parent_probs"] @ parent_weights.T
            parent_score /= np.maximum(parent_weights.sum(axis=1, keepdims=True).T, 1e-6)
            parent_score *= float(rulebook["coefficients"]["parent_weight"])
            result["branch_contribution"] = {
                "mean_abs_concept_score": float(np.abs(concept_score).mean()),
                "mean_abs_parent_score": float(np.abs(parent_score).mean()),
                "mean_abs_bias": float(np.abs(bias).mean()),
            }
            if "action_labels" in payload:
                labels = payload["action_labels"]
                result["branch_contribution"]["concept_only_top1"] = float(
                    (concept_score.argmax(axis=1) == labels).mean()
                )
                result["branch_contribution"]["parent_only_top1"] = float(
                    (parent_score.argmax(axis=1) == labels).mean()
                )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

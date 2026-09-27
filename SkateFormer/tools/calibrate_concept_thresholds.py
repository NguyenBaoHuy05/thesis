"""Fit per-concept thresholds on validation predictions only.

The input NPZ must contain ``concept_probs`` and ``concept_targets``. The
result can be referenced by ``concept_thresholds_path`` in a train config and
will then be used only for the final concept metric computation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def f1_score(probabilities: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    predicted = probabilities >= threshold
    targets = targets.astype(bool)
    tp = np.logical_and(predicted, targets).sum()
    fp = np.logical_and(predicted, ~targets).sum()
    fn = np.logical_and(~predicted, targets).sum()
    return float((2 * tp) / max(2 * tp + fp + fn, 1))


def calibrate(prediction_path: Path, output_path: Path) -> dict:
    with np.load(prediction_path) as payload:
        probabilities = np.asarray(payload["concept_probs"], dtype=np.float64)
        targets = np.asarray(payload["concept_targets"], dtype=np.float64)
    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("concept_probs and concept_targets must have shape [N,K]")

    candidates = np.linspace(0.05, 0.95, 91)
    thresholds = []
    per_concept_f1 = []
    for concept_id in range(probabilities.shape[1]):
        scores = [
            f1_score(probabilities[:, concept_id], targets[:, concept_id], threshold)
            for threshold in candidates
        ]
        best = max(scores)
        tied = [
            candidate
            for candidate, score in zip(candidates, scores)
            if abs(score - best) <= 1e-12
        ]
        selected = min(tied, key=lambda value: abs(float(value) - 0.5))
        thresholds.append(float(selected))
        per_concept_f1.append(float(best))

    result = {
        "thresholds": thresholds,
        "mean_per_concept_f1": float(np.mean(per_concept_f1)),
        "num_concepts": int(probabilities.shape[1]),
        "num_samples": int(probabilities.shape[0]),
        "source": str(prediction_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibrate(Path(args.predictions), Path(args.output))


if __name__ == "__main__":
    main()

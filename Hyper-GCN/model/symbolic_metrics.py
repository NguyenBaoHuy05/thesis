"""Metrics for weak concept targets supplied by action-class templates."""

import math

import numpy as np
from sklearn.metrics import average_precision_score


def concept_metrics(logits, targets, metadata, threshold=0.5):
    logits, targets = np.asarray(logits), np.asarray(targets)
    if (logits.ndim != 2 or logits.shape != targets.shape or logits.shape[0] == 0
            or logits.shape[1] != len(metadata)):
        raise ValueError("concept metrics require matching nonempty [samples, concepts] arrays")
    if not np.isfinite(logits).all() or not np.isin(targets, (0, 1)).all():
        raise ValueError("concept metrics require finite logits and binary targets")
    if not 0 < threshold < 1:
        raise ValueError("concept threshold must be between 0 and 1")
    predictions = logits >= math.log(threshold / (1.0 - threshold))
    truth = targets.astype(bool)
    tp = (predictions & truth).sum(axis=0)
    fp = (predictions & ~truth).sum(axis=0)
    fn = (~predictions & truth).sum(axis=0)
    positives = truth.sum(axis=0)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    valid = positives > 0
    ap = [
        float(average_precision_score(targets[:, index], logits[:, index])) if valid[index] else None
        for index in range(logits.shape[1])
    ]

    def summarize(indices):
        active = [index for index in indices if valid[index]]
        micro_denominator = int(denominator[indices].sum())
        return {
            "num_concepts": len(indices), "concepts_with_positives": len(active),
            "macro_f1": float(f1[active].mean()) if active else None,
            "macro_f1_all": float(f1[indices].mean()) if indices else None,
            "micro_f1": float(2 * tp[indices].sum() / micro_denominator) if micro_denominator else 0.0,
            "mAP": float(np.mean([ap[index] for index in active])) if active else None,
        }

    result = summarize(list(range(logits.shape[1])))
    result.update({
        "num_samples": int(logits.shape[0]), "threshold": float(threshold),
        "target_granularity": "action-class templates, not per-video annotations",
        "macro_policy": "macro_f1 and mAP exclude concepts with no positive evaluation targets",
        "scopes": {
            scope: summarize([i for i, row in enumerate(metadata) if row["scope"] == scope])
            for scope in sorted({row["scope"] for row in metadata})
        },
        "per_concept": [
            {
                "name": row["concept_name"], "scope": row["scope"],
                "positive_targets": int(positives[index]),
                "predicted_positives": int(predictions[:, index].sum()),
                "f1": float(f1[index]), "AP": ap[index],
            }
            for index, row in enumerate(metadata)
        ],
    })
    return result

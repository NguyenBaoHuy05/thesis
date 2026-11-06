"""Evaluate an HCG-NS-SkateFormer checkpoint on the configured test split.

This evaluator uses the same HCG-NS model, ontology, action-concept matrix and test
preprocessing as ``main_hcg_ns.py``.  It loads a full checkpoint such as
``work_dir/.../final_model.pt`` and does not retrain the model.

Example (run from the SkateFormer repository root)::

    python eval_hcg_concepts.py \
        --config config/train/HCG_NS_SkateFormer_ntu120_csub.yaml \
        --checkpoint work_dir/hcg_ns/main_ntu120_csub/final_model.pt

The evaluator writes ``concept_metrics.json`` and
``concept_metrics_per_concept.csv`` next to the checkpoint unless
``--output-dir`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from main_hcg_ns import (
    HCGNSSkateFormer,
    device_from_config,
    load_config,
    make_dataset,
    make_loader,
    move_batch,
    resolve,
    set_seed,
    validate_paths,
)


def load_full_checkpoint(model: HCGNSSkateFormer, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")

    state = {}
    for key, value in payload.items():
        clean = str(key)
        while clean.startswith(("module.", "model.")):
            clean = clean.split(".", 1)[1]
        state[clean] = value

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded HCG-NS checkpoint: {path}")
    print(f"Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}")
    if missing:
        print(f"First missing keys: {missing[:5]}")
    if unexpected:
        print(f"First unexpected keys: {unexpected[:5]}")
    decoder_missing = [
        key for key in missing if key.startswith("concept_decoder.")
    ]
    if decoder_missing:
        raise RuntimeError(
            "checkpoint is incompatible with the scope-routed pose/motion "
            "concept decoder; missing decoder keys include: "
            + ", ".join(decoder_missing[:5])
        )
    negative_key = "symbolic.negative_raw"
    if model.symbolic.negative_score_enabled and negative_key in missing:
        raise RuntimeError(
            "negative scoring is enabled by the config, but the checkpoint "
            "does not contain symbolic.negative_raw"
        )
    if not model.symbolic.negative_score_enabled and negative_key in unexpected:
        raise RuntimeError(
            "checkpoint contains symbolic negative rules, but the supplied "
            "config has negative_score_enabled=false"
        )


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0)


def per_label_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    names: list[str],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    target = targets.astype(bool)
    prediction = probabilities >= threshold
    total = np.full(probabilities.shape[1], probabilities.shape[0], dtype=np.float64)
    correct = (prediction == target).sum(axis=0).astype(np.float64)
    tp = np.logical_and(prediction, target).sum(axis=0).astype(np.float64)
    fp = np.logical_and(prediction, ~target).sum(axis=0).astype(np.float64)
    fn = np.logical_and(~prediction, target).sum(axis=0).astype(np.float64)
    positive = target.sum(axis=0).astype(np.float64)

    accuracy = safe_div(correct, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    rows = []
    for index in range(probabilities.shape[1]):
        rows.append({
            "concept_id": index,
            "concept_name": names[index] if index < len(names) else f"concept_{index}",
            "accuracy": float(accuracy[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "positive_support": int(positive[index]),
            "predicted_positive": int(prediction[:, index].sum()),
            "true_positive": int(tp[index]),
            "false_positive": int(fp[index]),
            "false_negative": int(fn[index]),
            "mean_probability": float(probabilities[:, index].mean()),
        })

    flat_target = target.reshape(-1)
    flat_prediction = prediction.reshape(-1)
    flat_tp = float(np.logical_and(flat_prediction, flat_target).sum())
    flat_fp = float(np.logical_and(flat_prediction, ~flat_target).sum())
    flat_fn = float(np.logical_and(~flat_prediction, flat_target).sum())
    micro_precision = flat_tp / max(flat_tp + flat_fp, 1.0)
    micro_recall = flat_tp / max(flat_tp + flat_fn, 1.0)
    micro_f1 = 2.0 * micro_precision * micro_recall / max(
        micro_precision + micro_recall, 1e-12
    )
    flat_probabilities = probabilities.reshape(-1)
    flat_targets = targets.astype(np.float64).reshape(-1)
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 11)[:-1]:
        upper = min(lower + 0.1, 1.0)
        mask = (flat_probabilities >= lower) & (
            flat_probabilities <= upper if upper == 1.0 else flat_probabilities < upper
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(flat_probabilities[mask].mean())
                - float(flat_targets[mask].mean())
            )

    aggregate = {
        "accuracy": float(accuracy.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "exact_match": float(np.all(prediction == target, axis=1).mean()),
        "sparsity": float(prediction.mean()),
        "brier": float(np.mean((probabilities - targets) ** 2)),
        "ece": float(ece),
    }
    return rows, aggregate


@torch.no_grad()
def evaluate(
    model: HCGNSSkateFormer,
    loader,
    device: torch.device,
    threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    model.eval()
    action_scores = []
    labels = []
    concept_probs = []
    concept_targets = []
    parent_probs = []
    parent_targets = []
    parent_for_child = []
    decoder = model.concept_decoder
    logit_accumulators = {
        name: {"sum": 0.0, "sum_sq": 0.0, "sum_abs": 0.0, "count": 0}
        for name in (
            "time_logits", "raw_sequence_logits", "refined_sequence_logits",
            "refiner_delta",
        )
    }
    saturated_low = 0
    saturated_high = 0
    sigmoid_count = 0

    for batch in loader:
        data, index_t, label, _, _ = move_batch(batch, device)
        output = model(data, index_t)
        action_scores.append(output["action_scores"].cpu().numpy())
        labels.append(label.cpu().numpy())
        concept_probs.append(output["sequence_probs"].cpu().numpy())
        concept_targets.append(model.symbolic.support[label].cpu().numpy())
        parent_probs.append(output["parent_probs"].cpu().numpy())
        parent_targets.append(model.symbolic.parent_support[label].cpu().numpy())
        parent_for_child.append(output["parent_for_child"].cpu().numpy())

        time_logits = output["time_logits"].float()
        raw_sequence_logits = output["raw_sequence_logits"].float()
        refined_sequence_logits = output["sequence_logits"].float()
        for name, values in (
            ("time_logits", time_logits),
            ("raw_sequence_logits", raw_sequence_logits),
            ("refined_sequence_logits", refined_sequence_logits),
            ("refiner_delta", refined_sequence_logits - raw_sequence_logits),
        ):
            values = values.detach()
            accumulator = logit_accumulators[name]
            accumulator["sum"] += values.sum().item()
            accumulator["sum_sq"] += values.square().sum().item()
            accumulator["sum_abs"] += values.abs().sum().item()
            accumulator["count"] += values.numel()

        refined_probs = torch.sigmoid(refined_sequence_logits)
        saturated_low += int((refined_probs < 0.01).sum().item())
        saturated_high += int((refined_probs > 0.99).sum().item())
        sigmoid_count += refined_probs.numel()

    scores = np.concatenate(action_scores)
    action_labels = np.concatenate(labels)
    probs = np.concatenate(concept_probs)
    targets = np.concatenate(concept_targets)
    parent = np.concatenate(parent_probs)
    parent_target = np.concatenate(parent_targets)
    parent_child = np.concatenate(parent_for_child)

    logit_summaries = {}
    for name, accumulator in logit_accumulators.items():
        count = max(accumulator["count"], 1)
        mean = accumulator["sum"] / count
        variance = max(accumulator["sum_sq"] / count - mean * mean, 0.0)
        logit_summaries[name] = {
            "mean": float(mean),
            "std": float(variance ** 0.5),
            "mean_abs": float(accumulator["sum_abs"] / count),
        }
    raw_mean_abs = logit_summaries["raw_sequence_logits"]["mean_abs"]
    raw_std = logit_summaries["raw_sequence_logits"]["std"]
    temperature = decoder.temperature
    decoder_type = decoder.decoder_type
    decoder_logit_diagnostics = {
        "architecture": {
            "prototype": "scope_routed_pose_motion_prototypes",
            "direct": "scope_routed_pose_motion_direct_heads",
            "grouped": "ontology_grouped_pose_motion_heads",
        }[decoder_type],
        "decoder_type": decoder_type,
        "scope_counts": {
            name: len(concept_ids)
            for name, concept_ids in decoder.scope_ids.items()
        },
        "scope_routing": {
            "spatial": "pose",
            "temporal": "motion_difference",
            "context": "pose_person_pair_when_available",
        },
        "refiner_enabled": decoder.refiner_enabled,
        "temperature": (
            float(temperature.item()) if temperature is not None else None
        ),
        "prototype_logit_bound": (
            float(1.0 / temperature.item()) if temperature is not None else None
        ),
        "head_group_counts": (
            {
                name: len(concept_ids)
                for name, concept_ids in decoder.base.group_ids.items()
            }
            if decoder_type == "grouped" else None
        ),
        **logit_summaries,
        "refiner_delta_to_raw_mean_abs_ratio": float(
            logit_summaries["refiner_delta"]["mean_abs"]
            / max(raw_mean_abs, 1e-12)
        ),
        "refiner_delta_to_raw_std_ratio": float(
            logit_summaries["refiner_delta"]["std"]
            / max(raw_std, 1e-12)
        ),
        "refined_sigmoid_saturation_below_0_01": float(
            saturated_low / max(sigmoid_count, 1)
        ),
        "refined_sigmoid_saturation_above_0_99": float(
            saturated_high / max(sigmoid_count, 1)
        ),
    }

    concept_names = list(model.symbolic.concept_names)
    parent_names = list(model.symbolic.parent_names)
    concept_rows, concept_summary = per_label_metrics(probs, targets, concept_names, threshold)
    _, parent_summary = per_label_metrics(parent, parent_target, parent_names, 0.5)

    prediction = scores.argmax(axis=1)
    top5 = np.argsort(scores, axis=1)[:, -min(5, scores.shape[1]):]
    confusion = np.zeros((scores.shape[1], scores.shape[1]), dtype=np.int64)
    np.add.at(confusion, (action_labels, prediction), 1)
    per_class_accuracy = [
        float(confusion[class_id, class_id] / max(confusion[class_id].sum(), 1))
        for class_id in range(scores.shape[1])
    ]
    negative_targets = model.symbolic.negative_support[action_labels].cpu().numpy()
    negative_mask = negative_targets.astype(bool)
    negative_probs = probs[negative_mask]
    hierarchy_delta = probs - parent_child

    result = {
        "threshold": float(threshold),
        "num_samples": int(len(action_labels)),
        "num_concepts": int(probs.shape[1]),
        "num_parents": int(parent.shape[1]),
        "action": {
            "top1": float((prediction == action_labels).mean()),
            "top5": float(np.array([
                action_labels[i] in top5[i] for i in range(len(action_labels))
            ]).mean()),
            "per_class_accuracy": per_class_accuracy,
            "confusion_matrix": confusion.tolist(),
        },
        "concept": concept_summary,
        "parent": parent_summary,
        "known_negative": {
            "count": int(negative_mask.sum()),
            "violation_rate": float((negative_probs >= 0.5).mean()) if negative_probs.size else 0.0,
            "mean_probability": float(negative_probs.mean()) if negative_probs.size else 0.0,
        },
        "hierarchy": {
            "violation_rate": float((hierarchy_delta > 0.0).mean()),
            "violation_mean": float(np.maximum(hierarchy_delta, 0.0).mean()),
        },
        "decoder_logit_diagnostics": decoder_logit_diagnostics,
        "per_concept": concept_rows,
    }
    return result, concept_names


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "concept_metrics.json"
    csv_path = output_dir / "concept_metrics_per_concept.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        rows = result["per_concept"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HCG-NS concepts from a checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    root = Path(__file__).resolve().parent
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 1)))
    device = device_from_config(cfg)
    validate_paths(root, cfg)

    test_dataset = make_dataset(root, cfg, "test", False, eval_mode=True)
    test_loader = make_loader(test_dataset, cfg, False)
    model = HCGNSSkateFormer(
        backbone_args=cfg["model_args"],
        concept_csv=resolve(root, cfg["concept_csv"]),
        metadata_csv=resolve(root, cfg["metadata_csv"]),
        num_classes=int(cfg["num_classes"]),
        num_concepts=int(cfg.get("num_concepts", 78)),
        residual_action_head=bool(cfg.get("residual_action_head", False)),
        residual_action_weight=float(cfg.get("residual_action_weight", 0.1)),
        prototype_args=cfg.get("prototype_args", {}),
        action_logit_scale=float(cfg.get("action_logit_scale", 1.0)),
        concept_score_mode=str(cfg.get("concept_score_mode", "probability")),
        concept_idf_power=float(cfg.get("concept_idf_power", 0.0)),
        negative_score_enabled=bool(cfg.get("negative_score_enabled", False)),
        negative_score_weight=float(cfg.get("negative_score_weight", 0.25)),
        negative_score_topk=int(cfg.get("negative_score_topk", 5)),
        negative_score_margin=(
            None if cfg.get("negative_score_margin") is None
            else float(cfg["negative_score_margin"])
        ),
        negative_score_temperature=float(
            cfg.get("negative_score_temperature", 0.25)
        ),
    ).to(device)
    load_full_checkpoint(model, checkpoint_path, device)
    print(f"Starting test evaluation on {len(test_dataset)} samples...")
    result, _ = evaluate(model, test_loader, device, args.threshold)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoint_path.parent
    write_outputs(result, output_dir)
    print(json.dumps({k: result[k] for k in (
        "action", "concept", "parent", "known_negative", "hierarchy",
        "decoder_logit_diagnostics",
    )}, indent=2))


if __name__ == "__main__":
    main()

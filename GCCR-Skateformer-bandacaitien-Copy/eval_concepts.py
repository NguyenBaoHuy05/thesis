"""
Evaluate concept predictions from a trained JointSpatialTemporalCBM checkpoint.

The concept target is the class-level concept annotation loaded by the model
from concepts/ntu_rgbd_60_120.csv.  Therefore, the reported metrics measure
agreement between the predicted concept probabilities and the concept vector
assigned to each ground-truth action class.

Example:
    python eval_concepts.py \
        --config config/ntu/skateformer/ntu_cs_j.yaml \
        --checkpoint work_dir/ntu_cs/skateformer_joint_symbolic/joint_best.pt

Outputs are written next to the checkpoint by default:
    concept_metrics.csv
    concept_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from train_joint import (
    build_loaders,
    build_model_from_args,
    get_parser as get_train_parser,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    # Reuse the training parser so model/data options stay identical to the
    # training pipeline.  Only evaluation-specific arguments are added here.
    parser = get_train_parser()
    parser.description = "Evaluate per-concept metrics for a trained Joint CBM"
    parser.set_defaults(phase="score")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert concept probabilities to 0/1.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for concept_metrics.csv/json. Defaults to checkpoint directory.",
    )
    parser.add_argument(
        "--include-sample-probs",
        action="store_true",
        help="Also save per-sample concept probabilities to concept_sample_probs.npz.",
    )

    # Apply YAML defaults before the final parse, matching train_joint.py.
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", default=None)
    config_args, _ = config_probe.parse_known_args()
    if config_args.config and os.path.isfile(config_args.config):
        with open(config_args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        parser.set_defaults(**{k: v for k, v in cfg.items() if not isinstance(v, (dict, list))})

    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def _load_full_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    raw = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw, dict):
        state = raw.get("state_dict", raw)
    else:
        state = raw
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(raw).__name__}")

    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"[ConceptEval][WARN] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"[ConceptEval][WARN] missing keys: {missing[:5]}")
        if unexpected:
            print(f"[ConceptEval][WARN] unexpected keys: {unexpected[:5]}")
    return missing, unexpected


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0)


@torch.no_grad()
def evaluate_concepts(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    threshold: float,
    save_sample_probs: bool = False,
) -> Dict[str, Any]:
    model.eval()

    num_concepts = int(model.num_concepts)
    correct = np.zeros(num_concepts, dtype=np.float64)
    target_positive = np.zeros(num_concepts, dtype=np.float64)
    predicted_positive = np.zeros(num_concepts, dtype=np.float64)
    true_positive = np.zeros(num_concepts, dtype=np.float64)
    false_positive = np.zeros(num_concepts, dtype=np.float64)
    false_negative = np.zeros(num_concepts, dtype=np.float64)
    probability_sum = np.zeros(num_concepts, dtype=np.float64)
    total_samples = 0

    sample_probs: List[np.ndarray] = []
    sample_targets: List[np.ndarray] = []
    sample_labels: List[np.ndarray] = []

    for batch in loader:
        if len(batch) == 4:
            data, index_t, labels, _ = batch
            index_t = index_t.to(device, non_blocking=True)
        else:
            data, labels, _ = batch
            index_t = None

        data = data.float().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            features = model.extract_feat(data, index_t)
            _, group_probs = model.encode_group_concepts(features)
            probs = torch.cat([group.float() for group in group_probs], dim=-1)

        targets = model.concept_matrix[labels].float()
        predictions = probs >= threshold
        target_bool = targets >= 0.5

        probs_np = probs.float().cpu().numpy()
        target_np = target_bool.cpu().numpy()
        pred_np = predictions.cpu().numpy()

        total_samples += probs_np.shape[0]
        correct += (pred_np == target_np).sum(axis=0)
        target_positive += target_np.sum(axis=0)
        predicted_positive += pred_np.sum(axis=0)
        true_positive += (pred_np & target_np).sum(axis=0)
        false_positive += (pred_np & ~target_np).sum(axis=0)
        false_negative += (~pred_np & target_np).sum(axis=0)
        probability_sum += probs_np.sum(axis=0)

        if save_sample_probs:
            sample_probs.append(probs_np)
            sample_targets.append(target_np.astype(np.int8))
            sample_labels.append(labels.cpu().numpy())

    total = float(total_samples)
    accuracy = correct / total if total else np.zeros(num_concepts, dtype=np.float64)
    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    mean_probability = probability_sum / total if total else np.zeros(num_concepts, dtype=np.float64)

    names = list(model.concept_names)
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "concept_index": i,
                "concept": name,
                "accuracy": float(accuracy[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "positive_support": int(target_positive[i]),
                "predicted_positive": int(predicted_positive[i]),
                "true_positive": int(true_positive[i]),
                "false_positive": int(false_positive[i]),
                "false_negative": int(false_negative[i]),
                "mean_probability": float(mean_probability[i]),
            }
        )

    micro_tp = float(true_positive.sum())
    micro_fp = float(false_positive.sum())
    micro_fn = float(false_negative.sum())
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    result: Dict[str, Any] = {
        "threshold": float(threshold),
        "num_samples": total_samples,
        "num_concepts": num_concepts,
        "mean_accuracy": float(accuracy.mean()) if num_concepts else 0.0,
        "mean_precision": float(precision.mean()) if num_concepts else 0.0,
        "mean_recall": float(recall.mean()) if num_concepts else 0.0,
        "mean_f1": float(f1.mean()) if num_concepts else 0.0,
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "concepts": rows,
    }

    if save_sample_probs:
        result["sample_probs"] = np.concatenate(sample_probs, axis=0)
        result["sample_targets"] = np.concatenate(sample_targets, axis=0)
        result["sample_labels"] = np.concatenate(sample_labels, axis=0)
    return result


def _write_outputs(result: Dict[str, Any], output_dir: str, save_sample_probs: bool):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "concept_metrics.csv")
    json_path = os.path.join(output_dir, "concept_metrics.json")

    rows = result["concepts"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {k: v for k, v in result.items() if k not in {"concepts", "sample_probs", "sample_targets", "sample_labels"}}
    summary["concepts"] = rows
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if save_sample_probs:
        np.savez_compressed(
            os.path.join(output_dir, "concept_sample_probs.npz"),
            probabilities=result["sample_probs"],
            targets=result["sample_targets"],
            labels=result["sample_labels"],
        )

    return csv_path, json_path


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    set_seed(args.seed)

    # A trained joint checkpoint already contains the backbone.  Do not pass it
    # as a backbone warm-start checkpoint during construction; load the full
    # state dict immediately afterward instead.
    args.pretrained_checkpoint = None
    args.concept_checkpoint = None
    args.resume_checkpoint = None
    args.checkpoint = None
    model = build_model_from_args(args, device)
    _load_full_checkpoint(model, checkpoint_path, device)
    model.eval()

    _, test_loader = build_loaders(args)
    result = evaluate_concepts(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=use_amp,
        threshold=args.threshold,
        save_sample_probs=args.include_sample_probs,
    )

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(checkpoint_path))
    csv_path, json_path = _write_outputs(result, output_dir, args.include_sample_probs)

    print(f"[ConceptEval] checkpoint: {checkpoint_path}")
    print(f"[ConceptEval] samples: {result['num_samples']} | concepts: {result['num_concepts']}")
    print(
        f"[ConceptEval] mean_acc={result['mean_accuracy']:.4f} | "
        f"mean_f1={result['mean_f1']:.4f} | micro_f1={result['micro_f1']:.4f}"
    )
    print(f"[ConceptEval] CSV:  {csv_path}")
    print(f"[ConceptEval] JSON: {json_path}")


if __name__ == "__main__":
    main()

"""Train/evaluate the V3 HCG-NS-SkateFormer-RF concept bottleneck.

This is the neuro-symbolic training entry point for the relation-free model.
It has no learned concept-relation graph, co-occurrence loss, exclusion loss,
or temporal-order loss. An optional fixed parent-concept ontology graph can
regularize child concepts during training. Temporal information remains inside
each unary concept through the configurable temporal MIL pool and sequence
statistics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from feeders.feeder_ntu_neurosymbolic import Feeder
from model.hcg_ns_model import HCGNSSkateFormer
from model.concept_ontology import load_action_concept_matrix_v2, load_ontology_metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve(root: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, optionally inheriting a local base config."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    if not isinstance(current, dict):
        raise ValueError(f"config must contain a mapping: {config_path}")
    base_reference = current.pop("base_config", None)
    if not base_reference:
        return current
    base_path = (config_path.parent / str(base_reference)).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"base config not found: {base_path}")
    return _deep_merge(load_config(base_path), current)


def device_from_config(cfg: dict[str, Any]) -> torch.device:
    requested = cfg.get("device", [0])
    if isinstance(requested, int):
        requested = [requested]
    if torch.cuda.is_available():
        return torch.device(f"cuda:{requested[0]}")
    return torch.device("cpu")


def make_dataset(root: Path, cfg: dict[str, Any], split: str, train: bool,
                 eval_mode: bool = False) -> Feeder:
    args = dict(cfg["train_feeder_args"] if train else cfg["test_feeder_args"])
    args["data_path"] = resolve(root, args.get("data_path"))
    args["split"] = split
    args["eval_mode"] = eval_mode
    if eval_mode and split == "test":
        args["p_interval"] = cfg.get("test_p_interval", [0.95])
        args["aug_method"] = ""
        args["intra_p"] = 0.0
        args["inter_p"] = 0.0
    return Feeder(**args)


def make_loader(dataset, cfg: dict[str, Any], train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size" if train else "test_batch_size", 32)),
        shuffle=train,
        drop_last=train,
        num_workers=int(cfg.get("num_worker", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def validate_paths(root: Path, cfg: dict[str, Any]) -> None:
    required = {
        "concept_csv": cfg.get("concept_csv"),
        "metadata_csv": cfg.get("metadata_csv"),
        "train_data": cfg.get("train_feeder_args", {}).get("data_path"),
        "test_data": cfg.get("test_feeder_args", {}).get("data_path"),
    }
    if cfg.get("weights"):
        required["weights"] = cfg["weights"]
    missing = []
    for name, value in required.items():
        resolved = resolve(root, value)
        if resolved is None or not Path(resolved).exists():
            missing.append(f"{name}: {resolved}")
    if missing:
        raise FileNotFoundError(
            "HCG-NS preflight failed; missing artifacts:\n  " + "\n  ".join(missing)
        )
    matrix, _, _ = load_action_concept_matrix_v2(
        resolve(root, cfg["concept_csv"]),
        num_classes=int(cfg["num_classes"]),
        num_concepts=int(cfg.get("num_concepts", 78)),
    )
    metadata = load_ontology_metadata(
        resolve(root, cfg["metadata_csv"]),
        num_concepts=int(cfg.get("num_concepts", 78)),
    )
    print(
        "Preflight OK: "
        f"{int(cfg['num_classes'])} actions, {matrix.shape[1]} concepts, "
        f"{len(metadata)} metadata rows, relation-free hierarchy"
    )
def load_checkpoint(model: HCGNSSkateFormer, path: str | None,
                    device: torch.device, max_missing_ratio: float = 0.02) -> None:
    if not path:
        return
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        for state_key in ("model_state", "state_dict", "model"):
            if state_key in payload and isinstance(payload[state_key], dict):
                payload = payload[state_key]
                break
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported checkpoint format: {path}")
    target = model.backbone.state_dict()
    state = {}
    for key, value in payload.items():
        clean = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone.", "model.", "encoder."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    changed = True
                    break
        if clean in target and tuple(target[clean].shape) == tuple(value.shape):
            state[clean] = value
    if not state:
        raise RuntimeError(f"checkpoint has no matching SkateFormer backbone weights: {path}")
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"Loaded backbone checkpoint: {path}")
    print(f"Matched: {len(state)}/{len(target)}; missing: {len(missing)}; unexpected: {len(unexpected)}")
    missing_ratio = len(missing) / max(len(target), 1)
    if missing_ratio > max_missing_ratio:
        raise RuntimeError(
            f"backbone checkpoint missing ratio {missing_ratio:.3f} exceeds "
            f"{max_missing_ratio:.3f}"
        )


def move_batch(batch, device: torch.device):
    if len(batch) == 5:
        data, index_t, label, sample_index, concept_prior = batch
    else:
        data, index_t, label, sample_index = batch
        concept_prior = None
    return (
        data.float().to(device, non_blocking=True),
        index_t.float().to(device, non_blocking=True),
        label.long().to(device, non_blocking=True),
        sample_index,
        concept_prior,
    )


def multilabel_metrics(probabilities: np.ndarray, targets: np.ndarray,
                       threshold: float | np.ndarray = 0.5) -> dict[str, float]:
    threshold_array = np.asarray(threshold, dtype=np.float64)
    if threshold_array.ndim == 1:
        threshold_array = threshold_array.reshape(1, -1)
    pred = probabilities >= threshold_array
    target = targets.astype(bool)
    tp = np.logical_and(pred, target).sum(axis=0).astype(np.float64)
    fp = np.logical_and(pred, ~target).sum(axis=0).astype(np.float64)
    fn = np.logical_and(~pred, target).sum(axis=0).astype(np.float64)
    micro_f1 = (2 * tp.sum()) / max(2 * tp.sum() + fp.sum() + fn.sum(), 1.0)
    per_class = (2 * tp) / np.maximum(2 * tp + fp + fn, 1.0)
    exact = np.all(pred == target, axis=1).mean()
    flat_probabilities = probabilities.reshape(-1)
    flat_targets = targets.astype(np.float64).reshape(-1)
    brier = float(np.mean((flat_probabilities - flat_targets) ** 2))
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
    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(per_class.mean()),
        "exact_match": float(exact),
        "sparsity": float(pred.mean()),
        "brier": brier,
        "ece": float(ece),
    }


@torch.no_grad()
def evaluate(model, loader, device, dump_path: Path | None = None,
             detailed: bool = False) -> dict[str, Any]:
    model.eval()
    scores, labels, concept_probs, parent_probs = [], [], [], []
    parent_for_child = []
    concept_targets, concept_negative_targets, parent_targets = [], [], []
    for batch in loader:
        data, index_t, label, _, _ = move_batch(batch, device)
        output = model(data, index_t)
        scores.append(output["action_scores"].cpu())
        labels.append(label.cpu())
        concept_probs.append(output["sequence_probs"].cpu().numpy())
        parent_probs.append(output["parent_probs"].cpu().numpy())
        parent_for_child.append(output["parent_for_child"].cpu().numpy())
        concept_targets.append(model.symbolic.support[label].cpu().numpy())
        concept_negative_targets.append(model.symbolic.negative_support[label].cpu().numpy())
        parent_targets.append(model.symbolic.parent_support[label].cpu().numpy())
    score_tensor = torch.cat(scores)
    label_tensor = torch.cat(labels)
    top1 = (score_tensor.argmax(dim=1) == label_tensor).float().mean().item()
    topk = score_tensor.topk(min(5, score_tensor.shape[1]), dim=1).indices
    top5 = (topk == label_tensor.unsqueeze(1)).any(dim=1).float().mean().item()
    concept_array = np.concatenate(concept_probs)
    concept_target_array = np.concatenate(concept_targets)
    concept_negative_array = np.concatenate(concept_negative_targets)
    parent_array = np.concatenate(parent_probs)
    parent_target_array = np.concatenate(parent_targets)
    parent_for_child_array = np.concatenate(parent_for_child)
    hierarchy_delta = concept_array - parent_for_child_array
    negative_mask = concept_negative_array.astype(bool)
    negative_probabilities = concept_array[negative_mask]
    concept_metric = multilabel_metrics(
        concept_array,
        concept_target_array,
        threshold=0.5,
    )
    parent_metric = multilabel_metrics(parent_array, parent_target_array, threshold=0.5)
    result: dict[str, Any] = {
        "top1": top1,
        "top5": top5,
        **concept_metric,
        "parent_micro_f1": parent_metric["micro_f1"],
        "parent_macro_f1": parent_metric["macro_f1"],
        "parent_exact_match": parent_metric["exact_match"],
        "parent_sparsity": parent_metric["sparsity"],
        "parent_brier": parent_metric["brier"],
        "parent_ece": parent_metric["ece"],
        "hierarchy_violation_rate": float((hierarchy_delta > 0.0).mean()),
        "hierarchy_violation_mean": float(np.maximum(hierarchy_delta, 0.0).mean()),
        "known_negative_count": int(negative_mask.sum()),
        "known_negative_violation_rate": float(
            (negative_probabilities >= 0.5).mean()
            if negative_probabilities.size else 0.0
        ),
        "known_negative_mean_probability": float(
            negative_probabilities.mean() if negative_probabilities.size else 0.0
        ),
    }
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_path,
            schema_version=np.asarray(6, dtype=np.int64),
            concept_probability_semantics=np.asarray(
                "scope_routed_pose_motion_gru_mil_with_ontology_refiner"
            ),
            parent_role=np.asarray(
                "auxiliary_supervision"
                if model.concept_decoder.parent_enabled else "disabled"
            ),
            concept_probs=concept_array,
            concept_targets=concept_target_array,
            concept_negative_targets=concept_negative_array,
            parent_probs=parent_array,
            parent_targets=parent_target_array,
            parent_for_child=parent_for_child_array,
            action_scores=score_tensor.numpy(),
            action_labels=label_tensor.numpy(),
        )
    if detailed:
        predictions = score_tensor.argmax(dim=1).numpy()
        label_array = label_tensor.numpy()
        num_classes = score_tensor.shape[1]
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        np.add.at(confusion, (label_array, predictions), 1)
        result["per_class_accuracy"] = [
            float(confusion[class_id, class_id] / max(confusion[class_id].sum(), 1))
            for class_id in range(num_classes)
        ]
        result["confusion_matrix"] = confusion.tolist()
    return result


def _gradient_norm(parameters) -> torch.Tensor:
    squared_norms = [
        parameter.grad.detach().float().norm(2).square()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared_norms:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(squared_norms).sum().sqrt()


def train_epoch(model, loader, optimizer, device, grad_clip: float | None,
                loss_weights: dict[str, float] | None = None) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    examples = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        data, index_t, label, _, _ = move_batch(batch, device)
        output = model(data, index_t)
        losses = model.compute_loss(output, label, loss_weights)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()

        base = model.concept_decoder.base
        gradient_diagnostics = {
            "concept_refiner_grad_norm": _gradient_norm(
                model.concept_decoder.concept_refiner.parameters()
            ),
            "parent_head_grad_norm": _gradient_norm(
                model.concept_decoder.parent_heads.parameters()
            ),
            "symbolic_rule_grad_norm": _gradient_norm(model.symbolic.parameters()),
        }
        pose_projector = (
            base.pose_scale_projectors if base.multi_scale else base.pose_projector
        )
        motion_projector = (
            base.motion_scale_projectors if base.multi_scale else base.motion_projector
        )
        centers = getattr(base, "centers", None)
        log_temperature = getattr(base, "log_temperature", None)
        concept_heads = getattr(base, "concept_heads", None)
        gradient_diagnostics.update({
            "center_grad_norm": _gradient_norm(
                [centers] if centers is not None else []
            ),
            "concept_head_grad_norm": _gradient_norm(
                concept_heads.parameters() if concept_heads is not None else []
            ),
            "pose_projector_grad_norm": _gradient_norm(pose_projector.parameters()),
            "motion_projector_grad_norm": _gradient_norm(
                motion_projector.parameters()
            ),
            "temporal_gru_grad_norm": _gradient_norm(
                base.temporal_gru.parameters()
                if base.temporal_gru is not None else []
            ),
            "temperature_grad_abs": _gradient_norm(
                [log_temperature] if log_temperature is not None else []
            ),
            "scale_logits_grad_norm": _gradient_norm(
                [base.scale_logits] if base.multi_scale else []
            ),
            "adaptive_mil_grad_norm": _gradient_norm(
                base.adaptive_mil_pool.parameters()
                if base.adaptive_mil_pool is not None else []
            ),
        })
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if grad_clip is not None:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, grad_clip,
            )
            gradient_diagnostics["gradient_clip_fraction"] = (
                total_grad_norm > grad_clip
            ).to(torch.float32)
        else:
            total_grad_norm = _gradient_norm(trainable_parameters)
            gradient_diagnostics["gradient_clip_fraction"] = torch.zeros(
                (), device=total_grad_norm.device
            )
        gradient_diagnostics["total_grad_norm_before_clip"] = total_grad_norm
        optimizer.step()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        for name, value in gradient_diagnostics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        steps += 1
        examples += int(label.shape[0])
    statistics = {name: value / max(steps, 1) for name, value in totals.items()}
    statistics.update({
        "train_steps": float(steps),
        "train_examples": float(examples),
    })
    if device.type == "cuda":
        statistics.update({
            "cuda_peak_allocated_mb": float(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            ),
            "cuda_peak_reserved_mb": float(
                torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            ),
        })
    return statistics


def replay_symbolic(rulebook: dict[str, Any], concept_probs: np.ndarray) -> np.ndarray:
    dense = rulebook["dense_weights"]
    coefficients = rulebook["coefficients"]
    concept = np.asarray(dense["concept"], dtype=np.float64)
    concept_idf = np.asarray(
        dense.get("concept_idf", np.ones(concept.shape[1])), dtype=np.float64
    )
    if coefficients.get("concept_score_mode", "probability") == "logit":
        clipped_concept = np.clip(concept_probs, 1e-4, 1.0 - 1e-4)
        concept_evidence = np.log(clipped_concept) - np.log1p(-clipped_concept)
    else:
        concept_evidence = concept_probs
    score_weights = concept * concept_idf[None, :]
    concept_score = concept_evidence @ score_weights.T
    concept_score /= np.maximum(score_weights.sum(axis=1, keepdims=True).T, 1e-6)
    action_logit_scale = float(coefficients.get("action_logit_scale", 1.0))
    score = action_logit_scale * concept_score
    if coefficients.get("negative_score_enabled", False):
        negative = np.asarray(dense["negative"], dtype=np.float64)
        margin = float(coefficients.get("negative_score_margin", 0.0))
        temperature = float(coefficients.get("negative_score_temperature", 0.25))
        violation = temperature * np.logaddexp(
            0.0, (concept_evidence - margin) / temperature
        )
        weighted = violation[:, None, :] * negative[None, :, :]
        weighted = np.where(negative[None, :, :] > 0.0, weighted, -np.inf)
        topk = min(int(coefficients.get("negative_score_topk", 5)), negative.shape[1])
        selected_ids = np.argpartition(weighted, -topk, axis=2)[:, :, -topk:]
        selected_values = np.take_along_axis(weighted, selected_ids, axis=2)
        expanded_weights = np.broadcast_to(negative[None, :, :], weighted.shape)
        selected_weights = np.take_along_axis(expanded_weights, selected_ids, axis=2)
        valid = np.isfinite(selected_values)
        penalty = np.where(valid, selected_values, 0.0).sum(axis=2)
        penalty /= np.maximum(
            np.where(valid, selected_weights, 0.0).sum(axis=2), 1e-6
        )
        score -= (
            action_logit_scale
            * float(coefficients.get("negative_score_weight", 0.25))
            * penalty
        )
    return score

def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=False,
            capture_output=True, text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def interpolate_loss_weights(
    target: dict[str, float] | None,
    start: dict[str, float] | None,
    epoch: int,
    warmup_epochs: int,
) -> dict[str, float] | None:
    """Linearly warm selected loss weights while leaving the rest unchanged."""
    if target is None:
        return None
    result = {name: float(value) for name, value in target.items()}
    if not start or warmup_epochs <= 0:
        return result
    unknown = set(start) - set(result)
    if unknown:
        raise ValueError(
            "loss warm-up contains weights absent from its target profile: "
            + ", ".join(sorted(unknown))
        )
    if warmup_epochs == 1:
        progress = 1.0
    else:
        progress = min(max((int(epoch) - 1) / (warmup_epochs - 1), 0.0), 1.0)
    for name, start_value in start.items():
        target_value = result[name]
        result[name] = float(start_value) + progress * (
            target_value - float(start_value)
        )
    return result


def scheduled_learning_rate(
    default: float, schedule: dict[int | str, float] | None, epoch: int
) -> float:
    """Return the latest 1-indexed stage-local learning-rate milestone."""
    learning_rate = float(default)
    for milestone, value in sorted(
        ((int(key), float(item)) for key, item in (schedule or {}).items()),
        key=lambda item: item[0],
    ):
        if milestone <= 0:
            raise ValueError("learning-rate milestones must be positive")
        if int(epoch) >= milestone:
            learning_rate = value
    return learning_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 1)))
    device = device_from_config(cfg)
    validate_paths(root, cfg)
    work_dir = Path(resolve(root, cfg.get("work_dir", "./work_dir/hcg_ns/ntu60_cs")))
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    train_dataset = make_dataset(root, cfg, "train", True, eval_mode=False)
    test_loader = make_loader(make_dataset(root, cfg, "test", False, eval_mode=True), cfg, False)
    train_loader = make_loader(train_dataset, cfg, True)

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
    load_checkpoint(
        model, resolve(root, cfg.get("weights")), device,
        max_missing_ratio=float(cfg.get("checkpoint_max_missing_ratio", 0.02)),
    )
    model.freeze_backbone()
    decoder_lr = float(cfg.get("base_lr", 1e-4))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=decoder_lr,
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    stage1 = int(cfg.get("stage1_epochs", 5))
    stage2 = int(cfg.get("stage2_epochs", 10))
    stage3 = int(cfg.get("stage3_epochs", 0))
    default_loss_weights = cfg.get("loss_weights")
    stage1_loss_weights = cfg.get("stage1_loss_weights", default_loss_weights)
    stage2_loss_weights = cfg.get("stage2_loss_weights", default_loss_weights)
    stage3_loss_weights = cfg.get("stage3_loss_weights", stage2_loss_weights)
    stage1_lr = float(cfg.get("stage1_lr", decoder_lr))
    stage2_lr = float(cfg.get("stage2_lr", decoder_lr))
    stage3_decoder_lr = float(cfg.get("stage3_decoder_lr", stage2_lr))
    stage3_backbone_lr = float(cfg.get("stage3_backbone_lr", decoder_lr * 0.1))
    for group in optimizer.param_groups:
        group["lr"] = stage1_lr
    total_epochs = stage1 + stage2 + stage3
    final_path = work_dir / "final_model.pt"
    history = []
    for epoch in range(total_epochs):
        if epoch < stage1:
            stage_epoch = epoch + 1
            stage_name = str(cfg.get("stage1_name", "parent_grounding"))
            active_loss_weights = interpolate_loss_weights(
                stage1_loss_weights,
                cfg.get("stage1_loss_warmup_start_weights"),
                stage_epoch,
                int(cfg.get("stage1_loss_warmup_epochs", 0)),
            )
            active_lr = scheduled_learning_rate(
                stage1_lr, cfg.get("stage1_lr_schedule"), stage_epoch
            )
            for group in optimizer.param_groups:
                group["lr"] = active_lr
        elif epoch < stage1 + stage2:
            stage_epoch = epoch - stage1 + 1
            stage_name = str(cfg.get("stage2_name", "unary_symbolic_composition"))
            active_loss_weights = interpolate_loss_weights(
                stage2_loss_weights,
                cfg.get("stage2_loss_warmup_start_weights"),
                stage_epoch,
                int(cfg.get("stage2_loss_warmup_epochs", 0)),
            )
            active_lr = scheduled_learning_rate(
                stage2_lr, cfg.get("stage2_lr_schedule"), stage_epoch
            )
            for group in optimizer.param_groups:
                group["lr"] = active_lr
        else:
            stage_name = "selective_backbone_adaptation"
            active_loss_weights = stage3_loss_weights
            active_lr = stage3_decoder_lr
        if epoch == stage1 + stage2 and stage3 > 0:
            model.unfreeze_last_stage()
            existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
            new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in existing]
            for group in optimizer.param_groups:
                group["lr"] = stage3_decoder_lr
            if new_params:
                optimizer.add_param_group({"params": new_params, "lr": stage3_backbone_lr})
        train_stats = train_epoch(
            model, train_loader, optimizer, device,
            float(cfg.get("grad_clip", 1.0)) if cfg.get("grad_clip", True) else None,
            active_loss_weights,
        )
        train_stats.update(model.training_diagnostics())
        row = {
            "epoch": epoch + 1,
            "stage": stage_name,
            "loss_profile": stage_name,
            "loss_weights": {
                name: float(value)
                for name, value in (active_loss_weights or {}).items()
            },
            "learning_rate": active_lr,
            "optimizer_learning_rates": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
            "grad_clip": (
                float(cfg.get("grad_clip", 1.0))
                if cfg.get("grad_clip", True) else None
            ),
            "model": "HCG-NS-SkateFormer-RF",
            "relation_components": (
                ["parent_concept_graph_regularizer"]
                if float((active_loss_weights or {}).get("graph", 0.0)) != 0.0 else []
            ),
            **train_stats,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_stats": train_stats,
            "selection": "final_epoch_no_validation",
        }, final_path)


    (work_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not final_path.exists():
        raise RuntimeError("training finished without producing final_model.pt")
    final_payload = torch.load(final_path, map_location=device)
    model.load_state_dict(final_payload["model_state"], strict=False)
    final_epoch = int(final_payload.get("epoch", -1))
    rulebook = model.symbolic.export_rulebook()
    (work_dir / "final_rules.json").write_text(
        json.dumps(rulebook, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Starting final test evaluation on {len(test_loader.dataset)} samples...")
    final_metrics = evaluate(
        model, test_loader, device,
        dump_path=work_dir / "test_predictions.npz", detailed=True,
    )
    print("Final test evaluation completed.")
    final_metrics.update({
        "final_epoch": final_epoch,
        "checkpoint": str(final_path),
        "evaluation_protocol": "full_train_then_test_once",
        "concept_target_source": "curated_action_concept_annotations",
    })
    (work_dir / "test_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with np.load(work_dir / "test_predictions.npz") as predictions:
        replay = replay_symbolic(rulebook, predictions["concept_probs"])
        reference = predictions["action_scores"].astype(np.float64)
        verification = {
            "format_version": 5,
            "model": "HCG-NS-SkateFormer-RF",
            "relation_components": [],
            "max_abs_error": float(np.max(np.abs(replay - reference))),
            "mean_abs_error": float(np.mean(np.abs(replay - reference))),
            "top1_mismatch_count": int(np.sum(replay.argmax(axis=1) != reference.argmax(axis=1))),
        }
        if not model.residual_action_head_enabled:
            if verification["max_abs_error"] >= 1e-3:
                raise RuntimeError(
                    "symbolic rule replay exceeded tolerance: "
                    f"{verification['max_abs_error']:.6g}"
                )
            if verification["top1_mismatch_count"] != 0:
                raise RuntimeError(
                    "symbolic rule replay changed "
                    f"{verification['top1_mismatch_count']} top-1 predictions"
                )
    (work_dir / "rule_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (work_dir / "symbolic_metrics.json").write_text(
        json.dumps(rulebook["rule_statistics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_path = work_dir / "config_resolved.yaml"
    graph_regularizer_enabled = any(
        float((profile or {}).get("graph", 0.0)) != 0.0
        for profile in (stage1_loss_weights, stage2_loss_weights, stage3_loss_weights)
    )
    training_relation_components = (
        ["parent_concept_graph_regularizer"] if graph_regularizer_enabled else []
    )
    manifest = {
        "model": "HCG-NS-SkateFormer-RF",
        "final_epoch": final_epoch,
        "checkpoint": str(final_path),
        "seed": int(cfg.get("seed", 1)),
        "relation_components_configured": training_relation_components,
        "relation_components_effective": training_relation_components,
        "evaluation": {
            "protocol": "full_train_then_test_once",
            "train_size": len(train_dataset),
            "test_is_not_used_for_checkpoint_selection": True,
        },
        "artifacts": {
            "concept_csv": str(resolve(root, cfg["concept_csv"])),
            "metadata_csv": str(resolve(root, cfg["metadata_csv"])),
        },
        "test_metrics": str(work_dir / "test_metrics.json"),
        "test_predictions": str(work_dir / "test_predictions.npz"),
        "rulebook": str(work_dir / "final_rules.json"),
        "config_resolved": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git_commit": git_commit(root),
    }
    (work_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"test": final_metrics}, ensure_ascii=False))
    print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

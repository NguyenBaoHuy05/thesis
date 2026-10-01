"""Train/evaluate HCG-NS-SkateFormer-RF.

This is the only neuro-symbolic training entry point in the cleaned v2 tree.
It has no relation graph, no co-occurrence loss, no exclusion loss, and no
temporal-order loss. Temporal information remains inside each unary concept
through the prototype decoder's time probabilities and sequence statistics.
"""

from __future__ import annotations

import argparse
import csv
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


def load_hard_negative_pairs(root: Path, cfg: dict[str, Any]) -> list[list[int]]:
    """Load reviewed action pairs and keep the config reproducible.

    The CSV is generated/reviewed from the v2 action-concept signatures.  It
    contains action ids rather than concept relations because the loss is
    applied to action logits.
    """
    configured = [list(pair) for pair in cfg.get("hard_negative_pairs", [])]
    csv_value = cfg.get("hard_negative_pairs_csv")
    if not csv_value:
        return configured
    csv_path = Path(resolve(root, csv_value))
    if not csv_path.exists():
        raise FileNotFoundError(f"hard-negative pairs CSV not found: {csv_path}")
    loaded: list[list[int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                loaded.append([
                    int(row["left_action_id"]),
                    int(row["right_action_id"]),
                ])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid hard-negative row in {csv_path}: {row}"
                ) from exc
    if configured and sorted(map(sorted, configured)) != sorted(map(sorted, loaded)):
        raise ValueError(
            "hard_negative_pairs and hard_negative_pairs_csv disagree; "
            "update both together or leave the inline list empty"
        )
    return loaded


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
        "labels_csv": cfg.get("labels_csv"),
        "hierarchy_csv": cfg.get("hierarchy_csv"),
        "train_data": cfg.get("train_feeder_args", {}).get("data_path"),
        "test_data": cfg.get("test_feeder_args", {}).get("data_path"),
    }
    if cfg.get("hard_negative_pairs_csv"):
        required["hard_negative_pairs_csv"] = cfg["hard_negative_pairs_csv"]
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
    pairs = load_hard_negative_pairs(root, cfg)
    print(f"Hard-negative action pairs: {len(pairs)} (concept-signature derived)")


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
             detailed: bool = False, concept_thresholds: np.ndarray | None = None) -> dict[str, Any]:
    model.eval()
    scores, labels, concept_probs, parent_probs = [], [], [], []
    raw_concept_probs, parent_for_child = [], []
    concept_targets, concept_negative_targets, parent_targets = [], [], []
    for batch in loader:
        data, index_t, label, _, _ = move_batch(batch, device)
        output = model(data, index_t)
        scores.append(output["action_scores"].cpu())
        labels.append(label.cpu())
        concept_probs.append(output["sequence_probs"].cpu().numpy())
        parent_probs.append(output["parent_probs"].cpu().numpy())
        raw_concept_probs.append(output["raw_sequence_probs"].cpu().numpy())
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
    raw_concept_array = np.concatenate(raw_concept_probs)
    parent_for_child_array = np.concatenate(parent_for_child)
    hierarchy_delta = raw_concept_array - parent_for_child_array
    negative_mask = concept_negative_array.astype(bool)
    negative_probabilities = concept_array[negative_mask]
    concept_metric = multilabel_metrics(
        concept_array,
        concept_target_array,
        threshold=0.5 if concept_thresholds is None else concept_thresholds,
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
            concept_probs=concept_array,
            concept_targets=concept_target_array,
            concept_negative_targets=concept_negative_array,
            parent_probs=parent_array,
            parent_targets=parent_target_array,
            raw_concept_probs=raw_concept_array,
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


def train_epoch(model, loader, optimizer, device, grad_clip: float | None,
                loss_weights: dict[str, float] | None = None) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    for batch in loader:
        data, index_t, label, _, _ = move_batch(batch, device)
        output = model(data, index_t)
        losses = model.compute_loss(output, label, loss_weights)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                grad_clip,
            )
        optimizer.step()
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        steps += 1
    return {name: value / max(steps, 1) for name, value in totals.items()}


def replay_symbolic(rulebook: dict[str, Any], concept_probs: np.ndarray,
                    parent_probs: np.ndarray) -> np.ndarray:
    dense = rulebook["dense_weights"]
    coefficients = rulebook["coefficients"]
    concept = np.asarray(dense["concept"], dtype=np.float64)
    parent = np.asarray(dense["parent"], dtype=np.float64)
    bias = np.asarray(dense["bias"], dtype=np.float64)
    concept_idf = np.asarray(
        dense.get("concept_idf", np.ones(concept.shape[1])), dtype=np.float64
    )
    if coefficients.get("concept_score_mode", "probability") == "logit":
        clipped = np.clip(concept_probs, 1e-4, 1.0 - 1e-4)
        evidence = np.log(clipped) - np.log1p(-clipped)
    else:
        evidence = concept_probs
    score_weights = concept * concept_idf[None, :]
    concept_score = evidence @ score_weights.T
    concept_score /= np.maximum(score_weights.sum(axis=1, keepdims=True).T, 1e-6)
    parent_score = parent_probs @ parent.T
    parent_score /= np.maximum(parent.sum(axis=1, keepdims=True).T, 1e-6)
    combined = concept_score + float(coefficients["parent_weight"]) * parent_score
    return float(coefficients.get("action_logit_scale", 1.0)) * combined + bias

def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=False,
            capture_output=True, text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
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

    hard_negative_pairs = load_hard_negative_pairs(root, cfg)
    model = HCGNSSkateFormer(
        backbone_args=cfg["model_args"],
        concept_csv=resolve(root, cfg["concept_csv"]),
        metadata_csv=resolve(root, cfg["metadata_csv"]),
        labels_csv=resolve(root, cfg.get("labels_csv")),
        zero_is_negative=bool(cfg.get("zero_is_negative", False)),
        num_classes=int(cfg["num_classes"]),
        num_concepts=int(cfg.get("num_concepts", 78)),
        hard_negative_pairs=hard_negative_pairs,
        hard_negative_margin=float(cfg.get("hard_negative_margin", 0.2)),
        residual_action_head=bool(cfg.get("residual_action_head", False)),
        residual_action_weight=float(cfg.get("residual_action_weight", 0.1)),
        prototype_args=cfg.get("prototype_args", {}),
        parent_weight=float(cfg.get("parent_weight", 0.35)),
        gate_floor=float(cfg.get("gate_floor", 0.05)),
        action_logit_scale=float(cfg.get("action_logit_scale", 1.0)),
        concept_score_mode=str(cfg.get("concept_score_mode", "probability")),
        concept_idf_power=float(cfg.get("concept_idf_power", 0.0)),
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
            stage_name = "parent_grounding"
            active_loss_weights = stage1_loss_weights
            active_lr = stage1_lr
        elif epoch < stage1 + stage2:
            stage_name = "unary_symbolic_composition"
            active_loss_weights = stage2_loss_weights
            active_lr = stage2_lr
            if epoch == stage1:
                for group in optimizer.param_groups:
                    group["lr"] = stage2_lr
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
        row = {
            "epoch": epoch + 1,
            "stage": stage_name,
            "loss_profile": stage_name,
            "learning_rate": active_lr,
            "model": "HCG-NS-SkateFormer-RF",
            "relation_components": [],
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
    final_metrics = evaluate(
        model, test_loader, device,
        dump_path=work_dir / "test_predictions.npz", detailed=True,
    )
    final_metrics.update({
        "final_epoch": final_epoch,
        "checkpoint": str(final_path),
        "evaluation_protocol": "full_train_then_test_once",
        "concept_target_source": "curated_action_concept_annotations",
        "hard_negative_pairs_source": str(resolve(root, cfg.get("hard_negative_pairs_csv")))
        if cfg.get("hard_negative_pairs_csv") else "inline_config",
        "hard_negative_pairs": [list(pair) for pair in hard_negative_pairs],
    })
    (work_dir / "test_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with np.load(work_dir / "test_predictions.npz") as predictions:
        replay = replay_symbolic(
            rulebook, predictions["concept_probs"], predictions["parent_probs"]
        )
        reference = predictions["action_scores"].astype(np.float64)
        verification = {
            "format_version": 2,
            "model": "HCG-NS-SkateFormer-RF",
            "relation_components": [],
            "max_abs_error": float(np.max(np.abs(replay - reference))),
            "mean_abs_error": float(np.mean(np.abs(replay - reference))),
            "top1_mismatch_count": int(np.sum(replay.argmax(axis=1) != reference.argmax(axis=1))),
        }
    (work_dir / "rule_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (work_dir / "symbolic_metrics.json").write_text(
        json.dumps(rulebook["rule_statistics"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_path = work_dir / "config_resolved.yaml"
    manifest = {
        "model": "HCG-NS-SkateFormer-RF",
        "final_epoch": final_epoch,
        "checkpoint": str(final_path),
        "seed": int(cfg.get("seed", 1)),
        "relation_components_configured": [],
        "relation_components_effective": [],
        "evaluation": {
            "protocol": "full_train_then_test_once",
            "train_size": len(train_dataset),
            "test_is_not_used_for_checkpoint_selection": True,
        },
        "artifacts": {
            "concept_csv": str(resolve(root, cfg["concept_csv"])),
            "metadata_csv": str(resolve(root, cfg["metadata_csv"])),
            "labels_csv": str(resolve(root, cfg.get("labels_csv"))),
            "hierarchy_csv": str(resolve(root, cfg["hierarchy_csv"])),
            "hard_negative_pairs_csv": str(resolve(root, cfg.get("hard_negative_pairs_csv")))
            if cfg.get("hard_negative_pairs_csv") else None,
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

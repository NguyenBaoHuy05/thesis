"""
Unified End-to-End Trainer for GRAFT-SAR Framework
==================================================
Supports:
- End-to-end training from Epoch 1 with Soft Temperature Annealing
- Joint multi-objective loss optimization (Action CE, Concept BCE, CLIP alignment, KG prior, Logic Sparsity)
- Dynamic Learning Rate scheduling with warm-up
- Periodic validation, checkpointing, and rule extraction logging
"""

import argparse
import os
import sys
import time
import yaml
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feeders.feeder_ntu import NTUFeeder, custom_collate_fn
from model.graft_sar import GRAFTSAR


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRAFT-SAR Model")
    parser.add_argument(
        "--config",
        default="config/ntu60_skateformer_logic.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--work-dir", default=None, help="Directory to save checkpoints")
    parser.add_argument("--device", default=None, help="Device to use: cuda, cuda:0, or cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--test-batch-size", type=int, default=None, help="Test batch size override")
    parser.add_argument("--num-epoch", type=int, default=None, help="Epoch count override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--pretrained-backbone", default=None, help="Path to pretrained skeleton backbone (.pt)")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Path to full model checkpoint to resume (.pt)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze skeleton backbone parameters")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with dummy data")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    # Update logic temperature annealing
    model.set_temperature(epoch, total_epochs)

    total_loss = 0.0
    correct_final, correct_neural, correct_logic = 0, 0, 0
    total_samples = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        data, labels, concept_gt, _, index_t, prompts_by_group = batch
        data = data.to(device)
        labels = labels.to(device)
        concept_gt = concept_gt.to(device)
        if index_t is not None:
            index_t = index_t.to(device)

        optimizer.zero_grad()

        outputs = model(
            x=data,
            index_t=index_t,
            labels=labels,
            concept_gt=concept_gt,
            prompts_by_group=prompts_by_group,
        )

        loss = outputs["loss"]
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        bs = labels.size(0)
        total_samples += bs
        total_loss += loss.item() * bs

        pred_final = outputs["y_final"].argmax(dim=-1)
        pred_neural = outputs["y_neural"].argmax(dim=-1)
        pred_logic = outputs["y_logic"].argmax(dim=-1)

        correct_final += (pred_final == labels).sum().item()
        correct_neural += (pred_neural == labels).sum().item()
        correct_logic += (pred_logic == labels).sum().item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(
                f"Epoch [{epoch}/{total_epochs}] Step [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss.item():.4f} | Acc Final: {100.0 * correct_final / total_samples:.2f}% | "
                f"Alpha: {outputs['alpha']:.3f}"
            )

    elapsed = time.time() - start_time
    metrics = {
        "train_loss": total_loss / max(total_samples, 1),
        "train_acc_final": 100.0 * correct_final / max(total_samples, 1),
        "train_acc_neural": 100.0 * correct_neural / max(total_samples, 1),
        "train_acc_logic": 100.0 * correct_logic / max(total_samples, 1),
        "epoch_time": elapsed,
    }
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct_final, correct_neural, correct_logic = 0, 0, 0
    total_samples = 0

    for batch in loader:
        data, labels, concept_gt, _, index_t, prompts_by_group = batch
        data = data.to(device)
        labels = labels.to(device)
        concept_gt = concept_gt.to(device)
        if index_t is not None:
            index_t = index_t.to(device)

        outputs = model(
            x=data,
            index_t=index_t,
            labels=labels,
            concept_gt=concept_gt,
            prompts_by_group=prompts_by_group,
        )

        bs = labels.size(0)
        total_samples += bs
        total_loss += outputs.get("loss", torch.tensor(0.0)).item() * bs

        pred_final = outputs["y_final"].argmax(dim=-1)
        pred_neural = outputs["y_neural"].argmax(dim=-1)
        pred_logic = outputs["y_logic"].argmax(dim=-1)

        correct_final += (pred_final == labels).sum().item()
        correct_neural += (pred_neural == labels).sum().item()
        correct_logic += (pred_logic == labels).sum().item()

    metrics = {
        "val_loss": total_loss / max(total_samples, 1),
        "val_acc_final": 100.0 * correct_final / max(total_samples, 1),
        "val_acc_neural": 100.0 * correct_neural / max(total_samples, 1),
        "val_acc_logic": 100.0 * correct_logic / max(total_samples, 1),
    }
    return metrics


def configure_training_phase(model: nn.Module, phase: str, allow_backbone: bool = True):
    """Set trainable modules for semantic, symbolic, or end-to-end warm-up."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    def enable(module):
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True

    if phase == "semantic":
        if allow_backbone:
            enable(model.backbone)
        enable(model.concept_decoder)
        enable(model.group_projections)
        enable(model.concept_proj_all)
        if model.text_encoder is not None:
            # Keep the pretrained CLIP text encoder frozen; train LoRA only.
            for name, parameter in model.text_encoder.named_parameters():
                parameter.requires_grad = "lora_" in name
    elif phase == "symbolic":
        enable(model.concept_decoder)
        enable(model.group_projections)
        enable(model.concept_proj_all)
        enable(model.logic_engine)
        model.alpha_param.requires_grad = True
    elif phase == "e2e":
        for parameter in model.parameters():
            parameter.requires_grad = True
        if model.text_encoder is not None:
            for name, parameter in model.text_encoder.named_parameters():
                parameter.requires_grad = "lora_" in name
        if not allow_backbone:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False
    else:
        raise ValueError(f"Unknown training phase: {phase}")


def make_phase_optimizer(model: nn.Module, cfg: dict, phase: str) -> torch.optim.Optimizer:
    """Create AdamW with a smaller learning rate for pretrained components."""
    if phase == "semantic":
        head_lr = cfg.get("semantic_head_lr", 1e-4)
        backbone_lr = cfg.get("semantic_backbone_lr", 1e-5)
        clip_lr = cfg.get("semantic_clip_lr", 1e-5)
    elif phase == "symbolic":
        head_lr = cfg.get("symbolic_lr", 1e-4)
        backbone_lr = head_lr
        clip_lr = head_lr
    else:
        head_lr = cfg.get("e2e_head_lr", 1e-4)
        backbone_lr = cfg.get("e2e_backbone_lr", 1e-5)
        clip_lr = cfg.get("e2e_clip_lr", 1e-5)

    groups = {"head": [], "backbone": [], "clip": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            groups["backbone"].append(parameter)
        elif name.startswith("text_encoder."):
            groups["clip"].append(parameter)
        else:
            groups["head"].append(parameter)

    param_groups = []
    if groups["head"]:
        param_groups.append({"params": groups["head"], "lr": head_lr})
    if groups["backbone"]:
        param_groups.append({"params": groups["backbone"], "lr": backbone_lr})
    if groups["clip"]:
        param_groups.append({"params": groups["clip"], "lr": clip_lr})

    if not param_groups:
        raise RuntimeError(f"No trainable parameters configured for phase '{phase}'")
    return torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.get("weight_decay", 0.0005),
    )


def configure_phase_losses(model: nn.Module, phase: str, base_weights: dict):
    """Use phase-specific objectives while preserving configured final weights."""
    weights = dict(base_weights)
    if phase == "semantic":
        weights.update({"final_ce": 0.0, "neural_ce": 0.0, "logic_ce": 0.0, "clip": 1.0})
    elif phase == "symbolic":
        weights.update({"final_ce": 0.0, "neural_ce": 0.0, "logic_ce": 1.0, "clip": 0.0})
    model.loss_weights.update(weights)


def get_phase(epoch: int, semantic_epochs: int, symbolic_epochs: int) -> str:
    if epoch <= semantic_epochs:
        return "semantic"
    if epoch <= semantic_epochs + symbolic_epochs:
        return "symbolic"
    return "e2e"


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Command line overrides
    if args.work_dir:
        cfg["work_dir"] = args.work_dir
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.test_batch_size:
        cfg["test_batch_size"] = args.test_batch_size
    if args.num_epoch:
        cfg["num_epoch"] = args.num_epoch
    if args.lr:
        cfg["base_lr"] = args.lr
    if args.debug:
        cfg["train_feeder_args"]["debug"] = True
        cfg["test_feeder_args"]["debug"] = True

    work_dir = cfg.get("work_dir", "./work_dir/default")
    os.makedirs(work_dir, exist_ok=True)

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Using device: {device}")

    # Dataset & Dataloader
    train_dataset = NTUFeeder(**cfg["train_feeder_args"])
    test_dataset = NTUFeeder(**cfg["test_feeder_args"])

    # Keep the backbone input projection consistent with the actual feeder
    # output.  Some NTU exports contain joint-only data (C=3), while others
    # contain concatenated modalities such as joint+bone (C=6).  The model
    # config cannot safely assume one of these layouts.
    train_channels = int(train_dataset.data.shape[1])
    test_channels = int(test_dataset.data.shape[1])
    if train_channels != test_channels:
        raise ValueError(
            "Train/test channel counts do not match: "
            f"train C={train_channels}, test C={test_channels}."
        )

    backbone_cfg = cfg.setdefault("model_args", {}).setdefault("backbone_cfg", {})
    configured_channels = backbone_cfg.get("in_channels")
    if configured_channels != train_channels:
        print(
            f"[Data] Detected skeleton input with C={train_channels}; "
            f"overriding backbone in_channels={configured_channels}.")
        backbone_cfg["in_channels"] = train_channels

    def skeleton_dimensions(dataset):
        # Feeder output is normally (N, C, T, V, M). Older/single-person
        # exports may be (N, C, T, V), which the adapter treats as M=1.
        data = dataset.data
        points = int(getattr(dataset, "num_points", data.shape[3]))
        return points, int(data.shape[4]) if data.ndim == 5 else 1

    train_points, train_people = skeleton_dimensions(train_dataset)
    test_points, test_people = skeleton_dimensions(test_dataset)
    if (train_points, train_people) != (test_points, test_people):
        raise ValueError(
            "Train/test skeleton layouts do not match: "
            f"train V/M={train_points}/{train_people}, "
            f"test V/M={test_points}/{test_people}."
        )
    for key, detected in (("num_points", train_points), ("num_people", train_people)):
        configured = backbone_cfg.get(key)
        if configured != detected:
            print(
                f"[Data] Detected skeleton {key}={detected}; "
                f"overriding backbone {key}={configured}."
            )
            backbone_cfg[key] = detected

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=cfg.get("num_worker", 0),
        collate_fn=custom_collate_fn,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.get("test_batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_worker", 0),
        collate_fn=custom_collate_fn,
    )

    if args.pretrained_backbone:
        cfg["model_args"]["pretrained_backbone"] = args.pretrained_backbone
    if args.freeze_backbone:
        cfg["model_args"]["freeze_backbone"] = True

    # Model
    model = GRAFTSAR(
        **cfg["model_args"],
        device=device,
    ).to(device)

    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        state_dict = torch.load(args.pretrained_checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[Model] Resumed full model weights from: {args.pretrained_checkpoint}")

    print(f"[Model] Initialized {cfg['model_args']['backbone_name']} with {sum(p.numel() for p in model.parameters()):,} parameters")

    best_val_acc = 0.0
    staged_training = cfg.get("staged_training", True)
    semantic_epochs = cfg.get("semantic_epochs", 5)
    symbolic_epochs = cfg.get("symbolic_epochs", 5)
    if staged_training:
        num_epoch = cfg.get("num_epoch", semantic_epochs + symbolic_epochs + cfg.get("e2e_epochs", 100))
    else:
        num_epoch = cfg.get("num_epoch", 90)

    base_loss_weights = dict(model.loss_weights)
    current_phase = None
    optimizer = None
    scheduler = None

    for epoch in range(1, num_epoch + 1):
        phase = get_phase(epoch, semantic_epochs, symbolic_epochs) if staged_training else "e2e"
        if phase != current_phase:
            allow_backbone = not args.freeze_backbone and not cfg["model_args"].get("freeze_backbone", False)
            configure_training_phase(model, phase, allow_backbone=allow_backbone)
            configure_phase_losses(model, phase, base_loss_weights)
            optimizer = make_phase_optimizer(model, cfg, phase)

            # Apply the LR milestones only to the long end-to-end phase.
            if phase == "e2e":
                scheduler = MultiStepLR(
                    optimizer,
                    milestones=cfg.get("step", [45, 65, 80]),
                    gamma=cfg.get("lr_decay_rate", 0.1),
                )
            else:
                scheduler = None
            current_phase = phase
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[Training] Entering phase '{phase}' with {trainable:,} trainable parameters")

        train_metrics = train_epoch(model, train_loader, optimizer, epoch, num_epoch, device)
        if scheduler is not None:
            scheduler.step()

        print(
            f"=== Epoch {epoch}/{num_epoch} Complete | "
            f"Train Acc Final: {train_metrics['train_acc_final']:.2f}% | "
            f"Neural: {train_metrics['train_acc_neural']:.2f}% | "
            f"Logic: {train_metrics['train_acc_logic']:.2f}% | "
            f"Time: {train_metrics['epoch_time']:.1f}s ==="
        )

        if epoch % cfg.get("eval_interval", 1) == 0:
            val_metrics = evaluate(model, test_loader, device)
            print(
                f"[Validation] Epoch {epoch} -> Val Acc Final: {val_metrics['val_acc_final']:.2f}% | "
                f"Neural: {val_metrics['val_acc_neural']:.2f}% | "
                f"Logic: {val_metrics['val_acc_logic']:.2f}%"
            )

            # Checkpoint saving
            if val_metrics["val_acc_final"] > best_val_acc:
                best_val_acc = val_metrics["val_acc_final"]
                best_path = os.path.join(work_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                print(f"[*] New Best Validation Accuracy: {best_val_acc:.2f}%. Saved to {best_path}")

        if epoch % cfg.get("save_interval", 10) == 0:
            ckpt_path = os.path.join(work_dir, f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n[Training Finished] Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()

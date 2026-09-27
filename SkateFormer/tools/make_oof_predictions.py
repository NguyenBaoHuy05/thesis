"""
make_oof_predictions.py
-----------------------
3-fold Out-Of-Fold (OOF) teacher predictions on the training split.

Each fold:
  - Train decoder on 2/3 of training samples (stratified by action class).
  - Predict concept probabilities for the held-out 1/3 (no augmentation).
  - All folds start from the same pretrained SkateFormer backbone.

Output:
    {out_dir}/oof_probs.npz
        oof_probs       [N_train, num_concepts]  float32
        oof_confidence  [N_train]                float32  (mean prob entropy as uncertainty)
        fold_assignment [N_train]                int32    (0, 1, or 2)
        action_labels   [N_train]                int64
        sample_names    [N_train]                str

Anti-leakage checks:
  - Prediction for each sample comes from a model that was NOT trained on that sample.
  - Test split is never touched.
  - No augmentation during prediction.

Usage:
    python tools/make_oof_predictions.py \\
        --config config/final/ntu60_cs.yaml \\
        --out_dir work_dir/final_pipeline/ntu60_cs/pseudo_labels \\
        --n_folds 3 \\
        --num_epoch 20 \\
        --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.model_selection import StratifiedKFold
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.concept_decoder import masked_bce_loss
from model.concept_model import ConceptModel
from feeders.feeder_ntu_concept import ConceptFeeder, ConceptFeederOOF
from tools.build_template_labels import get_template_matrix


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def init_seed(seed: int):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(
    cfg: dict,
    template_matrix: np.ndarray,
    device: int,
) -> ConceptModel:
    """Build a fresh ConceptModel from config."""
    template_tensor = torch.from_numpy(template_matrix)
    dk = dict(cfg.get("decoder_kwargs", {}))
    dk["mode"] = cfg.get("symbolic_mode", "learned")

    model = ConceptModel(
        backbone_ckpt=cfg.get("backbone_ckpt"),
        num_classes=cfg["num_classes"],
        num_concepts=cfg.get("num_concepts", 74),
        backbone_dim=cfg.get("backbone_dim", 192),
        backbone_kwargs=dict(cfg.get("backbone_kwargs", {})),
        decoder_kwargs=dk,
        freeze_backbone=cfg.get("freeze_backbone", True),
        use_symbolic=cfg.get("use_symbolic", True),
        template_matrix=template_tensor,
    ).cuda(device)
    return model


def train_one_fold(
    model: ConceptModel,
    train_loader: DataLoader,
    n_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_lr: float,
    warm_up_epoch: int,
    weight_decay: float,
    lambda_concept: float,
    lambda_action: float,
    grad_clip: bool,
    grad_max: float,
    device: int,
    early_stop_patience: int = 5,
    fold_id: int = 0,
) -> ConceptModel:
    """Train decoder for one OOF fold."""
    params = model.decoder_parameters()
    optimizer = optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    n_iter = len(train_loader)
    num_steps    = int(n_epochs * n_iter)
    warmup_steps = int(warm_up_epoch * n_iter)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps - warmup_steps,
        lr_min=min_lr,
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    global_step = 0
    best_loss = float("inf")
    no_improve = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        ep_loss, ep_c, ep_a = [], [], []

        for batch in tqdm(train_loader, desc=f"Fold {fold_id} Epoch {epoch + 1}/{n_epochs}", leave=False):
            scheduler.step_update(global_step)
            global_step += 1

            data, index_t, action_label, concept_target, concept_mask, _ = batch
            data           = data.float().cuda(device)
            index_t        = index_t.float().cuda(device)
            action_label   = action_label.long().cuda(device)
            concept_target = concept_target.float().cuda(device)
            concept_mask   = concept_mask.bool().cuda(device)

            concept_logits, action_logits = model(data, index_t)
            loss_c = masked_bce_loss(concept_logits, concept_target, concept_mask)
            loss_a = (
                F.cross_entropy(action_logits, action_label)
                if action_logits is not None else torch.tensor(0.0, device=device)
            )
            loss = lambda_concept * loss_c + lambda_action * loss_a

            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_max)
            optimizer.step()

            ep_loss.append(loss.item())
            ep_c.append(loss_c.item())
            ep_a.append(loss_a.item())

        mean_loss = np.mean(ep_loss)
        print(
            f"  [OOF fold {fold_id}] Epoch {epoch + 1}/{n_epochs} | "
            f"Loss={mean_loss:.4f} C={np.mean(ep_c):.4f} A={np.mean(ep_a):.4f}"
        )

        if mean_loss < best_loss - 1e-4:
            best_loss = mean_loss
            no_improve = 0
            best_state = OrderedDict(
                [(k, v.cpu().clone()) for k, v in model.state_dict().items()]
            )
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"  [OOF fold {fold_id}] Early stop at epoch {epoch + 1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def predict_fold(
    model: ConceptModel,
    val_loader: DataLoader,
    device: int,
) -> tuple:
    """
    Predict concept probabilities for the validation fold.

    Returns:
        probs:          [N_val, num_concepts]
        action_scores:  [N_val, num_classes]
        sample_indices: [N_val]
    """
    model.eval()
    all_probs, all_action, all_indices = [], [], []

    for batch in tqdm(val_loader, desc="Predicting", leave=False):
        data, index_t, _, _, _, sample_idx = batch
        data    = data.float().cuda(device)
        index_t = index_t.float().cuda(device)

        concept_logits, action_logits = model(data, index_t)
        probs = torch.sigmoid(concept_logits).cpu().numpy()
        all_probs.append(probs)
        if action_logits is not None:
            all_action.append(action_logits.cpu().numpy())
        all_indices.append(sample_idx.numpy())

    all_probs   = np.concatenate(all_probs)
    all_action  = np.concatenate(all_action) if all_action else np.array([])
    all_indices = np.concatenate(all_indices)
    return all_probs, all_action, all_indices


# ---------------------------------------------------------------------------
# Main OOF routine
# ---------------------------------------------------------------------------

def make_oof_predictions(
    cfg: dict,
    out_dir: str,
    n_folds: int = 3,
    num_epoch: int = 20,
    batch_size: int = 64,
    num_worker: int = 4,
    seed: int = 1,
    device: int = 0,
    early_stop_patience: int = 5,
):
    init_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load template ----
    template_matrix = get_template_matrix(
        num_classes=cfg["num_classes"],
        csv_path=cfg.get("concept_csv", "./concepts/ntu_rgbd_60_120.csv"),
    )

    # ---- Load full training dataset ----
    train_feeder_args = dict(cfg["train_feeder_args"])
    train_feeder_args["no_augment"] = False   # augment during training

    full_train_ds = ConceptFeeder(**train_feeder_args)
    action_labels = full_train_ds.label       # [N] for stratification
    N = len(full_train_ds)
    print(f"[OOF] Total training samples: {N}")

    # ---- Stratified K-Fold ----
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(N)

    oof_probs       = np.zeros((N, cfg.get("num_concepts", 74)), dtype=np.float32)
    oof_action      = np.zeros((N, cfg["num_classes"]), dtype=np.float32)
    fold_assignment = np.full(N, -1, dtype=np.int32)

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(indices, action_labels)):
        print(f"\n[OOF] === Fold {fold_id + 1}/{n_folds} | train={len(train_idx)} val={len(val_idx)} ===")

        # Build model fresh for each fold
        model = build_model(cfg, template_matrix, device)

        # ---- Training subset ----
        train_pseudo_npz = train_feeder_args.get("pseudo_label_npz")
        val_feeder_args  = dict(train_feeder_args)
        val_feeder_args["no_augment"] = True   # no aug during prediction

        # Subset datasets
        train_fold_ds = ConceptFeederOOF(
            indices=train_idx,
            **{**train_feeder_args, "no_augment": False},
        )
        val_fold_ds = ConceptFeederOOF(
            indices=val_idx,
            **{**val_feeder_args, "no_augment": True},
        )

        train_loader = DataLoader(
            train_fold_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_worker,
            drop_last=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_fold_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_worker,
            drop_last=False,
            pin_memory=True,
        )

        # ---- Train fold ----
        model = train_one_fold(
            model=model,
            train_loader=train_loader,
            n_epochs=num_epoch,
            base_lr=cfg.get("base_lr", 1e-3),
            min_lr=cfg.get("min_lr", 1e-5),
            warmup_lr=cfg.get("warmup_lr", 1e-7),
            warm_up_epoch=cfg.get("warm_up_epoch", 3),
            weight_decay=cfg.get("weight_decay", 0.05),
            lambda_concept=cfg.get("lambda_concept", 1.0),
            lambda_action=cfg.get("lambda_action", 1.0),
            grad_clip=cfg.get("grad_clip", True),
            grad_max=cfg.get("grad_max", 1.0),
            device=device,
            early_stop_patience=early_stop_patience,
            fold_id=fold_id,
        )

        # Save fold checkpoint
        fold_ckpt_dir = os.path.join(out_dir, "..", "oof", f"fold_{fold_id}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        ckpt_state = OrderedDict(
            [(k, v.cpu()) for k, v in model.state_dict().items()]
        )
        torch.save(ckpt_state, os.path.join(fold_ckpt_dir, "model.pt"))

        # ---- Predict on val fold ----
        probs, action_scores, sample_indices = predict_fold(model, val_loader, device)

        # Map back to global indices
        # val_fold_ds.indices are the indices in the full train dataset
        global_val_idx = val_fold_ds.indices
        oof_probs[global_val_idx] = probs
        if action_scores.size > 0:
            oof_action[global_val_idx] = action_scores
        fold_assignment[global_val_idx] = fold_id

        print(f"  Fold {fold_id} done. Mean concept prob: {probs.mean():.4f}")

        # Explicitly free GPU memory
        del model
        torch.cuda.empty_cache()

    # ---- Verify all samples predicted ----
    assert (fold_assignment >= 0).all(), "Some samples were not predicted!"

    # ---- Compute per-sample confidence (1 - entropy) ----
    eps = 1e-7
    entropy = -(
        oof_probs * np.log(oof_probs.clip(eps))
        + (1 - oof_probs) * np.log((1 - oof_probs).clip(eps))
    ).mean(axis=1)   # [N]
    oof_confidence = 1.0 - entropy / np.log(2.0)   # normalise to [0, 1]

    # ---- Save ----
    out_path = os.path.join(out_dir, "oof_probs.npz")
    np.savez_compressed(
        out_path,
        oof_probs=oof_probs,
        oof_action=oof_action,
        oof_confidence=oof_confidence,
        fold_assignment=fold_assignment,
        action_labels=action_labels,
        sample_names=np.array(full_train_ds.sample_name),
    )

    meta = {
        "n_folds": n_folds,
        "n_samples": int(N),
        "num_concepts": int(cfg.get("num_concepts", 74)),
        "num_classes": int(cfg["num_classes"]),
        "num_epoch_per_fold": num_epoch,
        "seed": seed,
        "out_path": out_path,
        "mean_confidence": float(oof_confidence.mean()),
        "mean_concept_prob": float(oof_probs.mean()),
    }
    with open(os.path.join(out_dir, "oof_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OOF] Saved predictions to {out_path}")
    print(f"  Mean confidence: {meta['mean_confidence']:.4f}")
    print(f"  Mean concept prob: {meta['mean_concept_prob']:.4f}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3-fold OOF concept predictions")
    parser.add_argument("--config",   required=True, help="Path to pipeline config YAML")
    parser.add_argument("--out_dir",  required=True, help="Output directory for NPZ files")
    parser.add_argument("--n_folds",  type=int, default=3)
    parser.add_argument("--num_epoch",type=int, default=20)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--num_worker",    type=int, default=4)
    parser.add_argument("--seed",          type=int, default=1)
    parser.add_argument("--device",        type=int, default=0)
    parser.add_argument("--early_stop",    type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    make_oof_predictions(
        cfg=cfg,
        out_dir=args.out_dir,
        n_folds=args.n_folds,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        num_worker=args.num_worker,
        seed=args.seed,
        device=args.device,
        early_stop_patience=args.early_stop,
    )


if __name__ == "__main__":
    main()

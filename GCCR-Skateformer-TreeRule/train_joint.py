"""
Universal Training & Evaluation Script for Joint Spatial-Temporal Symbolic CBM
==============================================================================
Author: Thesis Research Team & Antigravity AI

Supports training and evaluating Joint Symbolic Concept Bottleneck Models
on top of ANY neural skeleton backbone:
  - SkateFormer (Transformer)
  - CTR-GCN (Channel-Topology Refinement Graph Conv)
  - ST-GCN / Baseline (Spatial-Temporal Graph Conv)
  - Hyper-GCN (Hypergraph GCN)
  - SK-GCN (Symbolic-Knowledge GCN)
  - Custom registered backbones

Usage
-----
  # Train with SkateFormer:
  python train_joint.py --backbone skateformer --config config/joint/ntu_cs_skateformer_joint_symbolic.yaml

  # Train with CTR-GCN:
  python train_joint.py --backbone ctrgcn --config config/joint/ntu_cs_ctrgcn_joint_symbolic.yaml

  # Train with ST-GCN (Baseline):
  python train_joint.py --backbone stgcn --config config/joint/ntu_cs_stgcn_joint_symbolic.yaml

  # Generate score pickle file for ensemble:
  python train_joint.py --phase score --checkpoint work_dir/ntu_cs/ctrgcn_symbolic/joint_best.pt --data data/ntu/NTU60_CS.npz

  # Phase 2: resume a frozen-backbone symbolic run and continue end-to-end
  # (backbone unfrozen), picking the schedule up where Phase 1 left off:
  python train_joint.py --config config/ntu/ntu_cs_joint_symbolic_e2e.yaml \
      --resume-checkpoint work_dir/ntu_cs/joint_symbolic/joint_best.pt \
      --train-backbone --start-epoch 31 --epochs 50 --backbone-lr-mult 0.1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import pickle
from typing import Tuple, Dict, Any

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_symbolic_model import (
    build_joint_model,
    JointSpatialTemporalCBM,
    freeze_backbone,
    unfreeze_backbone,
)
from feeders.feeder_ntu import Feeder
from model.backbones import BACKBONE_REGISTRY


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Universal Trainer for Joint Spatial-Temporal Symbolic CBM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="YAML configuration file")
    p.add_argument("--phase", choices=["joint", "score"], default="joint",
                   help="joint = train | score = save .pkl from existing checkpoint")
    p.add_argument("--backbone", default="skateformer", choices=list(BACKBONE_REGISTRY.keys()),
                   help=f"Choice of backbone architecture: {list(BACKBONE_REGISTRY.keys())}")
    p.add_argument("--checkpoint", default=None,
                   help="Pretrained backbone weights OR trained joint_best.pt for score mode")
    p.add_argument("--pretrained-checkpoint", default=None,
                   help="Original pretrained backbone checkpoint (overrides --checkpoint for backbone init)")
    p.add_argument("--train-backbone", action="store_true",
                   help="Unfreeze backbone weights for end-to-end fine-tuning")
    p.add_argument("--resume-checkpoint", default=None,
                   help="Full JointSpatialTemporalCBM checkpoint (e.g. joint_best.pt from a prior "
                        "frozen-backbone symbolic phase) to resume ALL weights from -- backbone, "
                        "concept/group heads, symbolic action module and direct_action -- before "
                        "continuing training (typically combined with --train-backbone and "
                        "--start-epoch to switch into end-to-end fine-tuning).")
    p.add_argument("--start-epoch", type=int, default=1,
                   help="Epoch number to resume the training loop/schedule at (1-indexed, inclusive). "
                        "Use together with --resume-checkpoint so the symbolic/fusion-alpha schedule "
                        "and cosine LR schedule continue smoothly across the phase boundary instead "
                        "of resetting to their start-of-training values.")
    p.add_argument("--backbone-lr-mult", type=float, default=1.0,
                   help="Multiplier applied to --lr for backbone parameters only. Use <1.0 when "
                        "resuming into end-to-end fine-tuning so the already-adapted concept/symbolic "
                        "heads don't get destabilized by full-size backbone gradients.")
    p.add_argument("--data", default=None, help="Path to NTU60_CS.npz or NTU120_CSet.npz")
    p.add_argument("--work-dir", default="work_dir/joint_symbolic")

    # ── Fuzzy AND/OR Logic Tree ──────────────────────────────────────────────
    p.add_argument("--tree-depth",          type=int,   default=4,
                   help="Binary depth of the Fuzzy AND/OR Logic Tree (2^depth leaves)")
    p.add_argument("--tree-entropy-weight", type=float, default=0.01,
                   help="Weight of the entropy regularizer pushing leaf/gate selectors towards "
                        "clean (hardenable) discrete choices")
    p.add_argument("--tree-temp-start",     type=float, default=1.0,
                   help="Softmax/sigmoid temperature at epoch 1 (soft, easy to train)")
    p.add_argument("--tree-temp-end",       type=float, default=0.05,
                   help="Softmax/sigmoid temperature at the final epoch (sharp, close to hard tree)")
    p.add_argument("--tree-kg-neighbor-k",  type=int,   default=8,
                   help="Each class's leaf-selector may only pick concepts present in its own "
                        "ontology row or in its K nearest neighbors by ontology Jaccard similarity")
    p.add_argument("--tree-diversity-weight", type=float, default=0.05,
                   help="Weight penalizing pairwise concept-selection overlap between leaves of "
                        "the same class's tree, discouraging two leaves converging on one concept")
    p.add_argument("--concept-negative-weight", type=float, default=0.10)
    p.add_argument("--concept-checkpoint",    default=None)
    p.add_argument("--group-hidden-dim",      type=int,   default=128,
                   help="Hidden width of each concept-group MLP head (2 hidden layers)")
    p.add_argument("--concept-collapse-weight", type=float, default=0.02,
                   help="Weight of the regularizer penalizing concept heads whose predicted "
                        "probability has near-zero variance across the batch (collapsed heads)")

    # ── Training Hyperparameters ─────────────────────────────────────────────
    p.add_argument("--epochs",            type=int,   default=30)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--batch-size",        type=int,   default=64)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--concept-loss-weight", type=float, default=0.20)

    # ── Data Hyperparameters ─────────────────────────────────────────────────
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--num-class",   type=int, default=60)
    p.add_argument("--num-points",  type=int, default=25)
    p.add_argument("--num-people",  type=int, default=2)
    p.add_argument("--data-type",   choices=["j", "b", "jm", "bm"], default="j")

    # ── Infra ─────────────────────────────────────────────────────────────────
    p.add_argument("--device",           type=int, default=0)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--save-score",       action="store_true")
    p.add_argument("--no-amp",           action="store_true", help="Disable AMP and train in full FP32")
    return p


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class Logger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path

    def __call__(self, msg: str, also_print: bool = True):
        line = time.strftime("[%m.%d %H:%M:%S] ") + msg
        if also_print:
            print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def top1_acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def build_loaders(args) -> Tuple[DataLoader, DataLoader]:
    # SkateFormer uses 24 joints (partition=True drops joint 21).
    # All GCN backbones (CTR-GCN, ST-GCN, Hyper-GCN, SK-GCN) use standard 25 joints (partition=False).
    partition = (args.backbone.lower() == "skateformer")
    # Hyper-GCN's pretrained backbone was trained on joint coordinates centered
    # around joint 21 (spine base) -- see hypergcn/feeders/feeder_ntu.py. The
    # shared feeder must reproduce that or the frozen backbone sees out-of-
    # distribution (uncentered) input.
    center = (args.backbone.lower() == "hypergcn")

    common = dict(
        data_path   = args.data,
        window_size = args.window_size,
        debug       = False,
        thres       = args.window_size,
        uniform     = True,
        partition   = partition,
        center      = center,
        data_type   = args.data_type,
    )
    train_ds = Feeder(
        split      = "train",
        p_interval = [0.5, 1],
        aug_method = "a123489",
        intra_p    = 0.5,
        inter_p    = 0.2,
        **common,
    )
    test_ds = Feeder(
        split      = "test",
        p_interval = [0.95],
        aug_method = "z",
        intra_p    = 0.0,
        inter_p    = 0.0,
        **common,
    )
    kw = dict(num_workers=args.num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True,  **kw),
        DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False, **kw),
    )


def build_model_from_args(args, device: torch.device) -> JointSpatialTemporalCBM:
    """Builds JointSpatialTemporalCBM configured with the selected backbone."""
    pretrained = args.pretrained_checkpoint or args.checkpoint
    backbone_name = args.backbone.lower()

    # Backbone specific arguments
    backbone_cfg = dict(
        num_class   = args.num_class,
        num_point   = args.num_points,
        num_person  = args.num_people,
        in_channels = 3,
    )

    if backbone_name == "skateformer":
        backbone_cfg = dict(
            in_channels  = 3,
            num_people   = args.num_people,
            num_points   = args.num_points if args.num_points != 25 else 24,
            num_frames   = args.window_size,
            kernel_size  = 7,
            num_heads    = 32,
            attn_drop    = 0.0,
            head_drop    = 0.0,
            rel          = True,
            drop_path    = 0.0,
            type_1_size  = (8, 8),
            type_2_size  = (8, 12),
            type_3_size  = (8, 8),
            type_4_size  = (8, 12),
            mlp_ratio    = 4.0,
            index_t      = True,
        )

    model = build_joint_model(
        checkpoint_path         = pretrained,
        backbone_name           = backbone_name,
        backbone_cfg            = backbone_cfg,
        num_class               = args.num_class,
        concept_loss_weight     = args.concept_loss_weight,
        concept_negative_weight = args.concept_negative_weight,
        device                  = device,
        concept_checkpoint_path = args.concept_checkpoint,
        tree_depth              = args.tree_depth,
        tree_entropy_weight     = args.tree_entropy_weight,
        tree_diversity_weight   = args.tree_diversity_weight,
        tree_kg_neighbor_k      = args.tree_kg_neighbor_k,
        group_hidden_dim        = args.group_hidden_dim,
        concept_collapse_weight = args.concept_collapse_weight,
    )

    if args.resume_checkpoint:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"--resume-checkpoint not found: {args.resume_checkpoint}")
        raw = torch.load(args.resume_checkpoint, map_location="cpu")
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[JOINT] Resumed full model state <- {args.resume_checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if missing:
            print(f"[JOINT][WARN] resume missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"[JOINT][WARN] resume unexpected keys (first 5): {unexpected[:5]}")

    return model


def train_epoch(
    model: JointSpatialTemporalCBM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    use_amp: bool,
    train_backbone: bool = False,
) -> Tuple[float, float, float, float, float]:
    model.train()
    if not train_backbone and hasattr(model.backbone, "eval"):
        model.backbone.eval()

    losses, tree_ces, tree_entropies, concepts, collapses, diversities, accs = [], [], [], [], [], [], []

    for batch in loader:
        if len(batch) == 4:
            data, index_t, label, _ = batch
            index_t = index_t.to(device, non_blocking=True)
        else:
            data, label, _ = batch
            index_t = None
        data  = data.float().to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, ld = model(data, index_t=index_t, label=label, return_loss=True)

        if not torch.isfinite(loss).all():
            non_finites = {
                k: round(float(v.item()), 4)
                for k, v in ld.items()
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
            }
            print(f"[JOINT][WARN] Non-finite loss; skipping batch. Bad terms: {non_finites or 'loss sum overflow'}")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            logits = model(data, index_t=index_t)

        losses.append(loss.item())
        tree_ces.append(ld["tree_ce"].item())
        tree_entropies.append(ld["tree_entropy"].item())
        concepts.append(ld["concept"].item())
        collapses.append(ld["concept_collapse"].item())
        diversities.append(ld["leaf_diversity"].item())
        accs.append(top1_acc(logits, label))

    return (
        float(np.mean(losses)), float(np.mean(tree_ces)),
        float(np.mean(tree_entropies)), float(np.mean(concepts)),
        float(np.mean(collapses)), float(np.mean(diversities)), float(np.mean(accs)),
    )


@torch.no_grad()
def evaluate(
    model: JointSpatialTemporalCBM,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float, float]:
    """Returns (val_loss, top1_acc, top5_acc, neural_only_acc)."""
    model.eval()
    losses, logits_all, neural_logits_all, labels_all = [], [], [], []

    for batch in loader:
        if len(batch) == 4:
            data, index_t, label, _ = batch
            index_t = index_t.to(device, non_blocking=True)
        else:
            data, label, _ = batch
            index_t = None
        data  = data.float().to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(data, index_t=index_t)
            loss   = F.cross_entropy(logits, label)
            feat   = model.extract_feat(data, index_t)
            neural_logits = model.direct_action(feat)

        losses.append(loss.item())
        logits_all.append(logits.cpu())
        neural_logits_all.append(neural_logits.cpu())
        labels_all.append(label.cpu())

    all_logits = torch.cat(logits_all)
    all_neural_logits = torch.cat(neural_logits_all)
    all_labels = torch.cat(labels_all)
    acc1 = top1_acc(all_logits, all_labels)
    neural_acc = top1_acc(all_neural_logits, all_labels)

    _, preds_top5 = torch.topk(all_logits, k=min(5, all_logits.shape[1]), dim=1)
    acc5 = (preds_top5 == all_labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return float(np.mean(losses)), acc1, acc5, neural_acc


@torch.no_grad()
def save_scores(
    model: JointSpatialTemporalCBM,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    out_path: str,
):
    """Save predicted action logits as {data_type}_test_score.pkl for ensemble."""
    model.eval()
    all_logits = []
    for batch in loader:
        if len(batch) == 4:
            data, index_t, _, _ = batch
            index_t = index_t.to(device, non_blocking=True)
        else:
            data, _, _ = batch
            index_t = None
        data = data.float().to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(data, index_t=index_t, return_loss=False)
        all_logits.append(out.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    scores_dict = {i: logits[i] for i in range(len(logits))}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(scores_dict, f)
    print(f"[Joint] Saved test scores ({len(scores_dict)} samples) -> {out_path}")


def run_joint(args, device, use_amp, log):
    log("=" * 60)
    log("Joint Spatial-Temporal Concept Bottleneck + Fuzzy AND/OR Logic Tree")
    log(f"  Backbone    : {args.backbone.upper()}")
    log(f"  Pretrained  : {args.pretrained_checkpoint or args.checkpoint or 'None (Scratch)'}")
    log(f"  Data        : {args.data} (Type: {args.data_type})")
    log(f"  Groups      : 6 spatial + 3 temporal = 9")
    log(f"  Tree        : depth={args.tree_depth} | entropy_w={args.tree_entropy_weight} | "
        f"diversity_w={args.tree_diversity_weight} | "
        f"temp {args.tree_temp_start}->{args.tree_temp_end} | KG-neighbor-k={args.tree_kg_neighbor_k}")
    log(f"  Weights     : BCE w={args.concept_loss_weight} | collapse_w={args.concept_collapse_weight}")
    log(f"  ConceptHead : hidden={args.group_hidden_dim} (2 hidden layers)")
    log(f"  Epochs      : {args.start_epoch}->{args.epochs} | lr={args.lr} | wd={args.weight_decay}")
    if args.resume_checkpoint:
        log(f"  Resume from : {args.resume_checkpoint}")
    log("=" * 60)

    model = build_model_from_args(args, device)
    if args.train_backbone:
        unfreeze_backbone(model)
        log("  Backbone    : Trainable (full end-to-end fine-tuning)")
    else:
        freeze_backbone(model)
        log("  Backbone    : Frozen")

    train_loader, test_loader = build_loaders(args)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone")]
    head_params     = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")]
    param_groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        backbone_lr = args.lr * args.backbone_lr_mult
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
        log(f"  LR groups   : heads={args.lr:.2e} | backbone={backbone_lr:.2e} (mult={args.backbone_lr_mult})")
    last_epoch = (args.start_epoch - 2) if args.start_epoch > 1 else -1
    if last_epoch >= 0:
        # Resuming: CosineAnnealingLR requires 'initial_lr' to already be present
        # on each param group when last_epoch >= 0 (it's normally set the first
        # time a scheduler is constructed with last_epoch=-1, but here we build a
        # fresh optimizer and want the schedule to pick up mid-way).
        for group in param_groups:
            group.setdefault("initial_lr", group["lr"])
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01, last_epoch=last_epoch
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_ckpt = os.path.join(args.work_dir, "joint_best.pt")
    latest_ckpt = os.path.join(args.work_dir, "joint_latest.pt")

    best_acc = 0.0
    if args.resume_checkpoint:
        _, best_acc, _, _ = evaluate(model, test_loader, device, use_amp)
        log(f"  Resumed baseline val_acc={best_acc*100:.2f}% (epoch {args.start_epoch - 1})")

    for epoch in range(args.start_epoch, args.epochs + 1):
        t0 = time.time()

        # Anneal the tree's softmax/sigmoid temperature: soft (easy gradients) early,
        # sharp (near-discrete, hardenable) late.
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        temp = args.tree_temp_start + (args.tree_temp_end - args.tree_temp_start) * progress
        model.fuzzy_tree.temperature = temp

        loss, tree_ce, tree_entropy, concept, collapse, diversity, tr_acc = train_epoch(
            model, train_loader, optimizer, device, scaler, use_amp,
            train_backbone=args.train_backbone,
        )
        val_loss, val_acc, val_acc5, val_neural_acc = evaluate(model, test_loader, device, use_amp)
        scheduler.step()

        log(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={loss:.4f}  tree_ce={tree_ce:.4f}  tree_entropy={tree_entropy:.4f}  "
            f"concept={concept:.4f}  collapse={collapse:.4f}  diversity={diversity:.4f}  temp={temp:.3f}  "
            f"tr={tr_acc*100:.2f}%  val={val_acc*100:.2f}% (Top5: {val_acc5*100:.2f}%)  "
            f"neural={val_neural_acc*100:.2f}%  "
            f"({time.time()-t0:.1f}s)"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            log(f"  -> New best {best_acc*100:.2f}% -> {best_ckpt}")

        # Overwritten every epoch so a Kaggle session cutoff never loses more than
        # the epoch in progress -- resume with --resume-checkpoint joint_latest.pt
        # --start-epoch <this epoch + 1>, unlike joint_best.pt which can be many
        # epochs stale whenever val_acc plateaus or dips for a while.
        torch.save(model.state_dict(), latest_ckpt)
        log(f"  -> Latest -> {latest_ckpt} (epoch {epoch})")

    torch.save(model.state_dict(), os.path.join(args.work_dir, "joint_final.pt"))
    log(f"Training completed. Best val_acc={best_acc*100:.2f}%")

    if args.save_score:
        score_path = os.path.join(args.work_dir, f"{args.data_type}_test_score.pkl")
        save_scores(model, test_loader, device, use_amp, score_path)
        log(f"  Scores saved -> {score_path}")


def run_score(args, device, use_amp, log):
    """Load an existing checkpoint and generate score pickle."""
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --phase score")

    log("=" * 60)
    log("Score-only mode — Joint Symbolic CBM")
    log(f"  Backbone   : {args.backbone.upper()}")
    log(f"  Checkpoint : {args.checkpoint}")
    log(f"  Data       : {args.data}")
    log(f"  Data-type  : {args.data_type}")
    log(f"  Work-dir   : {args.work_dir}")
    log("=" * 60)

    model = build_model_from_args(args, device)
    missing, unexpected = model.load_state_dict(
        torch.load(args.checkpoint, map_location=device), strict=False
    )
    log(f"  Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    _, test_loader = build_loaders(args)
    val_loss, val_acc, val_acc5, val_neural_acc = evaluate(model, test_loader, device, use_amp)
    log(f"  Eval: val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  top5={val_acc5*100:.2f}%  neural={val_neural_acc*100:.2f}%")

    score_path = os.path.join(args.work_dir, f"{args.data_type}_test_score.pkl")
    save_scores(model, test_loader, device, use_amp, score_path)
    log(f"  Scores -> {score_path}")


def main():
    parser = get_parser()

    # Pre-parse --config if provided
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", default=None)
    config_args, _ = config_probe.parse_known_args()

    if config_args.config and os.path.isfile(config_args.config):
        with open(config_args.config, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}
            parser.set_defaults(**{k: v for k, v in cfg_dict.items() if not isinstance(v, (dict, list))})

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    os.makedirs(args.work_dir, exist_ok=True)
    log = Logger(os.path.join(args.work_dir, f"log_{args.phase}_{args.backbone}.txt"))
    set_seed(args.seed)

    if args.phase == "joint":
        run_joint(args, device, use_amp, log)
    elif args.phase == "score":
        run_score(args, device, use_amp, log)


if __name__ == "__main__":
    main()

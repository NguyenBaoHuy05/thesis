"""
Evaluate Branches: Neural-only, Symbolic-only, and Fused Accuracies
===================================================================
Compares:
  1. Pure Neural Branch    (alpha = 0.0): Direct action prediction from backbone
  2. Pure Symbolic Branch  (alpha = 1.0): Action prediction via Concept Bottleneck & KG
  3. Coupled Fused Model   (alpha = 0.5): Default fused model
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_symbolic_model import build_joint_model
from feeders.feeder_ntu import Feeder


def get_parser():
    p = argparse.ArgumentParser(description="Evaluate Neural vs Symbolic vs Fused branches")
    p.add_argument("--config", default="config/ntu/skateformer/ntu_cs_b.yaml",
                   help="Path to YAML config file")
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint (defaults to work_dir/joint_best.pt from config)")
    p.add_argument("--data", default=None,
                   help="Path to NTU60_CS.npz (defaults to path in config)")
    p.add_argument("--data-type", default=None, choices=["j", "b", "jm", "bm"],
                   help="Modality type (defaults to config)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=int, default=0)
    return p


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor):
    pred1 = logits.argmax(dim=-1)
    acc1 = (pred1 == targets).float().mean().item() * 100.0

    _, pred5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1)
    acc5 = (pred5 == targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0

    loss = F.cross_entropy(logits, targets).item()
    return acc1, acc5, loss


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load config defaults if provided
    cfg = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    data_path = args.data or cfg.get("data", "../data/ntu/NTU60_CS.npz")
    data_type = args.data_type or cfg.get("data_type", "b")
    checkpoint = args.checkpoint or os.path.join(cfg.get("work_dir", "work_dir/ntu_cs/skateformer_bone_symbolic"), "joint_best.pt")
    num_class = cfg.get("num_class", 60)
    window_size = cfg.get("window_size", 64)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("   GCCR-SkateFormer: Multi-Branch Evaluation (Ablation Study)")
    print("=" * 70)
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Data       : {data_path} (Modality: {data_type.upper()})")
    print(f"  Device     : {device}")
    print("=" * 70)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Build model
    backbone_cfg = dict(
        in_channels=3,
        num_people=cfg.get("num_people", 2),
        num_points=cfg.get("num_points", 24),
        num_frames=window_size,
        kernel_size=7,
        num_heads=32,
        attn_drop=0.0,
        head_drop=0.0,
        rel=True,
        drop_path=0.0,
        type_1_size=(8, 8),
        type_2_size=(8, 12),
        type_3_size=(8, 8),
        type_4_size=(8, 12),
        mlp_ratio=4.0,
        index_t=True,
    )

    model = build_joint_model(
        checkpoint_path=None,
        backbone_cfg=backbone_cfg,
        num_class=num_class,
        device=device,
        refine_steps=cfg.get("refine_steps", 1),
    )

    ckpt_data = torch.load(checkpoint, map_location=device)
    state = ckpt_data.get("state_dict", ckpt_data) if isinstance(ckpt_data, dict) else ckpt_data
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Checkpoint loaded successfully (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    # Build test dataset
    test_ds = Feeder(
        data_path=data_path,
        split="test",
        window_size=window_size,
        p_interval=[0.95],
        aug_method="z",
        intra_p=0.0,
        inter_p=0.0,
        debug=False,
        thres=window_size,
        uniform=True,
        partition=True,
        data_type=data_type,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"  Total test samples: {len(test_ds)}")
    print("  Evaluating branches...")

    neu_logits_list = []
    sym_logits_list = []
    fused_logits_list = []
    labels_list = []

    for batch in test_loader:
        if len(batch) == 4:
            data, index_t, label, _ = batch
            index_t = index_t.to(device, non_blocking=True)
        else:
            data, label, _ = batch
            index_t = None
        data = data.float().to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)

        feat_map = model.extract_feat_map(data, index_t)
        feat = model.pool_feat(feat_map)
        _, group_probs, sym_logits = model.reason(feat_map)
        neu_logits = model.direct_action(feat)
        fused_logits = 0.5 * neu_logits + 0.5 * sym_logits

        neu_logits_list.append(neu_logits.cpu())
        sym_logits_list.append(sym_logits.cpu())
        fused_logits_list.append(fused_logits.cpu())
        labels_list.append(label.cpu())

    all_neu = torch.cat(neu_logits_list, dim=0)
    all_sym = torch.cat(sym_logits_list, dim=0)
    all_fused = torch.cat(fused_logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    neu_acc1, neu_acc5, neu_loss = compute_metrics(all_neu, all_labels)
    sym_acc1, sym_acc5, sym_loss = compute_metrics(all_sym, all_labels)
    fused_acc1, fused_acc5, fused_loss = compute_metrics(all_fused, all_labels)

    print("\n" + "=" * 70)
    print("                      BRANCH ACCURACY RESULTS")
    print("=" * 70)
    header = f"{'Branch':<35} | {'Top-1 Acc':<11} | {'Top-5 Acc':<11} | {'Loss':<8}"
    print(header)
    print("-" * 70)
    print(f"{'1. Pure Neural (SkateFormer)':<35} | {neu_acc1:>9.2f}% | {neu_acc5:>9.2f}% | {neu_loss:>8.4f}")
    print(f"{'2. Pure Symbolic (Concept + KG)':<35} | {sym_acc1:>9.2f}% | {sym_acc5:>9.2f}% | {sym_loss:>8.4f}")
    print(f"{'3. Coupled Fused (50% Neural + 50% Sym)':<35} | {fused_acc1:>9.2f}% | {fused_acc5:>9.2f}% | {fused_loss:>8.4f}")
    print("=" * 70)
    print(f"-> Gain from Symbolic Fusion: {fused_acc1 - neu_acc1:+.2f}% over Pure Neural")
    print("=" * 70)


if __name__ == "__main__":
    main()

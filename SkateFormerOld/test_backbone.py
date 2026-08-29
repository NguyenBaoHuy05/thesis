"""
Universal Backbone Baseline Evaluation Script
=============================================
Author: Antigravity AI & Thesis Research Team

Evaluates the raw pretrained baseline accuracy of ANY skeleton backbone
(without CBM / DCR / Concept heads) on the specified dataset.

Usage
-----
  # Test SkateFormer baseline:
  python test_backbone.py --backbone skateformer --checkpoint pretrained/ntu_cs/SkateFormer_j.pth --data data/ntu/NTU60_CS.npz

  # Test CTR-GCN baseline:
  python test_backbone.py --backbone ctrgcn --checkpoint pretrained/ntu_cs/ctrgcn_j.pt --data data/ntu/NTU60_CS.npz

  # Test ST-GCN baseline:
  python test_backbone.py --backbone stgcn --checkpoint pretrained/ntu_cs/stgcn_j.pt --data data/ntu/NTU60_CS.npz
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.backbones import build_backbone, BACKBONE_REGISTRY
from feeders.feeder_ntu import Feeder


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate raw neural backbone baseline checkpoint")
    p.add_argument("--backbone", choices=list(BACKBONE_REGISTRY.keys()), default="skateformer",
                   help=f"Backbone type: {list(BACKBONE_REGISTRY.keys())}")
    p.add_argument("--checkpoint", required=True, help="Path to pretrained model checkpoint (.pt or .pth)")
    p.add_argument("--data", required=True, help="Path to NTU60_CS.npz or NTU120_CSet.npz")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--num-points", type=int, default=25)
    p.add_argument("--num-people", type=int, default=2)
    p.add_argument("--num-class", type=int, default=60)
    p.add_argument("--data-type", choices=["j", "b", "jm", "bm"], default="j",
                   help="j = joint, b = bone, jm = joint motion, bm = bone motion")
    p.add_argument("--save-score", action="store_true", help="Save scores as a pickle file for ensemble")
    p.add_argument("--work-dir", default="./work_dir/baseline", help="Directory to save test scores")
    return p


def build_loader(args) -> DataLoader:
    partition = (args.backbone.lower() == "skateformer")

    ds = Feeder(
        data_path   = args.data,
        split       = "test",
        p_interval  = [0.95],
        window_size = args.window_size,
        thres       = args.window_size,
        uniform     = True,
        partition   = partition,
        aug_method  = "z",
        intra_p     = 0.0,
        inter_p     = 0.0,
        debug       = False,
        data_type   = args.data_type,
    )
    return DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )


def load_model(args, device: torch.device):
    # 1. Load weights to detect num_classes
    raw = torch.load(args.checkpoint, map_location="cpu")
    ckpt = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    ckpt = OrderedDict((k.replace("module.", ""), v) for k, v in ckpt.items())

    # Detect class count from final linear layer
    num_classes = args.num_class
    for key in ["head.weight", "backbone.head.weight", "fc.weight", "backbone.fc.weight"]:
        if key in ckpt:
            num_classes = ckpt[key].shape[0]
            print(f"[Baseline] Auto-detected num_classes={num_classes} from checkpoint.")
            break

    # 2. Build configuration for backbone adapter
    backbone_name = args.backbone.lower()
    if backbone_name == "skateformer":
        cfg = dict(
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
            num_classes  = num_classes,
        )
    else:
        cfg = dict(
            in_channels = 3,
            num_point   = args.num_points,
            num_person  = args.num_people,
            num_class   = num_classes,
        )
        # Detect embedding_channels for HyperGCN
        for key in ["to_joint_embedding.weight", "backbone.to_joint_embedding.weight", "pos_embedding", "backbone.pos_embedding"]:
            if key in ckpt:
                c_embed = ckpt[key].shape[0] if "weight" in key else ckpt[key].shape[-1]
                cfg["embedding_channels"] = c_embed
                print(f"[Baseline] Auto-detected embedding_channels={c_embed} from checkpoint.")
                break

    adapter = build_backbone(backbone_name, cfg, num_class=num_classes)
    missing, unexpected = adapter.load_pretrained(args.checkpoint, device="cpu")
    if missing:
        print(f"[Baseline][WARN] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[Baseline][WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    print(f"[Baseline] Successfully loaded checkpoint <- {args.checkpoint}")
    return adapter.to(device)


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("=" * 60)
    print(f"Evaluating Raw Baseline: {args.backbone.upper()}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data} (Type: {args.data_type})")
    print(f"  Device     : {device}")
    print("=" * 60)

    model = load_model(args, device)
    model.eval()

    loader = build_loader(args)
    all_logits, all_labels = [], []
    t0 = time.time()

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
            logits = model(data, index_t)

        all_logits.append(logits.cpu())
        all_labels.append(label.cpu())

    elapsed = time.time() - t0
    logits  = torch.cat(all_logits)
    labels  = torch.cat(all_labels)

    preds_top1 = logits.argmax(dim=1)
    acc_top1   = (preds_top1 == labels).float().mean().item()

    _, preds_top5 = torch.topk(logits, k=min(5, logits.shape[1]), dim=1)
    acc_top5      = (preds_top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    print("\n" + "=" * 60)
    print("  Evaluation Results:")
    print("=" * 60)
    print(f"  Total Samples : {len(labels)}")
    print(f"  Time Taken    : {elapsed:.2f} seconds")
    print(f"  Top-1 Accuracy: {acc_top1 * 100:.2f}%")
    print(f"  Top-5 Accuracy: {acc_top5 * 100:.2f}%")
    print("=" * 60 + "\n")

    if args.save_score:
        os.makedirs(args.work_dir, exist_ok=True)
        score_path = os.path.join(args.work_dir, f"{args.backbone}_{args.data_type}_test_score.pkl")
        scores_dict = {i: logits.numpy()[i] for i in range(len(logits))}
        with open(score_path, "wb") as f:
            pickle.dump(scores_dict, f)
        print(f"[Baseline] Saved test scores to: {score_path}")


if __name__ == "__main__":
    main()

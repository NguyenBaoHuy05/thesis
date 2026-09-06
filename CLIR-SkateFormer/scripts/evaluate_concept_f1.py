"""Evaluate concept predictions before CRL refinement.

The script uses ``initial_concept_logits`` (z0), which is produced by the
spatial/temporal concept heads before the CRL reasoner updates the concepts.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Allow running this file directly with ``python scripts/...``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feeders.feeder_ntu import Feeder
from model.clir_skateformer import CLIR_SkateFormer


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    return OrderedDict((k.split("module.")[-1], v) for k, v in checkpoint.items())


def metric_report(targets, logits, names, threshold):
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    preds = (probs >= threshold).astype(np.int64)
    targets = targets.astype(np.int64)

    tp = np.sum((preds == 1) & (targets == 1), axis=0)
    fp = np.sum((preds == 1) & (targets == 0), axis=0)
    fn = np.sum((preds == 0) & (targets == 1), axis=0)
    support = np.sum(targets == 1, axis=0)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    micro_tp, micro_fp, micro_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    micro_precision = micro_tp / max(micro_tp + micro_fp, 1)
    micro_recall = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12)

    weighted_f1 = float(np.average(f1, weights=np.maximum(support, 1)))
    supported = support > 0
    print(f"threshold={threshold:.2f}")
    print(f"bit_accuracy={np.mean(preds == targets) * 100:.2f}%")
    print(f"macro_precision={precision.mean():.4f}")
    print(f"macro_recall={recall.mean():.4f}")
    print(f"macro_f1={f1.mean():.4f}")
    print(f"macro_f1_supported={f1[supported].mean():.4f} ({int(supported.sum())}/{len(supported)} concepts)")
    print(f"micro_f1={micro_f1:.4f}")
    print(f"weighted_f1={weighted_f1:.4f}")

    print("\nLowest concept F1:")
    for idx in np.argsort(f1)[:15]:
        print(f"  {idx:02d} {names[idx]}: F1={f1[idx]:.4f}, P={precision[idx]:.4f}, R={recall[idx]:.4f}, support={support[idx]}")

    focus = {
        "interaction_two_person",
        "temporal_direction_motion_forward",
        "temporal_direction_motion_backward",
        "temporal_direction_motion_upward",
        "temporal_direction_motion_downward",
        "temporal_sequence_seq_hands_first",
        "temporal_sequence_seq_body_first",
        "hand_manipulate",
        "hand_touch",
    }
    print("\nFocus concepts:")
    for idx, name in enumerate(names):
        if name in focus:
            print(f"  {idx:02d} {name}: F1={f1[idx]:.4f}, P={precision[idx]:.4f}, R={recall[idx]:.4f}, support={support[idx]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    feeder_args = dict(cfg["test_feeder_args"])
    dataset = Feeder(**feeder_args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CLIR_SkateFormer(**cfg["model_args"])
    model.load_state_dict(load_state_dict(args.checkpoint), strict=False)
    model.return_aux = True
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_targets = []
    all_z0 = []
    all_zk = []
    with torch.no_grad():
        for data, index_t, _label, _index, concept_targets in loader:
            data = data.float().to(device)
            index_t = index_t.float().to(device)
            output = model(data, index_t=index_t)
            all_z0.append(output["initial_concept_logits"].cpu().numpy())
            all_zk.append(output["concept_logits"].cpu().numpy())
            all_targets.append(concept_targets.numpy())

    targets = np.concatenate(all_targets, axis=0)
    z0 = np.concatenate(all_z0, axis=0)
    zk = np.concatenate(all_zk, axis=0)
    print(f"samples={len(targets)}, concepts={targets.shape[1]}, checkpoint={args.checkpoint}")
    print("\n=== BEFORE CRL: initial concept logits z0 ===")
    metric_report(targets, z0, dataset.concept_names, args.threshold)
    print("\n=== AFTER CRL: final concept logits zK (reference only) ===")
    metric_report(targets, zk, dataset.concept_names, args.threshold)


if __name__ == "__main__":
    main()

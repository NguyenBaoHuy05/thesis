"""
Benchmark Evaluation and Metric Reporting for Thesis Neurosymbolic Framework
============================================================================
Evaluates:
1. Top-1 & Top-5 Classification Accuracy (Final, Neural, Logic)
2. Mean Concept Precision, Recall, and F1-Score
3. Concept Exact Match Ratio (EMR)
4. Residual Fusion Weight Analysis (Alpha Distribution)
"""

import argparse
import os
import sys
import yaml
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feeders.feeder_ntu import NTUFeeder, custom_collate_fn
from model.thesis_model import ThesisNeurosymbolicModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Thesis Neurosymbolic Model")
    parser.add_argument("--config", default="config/ntu60_skateformer_logic.yaml", help="Path to config")
    parser.add_argument("--checkpoint", default=None, help="Path to model weights checkpoint (.pt)")
    parser.add_argument("--device", default=None, help="Device: cuda, cuda:0, or cpu")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--debug", action="store_true", help="Run with debug dummy dataset")
    return parser.parse_args()


def compute_topk_acc(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> Dict[str, float]:
    """Computes Top-K accuracies."""
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res[f"top{k}"] = float(correct_k.mul_(100.0 / batch_size).item())
    return res


def compute_concept_metrics(
    pred_probs: np.ndarray, gt_concepts: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes Concept-Level Micro/Macro Precision, Recall, and F1.
    """
    bin_preds = (pred_probs >= threshold).astype(int)
    bin_gt = (gt_concepts >= 0.5).astype(int)

    tp = np.sum((bin_preds == 1) & (bin_gt == 1))
    fp = np.sum((bin_preds == 1) & (bin_gt == 0))
    fn = np.sum((bin_preds == 0) & (bin_gt == 1))

    micro_precision = tp / max(tp + fp, 1)
    micro_recall = tp / max(tp + fn, 1)
    micro_f1 = (
        2 * (micro_precision * micro_recall) / max(micro_precision + micro_recall, 1e-6)
    )

    # Exact Match Ratio (All concepts in sample match ground truth)
    exact_matches = np.all(bin_preds == bin_gt, axis=-1).mean()

    return {
        "concept_micro_p": round(float(micro_precision) * 100.0, 2),
        "concept_micro_r": round(float(micro_recall) * 100.0, 2),
        "concept_micro_f1": round(float(micro_f1) * 100.0, 2),
        "concept_exact_match": round(float(exact_matches) * 100.0, 2),
    }


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()

    all_y_final = []
    all_y_neural = []
    all_y_logic = []
    all_targets = []
    all_concept_preds = []
    all_concept_gts = []

    print("[Evaluation] Processing test batches...")
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

        all_y_final.append(outputs["y_final"].cpu())
        all_y_neural.append(outputs["y_neural"].cpu())
        all_y_logic.append(outputs["y_logic"].cpu())
        all_targets.append(labels.cpu())
        all_concept_preds.append(outputs["concept_probs"].cpu().numpy())
        all_concept_gts.append(concept_gt.cpu().numpy())

    y_final = torch.cat(all_y_final, dim=0)
    y_neural = torch.cat(all_y_neural, dim=0)
    y_logic = torch.cat(all_y_logic, dim=0)
    targets = torch.cat(all_targets, dim=0)
    concept_preds = np.concatenate(all_concept_preds, axis=0)
    concept_gts = np.concatenate(all_concept_gts, axis=0)

    # Classification Metrics
    metrics_final = compute_topk_acc(y_final, targets, topk=(1, 5))
    metrics_neural = compute_topk_acc(y_neural, targets, topk=(1, 5))
    metrics_logic = compute_topk_acc(y_logic, targets, topk=(1, 5))

    # Concept Metrics
    concept_metrics = compute_concept_metrics(concept_preds, concept_gts)

    print("\n" + "=" * 60)
    print("           BENCHMARK EVALUATION RESULTS           ")
    print("=" * 60)
    print(f" Total Samples Evaluated: {len(targets)}")
    print(f" Residual Fusion Alpha  : {float(model.alpha):.4f}")
    print("-" * 60)
    print(f" [Final Ensemble] Top-1 Acc: {metrics_final['top1']:.2f}% | Top-5 Acc: {metrics_final['top5']:.2f}%")
    print(f" [Neural Branch] Top-1 Acc: {metrics_neural['top1']:.2f}% | Top-5 Acc: {metrics_neural['top5']:.2f}%")
    print(f" [Logic Branch]  Top-1 Acc: {metrics_logic['top1']:.2f}% | Top-5 Acc: {metrics_logic['top5']:.2f}%")
    print("-" * 60)
    print(f" [Concept F1 Score]     : {concept_metrics['concept_micro_f1']:.2f}%")
    print(f" [Concept Precision]    : {concept_metrics['concept_micro_p']:.2f}%")
    print(f" [Concept Recall]       : {concept_metrics['concept_micro_r']:.2f}%")
    print(f" [Exact Match Ratio]    : {concept_metrics['concept_exact_match']:.2f}%")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["test_feeder_args"]["debug"] = True

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = NTUFeeder(**cfg["test_feeder_args"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    model = ThesisNeurosymbolicModel(
        **cfg["model_args"],
        device=device,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Model] Loaded weights from {args.checkpoint}")

    run_evaluation(model, test_loader, device)


if __name__ == "__main__":
    main()

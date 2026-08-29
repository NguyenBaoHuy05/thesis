"""
Symbolic Reasoning & Logic Rule Explanation Script
==================================================
Extracts and visualizes First-Order Logic reasoning chains ($A \\land B \\implies \\text{Action}$)
and produces human-interpretable explanations for individual test instances.
"""

import argparse
import os
import sys
import yaml
import json

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feeders.feeder_ntu import NTUFeeder, custom_collate_fn
from model.thesis_model import ThesisNeurosymbolicModel


def parse_args():
    parser = argparse.ArgumentParser(description="Explain Actions with Symbolic Logic")
    parser.add_argument("--config", default="config/ntu60_skateformer_logic.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of test samples to explain")
    parser.add_argument("--output-json", default=None, help="Optional path to save JSON explanations")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["test_feeder_args"]["debug"] = True

    device = torch.device(args.device)

    # Load dataset
    test_dataset = NTUFeeder(**cfg["test_feeder_args"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.num_samples,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    # Initialize model
    model = ThesisNeurosymbolicModel(
        **cfg["model_args"],
        device=device,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Model] Loaded weights from {args.checkpoint}")

    # Fetch a batch
    batch = next(iter(test_loader))
    data, labels, _, _, index_t, _ = batch
    data = data.to(device)
    if index_t is not None:
        index_t = index_t.to(device)

    print("\n" + "=" * 70)
    print("             EXPLICIT LOGIC RULE EXPLANATIONS             ")
    print("=" * 70)

    # Extract explanations
    explanations = model.explain(data, index_t=index_t, top_k_concepts=6, threshold=0.3)

    for exp in explanations:
        s_idx = exp["sample_idx"]
        gt_label = int(labels[s_idx].item())
        gt_name = (
            model.class_names[gt_label]
            if gt_label < len(model.class_names)
            else f"Class_{gt_label}"
        )
        pred_name = exp["predicted_action"]
        is_correct = "[CORRECT]" if pred_name == gt_name else "[MISMATCH]"

        print(f"\n[Sample #{s_idx + 1}] {is_correct}")
        print(f"  * Ground Truth Action : {gt_name} (ID: {gt_label})")
        print(f"  * Predicted Action    : {pred_name} (ID: {exp['predicted_id']})")
        print(
            f"  * Confidences         : Neural = {exp['neural_confidence']:.2%}, "
            f"Logic = {exp['logic_confidence']:.2%}, Alpha = {exp['fusion_alpha']:.2f}"
        )

        print("  * Active Body-Part Concepts Detected:")
        if exp["active_concepts"]:
            for c in exp["active_concepts"]:
                print(f"      - {c['name']} (activation: {c['prob']:.2f})")
        else:
            print("      (None above threshold)")

        print("  * Extracted Symbolic First-Order Logic Rules:")
        if exp["logic_rules"]:
            for r in exp["logic_rules"]:
                print(f"      => IF ({r['rule']}) THEN Action='{pred_name}' [Weight: {r['weight']}]")
        else:
            print("      => General neural prior activated.")

    print("\n" + "=" * 70 + "\n")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(explanations, f, indent=2)
        print(f"[Explanation] Saved JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()

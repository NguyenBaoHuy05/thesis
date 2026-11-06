"""Script CLI de trich xuat va danh gia luat DNF tu GLIR-HyperGCN da huan luyen.

Ke thua nguyen ven tu CLIR-SkateFormer/scripts/extract_rules.py — chi doi
`CLIR_SkateFormer` -> `GLIR_HyperGCN` khi nap model tu checkpoint; RuleExtractor
va executable DNF engine khong doi (backbone-agnostic).

Su dung:
    python scripts/extract_rules.py \\
        --checkpoint work_dir/ntu60_cs/glir_hypergcn/best_model.pt \\
        --config config/ntu60_cs/glir_hypergcn.yaml \\
        --concept-csv concepts/ntu_rgbd_60_120.csv \\
        --output-dir work_dir/rules \\
        --eval-fidelity
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import GLIR_HyperGCN, RuleExtractor


def load_concept_csv(csv_path: str):
    import csv
    concept_names, action_names = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        concept_names = [h.strip() for h in header[1:] if h.strip()]
        for row in reader:
            if row and row[0].strip():
                action_names.append(row[0].strip())
    if len(concept_names) > 72:
        concept_names = [k for k in concept_names if k not in ("foot_tap", "temporal_sequence_seq_cascading")]
    return concept_names, action_names


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: str = "cpu"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_args = {k: cfg[k] for k in (
        "num_classes", "num_concepts", "spatial_concepts", "temporal_concepts",
        "num_person", "num_point", "in_channels", "embedding_channels",
        "num_loops", "r1", "r2a", "r2b", "beta_max", "beta_init",
        "literal_mode", "temperature", "hidden_dim", "dropout",
        "lambda_sup", "lambda_reg", "lambda_ground",
        "concept_negative_weight", "concept_positive_weight", "asymmetric_gamma_neg",
        "return_aux", "random_feedback", "use_grounding", "clip_backbone",
    ) if k in cfg}
    model = GLIR_HyperGCN(**model_args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    state_dict = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {len(state_dict)} tensors. Missing={len(missing)}, Unexpected={len(unexpected)}")

    model = model.to(device)
    model.eval()
    model.reasoner.literal_mode = "ste"
    print("[RuleExtractor] Switched literal_mode to 'ste' for rule extraction.")
    return model


def get_parser():
    parser = argparse.ArgumentParser(description="Extract DNF rules from trained GLIR-HyperGCN model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--concept-csv", default="concepts/ntu_rgbd_60_120.csv")
    parser.add_argument("--output-dir", default="work_dir/rules")
    parser.add_argument("--tau-and", type=float, default=0.5)
    parser.add_argument("--tau-and2", type=float, default=0.5)
    parser.add_argument("--tau-or", type=float, default=0.5)
    parser.add_argument("--top-k-l2", type=int, default=5)
    parser.add_argument("--top-k-l1", type=int, default=5)
    parser.add_argument("--top-k-literals", type=int, default=6)
    parser.add_argument("--eval-fidelity", action="store_true")
    parser.add_argument("--feeder", default="feeders.feeder_ntu.Feeder")
    parser.add_argument("--feeder-args", default="{}")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    return parser


def main():
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model_from_checkpoint(args.checkpoint, args.config, args.device)
    reasoner = model.reasoner.crl  # CRLReasoner — khong doi so voi CLIR

    concept_names, action_names = [], []
    if os.path.exists(args.concept_csv):
        concept_names, action_names = load_concept_csv(args.concept_csv)
        print(f"Loaded {len(concept_names)} concepts, {len(action_names)} actions from CSV.")
    else:
        print(f"[Warning] CSV not found: {args.concept_csv}. Using generic names.")

    extractor = RuleExtractor(
        reasoner=reasoner,
        concept_names=concept_names if concept_names else None,
        action_names=action_names if action_names else None,
        tau_and=args.tau_and, tau_and2=args.tau_and2, tau_or=args.tau_or,
        top_k_l2=args.top_k_l2, top_k_l1=args.top_k_l1, top_k_literals=args.top_k_literals,
    )

    print("Extracting DNF rules for all actions ...")
    all_rules = extractor.extract_all_rules()

    json_path = os.path.join(args.output_dir, "dnf_rules.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rules, f, ensure_ascii=False, indent=2)
    print(f"Rules saved to: {json_path}")

    txt_path = os.path.join(args.output_dir, "dnf_rules.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for rule in all_rules:
            f.write(RuleExtractor.format_rule_as_dnf(rule))
            f.write("\n" + "-" * 60 + "\n\n")
    print(f"Human-readable rules saved to: {txt_path}")

    print("\n=== Sample Rules (first 5 actions) ===")
    for rule in all_rules[:5]:
        print(RuleExtractor.format_rule_as_dnf(rule))
        print()

    if args.eval_fidelity:
        print("\nEvaluating fidelity ...")
        try:
            feeder_args = eval(f"dict({args.feeder_args})") if args.feeder_args.strip() != "{}" else {}
            mod_str, _, cls_str = args.feeder.rpartition(".")
            import importlib
            Feeder = getattr(importlib.import_module(mod_str), cls_str)

            data_loader = torch.utils.data.DataLoader(
                dataset=Feeder(**feeder_args), batch_size=16, shuffle=False, num_workers=2,
            )
            metrics = extractor.evaluate_fidelity(
                model=model, data_loader=data_loader, device=args.device, n_batches=args.eval_batches,
            )
            print(f"  Rule Coverage:  {metrics['coverage'] * 100:.2f}% ({metrics['covered_samples']}/{metrics['total_samples']})")
            print(f"  Rule Precision: {metrics['precision'] * 100:.2f}%")
            print(f"  Model-Rule Fidelity: {metrics['fidelity'] * 100:.2f}%")

            metrics_path = os.path.join(args.output_dir, "fidelity_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Metrics saved to: {metrics_path}")
        except Exception as e:
            print(f"[Warning] Fidelity evaluation failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""Concept-intervention tests for an HCG-NS-SkateFormer checkpoint.

The tests check whether action decisions actually flow through the concept
bottleneck and whether the learned rules are faithful to the ontology:

1. Score decomposition: how much of the top-1 vs top-2 margin comes from the
   concept term and the residual head.
2. Oracle intervention: replace predicted concepts by the action template
   (upper bound of the symbolic decoder).
3. Top-k intervention curve: correct the k most-wrong known concepts per
   sample (standard concept-bottleneck intervention), with a random-k control.
4. Shuffle test: feed another sample's concept vector; a faithful bottleneck
   should then predict the donor's label, not the original one.
5. Per-concept ablation: force one concept to 0 / 1 and measure the accuracy
   drop on actions that use it versus actions that do not.

Example (run from the SkateFormer repository root)::

    python eval_concept_intervention.py \
        --config config/train/HCG_NS_SkateFormer_ntu120_csub.yaml \
        --checkpoint work_dir/hcg_ns/main_ntu120_csub/final_model.pt

Writes ``intervention_metrics.json`` and ``intervention_per_concept.csv`` next
to the checkpoint unless ``--output-dir`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from eval_hcg_concepts import load_full_checkpoint
from main_hcg_ns import (
    HCGNSSkateFormer,
    device_from_config,
    load_config,
    make_dataset,
    make_loader,
    move_batch,
    resolve,
    set_seed,
    validate_paths,
)


@torch.no_grad()
def collect(model: HCGNSSkateFormer, loader, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    keys = ("sequence_probs", "action_scores", "residual_action_scores")
    buffers: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    labels = []
    for batch in loader:
        data, index_t, label, _, _ = move_batch(batch, device)
        output = model(data, index_t)
        for key in keys:
            if output.get(key) is not None:
                buffers[key].append(output[key].float())
        labels.append(label)
    result = {key: torch.cat(values) for key, values in buffers.items() if values}
    result["labels"] = torch.cat(labels).long()
    return result


class Scorer:
    """Recompute action scores from (possibly intervened) concept evidence."""

    def __init__(self, model: HCGNSSkateFormer, residual: torch.Tensor | None):
        self.symbolic = model.symbolic
        self.residual_term = None
        if residual is not None:
            self.residual_term = model.residual_action_weight * residual

    @torch.no_grad()
    def terms(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        concept_w = self.symbolic._weights()
        concept_evidence = self.symbolic._score_evidence(seq)
        score_w = concept_w * self.symbolic.concept_idf.unsqueeze(0)
        concept = (concept_evidence @ score_w.t()) / score_w.sum(dim=1).clamp_min(1e-6)
        concept = self.symbolic.action_logit_scale * concept
        terms = {"concept": concept}
        if self.residual_term is not None:
            terms["residual"] = self.residual_term
        return terms

    @torch.no_grad()
    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        return sum(self.terms(seq).values())


def accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return float("nan")
    return float((scores.argmax(dim=1) == labels).float().mean().item())


def score_decomposition(scorer: Scorer, seq, labels) -> dict[str, Any]:
    terms = scorer.terms(seq)
    scores = sum(terms.values())
    top2 = scores.topk(2, dim=1).indices
    first, second = top2[:, 0], top2[:, 1]
    rows = torch.arange(scores.shape[0], device=scores.device)
    margin = {
        name: value[rows, first] - value[rows, second] for name, value in terms.items()
    }
    total_abs = sum(value.abs() for value in margin.values()).clamp_min(1e-12)
    # Across-class spread of each term: a term that is nearly constant across
    # classes cannot drive the decision even if its absolute value is large.
    spread = {name: float(value.std(dim=1).mean().item()) for name, value in terms.items()}
    return {
        "margin_share": {
            name: float((value.abs() / total_abs).mean().item())
            for name, value in margin.items()
        },
        "mean_margin": {name: float(value.mean().item()) for name, value in margin.items()},
        "across_class_std": spread,
        "concept_only_top1": accuracy(terms["concept"], labels),
    }


def oracle_tests(scorer: Scorer, seq, labels) -> dict[str, float]:
    symbolic = scorer.symbolic
    concept_template = symbolic.support[labels]
    return {
        "predicted": accuracy(scorer(seq), labels),
        "oracle_concept": accuracy(scorer(concept_template), labels),
        "zero_concept": accuracy(scorer(torch.zeros_like(seq)), labels),
    }


def topk_intervention(
    scorer: Scorer, seq, labels, ks: list[int], generator: torch.Generator,
) -> list[dict[str, float]]:
    symbolic = scorer.symbolic
    target = symbolic.support[labels]
    known = ((symbolic.support + symbolic.negative_support) > 0).float()[labels]
    error = (seq - target).abs() * known
    random_key = torch.rand(seq.shape, generator=generator, device="cpu").to(seq.device)
    random_key = torch.where(known > 0, random_key, torch.full_like(random_key, -1.0))
    curve = []
    for k in ks:
        k_eff = min(int(k), seq.shape[1])
        row = {"k": k_eff}
        for name, key in (("worst", error), ("random", random_key)):
            intervened = seq.clone()
            if k_eff > 0:
                idx = key.topk(k_eff, dim=1).indices
                mask = torch.zeros_like(seq).scatter_(1, idx, 1.0) * known
                intervened = torch.where(mask > 0, target, intervened)
            row[f"{name}_top1"] = accuracy(scorer(intervened), labels)
        curve.append(row)
    return curve


def shuffle_test(scorer: Scorer, seq, labels, generator) -> dict[str, float]:
    perm = torch.randperm(labels.shape[0], generator=generator).to(labels.device)
    scores = scorer(seq[perm])
    return {
        "concept_acc_original_label": accuracy(scores, labels),
        "concept_acc_donor_label": accuracy(scores, labels[perm]),
    }


def per_concept_ablation(scorer: Scorer, seq, labels) -> list[dict[str, Any]]:
    symbolic = scorer.symbolic
    support = symbolic.support > 0
    base_pred = scorer(seq).argmax(dim=1)
    base_correct = base_pred == labels
    rows = []
    for concept_id, name in enumerate(symbolic.concept_names):
        users = support[:, concept_id]
        if not users.any():
            continue
        is_user = users[labels]
        row: dict[str, Any] = {
            "concept_id": concept_id,
            "concept_name": name,
            "num_actions_using": int(users.sum().item()),
            "mean_rule_weight": float(
                symbolic._weights()[users, concept_id].mean().item()
            ),
        }
        for value in (0.0, 1.0):
            intervened = seq.clone()
            intervened[:, concept_id] = value
            pred = scorer(intervened).argmax(dim=1)
            correct = pred == labels
            flipped = pred != base_pred
            tag = "off" if value == 0.0 else "on"
            row[f"{tag}_drop_users"] = float(
                (base_correct[is_user].float().mean() - correct[is_user].float().mean()).item()
            )
            row[f"{tag}_drop_others"] = float(
                (base_correct[~is_user].float().mean() - correct[~is_user].float().mean()).item()
            ) if (~is_user).any() else float("nan")
            row[f"{tag}_flip_rate"] = float(flipped.float().mean().item())
            if value == 1.0:
                # Faithfulness: when forcing a concept on flips a non-user
                # sample, the new prediction should be an action using it.
                moved = flipped & ~is_user
                row["on_flip_to_user_action"] = float(
                    users[pred[moved]].float().mean().item()
                ) if moved.any() else float("nan")
        rows.append(row)
    return rows


def write_outputs(result: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "intervention_metrics.json"
    csv_path = output_dir / "intervention_per_concept.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Concept-intervention tests for HCG-NS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ks", default="0,1,2,3,5,8,12,20,1000")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    for path in (config_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)

    root = Path(__file__).resolve().parent
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 1)))
    device = device_from_config(cfg)
    validate_paths(root, cfg)

    test_dataset = make_dataset(root, cfg, "test", False, eval_mode=True)
    test_loader = make_loader(test_dataset, cfg, False)
    model = HCGNSSkateFormer(
        backbone_args=cfg["model_args"],
        concept_csv=resolve(root, cfg["concept_csv"]),
        metadata_csv=resolve(root, cfg["metadata_csv"]),
        num_classes=int(cfg["num_classes"]),
        num_concepts=int(cfg.get("num_concepts", 78)),
        residual_action_head=bool(cfg.get("residual_action_head", False)),
        residual_action_weight=float(cfg.get("residual_action_weight", 0.1)),
        prototype_args=cfg.get("prototype_args", {}),
        action_logit_scale=float(cfg.get("action_logit_scale", 1.0)),
        concept_score_mode=str(cfg.get("concept_score_mode", "probability")),
        concept_idf_power=float(cfg.get("concept_idf_power", 0.0)),
    ).to(device)
    load_full_checkpoint(model, checkpoint_path, device)
    model.eval()

    data = collect(model, test_loader, device)
    seq, labels = data["sequence_probs"], data["labels"]
    scorer = Scorer(model, data.get("residual_action_scores"))
    recomputed = scorer(seq)
    max_diff = float((recomputed - data["action_scores"]).abs().max().item())
    if max_diff > 1e-3:
        raise RuntimeError(f"recomputed scores differ from model output by {max_diff}")

    generator = torch.Generator().manual_seed(int(args.seed))
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    rows = per_concept_ablation(scorer, seq, labels)
    result = {
        "num_samples": int(labels.shape[0]),
        "recompute_max_abs_diff": max_diff,
        "decomposition": score_decomposition(scorer, seq, labels),
        "oracle": oracle_tests(scorer, seq, labels),
        "topk_intervention": topk_intervention(scorer, seq, labels, ks, generator),
        "shuffle": shuffle_test(scorer, seq, labels, generator),
        "ablation_summary": {
            "concepts_tested": len(rows),
            "mean_off_drop_users": sum(r["off_drop_users"] for r in rows) / len(rows),
            "mean_off_drop_others": sum(r["off_drop_others"] for r in rows) / len(rows),
            "mean_on_flip_to_user_action": sum(
                r["on_flip_to_user_action"] for r in rows
                if r["on_flip_to_user_action"] == r["on_flip_to_user_action"]
            ) / max(1, sum(
                1 for r in rows if r["on_flip_to_user_action"] == r["on_flip_to_user_action"]
            )),
        },
    }
    output_dir = Path(args.output_dir).resolve() if args.output_dir else checkpoint_path.parent
    write_outputs(result, rows, output_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "topk_intervention"}, indent=2))
    for row in result["topk_intervention"]:
        print(row)


if __name__ == "__main__":
    main()

"""
Harden the Fuzzy AND/OR Logic Tree into an explicit Boolean formula
=====================================================================
Loads a trained JointSpatialTemporalCBM checkpoint, snaps every soft
AND/OR gate and concept-selector to its discrete counterpart, exports one
nested Boolean formula per class ("class_c <=> (a AND NOT b) OR c"), and
reports how much accuracy the hardening step costs (soft tree accuracy vs
hardened tree accuracy) plus how "clean" the hardening was (purity/entropy
of the learned selectors before snapping).

Usage
-----
  python harden_tree.py --config config/ntu/skateformer/ntu_cs_j.yaml \
      --checkpoint work_dir/ntu_cs/skateformer_joint_symbolic/joint_best.pt \
      --data <path/to/NTU60_CS.npz> --eval-split test
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_rulebook import get_parser, load_args, build_loader, build_model


@torch.no_grad()
def hardened_vs_soft_accuracy(model, loader, device, use_amp: bool):
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    n_correct_soft = 0
    n_correct_hard = 0
    n_correct_unique = 0
    n_correct_hybrid = 0
    n_unique = 0
    n_none = 0
    n_multi = 0
    n_total = 0

    for batch in loader:
        if len(batch) == 4:
            data, index_t, label, _ = batch
            index_t = index_t.to(device, non_blocking=True)
        else:
            data, label, _ = batch
            index_t = None
        data = data.float().to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            feat = model.extract_feat(data, index_t)
            _, gprobs = model.encode_group_concepts(feat)
            concept_probs = torch.cat([g.float() for g in gprobs], dim=-1)
            soft = model.fuzzy_tree(concept_probs, hard=False)
            hard = model.fuzzy_tree(concept_probs, hard=True)  # strict {0,1} per class

        n_fire = hard.sum(dim=1)
        unique_mask = n_fire == 1
        hard_preds = hard.argmax(1)
        soft_preds = soft.argmax(1)
        hybrid_preds = torch.where(unique_mask, hard_preds, soft_preds)

        n_correct_soft    += (soft_preds == label).sum().item()
        n_correct_hard    += (hard_preds == label).sum().item()
        n_correct_unique  += ((hard_preds == label) & unique_mask).sum().item()
        n_correct_hybrid  += (hybrid_preds == label).sum().item()
        n_unique += unique_mask.sum().item()
        n_none   += (n_fire == 0).sum().item()
        n_multi  += (n_fire > 1).sum().item()
        n_total  += label.numel()

    stats = {
        "soft_acc": n_correct_soft / n_total,
        "hard_acc": n_correct_hard / n_total,
        "hybrid_acc": n_correct_hybrid / n_total,
        "unique_acc": n_correct_unique / max(n_unique, 1),
        "coverage_unique": n_unique / n_total,
        "coverage_none": n_none / n_total,
        "coverage_multi": n_multi / n_total,
    }
    return stats


def main():
    p = get_parser()
    p.add_argument("--out", default=None,
                   help="Where to save the hardened formulas (default: alongside --checkpoint)")
    args = load_args(p)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    use_amp = getattr(args, "use_amp", True) and device.type == "cuda"
    print(f"[Harden] Device: {device}")

    print(f"\n[Step 1] Building model ({args.backbone.upper()}, stream={args.data_type})...")
    model = build_model(args, device)
    model.eval()

    print("\n[Step 2] Hardening report (purity/confidence BEFORE snapping to discrete)...")
    report = model.fuzzy_tree.hardening_report()
    for k, v in report.items():
        print(f"  {k:<30} {v:.4f}")
    if report["mean_leaf_selector_purity"] < 0.5 or report["mean_gate_confidence"] < 0.5:
        print(
            "  [WARN] Low purity/confidence -- hardening will likely lose real information. "
            "Consider raising --tree-entropy-weight or training longer with a lower final "
            "--tree-temp-end, then re-run this script."
        )

    print(f"\n[Step 3] Evaluating soft vs hardened accuracy on '{args.eval_split}' split...")
    eval_loader = build_loader(args, args.eval_split)
    stats = hardened_vs_soft_accuracy(model, eval_loader, device, use_amp)
    soft_acc, hard_acc = stats["soft_acc"], stats["hard_acc"]
    print(f"  Soft   (gradient) accuracy         : {soft_acc*100:.2f}%")
    print(f"  Hard   (Boolean, naive argmax)     : {hard_acc*100:.2f}%")
    print(f"  Hardening gap (soft - hard)        : {(soft_acc-hard_acc)*100:+.2f} pts")
    print(f"  Rule coverage: unique-fire={stats['coverage_unique']*100:.1f}%  "
          f"no-fire={stats['coverage_none']*100:.1f}%  multi-fire={stats['coverage_multi']*100:.1f}%")
    print(f"  Accuracy WHEN exactly 1 rule fires : {stats['unique_acc']*100:.2f}%  "
          f"(true faithfulness on samples the hardened rules can actually decide)")
    print(f"  Hybrid accuracy (rule if unique, else soft-tree fallback) : {stats['hybrid_acc']*100:.2f}%")

    print("\n[Step 4] Exporting Boolean formulas...")
    formulas = model.fuzzy_tree.export_formula(model.concept_names)
    out_path = args.out or os.path.join(os.path.dirname(args.checkpoint) or ".", "rule_tree_hardened.txt")
    lines = [
        "=" * 78,
        " Fuzzy AND/OR Logic Tree -- Hardened Boolean Formulas",
        "=" * 78,
        f"soft_accuracy={soft_acc*100:.2f}%  hard_accuracy={hard_acc*100:.2f}%  "
        f"gap={(soft_acc-hard_acc)*100:+.2f}pts",
        f"coverage: unique-fire={stats['coverage_unique']*100:.1f}%  "
        f"no-fire={stats['coverage_none']*100:.1f}%  multi-fire={stats['coverage_multi']*100:.1f}%  "
        f"| accuracy-when-unique={stats['unique_acc']*100:.2f}%  hybrid_accuracy={stats['hybrid_acc']*100:.2f}%",
        "",
    ] + formulas
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()

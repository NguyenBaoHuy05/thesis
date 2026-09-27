"""
select_audit_samples.py
-----------------------
Select a stratified audit subset (~100-300 samples) for manual annotation.

Sampling strategy (no audit labels used for training):
  1. Teacher-Template disagreement:   teacher strongly disagrees with template positive.
  2. High-confidence new positives:   teacher very confident but template=0.
  3. Low-confidence samples:          teacher uncertain (high entropy).
  4. Random control samples:          randomly sampled for unbiased reference.

Output:
    audit_samples.npz
        audit_indices      [M]        int64  — indices in the training split
        audit_probs        [M, C]     float32
        audit_template     [M, C]     float32
        audit_refined      [M, C]     float32  (if refined_npz provided)
        audit_action_labels[M]        int64
        audit_sample_names [M]        str
        audit_groups       [M]        str  (which sampling group)
    audit_report.json

Usage:
    python tools/select_audit_samples.py \\
        --oof_npz work_dir/final_pipeline/ntu60_cs/pseudo_labels/oof_probs.npz \\
        --template_npz work_dir/final_pipeline/ntu60_cs/pseudo_labels/template_labels_train.npz \\
        --out_dir work_dir/final_pipeline/ntu60_cs/audit \\
        --n_disagree 75 \\
        --n_new_pos 75 \\
        --n_low_conf 50 \\
        --n_random 50 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def select_audit_samples(
    oof_probs: np.ndarray,         # [N, C]
    oof_confidence: np.ndarray,    # [N]
    template_targets: np.ndarray,  # [N, C]  0/1
    action_labels: np.ndarray,     # [N]
    sample_names: np.ndarray,      # [N]
    n_disagree: int = 75,
    n_new_pos: int = 75,
    n_low_conf: int = 50,
    n_random: int = 50,
    disagree_threshold: float = 0.25,
    new_pos_threshold: float = 0.80,
    low_conf_threshold: float = 0.40,
    seed: int = 42,
    refined_targets: np.ndarray = None,
) -> tuple:
    rng = np.random.RandomState(seed)
    N = len(oof_probs)
    selected = {}

    # ---- Group 1: Teacher-Template Disagreement ----
    # Positions where template=1 but teacher prob very low
    disagree_score = np.zeros(N)
    t_pos_mask = (template_targets == 1.0)      # [N, C]
    disagree_per_sample = (
        t_pos_mask & (oof_probs <= disagree_threshold)
    ).sum(axis=1)   # [N] count of disagreements per sample
    disagree_score = disagree_per_sample.astype(float)

    disagree_idx = np.where(disagree_score > 0)[0]
    if len(disagree_idx) > n_disagree:
        # Prioritize most disagreement
        priority = disagree_score[disagree_idx]
        order = np.argsort(-priority)
        disagree_idx = disagree_idx[order[:n_disagree]]
    selected["disagree"] = set(disagree_idx.tolist())

    # ---- Group 2: High-Confidence New Positives ----
    # Template=0 but teacher very confident positive
    new_pos_score = np.zeros(N)
    t_neg_mask = (template_targets == 0.0)
    new_pos_per_sample = (
        t_neg_mask & (oof_probs >= new_pos_threshold)
    ).sum(axis=1)
    new_pos_score = new_pos_per_sample.astype(float)

    new_pos_idx = np.where(new_pos_score > 0)[0]
    # Exclude already selected
    new_pos_idx = np.array([i for i in new_pos_idx if i not in selected["disagree"]])
    if len(new_pos_idx) > n_new_pos:
        priority = new_pos_score[new_pos_idx]
        order = np.argsort(-priority)
        new_pos_idx = new_pos_idx[order[:n_new_pos]]
    selected["new_pos"] = set(new_pos_idx.tolist())

    # ---- Group 3: Low-Confidence Samples ----
    low_conf_idx = np.where(oof_confidence < low_conf_threshold)[0]
    already = selected["disagree"] | selected["new_pos"]
    low_conf_idx = np.array([i for i in low_conf_idx if i not in already])
    if len(low_conf_idx) > n_low_conf:
        low_conf_idx = rng.choice(low_conf_idx, n_low_conf, replace=False)
    selected["low_conf"] = set(low_conf_idx.tolist())

    # ---- Group 4: Random Control ----
    all_selected = selected["disagree"] | selected["new_pos"] | selected["low_conf"]
    remaining = np.array([i for i in range(N) if i not in all_selected])
    n_rand = min(n_random, len(remaining))
    rand_idx = rng.choice(remaining, n_rand, replace=False)
    selected["random"] = set(rand_idx.tolist())

    # ---- Merge ----
    all_idx = []
    group_labels = []
    for group_name, idx_set in selected.items():
        for i in sorted(idx_set):
            all_idx.append(i)
            group_labels.append(group_name)

    all_idx = np.array(all_idx, dtype=np.int64)
    group_labels = np.array(group_labels)

    print(f"[Audit] Total samples selected: {len(all_idx)}")
    for g in ["disagree", "new_pos", "low_conf", "random"]:
        print(f"  {g}: {(group_labels == g).sum()}")

    return all_idx, group_labels


def save_audit_samples(
    all_idx: np.ndarray,
    group_labels: np.ndarray,
    oof_probs: np.ndarray,
    template_targets: np.ndarray,
    action_labels: np.ndarray,
    sample_names: np.ndarray,
    out_dir: str,
    refined_targets: np.ndarray = None,
):
    os.makedirs(out_dir, exist_ok=True)

    save_dict = {
        "audit_indices":       all_idx,
        "audit_probs":         oof_probs[all_idx],
        "audit_template":      template_targets[all_idx],
        "audit_action_labels": action_labels[all_idx],
        "audit_sample_names":  sample_names[all_idx],
        "audit_groups":        group_labels,
    }
    if refined_targets is not None:
        save_dict["audit_refined"] = refined_targets[all_idx]

    out_path = os.path.join(out_dir, "audit_samples.npz")
    np.savez_compressed(out_path, **save_dict)

    # Report
    report = {
        "n_audit_samples": int(len(all_idx)),
        "groups": {
            g: int((group_labels == g).sum())
            for g in ["disagree", "new_pos", "low_conf", "random"]
        },
        "note": (
            "These samples are selected for manual annotation. "
            "Do NOT use audit labels for training in the first evaluation round."
        ),
    }
    report_path = os.path.join(out_dir, "audit_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[Audit] Saved to {out_path}")
    print(f"[Audit] Report: {report_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Select samples for manual audit")
    parser.add_argument("--oof_npz",      required=True)
    parser.add_argument("--template_npz", required=True)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--refined_npz",  default=None)
    parser.add_argument("--n_disagree",   type=int, default=75)
    parser.add_argument("--n_new_pos",    type=int, default=75)
    parser.add_argument("--n_low_conf",   type=int, default=50)
    parser.add_argument("--n_random",     type=int, default=50)
    parser.add_argument("--disagree_threshold",  type=float, default=0.25)
    parser.add_argument("--new_pos_threshold",   type=float, default=0.80)
    parser.add_argument("--low_conf_threshold",  type=float, default=0.40)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    oof = np.load(args.oof_npz)
    tpl = np.load(args.template_npz)

    oof_probs      = oof["oof_probs"]
    oof_confidence = oof["oof_confidence"]
    template_targets = tpl["concept_targets"]
    action_labels  = tpl["action_labels"]
    sample_names   = tpl.get("sample_names", np.arange(len(action_labels)).astype(str))

    refined_targets = None
    if args.refined_npz:
        refined_targets = np.load(args.refined_npz)["concept_targets"]

    all_idx, group_labels = select_audit_samples(
        oof_probs=oof_probs,
        oof_confidence=oof_confidence,
        template_targets=template_targets,
        action_labels=action_labels,
        sample_names=sample_names,
        n_disagree=args.n_disagree,
        n_new_pos=args.n_new_pos,
        n_low_conf=args.n_low_conf,
        n_random=args.n_random,
        disagree_threshold=args.disagree_threshold,
        new_pos_threshold=args.new_pos_threshold,
        low_conf_threshold=args.low_conf_threshold,
        seed=args.seed,
        refined_targets=refined_targets,
    )

    save_audit_samples(
        all_idx=all_idx,
        group_labels=group_labels,
        oof_probs=oof_probs,
        template_targets=template_targets,
        action_labels=action_labels,
        sample_names=sample_names,
        out_dir=args.out_dir,
        refined_targets=refined_targets,
    )


if __name__ == "__main__":
    main()

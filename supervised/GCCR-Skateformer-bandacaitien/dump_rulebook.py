"""
Symbolic Rule Book Generator
=============================
Thu thập luật symbolic từ TẤT CẢ samples trong validation set,
gom theo từng action class, tổng hợp thành Rule Book tổng hợp.

Với mỗi action, Rule Book ghi lại:
  - Tần suất xuất hiện của mỗi concept dương / âm cho từng group
  - Score trung bình của concept được chọn nhiều nhất
  - Concept "dominant" (xuất hiện nhiều nhất)

Usage
-----
  # Joint stream (best checkpoint):
  python dump_rulebook.py \\
      --config config/ntu/skateformer/ntu_cs_j.yaml \\
      --checkpoint work_dir/ntu_cs/skateformer_joint_symbolic/joint_best.pt \\
      --out work_dir/ntu_cs/skateformer_joint_symbolic/rulebook_joint.txt

  # Bone stream (best checkpoint):
  python dump_rulebook.py \\
      --config config/ntu/skateformer/ntu_cs_b.yaml \\
      --checkpoint work_dir/ntu_cs/skateformer_bone_symbolic/joint_best.pt \\
      --out work_dir/ntu_cs/skateformer_bone_symbolic/rulebook_bone.txt

  # Gioi han so action duoc gom (mac dinh = tat ca 60):
  python dump_rulebook.py ... --n-actions 10
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joint_symbolic_model import JointSpatialTemporalCBM, build_joint_model
from feeders.feeder_ntu import Feeder

# NTU-60 class names
NTU60_NAMES = [
    "drink_water", "eat_meal", "brush_teeth", "brush_hair", "drop",
    "pick_up", "throw", "sit_down", "stand_up", "clapping",
    "reading", "writing", "tear_up_paper", "put_on_jacket", "take_off_jacket",
    "put_on_shoe", "take_off_shoe", "put_on_glasses", "take_off_glasses",
    "put_on_hat", "take_off_hat", "cheer_up", "hand_waving",
    "kicking_something", "reach_into_pocket", "hopping", "jump_up",
    "phone_call", "play_phone_tablet", "type_keyboard", "point_something",
    "taking_selfie", "check_watch", "rub_two_hands", "nod_bow",
    "shake_head", "wipe_face", "salute", "put_palms_together",
    "cross_hands", "sneeze_cough", "staggering", "falling_down",
    "headache", "chest_pain", "back_pain", "neck_pain", "nausea_vomiting",
    "fan_self", "punch_slap", "kicking", "pushing", "pat_on_back",
    "point_finger", "hugging", "giving_object", "touch_pocket",
    "shaking_hands", "walking_towards", "walking_apart",
]


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Symbolic Rule Book Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",      default=None, help="YAML config (same as train_joint.py)")
    p.add_argument("--checkpoint",  default=None, help="Trained joint_best.pt checkpoint")
    p.add_argument("--data",        default=None, help="Path to NTU60_CS.npz")
    p.add_argument("--out",         default=None, help="Output rule book .txt path")
    p.add_argument("--data-type",   choices=["j", "b", "jm", "bm"], default="j")
    p.add_argument("--num-class",   type=int, default=60)
    p.add_argument("--num-points",  type=int, default=24)
    p.add_argument("--num-people",  type=int, default=2)
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device",      type=int, default=0)
    p.add_argument("--use-amp",     action="store_true", default=True,
                   help="Dung Automatic Mixed Precision (AMP) de tang toc tren GPU")
    p.add_argument("--n-actions",   type=int, default=60,
                   help="So action class muon gom (60 = tat ca NTU-60)")
    p.add_argument("--min-samples", type=int, default=1,
                   help="Bo qua action neu co it hon min-samples trong validation")
    p.add_argument("--symbolic-temperature", type=float, default=0.7)
    p.add_argument("--symbolic-margin",      type=float, default=0.15)
    p.add_argument("--concept-loss-weight",  type=float, default=0.20)
    p.add_argument("--concept-negative-weight", type=float, default=0.10)
    p.add_argument("--kg-s-weight",   type=float, default=0.02)
    p.add_argument("--kg-c-weight",   type=float, default=0.01)
    p.add_argument("--fusion-start-alpha", type=float, default=0.50)
    p.add_argument("--refine-steps", type=int, default=1,
                   help="Must match the checkpoint's training-time --refine-steps.")
    return p


def load_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    args = p.parse_args()
    # Load YAML config first, then let CLI override
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        defaults = {}
        for k, v in cfg.items():
            k2 = k.replace("-", "_")
            defaults[k2] = v
        p.set_defaults(**defaults)
        args = p.parse_args()
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Build loader (test split only)
# ─────────────────────────────────────────────────────────────────────────────

def build_test_loader(args) -> DataLoader:
    ds = Feeder(
        data_path   = args.data,
        split       = "test",
        window_size = args.window_size,
        debug       = False,
        thres       = args.window_size,
        uniform     = True,
        partition   = True,
        data_type   = args.data_type,
        p_interval  = [0.95],
        aug_method  = "z",
        intra_p     = 0.0,
        inter_p     = 0.0,
    )
    return DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        drop_last   = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args, device: torch.device) -> JointSpatialTemporalCBM:
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
        checkpoint_path         = None,
        backbone_cfg            = backbone_cfg,
        num_class               = args.num_class,
        concept_loss_weight     = args.concept_loss_weight,
        concept_negative_weight = args.concept_negative_weight,
        symbolic_temperature    = args.symbolic_temperature,
        symbolic_margin         = args.symbolic_margin,
        device                  = device,
        kg_s_weight             = args.kg_s_weight,
        kg_c_weight             = args.kg_c_weight,
        fusion_alpha            = args.fusion_start_alpha,
        refine_steps            = getattr(args, "refine_steps", 1),
    )

    ckpt_path = args.checkpoint
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint khong tim thay: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[RuleBook] Loaded checkpoint <- {ckpt_path}")
    if missing:
        print(f"  [WARN] missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  [WARN] unexpected keys (first 5): {unexpected[:5]}")

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Collect per-sample symbolic rules across the entire validation set
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_all_rules(
    model: JointSpatialTemporalCBM,
    loader: DataLoader,
    device: torch.device,
    n_actions: int,
    use_amp: bool = True,
) -> Dict[int, List[List[dict]]]:
    """
    Chay qua toan bo validation set, thu thap luat symbolic cho moi sample.

    Returns:
        action_rules[cls] = list of per-sample explanation lists
            e.g. action_rules[0] = [
                    [{"group": "head", "positive": "head_tilt_up", ...}, ...],  # sample 0
                    [{"group": "head", "positive": "head_tilt_up", ...}, ...],  # sample 1
                    ...
                 ]
    """
    model.eval()
    action_rules: Dict[int, List[List[dict]]] = defaultdict(list)
    amp_enabled = use_amp and device.type == "cuda"

    total_batches = len(loader)
    for bi, batch in enumerate(loader):
        if (bi + 1) % 20 == 0 or bi == 0:
            print(f"  [RuleBook] Processing batch {bi+1}/{total_batches}...")

        if len(batch) == 4:
            data, index_t, label, _ = batch
            index_t = index_t.to(device, non_blocking=True)
        else:
            data, label, _ = batch
            index_t = None

        data = data.float().to(device, non_blocking=True)

        # Forward: get concept probs for whole batch (with AMP if GPU)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            feat_map = model.extract_feat_map(data, index_t)
            _, group_probs, _ = model.reason(feat_map)

        for i, cls in enumerate(label.tolist()):
            if cls >= n_actions:
                continue
            # Extract explanation for sample i with its TRUE action label
            explanation = model.symbolic_action.explain(
                group_probs, i, cls, model.concept_names
            )
            action_rules[cls].append(explanation)

    return action_rules


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate per-action rules into Rule Book
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_rules(samples: List[List[dict]]) -> List[dict]:
    """
    Nhan danh sach explanation cua nhieu samples cho cung 1 action.
    Tra ve luat tong hop moi group voi thong ke day du.
    """
    if not samples:
        return []

    n = len(samples)
    groups = [row["group"] for row in samples[0]]
    aggregated = []

    for gi, group_name in enumerate(groups):
        pos_counts: Dict[str, int]   = defaultdict(int)
        pos_scores: Dict[str, float] = defaultdict(float)
        neg_counts: Dict[str, int]   = defaultdict(int)

        for sample_rows in samples:
            row = sample_rows[gi]
            pos = row["positive"]
            pos_counts[pos] += 1
            pos_scores[pos] += row["positive_score"]
            neg = row["negative"]
            if neg:
                neg_counts[neg] += 1

        # Dominant positive
        dom_pos = max(pos_counts, key=lambda k: pos_counts[k])
        dom_pos_freq  = pos_counts[dom_pos] / n
        dom_pos_score = pos_scores[dom_pos] / pos_counts[dom_pos]

        # Dominant negative
        if neg_counts:
            dom_neg = max(neg_counts, key=lambda k: neg_counts[k])
            dom_neg_freq = neg_counts[dom_neg] / n
        else:
            dom_neg = None
            dom_neg_freq = 0.0

        diversity = len(pos_counts)

        # Ranked list of all positive concepts
        ranked_pos = sorted(pos_counts.items(), key=lambda x: -x[1])
        ranked_pos_detail = [
            {
                "concept":   c,
                "freq":      pos_counts[c] / n,
                "avg_score": pos_scores[c] / pos_counts[c],
            }
            for c, _ in ranked_pos
        ]

        aggregated.append({
            "group":             group_name,
            "dominant_positive": dom_pos,
            "pos_freq":          dom_pos_freq,
            "avg_pos_score":     dom_pos_score,
            "dominant_negative": dom_neg,
            "neg_freq":          dom_neg_freq,
            "n_samples":         n,
            "diversity":         diversity,
            "all_positives":     ranked_pos_detail,
        })

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Format & Write Rule Book
# ─────────────────────────────────────────────────────────────────────────────

def format_rulebook(
    action_rules: Dict[int, List[List[dict]]],
    n_actions: int,
    min_samples: int,
    stream_tag: str = "",
) -> str:
    sep = "=" * 80
    lines = [
        sep,
        f"  SYMBOLIC RULE BOOK{(' -- ' + stream_tag) if stream_tag else ''}",
        f"  Gom luat tu toan bo validation samples cho tung action",
        sep,
        "",
    ]

    for cls in range(n_actions):
        if cls not in action_rules:
            continue
        samples = action_rules[cls]
        if len(samples) < min_samples:
            continue

        action_name = NTU60_NAMES[cls] if cls < len(NTU60_NAMES) else str(cls)
        agg = aggregate_rules(samples)

        lines.append("-" * 80)
        lines.append(f"Action [{cls:02d}] {action_name}  (n={len(samples)} validation samples)")
        lines.append("-" * 80)

        for row in agg:
            group     = row["group"]
            dom_pos   = row["dominant_positive"]
            pos_freq  = row["pos_freq"]
            pos_score = row["avg_pos_score"]
            dom_neg   = row["dominant_negative"]
            neg_freq  = row["neg_freq"]
            diversity = row["diversity"]

            neg_str = f"NOT({dom_neg})  [{neg_freq*100:.1f}%]" if dom_neg else "none"

            # Main rule line
            lines.append(
                f"  [{group:<20}]  "
                f"+{dom_pos}  [freq={pos_freq*100:.1f}%  score={pos_score:.3f}]"
                f"  -{neg_str}"
            )

            # If diversity > 1, show runner-up concepts
            if diversity > 1 and len(row["all_positives"]) > 1:
                others = row["all_positives"][1:4]  # top 3 runner-up
                details = ", ".join(
                    f"{c['concept']}({c['freq']*100:.0f}%)" for c in others
                )
                lines.append(f"  {'':22}  other: {details}")

        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = get_parser()
    args = load_args(p)

    # Resolve device
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
        use_amp = getattr(args, "use_amp", True)
    else:
        device = torch.device("cpu")
        use_amp = False  # AMP chi co hieu qua tren GPU
        if getattr(args, "use_amp", False):
            print("[RuleBook] WARN: CUDA khong co san, tat AMP, chay tren CPU")
    print(f"[RuleBook] Device: {device}  |  AMP: {use_amp}")

    # Resolve output path
    if args.out is None:
        work_dir = getattr(args, "work_dir", "work_dir/rulebook")
        os.makedirs(work_dir, exist_ok=True)
        args.out = os.path.join(work_dir, "rulebook.txt")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Build
    print(f"[RuleBook] Loading validation data: {args.data}")
    loader = build_test_loader(args)
    print(f"[RuleBook] Validation set: {len(loader.dataset)} samples, {len(loader)} batches")

    print(f"[RuleBook] Building model (SKATEFORMER, stream={args.data_type})")
    model = build_model(args, device)
    model.eval()

    # Collect
    print(f"[RuleBook] Collecting rules for {args.n_actions} action classes...")
    action_rules = collect_all_rules(model, loader, device, args.n_actions, use_amp=use_amp)
    print(f"[RuleBook] Done. Classes found: {sorted(action_rules.keys())}")

    for cls in sorted(action_rules.keys()):
        name = NTU60_NAMES[cls] if cls < len(NTU60_NAMES) else str(cls)
        print(f"  Action [{cls:02d}] {name:<30} : {len(action_rules[cls])} samples")

    # Format & Save
    stream_tag = f"stream={args.data_type.upper()}  backbone=SKATEFORMER"
    text = format_rulebook(
        action_rules,
        n_actions   = args.n_actions,
        min_samples = args.min_samples,
        stream_tag  = stream_tag,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[RuleBook] Rule Book saved -> {args.out}")
    # Preview first 3000 chars
    print(text[:3000])


if __name__ == "__main__":
    main()

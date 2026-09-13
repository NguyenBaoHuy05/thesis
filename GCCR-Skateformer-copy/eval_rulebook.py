"""
KMeans Rule Book Classifier — Pure Symbolic Inference & Evaluation
==================================================================
Xay dung K prototype (cluster centroid) moi action tu concept prob vectors.
Score = max cosine similarity giua sample va bat ky prototype nao cua action.

So sanh:
  1. Neural only   (direct_action head)
  2. Symbolic      (JointSymbolicActionCompatibility)
  3. Fused         (1-alpha)*neural + alpha*symbolic
  4. KMeans-RB     K prototypes per action — pure rule matching

Usage
-----
  python eval_rulebook.py --config config/ntu/skateformer/ntu_cs_j.yaml
      --checkpoint work_dir/ntu_cs/skateformer_joint_symbolic/joint_best.pt
      --data D:\\KLTN_Huy\\thesis\\data\\ntu\\NTU60_CS.npz
      --n-actions 60 --device 0
      --rulebook-split train --eval-split test
      --max-per-class 100
      --kmeans-k 3
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import yaml
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joint_symbolic_model import JointSpatialTemporalCBM, build_joint_model
from model.backbones import BACKBONE_REGISTRY
from feeders.feeder_ntu import Feeder

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
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="KMeans Rule Book Classifier — Pure Symbolic Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",      default=None)
    p.add_argument("--checkpoint",  default=None, help="joint_best.pt")
    p.add_argument("--data",        default=None, help="NTU60_CS.npz")
    p.add_argument("--backbone",    default="skateformer", choices=list(BACKBONE_REGISTRY.keys()))
    p.add_argument("--data-type",   choices=["j", "b", "jm", "bm"], default="j")
    p.add_argument("--num-class",   type=int, default=60)
    p.add_argument("--num-points",  type=int, default=24)
    p.add_argument("--num-people",  type=int, default=2)
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device",      type=int, default=0)
    p.add_argument("--n-actions",   type=int, default=60)
    p.add_argument("--max-per-class", type=int, default=100,
                   help="Max samples/class khi build KMeans (0 = khong gioi han)")
    p.add_argument("--use-amp",     action="store_true", default=True)
    p.add_argument("--rulebook-split", choices=["train", "test"], default="train",
                   help="Split de build KMeans prototypes")
    p.add_argument("--eval-split",  choices=["train", "test"], default="test",
                   help="Split de evaluate")
    p.add_argument("--kmeans-k",    type=int, default=3,
                   help="So clusters K moi action (K=1 = mean, K>1 = capture variation)")
    # Model hyperparams (overridden by yaml)
    p.add_argument("--symbolic-temperature",    type=float, default=0.7)
    p.add_argument("--symbolic-margin",         type=float, default=0.15)
    p.add_argument("--concept-loss-weight",     type=float, default=0.20)
    p.add_argument("--concept-negative-weight", type=float, default=0.10)
    p.add_argument("--kg-s-weight",             type=float, default=0.02)
    p.add_argument("--kg-c-weight",             type=float, default=0.01)
    p.add_argument("--fusion-start-alpha",      type=float, default=0.50)
    p.add_argument("--km-alpha",    type=float, default=0.5,
                   help="Beta: ty le kmeans_rb trong fusion neural+kmeans (0=pure neural, 1=pure kmeans)")
    p.add_argument("--km-alpha-sweep", action="store_true", default=False,
                   help="Tu dong sweep beta tu 0.0 den 1.0 tim best fusion")
    return p


def load_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    args = p.parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        p.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
        args = p.parse_args()
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Build helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_loader(args, split: str) -> DataLoader:
    partition = (args.backbone.lower() == "skateformer")
    ds = Feeder(
        data_path   = args.data,
        split       = split,
        window_size = args.window_size,
        debug       = False,
        thres       = args.window_size,
        uniform     = True,
        partition   = partition,
        data_type   = args.data_type,
        p_interval  = [0.5, 1] if split == "train" else [0.95],
        aug_method  = "z",
        intra_p     = 0.0,
        inter_p     = 0.0,
    )
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers, pin_memory=True,
    )


def build_model(args, device: torch.device) -> JointSpatialTemporalCBM:
    backbone_name = args.backbone.lower()
    if backbone_name == "skateformer":
        backbone_cfg = dict(
            in_channels=3, num_people=args.num_people,
            num_points=args.num_points if args.num_points != 25 else 24,
            num_frames=args.window_size, kernel_size=7, num_heads=32,
            attn_drop=0.0, head_drop=0.0, rel=True, drop_path=0.0,
            type_1_size=(8, 8), type_2_size=(8, 12),
            type_3_size=(8, 8), type_4_size=(8, 12),
            mlp_ratio=4.0, index_t=True,
        )
    else:
        backbone_cfg = dict(
            num_class=args.num_class, num_point=args.num_points,
            num_person=args.num_people, in_channels=3,
        )

    model = build_joint_model(
        checkpoint_path=None, backbone_name=backbone_name,
        backbone_cfg=backbone_cfg, num_class=args.num_class,
        concept_loss_weight=args.concept_loss_weight,
        concept_negative_weight=args.concept_negative_weight,
        symbolic_temperature=args.symbolic_temperature,
        symbolic_margin=args.symbolic_margin, device=device,
        kg_s_weight=args.kg_s_weight, kg_c_weight=args.kg_c_weight,
        fusion_alpha=args.fusion_start_alpha,
    )
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Eval] Loaded <- {args.checkpoint}  (missing={len(missing)}, unexpected={len(unexpected)})")
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Collect concept prob vectors from a data split
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_concept_vectors(
    model: JointSpatialTemporalCBM,
    loader: DataLoader,
    device: torch.device,
    n_actions: int,
    use_amp: bool = True,
    max_per_class: int = 0,
) -> Dict[int, np.ndarray]:
    """
    Chay forward qua loader, thu thap full concept prob vector
    (concat 9 groups) cho tung sample.

    Returns:
        concept_vecs[cls] = ndarray shape (n_samples, total_concepts)
    """
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    cap = max_per_class if max_per_class > 0 else float("inf")
    raw: Dict[int, List[np.ndarray]] = defaultdict(list)
    total = len(loader)

    for bi, batch in enumerate(loader):
        if (bi + 1) % 30 == 0 or bi == 0:
            print(f"  [Collect] batch {bi+1}/{total}", end="\r")

        if len(batch) == 4:
            data, index_t, label, _ = batch
            index_t = index_t.to(device, non_blocking=True)
        else:
            data, label, _ = batch
            index_t = None

        data = data.float().to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            feat = model.extract_feat(data, index_t)
            _, group_probs = model.encode_group_concepts(feat)

        # Concat all group probs -> full concept vector (B, total_concepts)
        vec = torch.cat([p.float() for p in group_probs], dim=-1).cpu().numpy()

        for i, cls in enumerate(label.tolist()):
            if cls >= n_actions or len(raw[cls]) >= cap:
                continue
            raw[cls].append(vec[i])

        # Early stop khi du samples cho tat ca classes
        if max_per_class > 0 and all(len(raw[c]) >= cap for c in range(n_actions)):
            print(f"\n  [Collect] Early stop: du {max_per_class} samples/class")
            break

    print()
    return {cls: np.stack(vecs) for cls, vecs in raw.items() if vecs}


# ─────────────────────────────────────────────────────────────────────────────
# KMeans Rule Book Classifier
# ─────────────────────────────────────────────────────────────────────────────

class KMeansRuleBookClassifier:
    """
    Moi action duoc dai dien boi K cluster centroids (rule prototypes).
    Scoring: score(action_c) = max_k cos_sim(sample, centroid_k_c)
    Predict: argmax_c score(c)

    K=1  -> equivalent to mean prototype (no variation captured)
    K>1  -> captures intra-class variation (different "styles" of same action)
    """

    def __init__(
        self,
        concept_vecs: Dict[int, np.ndarray],
        K: int = 3,
        num_class: int = 60,
    ):
        self.K = K
        self.num_class = num_class
        self.centroids: Dict[int, np.ndarray] = {}  # (K_actual, D) L2-normalized
        self.total_concepts: int = 0

        print(f"  [KMeans] Fitting K={K} prototypes per action...")
        for cls, X in sorted(concept_vecs.items()):
            n, D = X.shape
            if self.total_concepts == 0:
                self.total_concepts = D
            k_actual = min(K, n)

            if k_actual == 1 or n <= k_actual:
                center = X.mean(axis=0, keepdims=True)          # (1, D)
            else:
                km = KMeans(n_clusters=k_actual, random_state=42, n_init="auto")
                km.fit(X)
                center = km.cluster_centers_                     # (K, D)

            # L2-normalize -> cosine similarity = dot product
            norms = np.linalg.norm(center, axis=1, keepdims=True).clip(min=1e-8)
            self.centroids[cls] = (center / norms).astype(np.float32)

        print(f"  [KMeans] Done. {len(self.centroids)} actions, "
              f"{self.total_concepts} concepts, K={K}")

        # Pre-build padded centroid tensor (num_class, K_max, D) — L2-normalized
        # Built once, moved to device on first call -> zero per-batch allocation
        K_max = max(c.shape[0] for c in self.centroids.values())
        C_pad  = np.zeros((num_class, K_max, self.total_concepts), dtype=np.float32)
        C_mask = np.zeros((num_class, K_max), dtype=bool)
        for cls, center in self.centroids.items():
            k = center.shape[0]
            C_pad[cls, :k]  = center
            C_mask[cls, :k] = True
        # Store as numpy; will be moved to GPU once on first score() call
        self._C_pad  = C_pad   # (num_class, K_max, D)
        self._C_mask = C_mask  # (num_class, K_max)
        self._C_device: Optional[str] = None
        self._C_gpu:  Optional[torch.Tensor] = None   # (num_class, K_max, D)
        self._M_gpu:  Optional[torch.Tensor] = None   # (num_class, K_max)

    def _to_device(self, device: torch.device):
        """Move centroid tensors to device once and cache."""
        dev_str = str(device)
        if self._C_device != dev_str:
            self._C_gpu = torch.tensor(self._C_pad,  device=device)
            self._M_gpu = torch.tensor(self._C_mask, device=device)  # bool mask
            self._C_device = dev_str

    @torch.no_grad()
    def score(self, group_probs: List[torch.Tensor]) -> torch.Tensor:
        """
        group_probs: list of (B, n_gi) tensors
        Returns:     (B, num_class) score tensor  — single batched matmul, no per-class loop
        """
        device = group_probs[0].device
        self._to_device(device)  # no-op if already cached

        p      = torch.cat([g.float() for g in group_probs], dim=-1)  # (B, D)
        p_norm = F.normalize(p, dim=-1)                                # (B, D)

        # Batched cosine similarity: (B, D) x (D, num_class*K_max)
        # -> reshape C to (num_class*K_max, D) then matmul
        NC, KM, D = self._C_gpu.shape
        C_flat = self._C_gpu.view(NC * KM, D)          # (NC*KM, D)
        sim    = p_norm @ C_flat.T                      # (B, NC*KM)
        sim    = sim.view(-1, NC, KM)                   # (B, NC, KM)

        # Mask out padding slots with -inf before max
        mask = self._M_gpu.unsqueeze(0)                 # (1, NC, KM)
        sim  = sim.masked_fill(~mask, float("-inf"))    # (B, NC, KM)

        scores = sim.max(dim=-1).values                 # (B, NC)
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: JointSpatialTemporalCBM,
    kmeans_clf: KMeansRuleBookClassifier,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    num_class: int,
) -> tuple[Dict[str, float], Dict[str, Dict[int, float]], dict]:
    """
    Single-pass evaluation.
    Returns: (overall_acc, per_class_acc, raw_logits)
        raw_logits keys: 'neural', 'symbolic', 'kmeans_rb', 'labels'
    """
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    # Store raw logits for post-hoc fusion sweep
    raw: Dict[str, List[torch.Tensor]] = {
        "neural": [], "symbolic": [], "fused": [], "kmeans_rb": []
    }
    labels_all = []

    # Per-class accumulators
    correct = {
        "neural": defaultdict(int),
        "symbolic": defaultdict(int),
        "kmeans_rb": defaultdict(int),
    }
    counts  = defaultdict(int)

    total = len(loader)
    for bi, batch in enumerate(loader):
        if (bi + 1) % 30 == 0 or bi == 0:
            print(f"  [Eval] batch {bi+1}/{total}", end="\r")

        if len(batch) == 4:
            data, index_t, label, _ = batch
            index_t = index_t.to(device, non_blocking=True)
        else:
            data, label, _ = batch
            index_t = None

        data  = data.float().to(device, non_blocking=True)
        label = label.long()

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            feat          = model.extract_feat(data, index_t)
            _, gprobs     = model.encode_group_concepts(feat)
            sym_logits    = model.symbolic_action(gprobs)
            neural_logits = model.direct_action(feat)

        alpha        = model.fusion_alpha
        fused_logits = (1.0 - alpha) * neural_logits + alpha * sym_logits
        km_logits    = kmeans_clf.score(gprobs)

        # Luu logits tren CPU de sweep sau
        raw["neural"].append(neural_logits.cpu())
        raw["symbolic"].append(sym_logits.cpu())
        raw["fused"].append(fused_logits.cpu())
        raw["kmeans_rb"].append(km_logits.cpu())
        labels_all.append(label.cpu())

        # Per-class
        neu_pred = neural_logits.argmax(1).cpu()
        sym_pred = sym_logits.argmax(1).cpu()
        km_pred  = km_logits.argmax(1).cpu()
        for i, cls in enumerate(label.tolist()):
            counts[cls] += 1
            if neu_pred[i] == cls:
                correct["neural"][cls] += 1
            if sym_pred[i] == cls:
                correct["symbolic"][cls] += 1
            if km_pred[i] == cls:
                correct["kmeans_rb"][cls] += 1

    print()

    all_labels = torch.cat(labels_all)  # (N,)
    # Concat logits
    cat_raw = {k: torch.cat(v, dim=0) for k, v in raw.items()}  # (N, num_class)
    cat_raw["labels"] = all_labels

    overall = {
        mode: (cat_raw[mode].argmax(1) == all_labels).float().mean().item()
        for mode in ["neural", "symbolic", "fused", "kmeans_rb"]
    }
    per_class = {
        mode: {cls: correct[mode][cls] / counts[cls] for cls in sorted(counts)}
        for mode in correct
    }
    return overall, per_class, cat_raw


def fusion_sweep(
    cat_raw: dict,
    alphas: List[float],
) -> List[tuple[float, float]]:
    """
    Sweep beta: fused = (1-beta)*neural + beta*kmeans_rb
    Tra ve: [(beta, acc), ...] sorted by acc desc
    """
    neural  = cat_raw["neural"]    # (N, C)
    km      = cat_raw["kmeans_rb"] # (N, C)
    labels  = cat_raw["labels"]    # (N,)

    # Normalize logit scales truoc khi fuse
    # neural: softmax-scale | kmeans: cosine-sim-scale -> normalize
    neural_n = F.normalize(neural, dim=-1)
    km_n     = F.normalize(km,     dim=-1)

    results = []
    for beta in alphas:
        fused = (1.0 - beta) * neural_n + beta * km_n
        acc   = (fused.argmax(1) == labels).float().mean().item()
        results.append((beta, acc))
    return sorted(results, key=lambda x: -x[1])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p    = get_parser()
    args = load_args(p)

    # Device
    if torch.cuda.is_available() and args.device >= 0:
        device  = torch.device(f"cuda:{args.device}")
        use_amp = getattr(args, "use_amp", True)
    else:
        device  = torch.device("cpu")
        use_amp = False
        print("[Eval] WARN: CUDA not available, running on CPU")
    print(f"[Eval] Device: {device}  |  AMP: {use_amp}")

    K = getattr(args, "kmeans_k", 3)

    # ── Build model ────────────────────────────────────────────────────────────
    print(f"\n[Step 1] Building model ({args.backbone.upper()}, stream={args.data_type})...")
    model = build_model(args, device)
    model.eval()

    # ── Collect concept vectors from rulebook_split ────────────────────────────
    max_pc   = getattr(args, "max_per_class", 100)
    cap_str  = f"{max_pc} samples/class" if max_pc > 0 else "all samples"
    print(f"\n[Step 2] Collecting concept vectors from '{args.rulebook_split}' split (cap={cap_str})...")
    rb_loader = build_loader(args, args.rulebook_split)
    print(f"  {args.rulebook_split} set: {len(rb_loader.dataset)} samples, {len(rb_loader)} batches")
    concept_vecs = collect_concept_vectors(
        model, rb_loader, device, args.n_actions, use_amp, max_per_class=max_pc
    )
    n_total = sum(len(v) for v in concept_vecs.values())
    print(f"  Collected {n_total} vectors across {len(concept_vecs)} classes")

    # ── Fit KMeans ─────────────────────────────────────────────────────────────
    print(f"\n[Step 3] Fitting KMeans (K={K})...")
    kmeans_clf = KMeansRuleBookClassifier(concept_vecs, K=K, num_class=args.num_class)

    # ── Evaluate (single pass) ─────────────────────────────────────────────────
    eval_loader = (rb_loader if args.eval_split == args.rulebook_split
                   else build_loader(args, args.eval_split))
    print(f"\n[Step 4] Evaluating on '{args.eval_split}' split ({len(eval_loader.dataset)} samples)...")
    results, per_class, cat_raw = evaluate(model, kmeans_clf, eval_loader, device, use_amp, args.num_class)

    # ── Print summary ──────────────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  KMeans Rule Book Evaluation  (K={K})")
    print(f"  Stream: {args.data_type.upper()}  |  Backbone: {args.backbone.upper()}")
    print(f"  Build split: '{args.rulebook_split}'  |  Eval split: '{args.eval_split}'")
    print(sep)
    labels_col = {
        "neural":    "",
        "symbolic":  "",
        "fused":     f"  <- (1-a)*neural + a*symbolic, a={model.fusion_alpha:.2f}",
        "kmeans_rb": f"  <- K-means K={K} prototypes (pure symbolic)",
    }
    print(f"  {'Mode':<16}  {'Top-1 Acc':>10}")
    print(f"  {'-'*16}  {'-'*10}")
    for mode, acc in results.items():
        print(f"  {mode:<16}  {acc*100:>9.2f}%{labels_col.get(mode, '')}")

    # ── Neural + KMeans fusion sweep ───────────────────────────────────────────
    km_alpha = getattr(args, "km_alpha", 0.5)
    do_sweep = getattr(args, "km_alpha_sweep", False)

    if do_sweep:
        alphas = [round(a * 0.05, 2) for a in range(21)]  # 0.00, 0.05, ..., 1.00
        sweep_results = fusion_sweep(cat_raw, alphas)
        best_beta, best_acc = sweep_results[0]
        print(f"\n  Neural + KMeans Fusion Sweep:")
        print(f"  {'Beta':>6}  {'Acc':>9}")
        print(f"  {'-'*6}  {'-'*9}")
        for beta, acc in sorted(sweep_results, key=lambda x: x[0]):
            flag = "  <- BEST" if beta == best_beta else ""
            print(f"  {beta:>6.2f}  {acc*100:>8.2f}%{flag}")
        print(f"\n  Best beta={best_beta:.2f}  -> Acc={best_acc*100:.2f}%")
    else:
        # Single beta evaluation
        from torch.nn import functional as F2
        neural_n = F.normalize(cat_raw["neural"],    dim=-1)
        km_n     = F.normalize(cat_raw["kmeans_rb"], dim=-1)
        labels   = cat_raw["labels"]
        fused_km = (1.0 - km_alpha) * neural_n + km_alpha * km_n
        km_fused_acc = (fused_km.argmax(1) == labels).float().mean().item()
        print(f"  {'km_fused':<16}  {km_fused_acc*100:>9.2f}%  <- (1-{km_alpha})*neural + {km_alpha}*kmeans")

    print(sep)

    # ── Per-class breakdown ────────────────────────────────────────────────────
    neu_cls = per_class["neural"]
    sym_cls = per_class["symbolic"]
    km_cls  = per_class["kmeans_rb"]
    print(f"\n  Per-class accuracy — Neural vs Symbolic vs KMeans K={K}:")
    print(f"  {'Cls':<4}  {'Action':<26}  {'Neural':>7}  {'Symbolic':>8}  {'KMeans':>7}  {'d(KM-Neu)':>9}")
    print(f"  {'-'*4}  {'-'*26}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*9}")
    for cls in sorted(sym_cls):
        name  = NTU60_NAMES[cls] if cls < len(NTU60_NAMES) else str(cls)
        n_acc = neu_cls.get(cls, 0.0) * 100
        s_acc = sym_cls[cls] * 100
        k_acc = km_cls.get(cls, 0.0) * 100
        delta = k_acc - n_acc
        marker = " +" if delta > 1 else (" --" if delta < -3 else "")
        print(f"  {cls:>3}   {name:<26}  {n_acc:>6.1f}%  {s_acc:>7.1f}%  {k_acc:>6.1f}%  {delta:>+8.1f}%{marker}")

    neu_mean = np.mean(list(neu_cls.values())) * 100
    sym_mean = np.mean(list(sym_cls.values())) * 100
    km_mean  = np.mean(list(km_cls.values())) * 100
    print(f"  {'':4}   {'Mean per-class':<26}  {neu_mean:>6.1f}%  {sym_mean:>7.1f}%  {km_mean:>6.1f}%  {km_mean-neu_mean:>+8.1f}%")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()

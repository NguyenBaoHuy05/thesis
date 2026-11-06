"""
Rule Comparison & Case Study — Neural vs Fuzzy AND/OR Tree vs KMeans Rule Book
================================================================================
So sanh 4 diem tren pho symbolic->neural, deu doc tu 1 checkpoint da train:
  1. Neural       (direct_action head — mo hinh tham chieu, khong dung de "giai thich")
  2. Tree (soft)  (FuzzyLogicTree, gate AND/OR mem — co gradient, upper bound cua tree)
  3. Tree (hard)  (FuzzyLogicTree da harden — cong thuc Boolean AND/OR/NOT that)
  4. KMeans-RB    (K prototypes per action — case-based, khong hoc)
  + Tree+KMeans Ensemble (--tree-km-gamma / --tree-km-sweep): ket hop 2 phuong an
    symbolic (soft tree + case-based KMeans), 0% neural, xem co cong huong khong.

Usage
-----
  python eval_rulebook.py --config config/ntu/skateformer/ntu_cs_j.yaml
      --checkpoint work_dir/ntu_cs/skateformer_joint_symbolic/joint_best.pt
      --data D:\\KLTN_Huy\\thesis\\data\\ntu\\NTU60_CS.npz
      --n-actions 60 --device 0
      --rulebook-split train --eval-split test
      --max-per-class 100
      --kmeans-k 3
      --tree-km-sweep
      --case-study --n-case-per-class 2
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
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None
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
        description="Rule Comparison & Case Study — Neural vs Fuzzy AND/OR Tree vs KMeans",
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
    p.add_argument("--no-amp",      dest="use_amp", action="store_false", help="Disable AMP and evaluate in full FP32")
    p.add_argument("--rulebook-split", choices=["train", "test"], default="train",
                   help="Split de build KMeans prototypes")
    p.add_argument("--eval-split",  choices=["train", "test"], default="test",
                   help="Split de evaluate")
    p.add_argument("--kmeans-k",    type=int, default=3,
                   help="So clusters K moi action (K=1 = mean, K>1 = capture variation)")
    # Tree + KMeans ensemble (both symbolic/interpretable, 0% neural)
    p.add_argument("--tree-km-gamma", type=float, default=0.5,
                   help="Ty le kmeans_rb trong ensemble Tree(soft)+KMeans: "
                        "(1-gamma)*tree_soft + gamma*kmeans_rb")
    p.add_argument("--tree-km-sweep", action="store_true", default=False,
                   help="Tu dong sweep gamma tu 0.0 den 1.0 cho ensemble Tree(soft)+KMeans")
    # Model hyperparams (overridden by yaml) — must match the checkpoint being loaded
    p.add_argument("--concept-loss-weight",     type=float, default=0.20)
    p.add_argument("--concept-negative-weight", type=float, default=0.10)
    p.add_argument("--tree-depth",              type=int,   default=4)
    p.add_argument("--tree-entropy-weight",     type=float, default=0.01)
    p.add_argument("--tree-diversity-weight",   type=float, default=0.05)
    p.add_argument("--tree-kg-neighbor-k",      type=int,   default=8)
    p.add_argument("--group-hidden-dim",        type=int,   default=128,
                   help="Must match the checkpoint's training-time value -- it changes "
                        "concept-head parameter shapes, unlike the other tree/collapse weights")
    p.add_argument("--concept-collapse-weight", type=float, default=0.02)
    # Case study
    p.add_argument("--case-study", action="store_true", default=False,
                   help="Xuat case study dinh tinh: N sample/class, cong thuc Boolean da harden, "
                        "trace tung literal dung/sai, true vs predicted label")
    p.add_argument("--n-case-per-class", type=int, default=2)
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
    center = (args.backbone.lower() == "hypergcn")
    ds = Feeder(
        data_path   = args.data,
        split       = split,
        window_size = args.window_size,
        debug       = False,
        thres       = args.window_size,
        uniform     = True,
        partition   = partition,
        center      = center,
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
        device=device,
        tree_depth=args.tree_depth,
        tree_entropy_weight=args.tree_entropy_weight,
        tree_diversity_weight=args.tree_diversity_weight,
        tree_kg_neighbor_k=args.tree_kg_neighbor_k,
        group_hidden_dim=args.group_hidden_dim,
        concept_collapse_weight=args.concept_collapse_weight,
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

def _fit_kmeans(X: np.ndarray, k: int, max_iter: int = 20, seed: int = 42) -> np.ndarray:
    """Fit KMeans using sklearn if available, else lightweight numpy fallback."""
    if KMeans is not None:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        km.fit(X)
        return km.cluster_centers_
    # Lightweight numpy fallback for environments without scikit-learn
    np.random.seed(seed)
    n = len(X)
    idx = np.random.choice(n, size=k, replace=False)
    centers = X[idx].copy()
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=-1)
        labels = dists.argmin(axis=-1)
        new_centers = np.array([
            X[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
            for j in range(k)
        ])
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers
    return centers


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
                center = _fit_kmeans(X, k_actual)               # (K, D)

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
    Single-pass evaluation across the 4 comparison points.
    Returns: (overall_acc, per_class_acc, raw_scores)
        raw_scores keys: 'neural', 'tree_soft', 'tree_hard', 'kmeans_rb', 'labels'
    """
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    raw: Dict[str, List[torch.Tensor]] = {
        "neural": [], "tree_soft": [], "tree_hard": [], "kmeans_rb": []
    }
    labels_all = []
    counts = defaultdict(int)

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
            concept_probs = torch.cat([g.float() for g in gprobs], dim=-1)
            tree_soft     = model.fuzzy_tree(concept_probs, hard=False)
            tree_hard     = model.fuzzy_tree(concept_probs, hard=True)
            neural_logits = model.direct_action(feat)

        km_logits = kmeans_clf.score(gprobs)

        raw["neural"].append(neural_logits.cpu())
        raw["tree_soft"].append(tree_soft.cpu())
        raw["tree_hard"].append(tree_hard.cpu())
        raw["kmeans_rb"].append(km_logits.cpu())
        labels_all.append(label.cpu())

        for cls in label.tolist():
            counts[cls] += 1

    print()

    all_labels = torch.cat(labels_all)  # (N,)
    cat_raw = {k: torch.cat(v, dim=0) for k, v in raw.items()}  # (N, num_class)
    cat_raw["labels"] = all_labels

    modes = ["neural", "tree_soft", "tree_hard", "kmeans_rb"]
    overall = {
        mode: (cat_raw[mode].argmax(1) == all_labels).float().mean().item()
        for mode in modes
    }
    per_class = {
        mode: {
            cls: ((cat_raw[mode].argmax(1) == cls) & (all_labels == cls)).sum().item() / counts[cls]
            for cls in sorted(counts) if counts[cls] > 0
        }
        for mode in modes
    }
    return overall, per_class, cat_raw


def tree_kmeans_sweep(
    cat_raw: dict,
    gammas: List[float],
) -> List[tuple[float, float]]:
    """
    Sweep gamma: ensemble = (1-gamma)*tree_soft + gamma*kmeans_rb
    (Ensemble 2 phuong an symbolic/interpretable, 0% neural.)
    Tra ve: [(gamma, acc), ...] sorted by acc desc
    """
    tree_n = F.normalize(cat_raw["tree_soft"], dim=-1)
    km_n   = F.normalize(cat_raw["kmeans_rb"], dim=-1)
    labels = cat_raw["labels"]

    results = []
    for g in gammas:
        fused = (1.0 - g) * tree_n + g * km_n
        acc   = (fused.argmax(1) == labels).float().mean().item()
        results.append((g, acc))
    return sorted(results, key=lambda x: -x[1])


@torch.no_grad()
def case_study(
    model: JointSpatialTemporalCBM,
    loader: DataLoader,
    device: torch.device,
    n_actions: int,
    n_per_class: int,
    out_path: str,
):
    """N sample/class: cong thuc Boolean da harden cua true class, trace tung literal
    dung/sai tren chinh sample do, va predicted label (argmax tree cung) so voi true label."""
    model.eval()
    formulas = model.fuzzy_tree.export_formula(model.concept_names)

    samples: Dict[int, List[tuple]] = defaultdict(list)
    for batch in loader:
        if len(batch) == 4:
            data, index_t, label, _ = batch
        else:
            data, label, _ = batch
            index_t = None
        for i, cls in enumerate(label.tolist()):
            if cls < n_actions and len(samples[cls]) < n_per_class:
                samples[cls].append((
                    data[i:i + 1].float().to(device),
                    index_t[i:i + 1].to(device) if index_t is not None else None,
                ))
        if all(len(samples[c]) >= n_per_class for c in range(n_actions)):
            break

    lines = ["=" * 78, " Fuzzy AND/OR Logic Tree -- Case Study (hardened rules)", "=" * 78]
    for cls in range(n_actions):
        if cls not in samples:
            continue
        name = NTU60_NAMES[cls] if cls < len(NTU60_NAMES) else str(cls)
        lines.append(f"\nAction [{cls:02d}] {name}")
        lines.append(f"  Rule: {formulas[cls]}")
        for x, it in samples[cls]:
            feat = model.extract_feat(x, it)
            _, gprobs = model.encode_group_concepts(feat)
            concept_probs = torch.cat([g.float() for g in gprobs], dim=-1)
            trace = model.fuzzy_tree.explain_hard(concept_probs, cls, model.concept_names)
            tree_hard = model.fuzzy_tree(concept_probs, hard=True)
            pred = int(tree_hard.argmax(1).item())
            pred_name = NTU60_NAMES[pred] if pred < len(NTU60_NAMES) else str(pred)
            correct = "OK" if pred == cls else "WRONG"
            lit_str = ", ".join(
                f"{l['literal']}={'T' if l['satisfied'] else 'F'}({l['concept_prob']:.2f})"
                for l in trace["literals"]
            )
            lines.append(f"    literals: {lit_str}")
            lines.append(
                f"    rule({name})={trace['result']}  predicted={pred_name}  true={name}  [{correct}]"
            )

    text = "\n".join(lines)
    print(text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Eval] Case study saved -> {out_path}")


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

    # ── Concept head diagnostic: flag heads that never activate (collapsed) ────
    all_vecs = np.concatenate(list(concept_vecs.values()), axis=0)  # (N, num_concepts)
    concept_mean = all_vecs.mean(axis=0)
    concept_std  = all_vecs.std(axis=0)
    dead_idx = np.where(concept_std < 0.02)[0]
    if len(dead_idx) > 0:
        print(f"\n  [WARN] {len(dead_idx)}/{len(concept_mean)} concept(s) have near-zero variance "
              f"across ALL {n_total} samples (likely collapsed/dead head -- always predicting the "
              f"same probability regardless of input, not a genuine per-sample signal):")
        for i in dead_idx:
            name = model.concept_names[i] if i < len(model.concept_names) else str(i)
            print(f"      {name:<40} mean={concept_mean[i]:.4f}  std={concept_std[i]:.4f}")
        print("      -> the tree may be exploiting these as 'free' NOT-literals (always-true, "
              "uninformative) instead of learning real discriminative concepts.")

    # ── Fit KMeans ─────────────────────────────────────────────────────────────
    print(f"\n[Step 3] Fitting KMeans (K={K})...")
    kmeans_clf = KMeansRuleBookClassifier(concept_vecs, K=K, num_class=args.num_class)

    # ── Evaluate (single pass) ─────────────────────────────────────────────────
    eval_loader = (rb_loader if args.eval_split == args.rulebook_split
                   else build_loader(args, args.eval_split))
    print(f"\n[Step 4] Evaluating on '{args.eval_split}' split ({len(eval_loader.dataset)} samples)...")
    results, per_class, cat_raw = evaluate(
        model, kmeans_clf, eval_loader, device, use_amp, args.num_class
    )

    # ── Rule coverage + hard-with-soft-fallback (hard formula is a strict boolean
    # per class -- 0, 1, or >1 classes can fire for the same sample) ────────────
    hard_all = cat_raw["tree_hard"]          # (N, num_class) in {0., 1.}
    n_fire   = hard_all.sum(dim=1)           # (N,)
    labels   = cat_raw["labels"]
    coverage_unique = (n_fire == 1).float().mean().item()
    coverage_none   = (n_fire == 0).float().mean().item()
    coverage_multi  = (n_fire > 1).float().mean().item()

    hard_preds   = hard_all.argmax(1)
    soft_preds   = cat_raw["tree_soft"].argmax(1)
    unique_mask  = (n_fire == 1)
    hybrid_preds = torch.where(unique_mask, hard_preds, soft_preds)
    hybrid_acc   = (hybrid_preds == labels).float().mean().item()
    unique_acc   = ((hard_preds == labels) & unique_mask).float().sum().item() / max(unique_mask.sum().item(), 1)

    print(f"\n  Rule coverage (hardened tree): unique-fire={coverage_unique*100:.1f}%  "
          f"no-fire={coverage_none*100:.1f}%  multi-fire={coverage_multi*100:.1f}%")
    print(f"  Accuracy WHEN exactly 1 rule fires : {unique_acc*100:.2f}%  "
          f"(true faithfulness of the hardened rules, on the samples they can actually decide)")
    print(f"  Hybrid accuracy (rule if unique, else soft-tree fallback) : {hybrid_acc*100:.2f}%")

    # ── Print summary ──────────────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  Rule Comparison Evaluation  (K={K})")
    print(f"  Stream: {args.data_type.upper()}  |  Backbone: {args.backbone.upper()}")
    print(f"  Build split: '{args.rulebook_split}'  |  Eval split: '{args.eval_split}'")
    print(sep)
    labels_col = {
        "neural":    "  <- reference upper bound, not 'explainable'",
        "tree_soft": "  <- Fuzzy AND/OR Tree, soft gates (has gradient)",
        "tree_hard": "  <- Fuzzy AND/OR Tree, HARDENED (real Boolean AND/OR/NOT)",
        "kmeans_rb": f"  <- K-means K={K} prototypes (case-based, not learned)",
    }
    print(f"  {'Mode':<12}  {'Top-1 Acc':>10}")
    print(f"  {'-'*12}  {'-'*10}")
    for mode, acc in results.items():
        print(f"  {mode:<12}  {acc*100:>9.2f}%{labels_col.get(mode, '')}")
    hardening_gap = (results["tree_soft"] - results["tree_hard"]) * 100
    print(f"\n  Hardening gap (soft - hard) = {hardening_gap:+.2f} pts "
          f"-- cai gia phai tra khi cung hoa logic thanh AND/OR/NOT that")
    print(sep)

    # ── Tree(soft) + KMeans-RB ensemble (2 diem symbolic, 0% neural) ────────────
    active_gamma = args.tree_km_gamma
    if getattr(args, "tree_km_sweep", False):
        gammas = [round(g * 0.05, 2) for g in range(21)]  # 0.00, 0.05, ..., 1.00
        tk_results = tree_kmeans_sweep(cat_raw, gammas)
        best_gamma, best_tk_acc = tk_results[0]
        active_gamma = best_gamma
        print(f"\n  Tree(soft) + KMeans-RB Ensemble Sweep ((1-g)*tree_soft + g*kmeans_rb, 0% Neural):")
        print(f"  {'Gamma':>6}  {'Acc':>9}")
        print(f"  {'-'*6}  {'-'*9}")
        for g, acc in sorted(tk_results, key=lambda x: x[0]):
            flag = "  <- BEST" if g == best_gamma else ""
            print(f"  {g:>6.2f}  {acc*100:>8.2f}%{flag}")
        print(f"\n  Best gamma={best_gamma:.2f} -> Acc={best_tk_acc*100:.2f}% (2 phuong an symbolic ket hop)")

    tree_n  = F.normalize(cat_raw["tree_soft"], dim=-1)
    km_n    = F.normalize(cat_raw["kmeans_rb"], dim=-1)
    tree_km = (1.0 - active_gamma) * tree_n + active_gamma * km_n
    tk_preds = tree_km.argmax(1)
    tk_acc   = (tk_preds == labels).float().mean().item()
    per_class["tree_km"] = {
        cls: ((tk_preds == cls) & (labels == cls)).sum().item() / max(((labels == cls).sum().item()), 1)
        for cls in sorted(per_class["kmeans_rb"])
    }
    print(f"  {'tree_km':<12}  {tk_acc*100:>9.2f}%  <- (1-{active_gamma:.2f})*tree_soft + {active_gamma:.2f}*kmeans_rb")
    print(sep)

    # ── Per-class breakdown ────────────────────────────────────────────────────
    neu_cls  = per_class["neural"]
    soft_cls = per_class["tree_soft"]
    hard_cls = per_class["tree_hard"]
    km_cls   = per_class["kmeans_rb"]
    tk_cls   = per_class["tree_km"]
    print(f"\n  Per-class accuracy — Neural vs Tree(soft) vs Tree(hard) vs KMeans K={K} vs Tree+KM (g={active_gamma:.2f}):")
    print(f"  {'Cls':<4}  {'Action':<26}  {'Neural':>7}  {'Tree-S':>7}  {'Tree-H':>7}  {'KMeans':>7}  {'Tree+KM':>7}  {'d(H-Neu)':>8}")
    print(f"  {'-'*4}  {'-'*26}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")
    for cls in sorted(neu_cls):
        name   = NTU60_NAMES[cls] if cls < len(NTU60_NAMES) else str(cls)
        n_acc  = neu_cls.get(cls, 0.0) * 100
        s_acc  = soft_cls.get(cls, 0.0) * 100
        h_acc  = hard_cls.get(cls, 0.0) * 100
        k_acc  = km_cls.get(cls, 0.0) * 100
        t_acc  = tk_cls.get(cls, 0.0) * 100
        delta  = h_acc - n_acc
        marker = " +" if delta > 1 else (" --" if delta < -3 else "")
        print(f"  {cls:>3}   {name:<26}  {n_acc:>6.1f}%  {s_acc:>6.1f}%  {h_acc:>6.1f}%  {k_acc:>6.1f}%  {t_acc:>6.1f}%  {delta:>+7.1f}%{marker}")

    neu_mean  = np.mean(list(neu_cls.values())) * 100
    soft_mean = np.mean(list(soft_cls.values())) * 100
    hard_mean = np.mean(list(hard_cls.values())) * 100
    km_mean   = np.mean(list(km_cls.values())) * 100
    tk_mean   = np.mean(list(tk_cls.values())) * 100
    print(f"  {'':4}   {'Mean per-class':<26}  {neu_mean:>6.1f}%  {soft_mean:>6.1f}%  {hard_mean:>6.1f}%  {km_mean:>6.1f}%  {tk_mean:>6.1f}%  {hard_mean-neu_mean:>+7.1f}%")
    print(f"\n{sep}\n")

    # ── Case study ──────────────────────────────────────────────────────────
    if getattr(args, "case_study", False):
        case_out = os.path.join(os.path.dirname(args.checkpoint) or ".", "case_study.txt")
        print(f"\n[Step 5] Case study ({args.n_case_per_class} samples/class)...")
        case_study(
            model, eval_loader, device,
            n_actions=args.n_actions,
            n_per_class=args.n_case_per_class,
            out_path=case_out,
        )


if __name__ == "__main__":
    main()

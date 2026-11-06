"""
Joint + Bone Ensemble Evaluation
==================================
Ket hop hai stream (joint va bone) voi KMeans Rule Book.

Cac mode duoc so sanh:
  1. joint_neural   : chỉ joint direct_action head
  2. bone_neural    : chỉ bone direct_action head
  3. neural_ens     : joint_neural + bone_neural (trung binh)
  4. joint_km       : KMeans RuleBook joint stream
  5. bone_km        : KMeans RuleBook bone stream
  6. km_ens         : joint_km + bone_km (trung binh)
  7. full_ens       : (1-α)*neural_ens + α*km_ens (sweep α)

Usage
-----
  python eval_ensemble.py
    --joint-config   config/ntu/skateformer/ntu_cs_j.yaml
    --joint-ckpt     work_dir/ntu_cs/skateformer_joint_symbolic/joint_best.pt
    --bone-config    config/ntu/skateformer/ntu_cs_b.yaml
    --bone-ckpt      work_dir/ntu_cs/skateformer_bone_symbolic/joint_best.pt
    --data           D:\\KLTN_Huy\\thesis\\data\\ntu\\NTU60_CS.npz
    --n-actions 60 --device 0 --kmeans-k 1 --max-per-class 0
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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
        description="Joint + Bone Ensemble with KMeans Rule Book",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Joint stream
    p.add_argument("--joint-config", default="config/ntu/skateformer/ntu_cs_j.yaml")
    p.add_argument("--joint-ckpt",   default="work_dir/ntu_cs/skateformer_joint_symbolic/joint_best.pt")
    # Bone stream
    p.add_argument("--bone-config",  default="config/ntu/skateformer/ntu_cs_b.yaml")
    p.add_argument("--bone-ckpt",    default="work_dir/ntu_cs/skateformer_bone_symbolic/joint_best.pt")
    # Data
    p.add_argument("--data",         default=None, help="NTU60_CS.npz")
    p.add_argument("--n-actions",    type=int, default=60)
    p.add_argument("--num-class",    type=int, default=60)
    p.add_argument("--window-size",  type=int, default=64)
    p.add_argument("--num-people",   type=int, default=2)
    p.add_argument("--num-points",   type=int, default=24)
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--device",       type=int, default=0)
    p.add_argument("--use-amp",      action="store_true", default=True)
    # KMeans
    p.add_argument("--kmeans-k",     type=int, default=1)
    p.add_argument("--max-per-class",type=int, default=100)
    p.add_argument("--rulebook-split", choices=["train","test"], default="train")
    p.add_argument("--eval-split",   choices=["train","test"], default="test")
    # Fusion sweep
    p.add_argument("--km-alpha-sweep", action="store_true", default=False)
    p.add_argument("--km-alpha",     type=float, default=0.5)
    p.add_argument("--tri-sweep",    action="store_true", default=True, help="Sweep 3-way weights (neural, symbolic, kmeans)")
    p.add_argument("--out-file",     type=str, default="work_dir/ntu_cs/ensemble_tri_results.txt")
    # Model hyperparams
    p.add_argument("--symbolic-temperature",    type=float, default=0.7)
    p.add_argument("--symbolic-margin",         type=float, default=0.15)
    p.add_argument("--concept-loss-weight",     type=float, default=0.20)
    p.add_argument("--concept-negative-weight", type=float, default=0.10)
    p.add_argument("--kg-s-weight",             type=float, default=0.02)
    p.add_argument("--kg-c-weight",             type=float, default=0.01)
    p.add_argument("--fusion-start-alpha",      type=float, default=0.50)
    return p


def load_yaml(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_args(p: argparse.ArgumentParser) -> argparse.Namespace:
    # First parse known args to find config paths
    args, _ = p.parse_known_args()
    j_cfg = load_yaml(args.joint_config)
    b_cfg = load_yaml(args.bone_config)

    # Exclude keys that shouldn't override CLI arguments
    exclude = {"data_type", "checkpoint", "joint_ckpt", "bone_ckpt", "data"}
    yaml_defaults = {k.replace("-", "_"): v for k, v in j_cfg.items() if k not in exclude}
    p.set_defaults(**yaml_defaults)

    # Final parse so CLI arguments always take precedence
    args = p.parse_args()
    if not args.data:
        args.data = j_cfg.get("data", None)
    return args


# ─────────────────────────────────────────────────────────────────────────────
# Build helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_loader(args, data_type: str, split: str) -> DataLoader:
    backbone = "skateformer"
    num_points = 24
    ds = Feeder(
        data_path   = args.data,
        split       = split,
        window_size = args.window_size,
        debug       = False,
        thres       = args.window_size,
        uniform     = True,
        partition   = True,  # skateformer always True
        data_type   = data_type,
        p_interval  = [0.5, 1] if split == "train" else [0.95],
        aug_method  = "z",
        intra_p     = 0.0,
        inter_p     = 0.0,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      drop_last=False, num_workers=args.num_workers, pin_memory=True)


def build_stream_model(ckpt: str, data_type: str, args, device: torch.device) -> JointSpatialTemporalCBM:
    backbone_cfg = dict(
        in_channels=3, num_people=args.num_people, num_points=24,
        num_frames=args.window_size, kernel_size=7, num_heads=32,
        attn_drop=0.0, head_drop=0.0, rel=True, drop_path=0.0,
        type_1_size=(8, 8), type_2_size=(8, 12),
        type_3_size=(8, 8), type_4_size=(8, 12),
        mlp_ratio=4.0, index_t=True,
    )
    model = build_joint_model(
        checkpoint_path=None, backbone_name="skateformer",
        backbone_cfg=backbone_cfg, num_class=args.num_class,
        concept_loss_weight=args.concept_loss_weight,
        concept_negative_weight=args.concept_negative_weight,
        symbolic_temperature=args.symbolic_temperature,
        symbolic_margin=args.symbolic_margin, device=device,
        kg_s_weight=args.kg_s_weight, kg_c_weight=args.kg_c_weight,
        fusion_alpha=args.fusion_start_alpha,
    )
    raw   = torch.load(ckpt, map_location="cpu")
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    state = {k.replace("module.", ""): v for k, v in state.items()}
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"  Loaded {data_type.upper()} <- {ckpt}  (missing={len(miss)}, unexpected={len(unexp)})")
    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# KMeans Classifier (same as eval_rulebook.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_concept_vectors(
    model: JointSpatialTemporalCBM,
    loader: DataLoader,
    device: torch.device,
    n_actions: int,
    use_amp: bool,
    max_per_class: int,
) -> Dict[int, np.ndarray]:
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    cap = max_per_class if max_per_class > 0 else float("inf")
    raw: Dict[int, List[np.ndarray]] = defaultdict(list)
    total = len(loader)

    for bi, batch in enumerate(loader):
        if (bi + 1) % 30 == 0 or bi == 0:
            print(f"    batch {bi+1}/{total}", end="\r")

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

        vec = torch.cat([p.float() for p in group_probs], dim=-1).cpu().numpy()
        for i, cls in enumerate(label.tolist()):
            if cls >= n_actions or len(raw[cls]) >= cap:
                continue
            raw[cls].append(vec[i])

        if max_per_class > 0 and all(len(raw[c]) >= cap for c in range(n_actions)):
            print(f"\n    Early stop: du {max_per_class} samples/class")
            break

    print()
    return {cls: np.stack(vecs) for cls, vecs in raw.items() if vecs}


class KMeansRuleBookClassifier:
    def __init__(self, concept_vecs: Dict[int, np.ndarray], K: int, num_class: int):
        self.K = K
        self.num_class = num_class
        centroids: Dict[int, np.ndarray] = {}
        total_concepts = 0

        for cls, X in sorted(concept_vecs.items()):
            n, D = X.shape
            total_concepts = D
            k_actual = min(K, n)
            if k_actual == 1 or n <= k_actual:
                center = X.mean(axis=0, keepdims=True)
            else:
                km = KMeans(n_clusters=k_actual, random_state=42, n_init="auto")
                km.fit(X)
                center = km.cluster_centers_
            norms = np.linalg.norm(center, axis=1, keepdims=True).clip(min=1e-8)
            centroids[cls] = (center / norms).astype(np.float32)

        # Pre-build padded tensor
        K_max  = max(c.shape[0] for c in centroids.values())
        C_pad  = np.zeros((num_class, K_max, total_concepts), dtype=np.float32)
        C_mask = np.zeros((num_class, K_max), dtype=bool)
        for cls, center in centroids.items():
            k = center.shape[0]
            C_pad[cls, :k]  = center
            C_mask[cls, :k] = True
        self._C_pad  = C_pad
        self._C_mask = C_mask
        self._C_device: Optional[str] = None
        self._C_gpu: Optional[torch.Tensor] = None
        self._M_gpu: Optional[torch.Tensor] = None

    def _to_device(self, device: torch.device):
        dev_str = str(device)
        if self._C_device != dev_str:
            self._C_gpu = torch.tensor(self._C_pad,  device=device)
            self._M_gpu = torch.tensor(self._C_mask, device=device)
            self._C_device = dev_str

    @torch.no_grad()
    def score(self, group_probs: List[torch.Tensor]) -> torch.Tensor:
        device = group_probs[0].device
        self._to_device(device)
        p      = torch.cat([g.float() for g in group_probs], dim=-1)
        p_norm = F.normalize(p, dim=-1)
        NC, KM, D = self._C_gpu.shape
        C_flat = self._C_gpu.view(NC * KM, D)
        sim    = p_norm @ C_flat.T
        sim    = sim.view(-1, NC, KM)
        mask   = self._M_gpu.unsqueeze(0)
        sim    = sim.masked_fill(~mask, float("-inf"))
        return sim.max(dim=-1).values


# ─────────────────────────────────────────────────────────────────────────────
# Collect logits from one stream
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_stream_logits(
    model: JointSpatialTemporalCBM,
    km_clf: KMeansRuleBookClassifier,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    tag: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (neural_logits, sym_logits, km_logits, labels)  all on CPU, shape (N, C) / (N,)
    """
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    all_neural, all_sym, all_km, all_labels = [], [], [], []
    total = len(loader)

    for bi, batch in enumerate(loader):
        if (bi + 1) % 30 == 0 or bi == 0:
            print(f"  [{tag}] batch {bi+1}/{total}", end="\r")

        if len(batch) == 4:
            data, index_t, label, _ = batch
            index_t = index_t.to(device, non_blocking=True)
        else:
            data, label, _ = batch
            index_t = None

        data = data.float().to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            feat          = model.extract_feat(data, index_t)
            _, gprobs     = model.encode_group_concepts(feat)
            neural_logits = model.direct_action(feat)
            sym_logits    = model.symbolic_action(gprobs)

        km_logits = km_clf.score(gprobs)
        all_neural.append(neural_logits.cpu())
        all_sym.append(sym_logits.cpu())
        all_km.append(km_logits.cpu())
        all_labels.append(label.cpu())

    print()
    return (
        torch.cat(all_neural),   # (N, C)
        torch.cat(all_sym),      # (N, C)
        torch.cat(all_km),       # (N, C)
        torch.cat(all_labels),   # (N,)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble evaluation
# ─────────────────────────────────────────────────────────────────────────────

def acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def ensemble_sweep(
    j_neural: torch.Tensor,
    b_neural: torch.Tensor,
    j_km: torch.Tensor,
    b_km: torch.Tensor,
    labels: torch.Tensor,
    alphas: List[float],
) -> List[Tuple[float, float]]:
    """
    Sweep alpha: full_ens = (1-α) * neural_ens + α * km_ens
    """
    jn = norm(j_neural)
    bn = norm(b_neural)
    jk = norm(j_km)
    bk = norm(b_km)
    neural_ens = jn + bn  # (N, C)
    km_ens     = jk + bk  # (N, C)

    results = []
    for alpha in alphas:
        fused = (1.0 - alpha) * neural_ens + alpha * km_ens
        results.append((alpha, acc(fused, labels)))
    return sorted(results, key=lambda x: -x[1])


def tri_fusion_sweep(
    neural_ens: torch.Tensor,
    sym_ens: torch.Tensor,
    km_ens: torch.Tensor,
    labels: torch.Tensor,
    step: float = 0.05,
) -> List[Tuple[float, float, float, float]]:
    """
    Sweep (w_neu, w_sym, w_km) on simplex: w_neu + w_sym + w_km == 1.0.
    Returns: sorted list of (wn, ws, wk, acc) descending by acc.
    """
    results = []
    steps = int(round(1.0 / step))
    for i in range(steps + 1):
        wn = round(i * step, 2)
        for j in range(steps + 1 - i):
            ws = round(j * step, 2)
            wk = round(1.0 - wn - ws, 2)
            if wk < -1e-5:
                continue
            wk = max(0.0, wk)
            fused = wn * neural_ens + ws * sym_ens + wk * km_ens
            a = acc(fused, labels)
            results.append((wn, ws, wk, a))
    return sorted(results, key=lambda x: -x[3])


# ─────────────────────────────────────────────────────────────────────────────
# Per-class breakdown
# ─────────────────────────────────────────────────────────────────────────────

def per_class_acc(logits: torch.Tensor, labels: torch.Tensor, num_class: int) -> Dict[int, float]:
    correct = defaultdict(int)
    counts  = defaultdict(int)
    preds   = logits.argmax(1)
    for i in range(len(labels)):
        cls = labels[i].item()
        counts[cls] += 1
        if preds[i].item() == cls:
            correct[cls] += 1
    return {cls: correct[cls] / counts[cls] for cls in sorted(counts)}


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
    print(f"[Ensemble] Device: {device}  |  AMP: {use_amp}")

    K      = getattr(args, "kmeans_k", 1)
    max_pc = getattr(args, "max_per_class", 100)
    cap_str = f"{max_pc}/class" if max_pc > 0 else "all"

    # ── Load both models ──────────────────────────────────────────────────────
    print("\n[Step 1] Loading models...")
    model_j = build_stream_model(args.joint_ckpt, "j", args, device)
    model_b = build_stream_model(args.bone_ckpt,  "b", args, device)

    # ── Build KMeans from rulebook_split ─────────────────────────────────────
    print(f"\n[Step 2] Building KMeans (K={K}, cap={cap_str}) from '{args.rulebook_split}'...")
    rb_loader_j = build_loader(args, "j", args.rulebook_split)
    rb_loader_b = build_loader(args, "b", args.rulebook_split)

    print(f"  Joint: collecting concept vectors...")
    cvecs_j = collect_concept_vectors(model_j, rb_loader_j, device, args.n_actions, use_amp, max_pc)
    print(f"  Bone:  collecting concept vectors...")
    cvecs_b = collect_concept_vectors(model_b, rb_loader_b, device, args.n_actions, use_amp, max_pc)

    print(f"  Fitting KMeans K={K}...")
    km_j = KMeansRuleBookClassifier(cvecs_j, K=K, num_class=args.num_class)
    km_b = KMeansRuleBookClassifier(cvecs_b, K=K, num_class=args.num_class)
    print(f"  Done.")

    # ── Collect logits from eval_split ────────────────────────────────────────
    eval_loader_j = (rb_loader_j if args.eval_split == args.rulebook_split
                     else build_loader(args, "j", args.eval_split))
    eval_loader_b = (rb_loader_b if args.eval_split == args.rulebook_split
                     else build_loader(args, "b", args.eval_split))

    print(f"\n[Step 3] Collecting logits from '{args.eval_split}' ({len(eval_loader_j.dataset)} samples)...")
    j_neural, j_sym, j_km, labels = collect_stream_logits(model_j, km_j, eval_loader_j, device, use_amp, "Joint")
    b_neural, b_sym, b_km, _      = collect_stream_logits(model_b, km_b, eval_loader_b, device, use_amp, "Bone")

    # ── Compute all modes ─────────────────────────────────────────────────────
    jn = norm(j_neural);  bn = norm(b_neural)
    js = norm(j_sym);     bs = norm(b_sym)
    jk = norm(j_km);      bk = norm(b_km)

    neural_ens   = jn + bn
    symbolic_ens = js + bs
    km_ens       = jk + bk
    tri_equal_ens = (neural_ens + symbolic_ens + km_ens) / 3.0

    # ── Tri-Fusion Sweep on Simplex (wn + ws + wk == 1.0) ─────────────────────
    tri_sweep = tri_fusion_sweep(neural_ens, symbolic_ens, km_ens, labels, step=0.05)
    best_wn, best_ws, best_wk, best_tri_acc = tri_sweep[0]
    best_tri_ens = best_wn * neural_ens + best_ws * symbolic_ens + best_wk * km_ens
    best_tri_tag = f"tri_ens({best_wn:.2f},{best_ws:.2f},{best_wk:.2f})"

    modes = {
        "joint_neural":   j_neural,
        "joint_sym":      j_sym,
        "joint_km":       j_km,
        "bone_neural":    b_neural,
        "bone_sym":       b_sym,
        "bone_km":        b_km,
        "neural_ens":     neural_ens,
        "symbolic_ens":   symbolic_ens,
        "km_ens":         km_ens,
        "tri_ens(equal)": tri_equal_ens,
        best_tri_tag:     best_tri_ens,
    }

    # ── Print overall summary ─────────────────────────────────────────────────
    output_lines = []
    def log(s=""):
        print(s)
        output_lines.append(s)

    sep = "=" * 70
    log(f"\n{sep}")
    log(f"  Joint + Bone Tri-Fusion Evaluation  (K={K})")
    log(f"  Build: '{args.rulebook_split}'  |  Eval: '{args.eval_split}'")
    log(sep)
    log(f"  {'Mode':<26}  {'Top-1 Acc':>10}")
    log(f"  {'-'*26}  {'-'*10}")
    for mode, logits in modes.items():
        a = acc(logits, labels)
        tag = ""
        if mode == "neural_ens":
            tag = "  <- Pure Neural Ensemble"
        elif mode == "symbolic_ens":
            tag = "  <- Pure Symbolic Ensemble"
        elif mode == "km_ens":
            tag = "  <- Pure KMeans Ensemble"
        elif mode == "tri_ens(equal)":
            tag = "  <- 1/3 Neural + 1/3 Sym + 1/3 KM"
        elif mode == best_tri_tag:
            tag = "  <- BEST Tri-Fusion"
        log(f"  {mode:<26}  {a*100:>9.2f}%{tag}")
    log(sep)

    # ── Tri-Fusion Sweep Top Results ──────────────────────────────────────────
    log(f"\n  Tri-Fusion Simplex Sweep (Top 10 configurations):")
    log(f"  {'Rank':<5}  {'W_neural':>8}  {'W_symbolic':>10}  {'W_kmeans':>8}  {'Top-1 Acc':>10}")
    log(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
    for rank, (wn, ws, wk, a) in enumerate(tri_sweep[:10], start=1):
        flag = "  <- BEST" if rank == 1 else ""
        log(f"  #{rank:<4}  {wn:>8.2f}  {ws:>10.2f}  {wk:>8.2f}  {a*100:>9.2f}%{flag}")

    # Specific reference anchors
    anchors = [
        ("Pure Neural",       1.0, 0.0, 0.0),
        ("Pure Symbolic",     0.0, 1.0, 0.0),
        ("Pure KMeans",       0.0, 0.0, 1.0),
        ("Neural + KM (50/50)",0.5, 0.0, 0.5),
        ("Neural + KM (10/90)",0.1, 0.0, 0.9),
        ("Equal Tri-Fusion",  0.35, 0.35, 0.30),
    ]
    log(f"\n  Key Fusion Reference Points:")
    log(f"  {'Configuration':<22}  {'W_neu':>6}  {'W_sym':>6}  {'W_km':>6}  {'Top-1 Acc':>10}")
    log(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}")
    for name, wn, ws, wk in anchors:
        fused = wn * neural_ens + ws * symbolic_ens + wk * km_ens
        a = acc(fused, labels)
        log(f"  {name:<22}  {wn:>6.2f}  {ws:>6.2f}  {wk:>6.2f}  {a*100:>9.2f}%")
    log(f"  {'Best Tri-Fusion':<22}  {best_wn:>6.2f}  {best_ws:>6.2f}  {best_wk:>6.2f}  {best_tri_acc*100:>9.2f}%")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    key_modes = {
        "neural_ens":   neural_ens,
        "symbolic_ens": symbolic_ens,
        "km_ens":       km_ens,
        "tri_best":     best_tri_ens,
    }
    log(f"\n  Per-class: Neural_ens vs Symbolic_ens vs KM_ens vs Best Tri-Fusion:")
    log(f"  {'Cls':<4}  {'Action':<24}  {'N_ens':>6}  {'Sym_ens':>7}  {'KM_ens':>7}  {'Tri_best':>8}  {'d(Tri-N)':>8}")
    log(f"  {'-'*4}  {'-'*24}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")
    pc = {m: per_class_acc(l, labels, args.num_class) for m, l in key_modes.items()}
    for cls in sorted(pc["neural_ens"]):
        name  = NTU60_NAMES[cls] if cls < len(NTU60_NAMES) else str(cls)
        n_acc = pc["neural_ens"].get(cls, 0)   * 100
        s_acc = pc["symbolic_ens"].get(cls, 0) * 100
        k_acc = pc["km_ens"].get(cls, 0)       * 100
        t_acc = pc["tri_best"].get(cls, 0)     * 100
        dtri  = t_acc - n_acc
        marker = " +" if dtri > 1.0 else (" --" if dtri < -2.0 else "")
        log(f"  {cls:>3}   {name:<24}  {n_acc:>5.1f}%  {s_acc:>6.1f}%  {k_acc:>6.1f}%  {t_acc:>7.1f}%  {dtri:>+7.1f}%{marker}")

    for m in ["neural_ens", "symbolic_ens", "km_ens", "tri_best"]:
        mean = np.mean(list(pc[m].values())) * 100
        log(f"  Mean [{m}]: {mean:.2f}%")
    log(f"\n{sep}\n")

    if getattr(args, "out_file", None):
        out_dir = os.path.dirname(os.path.abspath(args.out_file))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines) + "\n")
        print(f"  [Saved] Ensemble results saved to -> {args.out_file}\n")


if __name__ == "__main__":
    main()

"""
Joint Spatial-Temporal Concept Bottleneck with a Fuzzy AND/OR Logic Tree
=========================================================================
A Neuro-Symbolic Human Action Recognition Framework.

Architecture
------------
Input (B, C, T, V, M) -> SkateFormer Backbone -> Feature (B, feat_dim)
    |
    +---> 9 Concept Group Heads (Head, Hand, Arm, Hip, Leg, Foot, Temporal...) -> Concept Probs
    |         |
    |         +---> FuzzyLogicTree (real AND/OR/NOT gates, t-norm/t-conorm; each leaf's candidate
    |               concepts are restricted to this class's KG-nearest-neighbor ontology
    |               concepts) -> tree_logits (B, num_class)
    |
    +---> Direct Neural Action Classifier (Linear) -> neural_logits (B, num_class), kept only as a
          reference/upper-bound accuracy signal — NOT fused into the tree's prediction.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones import build_backbone, BaseBackboneAdapter
from model.backbones.skateformer_adapter import DEFAULT_SKATEFORMER_CFG
from knowledge_graph import load_concept_matrix, build_semantic_similarity_from_concepts
from fuzzy_logic_tree import FuzzyLogicTree

CONCEPTS_CSV = os.path.join(os.path.dirname(__file__), "concepts", "ntu_rgbd_60_120.csv")

GROUP_SPECS = [
    ("head", ("head_",)),
    ("hand", ("hand_", "interaction_two_person")),
    ("arm", ("arm_",)),
    ("hip", ("hip_",)),
    ("leg", ("leg_",)),
    ("foot", ("foot_",)),
    ("temporal_direction", ("temporal_direction_",)),
    ("temporal_sequence", ("temporal_sequence_",)),
    ("temporal_dynamics", ("temporal_dynamics_",)),
]


def build_concept_groups(concept_names: List[str]) -> Dict[str, List[int]]:
    """Partitions concept names into 9 semantically grouped anatomical & temporal regions."""
    groups = {name: [] for name, _ in GROUP_SPECS}
    used = set()
    for name, prefixes in GROUP_SPECS:
        for idx, concept in enumerate(concept_names):
            if idx in used:
                continue
            if any(concept.startswith(prefix) or concept == prefix for prefix in prefixes):
                groups[name].append(idx)
                used.add(idx)
    missing = [i for i in range(len(concept_names)) if i not in used]
    if missing:
        raise ValueError(f"Unassigned concepts: {[concept_names[i] for i in missing]}")
    if any(len(v) == 0 for v in groups.values()):
        raise ValueError(f"Empty concept group: {groups}")
    return groups


class JointSpatialTemporalCBM(nn.Module):
    """
    Joint Spatial-Temporal Concept Bottleneck + Fuzzy AND/OR Logic Tree.
    Uses SkateFormer as the default neural backbone.
    """

    def __init__(
        self,
        backbone: Union[str, BaseBackboneAdapter, nn.Module] = "skateformer",
        backbone_cfg: Optional[dict] = None,
        num_class: int = 60,
        concept_loss_weight: float = 0.5,
        concept_negative_weight: float = 0.10,
        concepts_csv: Optional[str] = None,
        tree_depth: int = 4,
        tree_entropy_weight: float = 0.01,
        tree_diversity_weight: float = 0.05,
        tree_kg_neighbor_k: int = 8,
        group_hidden_dim: int = 128,
        concept_collapse_weight: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        if isinstance(backbone, str):
            cfg = dict(backbone_cfg or {})
            self.backbone_adapter = build_backbone(backbone, backbone_cfg=cfg, num_class=num_class)
        elif isinstance(backbone, (BaseBackboneAdapter, nn.Module)):
            self.backbone_adapter = backbone
        else:
            raise ValueError(f"Invalid backbone parameter: {backbone}")

        if hasattr(self.backbone_adapter, "backbone"):
            self.backbone = self.backbone_adapter.backbone
        else:
            self.backbone = self.backbone_adapter

        if hasattr(self.backbone_adapter, "feat_dim"):
            self.feat_dim: int = self.backbone_adapter.feat_dim
        else:
            self.feat_dim: int = 192

        self.num_class = num_class
        self.concept_loss_weight = float(concept_loss_weight)
        self.concept_negative_weight = float(concept_negative_weight)
        self.concept_collapse_weight = float(concept_collapse_weight)

        # Load concept ontology from CSV
        csv_file = concepts_csv or CONCEPTS_CSV
        concept_matrix, concept_names = load_concept_matrix(csv_file, num_class)
        self.register_buffer("concept_matrix", concept_matrix)
        self.concept_names: List[str] = concept_names
        self.num_concepts: int = concept_matrix.shape[1]

        concept_pos = concept_matrix.float().sum(dim=0)
        concept_neg = float(num_class) - concept_pos
        pos_weight = (concept_neg / concept_pos.clamp_min(1.0)).clamp(1.0, 20.0)
        self.register_buffer("concept_pos_weight", pos_weight)

        # KG-constrained leaf candidates: each class's leaf-selector may only pick a
        # concept that is actually present in this class's own ontology row, or in
        # one of its `tree_kg_neighbor_k` nearest neighbors by ontology Jaccard
        # similarity -- narrows the free 74-way softmax per leaf down to a
        # semantically grounded pool (own concepts for positive literals, confusable
        # neighbors' concepts for meaningful NOT literals), instead of letting every
        # leaf pick from the full concept vocabulary regardless of relevance.
        S_KG = build_semantic_similarity_from_concepts(concept_matrix)
        k = min(int(tree_kg_neighbor_k), num_class)
        neighbor_idx = S_KG.topk(k, dim=-1).indices
        neighbor_mask = torch.zeros(num_class, num_class).scatter_(1, neighbor_idx, 1.0)
        candidate_mask = (neighbor_mask @ concept_matrix.float() > 0).float()

        pool_sizes = candidate_mask.sum(dim=-1)
        num_leaves = 2 ** tree_depth
        starved = int((pool_sizes < num_leaves).sum())
        print(
            f"[JOINT] Leaf candidate pool per class (KG-neighbor-k={k}): "
            f"min={int(pool_sizes.min())} mean={float(pool_sizes.mean()):.1f} max={int(pool_sizes.max())} "
            f"| num_leaves={num_leaves} | {starved}/{num_class} class(es) have fewer candidates than "
            f"leaves -- diversity regularizer cannot prevent duplication there, only depth or a "
            f"larger tree_kg_neighbor_k can"
        )

        # 9 Concept Groups
        self.group_order = [name for name, _ in GROUP_SPECS]
        self.concept_groups = build_concept_groups(self.concept_names)
        self.group_hidden_dim = int(group_hidden_dim)
        self.group_heads = nn.ModuleDict()

        for gi, name in enumerate(self.group_order):
            n = len(self.concept_groups[name])
            self.group_heads[name] = nn.Sequential(
                nn.Linear(self.feat_dim, self.group_hidden_dim),
                nn.GELU(),
                nn.Linear(self.group_hidden_dim, self.group_hidden_dim),
                nn.GELU(),
                nn.Linear(self.group_hidden_dim, n),
            )
            idx = torch.tensor(self.concept_groups[name], dtype=torch.long)
            self.register_buffer(f"concept_idx_{gi}", idx)

        self.fuzzy_tree = FuzzyLogicTree(
            num_concepts=self.num_concepts,
            num_class=num_class,
            depth=tree_depth,
            entropy_weight=tree_entropy_weight,
            diversity_weight=tree_diversity_weight,
            concept_candidate_mask=candidate_mask,
        )

        self.direct_action = nn.Linear(self.feat_dim, num_class)

    def extract_feat(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hasattr(self.backbone_adapter, "extract_feat"):
            return self.backbone_adapter.extract_feat(x, index_t)
        return self.backbone_adapter(x, index_t)

    def encode_group_concepts(self, feat: torch.Tensor):
        logits, probs = [], []
        for name in self.group_order:
            z_group = self.group_heads[name](feat)
            logits.append(z_group)
            probs.append(torch.sigmoid(z_group))
        return logits, probs

    @staticmethod
    def _concept_collapse_penalty(group_logits: List[torch.Tensor]) -> torch.Tensor:
        """Penalize concept heads whose predicted probability barely varies across
        the batch -- a head stuck near a constant offers no real per-sample signal,
        yet a collapsed-to-0 head still lets NOT-that-concept look "true" for every
        sample once fed into the fuzzy tree's product t-norm AND."""
        terms = []
        for z in group_logits:
            p = torch.sigmoid(z.float())
            var = p.var(dim=0, unbiased=False)
            # `var + eps`, not `clamp_min` -- clamp_min has zero gradient below the
            # floor, so once a head is already fully collapsed (var < eps) the
            # penalty would stop pushing on it entirely. The additive eps keeps
            # -1/(var+eps) finite and non-zero everywhere, including var == 0.
            terms.append((-torch.log(var + 1e-4)).mean())
        return torch.stack(terms).mean()

    def freeze_backbone(self) -> Tuple[int, int]:
        """Freezes all backbone parameters."""
        frozen, trainable = 0, 0
        target = getattr(self.backbone_adapter, "backbone", self.backbone_adapter)
        for p in target.parameters():
            p.requires_grad = False
            frozen += p.numel()
        return frozen, trainable

    def unfreeze_backbone(self) -> Tuple[int, int]:
        """Unfreezes all backbone parameters for end-to-end training."""
        frozen, trainable = 0, 0
        target = getattr(self.backbone_adapter, "backbone", self.backbone_adapter)
        for p in target.parameters():
            p.requires_grad = True
            trainable += p.numel()
        return frozen, trainable

    def forward(
        self,
        x: torch.Tensor,
        index_t: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        feat = self.extract_feat(x, index_t)
        group_logits, group_probs = self.encode_group_concepts(feat)
        concept_probs = torch.cat(group_probs, dim=-1)
        tree_logits = self.fuzzy_tree(concept_probs)

        if not return_loss or label is None:
            return tree_logits

        neural_logits = self.direct_action(feat)
        tree_total, tree_dict = self.fuzzy_tree.loss(concept_probs, label)

        # Weighted BCE concept loss
        concept_losses = []
        for gi, name in enumerate(self.group_order):
            idx = getattr(self, f"concept_idx_{gi}")
            target = self.concept_matrix[label][:, idx].float()
            pos_weight = self.concept_pos_weight[idx].to(group_logits[gi].device)
            bce = F.binary_cross_entropy_with_logits(
                group_logits[gi], target,
                pos_weight=pos_weight, reduction="none"
            )
            element_weight = target + (1.0 - target) * self.concept_negative_weight
            concept_losses.append((bce * element_weight).sum() / element_weight.sum().clamp_min(1.0))
        concept_bce = torch.stack(concept_losses).mean()
        collapse_penalty = self._concept_collapse_penalty(group_logits)

        neural_action_loss = F.cross_entropy(neural_logits, label)
        total = (
            tree_total
            + self.concept_loss_weight * concept_bce
            + self.concept_collapse_weight * collapse_penalty
            + 0.20 * neural_action_loss
        )
        return total, {
            "tree_ce": tree_dict["ce"].detach(),
            "tree_entropy": tree_dict["entropy"].detach(),
            "leaf_diversity": tree_dict["leaf_diversity"].detach(),
            "neural_action": neural_action_loss.detach(),
            "concept": concept_bce.detach(),
            "concept_collapse": collapse_penalty.detach(),
        }


def freeze_backbone(model: nn.Module) -> Tuple[int, int]:
    """Helper to freeze backbone parameters."""
    if hasattr(model, "freeze_backbone"):
        return model.freeze_backbone()
    frozen, trainable = 0, 0
    target = getattr(model, "backbone", model)
    for p in target.parameters():
        p.requires_grad = False
        frozen += p.numel()
    return frozen, trainable


def unfreeze_backbone(model: nn.Module) -> Tuple[int, int]:
    """Helper to unfreeze backbone parameters."""
    if hasattr(model, "unfreeze_backbone"):
        return model.unfreeze_backbone()
    frozen, trainable = 0, 0
    target = getattr(model, "backbone", model)
    for p in target.parameters():
        p.requires_grad = True
        trainable += p.numel()
    return frozen, trainable


def build_joint_model(
    checkpoint_path: Optional[str] = None,
    backbone_name: str = "skateformer",
    backbone_cfg: Optional[dict] = None,
    num_class: int = 60,
    concept_loss_weight: float = 0.5,
    concept_negative_weight: float = 0.10,
    device: torch.device = torch.device("cpu"),
    concept_checkpoint_path: Optional[str] = None,
    tree_depth: int = 4,
    tree_entropy_weight: float = 0.01,
    tree_diversity_weight: float = 0.05,
    tree_kg_neighbor_k: int = 8,
    group_hidden_dim: int = 128,
    concept_collapse_weight: float = 0.02,
    **kwargs,
) -> JointSpatialTemporalCBM:
    """
    Factory to construct JointSpatialTemporalCBM with SkateFormer and load pretrained weights.
    """
    backbone_cfg = dict(backbone_cfg or {})
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            raw = torch.load(checkpoint_path, map_location="cpu")
            state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
            state = {k.replace("module.", ""): v for k, v in state.items()}

            for key in ["head.weight", "backbone.head.weight", "fc.weight", "backbone.fc.weight", "direct_action.weight"]:
                if key in state:
                    detected = state[key].shape[0]
                    if detected != num_class:
                        print(f"[JOINT] Auto-detected num_class={detected} from checkpoint (was {num_class})")
                        num_class = detected
                    break
        except Exception as e:
            print(f"[JOINT][WARN] Could not inspect checkpoint for configuration: {e}")

    model = JointSpatialTemporalCBM(
        backbone=backbone_name,
        backbone_cfg=backbone_cfg,
        num_class=num_class,
        concept_loss_weight=concept_loss_weight,
        concept_negative_weight=concept_negative_weight,
        tree_depth=tree_depth,
        tree_entropy_weight=tree_entropy_weight,
        tree_diversity_weight=tree_diversity_weight,
        tree_kg_neighbor_k=tree_kg_neighbor_k,
        group_hidden_dim=group_hidden_dim,
        concept_collapse_weight=concept_collapse_weight,
    )

    # 1. Load Pretrained Backbone
    if checkpoint_path and os.path.exists(checkpoint_path):
        if hasattr(model.backbone_adapter, "load_pretrained"):
            missing, unexpected = model.backbone_adapter.load_pretrained(checkpoint_path, device=device)
            print(f"[JOINT] Backbone loaded <- {checkpoint_path}; missing={len(missing)}, unexpected={len(unexpected)}")

            # Warm-start direct_action from backbone classification head if matching shape
            cls_weights = model.backbone_adapter.get_classifier_weights()
            if cls_weights is not None:
                w, b = cls_weights
                if w is not None and w.shape == model.direct_action.weight.shape:
                    with torch.no_grad():
                        model.direct_action.weight.copy_(w)
                        if b is not None and model.direct_action.bias is not None:
                            model.direct_action.bias.copy_(b)
                    print(f"[JOINT] Successfully warm-started direct_action classifier weights from backbone head.")
    else:
        if checkpoint_path:
            print(f"[JOINT][WARN] Checkpoint path not found: {checkpoint_path}. Initializing backbone from scratch.")
        else:
            print(f"[JOINT] Initializing backbone from scratch.")

    return model.to(device)

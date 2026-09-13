"""
Joint Spatial-Temporal Concept Bottleneck with Symbolic Action Compatibility
============================================================================
A Neuro-Symbolic Human Action Recognition Framework.

Architecture
------------
Input (B, C, T, V, M) -> SkateFormer Backbone -> Feature (B, feat_dim)
    |
    +---> 9 Concept Group Heads (Head, Hand, Arm, Hip, Leg, Foot, Temporal...) -> Concept Probs
    |         |
    |         +---> Joint Symbolic Action Compatibility Scorer -> symbolic_logits (B, num_class)
    |
    +---> Direct Neural Action Classifier (Linear) -> neural_logits (B, num_class)

Fused Prediction: (1 - α) * neural_logits + α * symbolic_logits
"""

from __future__ import annotations
import os
import math
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.base_adapter import BaseBackboneAdapter
from model.backbones.skateformer_adapter import SkateFormerAdapter, DEFAULT_SKATEFORMER_CFG
from knowledge_graph import (
    load_concept_matrix,
    build_semantic_similarity_from_concepts,
    build_concept_coactivation_static,
)

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


class JointSymbolicActionCompatibility(nn.Module):
    """Differentiable action compatibility over grouped concept probabilities."""

    def __init__(
        self,
        concept_matrix: torch.Tensor,
        groups: Dict[str, List[int]],
        group_order: List[str],
        temperature: float = 0.7,
        margin: float = 0.15,
    ):
        super().__init__()
        self.num_class = concept_matrix.size(0)
        self.group_order = group_order
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.register_buffer("concept_matrix", concept_matrix.float())

        for gi, name in enumerate(group_order):
            idx = torch.tensor(groups[name], dtype=torch.long)
            self.register_buffer(f"indices_{gi}", idx)
            template = concept_matrix[:, idx].float()
            self.register_buffer(f"template_{gi}", template)
            self.register_buffer(f"positive_count_{gi}", template.sum(-1).clamp_min(1.0))
            self.register_buffer(f"active_{gi}", (template.sum(-1) > 0).float())

        self.group_logit = nn.Parameter(torch.zeros(len(group_order)))
        self.class_bias = nn.Parameter(torch.zeros(self.num_class))

    def _group_tensors(self, probs: List[torch.Tensor], gi: int):
        template = getattr(self, f"template_{gi}")
        count = getattr(self, f"positive_count_{gi}")
        p = probs[gi]
        score = (p[:, None, :] * template[None]).sum(-1) / count[None]
        active = getattr(self, f"active_{gi}")
        return score, active

    def forward(self, probs: List[torch.Tensor]) -> torch.Tensor:
        scores, active = [], []
        for gi in range(len(self.group_order)):
            s, m = self._group_tensors(probs, gi)
            scores.append(s)
            active.append(m)
        score = torch.stack(scores, dim=-1)
        active = torch.stack(active, dim=-1)
        weights = torch.softmax(self.group_logit, dim=0)
        weighted_active = active * weights[None, None]
        score = (score * weighted_active).sum(-1) / weighted_active.sum(-1).clamp_min(1e-6)
        return score / max(self.temperature, 1e-4) + self.class_bias

    def symbolic_loss(
        self,
        probs: List[torch.Tensor],
        labels: torch.Tensor,
        logits: Optional[List[torch.Tensor]] = None,
        selection_weight: float = 0.01,
        margin_weight: float = 0.10,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        score_parts, active_parts = [], []
        for gi in range(len(self.group_order)):
            s, m = self._group_tensors(probs, gi)
            score_parts.append(s)
            active_parts.append(m)
        all_scores = torch.stack(score_parts, dim=-1)
        active = torch.stack(active_parts, dim=-1)
        weights = torch.softmax(self.group_logit, dim=0)
        weighted_active = active * weights[None, None]
        class_scores = (all_scores * weighted_active).sum(-1) / weighted_active.sum(-1).clamp_min(1e-6)
        target_score = class_scores.gather(1, labels[:, None]).squeeze(1)

        positive_loss = 1.0 - target_score.clamp(0.0, 1.0).mean()
        negative_scores = class_scores.scatter(1, labels[:, None], -1e4)
        strongest_negative = negative_scores.max(dim=1).values
        ranking_loss = F.relu(self.margin - target_score + strongest_negative).mean()

        entropy_terms = []
        for gi, p in enumerate(probs):
            z = logits[gi].float() if logits is not None else torch.logit(
                p.float().clamp(1e-4, 1.0 - 1e-4)
            )
            q = torch.softmax(z / max(self.temperature, 1e-4), dim=-1)
            q = torch.nan_to_num(q, nan=0.0, posinf=1.0, neginf=0.0)
            entropy_terms.append((-(q * q.clamp_min(1e-8).log()).sum(-1) /
                                  math.log(max(q.size(-1), 2))).mean())
        entropy = torch.stack(entropy_terms).mean()
        total = positive_loss + margin_weight * ranking_loss + selection_weight * entropy
        return total, {
            "positive": positive_loss.detach(),
            "ranking": ranking_loss.detach(),
            "selection": entropy.detach(),
        }

    @torch.no_grad()
    def explain(
        self,
        probs: List[torch.Tensor],
        sample_idx: int,
        action_idx: int,
        concept_names: List[str],
        k_groups: int = 9,
    ) -> List[dict]:
        out = []
        for gi, group_name in enumerate(self.group_order[:k_groups]):
            p = probs[gi][sample_idx]
            indices = getattr(self, f"indices_{gi}")
            names = [concept_names[i] for i in indices.tolist()]
            vals = p
            target_template = getattr(self, f"template_{gi}")[action_idx]
            if not bool((target_template > 0).any()):
                out.append({
                    "group": group_name,
                    "positive": "NONE",
                    "positive_score": round(float(1.0 - vals.max()), 6),
                    "negative": None,
                    "negative_score": None,
                })
                continue
            pos_idx = int(vals.argmax())
            target = target_template > 0
            competitor = getattr(self, f"template_{gi}").sum(0) > 0
            neg_candidates = competitor & ~target
            if bool(neg_candidates.any()):
                neg_vals = vals.masked_fill(~neg_candidates, 1.0)
                neg_idx = int(neg_vals.argmin())
                negative = names[neg_idx]
                negative_score = float(1.0 - vals[neg_idx])
            else:
                negative, negative_score = None, None
            out.append({
                "group": group_name,
                "positive": names[pos_idx],
                "positive_score": round(float(vals[pos_idx]), 6),
                "negative": negative,
                "negative_score": negative_score,
            })
        return out


class JointSpatialTemporalCBM(nn.Module):
    """
    Joint Spatial-Temporal Concept Bottleneck + Symbolic Compatibility Model.
    Uses SkateFormer as the default neural backbone.
    """

    def __init__(
        self,
        backbone: Union[str, BaseBackboneAdapter, nn.Module] = "skateformer",
        backbone_cfg: Optional[dict] = None,
        num_class: int = 60,
        concept_loss_weight: float = 0.5,
        concept_negative_weight: float = 0.10,
        kg_s_weight: float = 0.05,
        kg_c_weight: float = 0.05,
        fusion_alpha: float = 0.30,
        symbolic_temperature: float = 0.7,
        symbolic_margin: float = 0.15,
        concepts_csv: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(backbone, str):
            cfg = dict(backbone_cfg or {})
            cfg["num_classes"] = num_class
            self.backbone_adapter = SkateFormerAdapter(**cfg)
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
        self.kg_s_weight = float(kg_s_weight)
        self.kg_c_weight = float(kg_c_weight)
        self.fusion_alpha = float(fusion_alpha)

        # Load knowledge graph buffers from CSV
        csv_file = concepts_csv or CONCEPTS_CSV
        concept_matrix, concept_names = load_concept_matrix(csv_file, num_class)
        self.register_buffer("concept_matrix", concept_matrix)
        self.concept_names: List[str] = concept_names
        self.num_concepts: int = concept_matrix.shape[1]

        S_KG = build_semantic_similarity_from_concepts(concept_matrix)
        C_KG = build_concept_coactivation_static(concept_matrix)
        self.register_buffer("S_KG_static", S_KG)
        self.register_buffer("C_KG_static", C_KG)

        concept_pos = concept_matrix.float().sum(dim=0)
        concept_neg = float(num_class) - concept_pos
        pos_weight = (concept_neg / concept_pos.clamp_min(1.0)).clamp(1.0, 20.0)
        self.register_buffer("concept_pos_weight", pos_weight)

        # 9 Concept Groups
        self.group_order = [name for name, _ in GROUP_SPECS]
        self.concept_groups = build_concept_groups(self.concept_names)
        self.group_hidden_dim = 64
        self.group_heads = nn.ModuleDict()

        for name in self.group_order:
            n = len(self.concept_groups[name])
            self.group_heads[name] = nn.Sequential(
                nn.Linear(self.feat_dim, self.group_hidden_dim),
                nn.GELU(),
                nn.Linear(self.group_hidden_dim, n),
            )

        self.symbolic_action = JointSymbolicActionCompatibility(
            self.concept_matrix,
            self.concept_groups,
            self.group_order,
            temperature=symbolic_temperature,
            margin=symbolic_margin,
        )

        self.direct_action = nn.Linear(self.feat_dim, num_class)

    def extract_feat(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hasattr(self.backbone_adapter, "extract_feat"):
            return self.backbone_adapter.extract_feat(x, index_t)
        return self.backbone_adapter(x, index_t)

    def encode_group_concepts(self, feat: torch.Tensor):
        logits, probs = [], []
        for name in self.group_order:
            z = self.group_heads[name](feat)
            logits.append(z)
            probs.append(torch.sigmoid(z))
        return logits, probs

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
        fusion_alpha: Optional[float] = None,
    ):
        feat = self.extract_feat(x, index_t)
        group_logits, group_probs = self.encode_group_concepts(feat)
        symbolic_logits = self.symbolic_action(group_probs)
        neural_logits = self.direct_action(feat)

        alpha = self.fusion_alpha if fusion_alpha is None else float(fusion_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        action_logits = (1.0 - alpha) * neural_logits + alpha * symbolic_logits

        if not return_loss or label is None:
            return action_logits

        symbolic, symbolic_dict = self.symbolic_action.symbolic_loss(
            group_probs, label, logits=group_logits
        )

        # S_KG loss
        s = self.S_KG_static.to(action_logits.device).float()
        s_target = s[label]
        s_target = s_target / s_target.sum(-1, keepdim=True).clamp_min(1e-6)
        loss_s = F.kl_div(
            F.log_softmax(action_logits.float(), dim=-1),
            s_target,
            reduction="batchmean",
        )

        # C_KG loss
        p_all = torch.cat([v.float() for v in group_probs], dim=-1)
        centered = p_all - p_all.mean(dim=0, keepdim=True)
        norm = torch.sqrt((centered.square()).sum(dim=0) + 1e-3)
        pred_c = (centered.transpose(0, 1) @ centered) / (norm[:, None] * norm[None, :])
        pred_c = ((pred_c.clamp(-1.0, 1.0) + 1.0) * 0.5).float()
        pred_c = torch.nan_to_num(pred_c, nan=0.5, posinf=1.0, neginf=0.0)
        target_c = self.C_KG_static.to(p_all.device).float()
        loss_c = F.smooth_l1_loss(pred_c, target_c)
        kg = self.kg_s_weight * loss_s + self.kg_c_weight * loss_c

        # Weighted BCE concept loss
        concept_losses = []
        for gi, name in enumerate(self.group_order):
            idx = getattr(self.symbolic_action, f"indices_{gi}")
            target = self.symbolic_action.concept_matrix[label][:, idx]
            pos_weight = self.concept_pos_weight[idx].to(group_logits[gi].device)
            bce = F.binary_cross_entropy_with_logits(
                group_logits[gi], target,
                pos_weight=pos_weight, reduction="none"
            )
            element_weight = target + (1.0 - target) * self.concept_negative_weight
            concept_losses.append((bce * element_weight).sum() / element_weight.sum().clamp_min(1.0))
        concept_bce = torch.stack(concept_losses).mean()

        action_loss = F.cross_entropy(action_logits, label)
        neural_action_loss = F.cross_entropy(neural_logits, label)
        total = (
            action_loss
            + 0.20 * neural_action_loss
            + self.concept_loss_weight * concept_bce
            + symbolic
            + kg
        )
        return total, {
            "action": action_loss.detach(),
            "action_raw": action_loss,
            "neural_action": neural_action_loss.detach(),
            "symbolic": symbolic.detach(),
            "symbolic_raw": symbolic,
            "concept": concept_bce.detach(),
            "kg": kg.detach(),
            "fusion_alpha": torch.tensor(alpha, device=action_logits.device),
            "kg_s": loss_s.detach(),
            "kg_c": loss_c.detach(),
            **symbolic_dict,
        }

    @torch.no_grad()
    def explain_joint(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None, action_idx: int = 0):
        feat = self.extract_feat(x, index_t)
        _, probs = self.encode_group_concepts(feat)
        return self.symbolic_action.explain(probs, 0, action_idx, self.concept_names)


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
    symbolic_temperature: float = 0.7,
    symbolic_margin: float = 0.15,
    concept_negative_weight: float = 0.10,
    device: torch.device = torch.device("cpu"),
    concept_checkpoint_path: Optional[str] = None,
    kg_s_weight: float = 0.05,
    kg_c_weight: float = 0.05,
    fusion_alpha: float = 0.30,
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
        kg_s_weight=kg_s_weight,
        kg_c_weight=kg_c_weight,
        fusion_alpha=fusion_alpha,
        symbolic_temperature=symbolic_temperature,
        symbolic_margin=symbolic_margin,
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

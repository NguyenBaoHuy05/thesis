"""
Joint Spatial-Temporal Concept Bottleneck with Auxiliary KG Branch (AuxKGBranch)
================================================================================
A Neuro-Symbolic Human Action Recognition Framework.

Architecture
------------
Input (B, C, T, V, M) -> SkateFormer Backbone -> Feature (B, feat_dim)
    |
    +---> Direct Neural Action Classifier (Linear) -> action_logits (B, num_class)   [Inference Path]
    |
    [Training Only Regularization Path]:
    +---> 9 Concept Group Heads (Head, Hand, Arm, Hip, Leg, Foot, Temporal...) -> group_probs
              |
              +---> AuxKGBranch (9 x KGUnionLayer AND/OR) -> aux_logits (B, num_class)
              +---> Concept Supervised BCE Loss (from CSV ground truth)
              +---> S_KG Loss (Semantic similarity) & C_KG Loss (Co-activation)

Inference: Pure neural forward path via direct_action(feat) -> zero symbolic overhead.
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


def _logit(probability: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Stable logit function mapping probabilities in (0, 1) to (-inf, +inf)."""
    probability = probability.clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)


class KGUnionLayer(nn.Module):
    """
    Differentiable conjunction (AND) / disjunction (OR) rule bank.
    Weights are stored in logit space and constrained to (0, 1) via sigmoid.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        prior: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if prior is None:
            prior = torch.full((input_dim, output_dim), 0.05)
            diag_len = min(input_dim, output_dim)
            prior[torch.arange(diag_len), torch.arange(diag_len)] = 0.9

        prior = prior.float().clamp(1e-4, 1.0 - 1e-4)
        self.register_buffer("prior", prior.clone())

        noise = 0.02 * torch.randn_like(prior)
        self.and_weight = nn.Parameter(_logit((prior + noise).clamp(1e-4, 1.0 - 1e-4)))
        self.or_weight = nn.Parameter(_logit((prior + noise).clamp(1e-4, 1.0 - 1e-4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying soft AND and OR operations.

        Args:
            x: Input literals [B, input_dim] in [0, 1].

        Returns:
            Concatenated [conjunction, disjunction] tensor of shape [B, 2 * output_dim].
        """
        and_weight = torch.sigmoid(self.and_weight)  # [input_dim, output_dim]
        or_weight = torch.sigmoid(self.or_weight)    # [input_dim, output_dim]

        # Soft conjunction (AND): prod(1 - w_i * (1 - x_i))
        conjunction = torch.prod(
            1.0 - and_weight.unsqueeze(0) * (1.0 - x.unsqueeze(-1)), dim=1
        )
        # Soft disjunction (OR): 1 - prod(1 - w_i * x_i)
        disjunction = 1.0 - torch.prod(
            1.0 - or_weight.unsqueeze(0) * x.unsqueeze(-1), dim=1
        )
        return torch.cat([conjunction, disjunction], dim=1)  # [B, 2 * output_dim]

    def weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.and_weight), torch.sigmoid(self.or_weight)


class AuxKGBranch(nn.Module):
    """
    9 x KGUnionLayer per concept group — used in training only.
    Forces concept heads to learn AND/OR-composable representations via multi-task gradient.
    Completely dropped at inference -> zero symbolic overhead.
    """

    def __init__(
        self,
        concept_groups: Dict[str, List[int]],
        group_order: List[str],
        num_class: int,
        l_small: int = 16,
    ):
        super().__init__()
        self.concept_groups = concept_groups
        self.group_order = group_order
        self.num_class = num_class
        self.l_small = l_small

        self.layers = nn.ModuleDict()
        total_out = 0
        for name in group_order:
            n = len(concept_groups[name])
            self.layers[name] = KGUnionLayer(input_dim=2 * n, output_dim=l_small)
            total_out += 2 * l_small

        self.aux_head = nn.Linear(total_out, num_class)

    def forward(self, group_probs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            group_probs: List of probabilities for each group, each shape [B, n_i] in [0, 1].
        Returns:
            aux_logits: [B, num_class]
        """
        out = []
        for i, name in enumerate(self.group_order):
            p = group_probs[i]
            lit = torch.cat([p, 1.0 - p], dim=-1)  # [B, 2 * n_i]
            out.append(self.layers[name](lit))      # [B, 2 * l_small]
        return self.aux_head(torch.cat(out, dim=-1))  # [B, num_class]


# ============================================================================
# [DEPRECATED / COMMENTED OUT] JointSymbolicActionCompatibility
# Retained for archival reference; replaced by AuxKGBranch training regularizer.
# ============================================================================
# class JointSymbolicActionCompatibility(nn.Module):
#     """Differentiable action compatibility over grouped concept probabilities."""
#
#     def __init__(
#         self,
#         concept_matrix: torch.Tensor,
#         groups: Dict[str, List[int]],
#         group_order: List[str],
#         temperature: float = 0.7,
#         margin: float = 0.15,
#     ):
#         super().__init__()
#         self.num_class = concept_matrix.size(0)
#         self.group_order = group_order
#         self.temperature = float(temperature)
#         self.margin = float(margin)
#         self.register_buffer("concept_matrix", concept_matrix.float())
#
#         for gi, name in enumerate(group_order):
#             idx = torch.tensor(groups[name], dtype=torch.long)
#             self.register_buffer(f"indices_{gi}", idx)
#             template = concept_matrix[:, idx].float()
#             self.register_buffer(f"template_{gi}", template)
#             self.register_buffer(f"positive_count_{gi}", template.sum(-1).clamp_min(1.0))
#             self.register_buffer(f"active_{gi}", (template.sum(-1) > 0).float())
#
#         self.group_logit = nn.Parameter(torch.zeros(len(group_order)))
#         self.class_bias = nn.Parameter(torch.zeros(self.num_class))
#
#     def _group_tensors(self, probs: List[torch.Tensor], gi: int):
#         template = getattr(self, f"template_{gi}")
#         count = getattr(self, f"positive_count_{gi}")
#         p = probs[gi]
#         score = (p[:, None, :] * template[None]).sum(-1) / count[None]
#         active = getattr(self, f"active_{gi}")
#         return score, active
#
#     def forward(self, probs: List[torch.Tensor]) -> torch.Tensor:
#         scores, active = [], []
#         for gi in range(len(self.group_order)):
#             s, m = self._group_tensors(probs, gi)
#             scores.append(s)
#             active.append(m)
#         score = torch.stack(scores, dim=-1)
#         active = torch.stack(active, dim=-1)
#         weights = torch.softmax(self.group_logit, dim=0)
#         weighted_active = active * weights[None, None]
#         score = (score * weighted_active).sum(-1) / weighted_active.sum(-1).clamp_min(1e-6)
#         return score / max(self.temperature, 1e-4) + self.class_bias
#
#     def symbolic_loss(
#         self,
#         probs: List[torch.Tensor],
#         labels: torch.Tensor,
#         logits: Optional[List[torch.Tensor]] = None,
#         selection_weight: float = 0.01,
#         margin_weight: float = 0.10,
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         score_parts, active_parts = [], []
#         for gi in range(len(self.group_order)):
#             s, m = self._group_tensors(probs, gi)
#             score_parts.append(s)
#             active_parts.append(m)
#         all_scores = torch.stack(score_parts, dim=-1)
#         active = torch.stack(active_parts, dim=-1)
#         weights = torch.softmax(self.group_logit, dim=0)
#         weighted_active = active * weights[None, None]
#         class_scores = (all_scores * weighted_active).sum(-1) / weighted_active.sum(-1).clamp_min(1e-6)
#         target_score = class_scores.gather(1, labels[:, None]).squeeze(1)
#
#         positive_loss = 1.0 - target_score.clamp(0.0, 1.0).mean()
#         negative_scores = class_scores.scatter(1, labels[:, None], -1e4)
#         strongest_negative = negative_scores.max(dim=1).values
#         ranking_loss = F.relu(self.margin - target_score + strongest_negative).mean()
#
#         entropy_terms = []
#         for gi, p in enumerate(probs):
#             z = logits[gi].float() if logits is not None else torch.logit(
#                 p.float().clamp(1e-4, 1.0 - 1e-4)
#             )
#             q = torch.softmax(z / max(self.temperature, 1e-4), dim=-1)
#             q = torch.nan_to_num(q, nan=0.0, posinf=1.0, neginf=0.0)
#             entropy_terms.append((-(q * q.clamp_min(1e-8).log()).sum(-1) /
#                                   math.log(max(q.size(-1), 2))).mean())
#         entropy = torch.stack(entropy_terms).mean()
#         total = positive_loss + margin_weight * ranking_loss + selection_weight * entropy
#         return total, {
#             "positive": positive_loss.detach(),
#             "ranking": ranking_loss.detach(),
#             "selection": entropy.detach(),
#         }


class JointSpatialTemporalCBM(nn.Module):
    """
    Joint Spatial-Temporal Concept Bottleneck with Auxiliary KG Branch.
    - Main branch: direct_action(feat) (pure neural, unchanged at inference)
    - Aux branch: AuxKGBranch(group_probs) (training regularizer only, zero overhead at inference)
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
        aux_kg_weight: float = 0.05,
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
        self.aux_kg_weight = float(aux_kg_weight)

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

        for gi, name in enumerate(self.group_order):
            n = len(self.concept_groups[name])
            self.group_heads[name] = nn.Sequential(
                nn.Linear(self.feat_dim, self.group_hidden_dim),
                nn.GELU(),
                nn.Linear(self.group_hidden_dim, n),
            )
            # Register group indices for fast indexing in BCE and explain
            idx = torch.tensor(self.concept_groups[name], dtype=torch.long)
            self.register_buffer(f"group_indices_{gi}", idx)

        # Auxiliary KG Branch (9 x KGUnionLayer per-group)
        self.aux_kg = AuxKGBranch(
            concept_groups=self.concept_groups,
            group_order=self.group_order,
            num_class=num_class,
            l_small=16,
        )

        # Main neural action classifier
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
        **kwargs,
    ):
        feat = self.extract_feat(x, index_t)
        action_logits = self.direct_action(feat)

        if not return_loss or label is None:
            return action_logits

        # Auxiliary path & Concept heads (only evaluated during training loss)
        group_logits, group_probs = self.encode_group_concepts(feat)
        aux_logits = self.aux_kg(group_probs)
        aux_loss = F.cross_entropy(aux_logits, label) * self.aux_kg_weight

        # S_KG loss (semantic similarity regularizer)
        s = self.S_KG_static.to(action_logits.device).float()
        s_target = s[label]
        s_target = s_target / s_target.sum(-1, keepdim=True).clamp_min(1e-6)
        loss_s = F.kl_div(
            F.log_softmax(action_logits.float(), dim=-1),
            s_target,
            reduction="batchmean",
        )

        # C_KG loss (concept co-activation regularizer)
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
            idx = getattr(self, f"group_indices_{gi}")
            target = self.concept_matrix[label][:, idx]
            pos_weight = self.concept_pos_weight[idx].to(group_logits[gi].device)
            bce = F.binary_cross_entropy_with_logits(
                group_logits[gi], target,
                pos_weight=pos_weight, reduction="none"
            )
            element_weight = target + (1.0 - target) * self.concept_negative_weight
            concept_losses.append((bce * element_weight).sum() / element_weight.sum().clamp_min(1.0))
        concept_bce = torch.stack(concept_losses).mean()

        action_loss = F.cross_entropy(action_logits, label)
        total = (
            action_loss
            + self.concept_loss_weight * concept_bce
            + kg
            + aux_loss
        )
        return total, {
            "action": action_loss.detach(),
            "action_raw": action_loss,
            "concept": concept_bce.detach(),
            "kg": kg.detach(),
            "kg_s": loss_s.detach(),
            "kg_c": loss_c.detach(),
            "aux_kg": aux_loss.detach(),
            "aux_kg_raw": aux_loss,
        }

    @torch.no_grad()
    def explain_joint(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None, action_idx: int = 0):
        feat = self.extract_feat(x, index_t)
        _, probs = self.encode_group_concepts(feat)
        out = []
        for gi, group_name in enumerate(self.group_order):
            p = probs[gi][0]
            idx = getattr(self, f"group_indices_{gi}")
            names = [self.concept_names[i] for i in idx.tolist()]
            vals = p
            template_gi = self.concept_matrix[:, idx].float()
            target_template = template_gi[action_idx]
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
            competitor = template_gi.sum(0) > 0
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
    aux_kg_weight: float = 0.05,
    device: torch.device = torch.device("cpu"),
    concept_checkpoint_path: Optional[str] = None,
    kg_s_weight: float = 0.05,
    kg_c_weight: float = 0.05,
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
        kg_s_weight=kg_s_weight,
        kg_c_weight=kg_c_weight,
        aux_kg_weight=aux_kg_weight,
        **kwargs,
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

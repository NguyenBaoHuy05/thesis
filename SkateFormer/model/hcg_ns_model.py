"""HCG-NS-SkateFormer-RF: relation-free hierarchical neuro-symbolic model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from model.SkateFormer import SkateFormer_
from model.concept_ontology import load_ontology_metadata
from model.hierarchical_concept_decoder import HierarchicalConceptDecoder
from model.hierarchical_symbolic_decoder import HierarchicalSymbolicDecoder
from model.kinematic_encoder import KinematicEncoder


class HCGNSSkateFormer(nn.Module):
    """SkateFormer backbone + temporal unary concepts + hierarchy + composition."""

    def __init__(
        self,
        backbone_args: dict[str, Any],
        concept_csv: str | Path,
        metadata_csv: str | Path,
        num_classes: int,
        labels_csv: str | Path | None = None,
        zero_is_negative: bool = False,
        num_concepts: int = 78,
        hard_negative_pairs: list[list[int]] | None = None,
        hard_negative_margin: float = 0.2,
        residual_action_head: bool = False,
        residual_action_weight: float = 0.1,
        prototype_args: dict[str, Any] | None = None,
        parent_weight: float = 0.35,
        gate_floor: float = 0.05,
    ):
        super().__init__()
        args = dict(backbone_args)
        args["num_classes"] = int(num_classes)
        self.backbone = SkateFormer_(**args)
        feature_dim = int(args.get("channels", (96, 192, 192, 192))[-1])
        self.feature_dim = feature_dim
        self.num_classes = int(num_classes)
        self.num_concepts = int(num_concepts)
        metadata = load_ontology_metadata(metadata_csv, num_concepts=num_concepts)
        prototype_args = dict(prototype_args or {})
        parent_enabled = bool(prototype_args.get("parent_enabled", True))
        all_pyramid_dims = tuple(args.get("channels", (96, 192, 192, 192)))
        pyramid_levels = tuple(prototype_args.get("pyramid_levels", (2, 3)))
        try:
            pyramid_dims = tuple(all_pyramid_dims[level] for level in pyramid_levels)
        except IndexError as exc:
            raise ValueError(
                f"prototype pyramid levels {pyramid_levels} exceed backbone stages"
            ) from exc
        self.concept_decoder = HierarchicalConceptDecoder(
            feature_dim=feature_dim,
            metadata=metadata,
            num_concepts=num_concepts,
            temperature=float(prototype_args.get("temperature", 0.07)),
            multi_scale=bool(prototype_args.get("multi_scale", False)),
            pyramid_dims=pyramid_dims,
            pyramid_levels=pyramid_levels,
            num_people=int(args.get("num_people", 1)),
            num_points=int(args.get("num_points", 0)) or None,
            interaction_reasoning=bool(prototype_args.get("interaction_reasoning", True)),
            gate_floor=float(gate_floor),
            parent_enabled=parent_enabled,
            region_pooling=bool(prototype_args.get("region_pooling", False)),
            region_tokens=prototype_args.get("region_tokens"),
            scope_fusion=str(prototype_args.get("scope_fusion", "fixed")),
            scope_scale=float(prototype_args.get("scope_scale", 0.25)),
        )
        self.kinematic = None
        if bool(prototype_args.get("motion_branch", False)):
            self.kinematic = KinematicEncoder(
                in_channels=int(args.get("in_channels", 3)),
                hidden_dim=int(prototype_args.get("motion_hidden_dim", 64)),
                out_dim=feature_dim,
            )
        self.symbolic = HierarchicalSymbolicDecoder(
            concept_csv=concept_csv,
            metadata_csv=metadata_csv,
            labels_csv=labels_csv,
            zero_is_negative=zero_is_negative,
            num_classes=num_classes,
            num_concepts=num_concepts,
            parent_weight=float(parent_weight),
            parent_enabled=parent_enabled,
        )
        normalized = set()
        for pair in hard_negative_pairs or []:
            if len(pair) != 2:
                raise ValueError(f"hard-negative pair must contain two ids: {pair}")
            first, second = sorted((int(pair[0]), int(pair[1])))
            if first == second or not (0 <= first < num_classes and 0 <= second < num_classes):
                raise ValueError(f"invalid hard-negative action pair: {pair}")
            normalized.add((first, second))
        self.hard_negative_pairs = tuple(sorted(normalized))
        self.hard_negative_margin = float(hard_negative_margin)
        self.residual_action_head_enabled = bool(residual_action_head)
        self.residual_action_weight = float(residual_action_weight)
        if self.residual_action_head_enabled:
            self.residual_action_head = nn.Linear(feature_dim, num_classes)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if not any(parameter.requires_grad for parameter in module.parameters()):
                    module.eval()
        return self

    def unfreeze_last_stage(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.stages[-1].parameters():
            parameter.requires_grad = True

    def hard_negative_loss(
        self, action_scores: torch.Tensor, action_labels: torch.Tensor,
    ) -> torch.Tensor:
        if not self.hard_negative_pairs:
            return action_scores.new_zeros(())
        pairs = torch.tensor(self.hard_negative_pairs, dtype=torch.long, device=action_scores.device)
        left, right = pairs[:, 0], pairs[:, 1]
        is_left = action_labels.unsqueeze(1).eq(left.unsqueeze(0))
        is_right = action_labels.unsqueeze(1).eq(right.unsqueeze(0))
        valid = is_left | is_right
        if not valid.any():
            return action_scores.new_zeros(())
        positive_score = torch.where(is_left, action_scores[:, left], action_scores[:, right])
        negative_score = torch.where(is_left, action_scores[:, right], action_scores[:, left])
        return F.relu(self.hard_negative_margin + negative_score - positive_score)[valid].mean()

    def forward(self, x: torch.Tensor, index_t: torch.Tensor | None = None) -> dict[str, Any]:
        feature_output = self.backbone.forward_features_raw(
            x, index_t, return_pyramid=self.concept_decoder.base.multi_scale
        )
        feature_map = feature_output[-1] if isinstance(feature_output, (list, tuple)) else feature_output
        if self.kinematic is not None:
            # Motion evidence is fused into the final-stage tokens only.
            feature_map = feature_map + self.kinematic(x, tuple(feature_map.shape[2:]))
            if isinstance(feature_output, (list, tuple)):
                feature_output = [*feature_output[:-1], feature_map]
            else:
                feature_output = feature_map
        concepts = self.concept_decoder(feature_output)
        action_scores = self.symbolic(concepts["sequence_probs"], concepts["parent_probs"])
        residual_scores = None
        if self.residual_action_head_enabled:
            residual_scores = self.residual_action_head(feature_map.mean(dim=(2, 3)))
            action_scores = action_scores + self.residual_action_weight * residual_scores
        return {
            "feature_map": feature_map,
            "action_scores": action_scores,
            "time_logits": concepts["time_logits"],
            "time_probs": concepts["time_probs"],
            "raw_sequence_probs": concepts["raw_sequence_probs"],
            "sequence_probs": concepts["sequence_probs"],
            "expected_time": concepts["expected_time"],
            "peak_time": concepts["peak_time"],
            "duration": concepts["duration"],
            "rising_activity": concepts["rising_activity"],
            "falling_activity": concepts["falling_activity"],
            "parent_logits": concepts["parent_logits"],
            "parent_probs": concepts["parent_probs"],
            "parent_for_child": concepts["parent_for_child"],
            "soft_gate": concepts["soft_gate"],
            "scope_logits": concepts["scope_logits"],
            "scope_probs": concepts["scope_probs"],
            "scope_term": concepts["scope_term"],
            "cosine_similarity": concepts["cosine_similarity"],
            "centers": concepts["centers"],
            "residual_action_scores": residual_scores,
        }

    def compute_loss(
        self, output: dict[str, Any], action_labels: torch.Tensor,
        weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        weights = {
            "parent": 0.25,
            "weak_concept": 0.25,
            "weak_positive": 1.0,
            "weak_negative": 0.0,
            "hierarchy": 0.10,
            "density": 0.05,
            "sparse": 0.001,
            "center": 0.05,
            "hard_negative": 0.25,
            **(weights or {}),
        }
        action_loss = F.cross_entropy(output["action_scores"], action_labels)
        hard_negative = self.hard_negative_loss(output["action_scores"], action_labels)
        concept_loss = self.symbolic.concept_loss(
            output["sequence_probs"], action_labels,
            positive_weight=float(weights.get("weak_positive", 1.0)),
            negative_weight=float(weights.get("weak_negative", 0.0)),
        )
        parent_loss = self.symbolic.parent_loss(output["parent_probs"], action_labels)
        hierarchy_loss = self.symbolic.hierarchy_loss(
            output["raw_sequence_probs"], output["parent_probs"]
        )
        density_loss = self.symbolic.density_loss(
            output["sequence_probs"], action_labels
        )
        sparse_loss = self.symbolic.sparsity_loss()
        center_loss = (
            self.concept_decoder.center_grounding_loss(
                output["cosine_similarity"], action_labels, self.symbolic.support
            ) + self.concept_decoder.center_separation_loss()
        )
        total = (
            action_loss
            + weights["hard_negative"] * hard_negative
            + weights["weak_concept"] * concept_loss
            + weights["parent"] * parent_loss
            + weights["hierarchy"] * hierarchy_loss
            + weights["density"] * density_loss
            + weights["sparse"] * sparse_loss
            + weights["center"] * center_loss
        )
        return {
            "total": total,
            "action": action_loss,
            "hard_negative": hard_negative,
            "weak_concept": concept_loss,
            "parent": parent_loss,
            "hierarchy": hierarchy_loss,
            "density": density_loss,
            "sparse": sparse_loss,
            "center": center_loss,
        }

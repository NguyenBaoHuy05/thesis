"""Unary hierarchical symbolic composition for HCG-NS SkateFormer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from model.concept_ontology import (
    build_parent_support,
    load_action_concept_annotations,
    load_action_concept_matrix_v2,
    load_ontology_metadata,
    build_parent_index,
)


class HierarchicalSymbolicDecoder(nn.Module):
    """Compose unary concept and parent evidence into action scores.

    The support matrix is a weak positive ontology. A zero in that matrix is
    unknown rather than a negative label; explicit negative states come only
    from the companion annotation file. Unknown entries do not contribute to
    the concept loss or to an action rule.
    """

    def __init__(
        self,
        concept_csv: str | Path,
        metadata_csv: str | Path,
        num_classes: int,
        labels_csv: str | Path | None = None,
        zero_is_negative: bool = False,
        num_concepts: int = 78,
        parent_weight: float = 0.35,
        parent_enabled: bool = True,
    ):
        super().__init__()
        metadata = load_ontology_metadata(metadata_csv, num_concepts=num_concepts)
        matrix, action_names, concept_names = load_action_concept_matrix_v2(
            concept_csv, num_classes=num_classes, num_concepts=num_concepts
        )
        metadata_names = [row["concept_name"] for row in metadata]
        if concept_names != metadata_names:
            raise ValueError(
                "action-concept header and ontology metadata use different concept ordering"
            )
        if labels_csv is None:
            positive_mask = matrix.clone()
            negative_mask = (1.0 - matrix) if zero_is_negative else torch.zeros_like(matrix)
            confidence = positive_mask.clone()
        else:
            positive_mask, negative_mask, confidence = load_action_concept_annotations(
                labels_csv,
                num_classes=num_classes,
                num_concepts=num_concepts,
                action_names=action_names,
                concept_names=concept_names,
                zero_is_negative=zero_is_negative,
            )
            if not torch.equal(positive_mask, matrix):
                raise ValueError(
                    "action-concept matrix and annotation positive states disagree"
                )
        parent_names, concept_to_parent, children, parent_levels = build_parent_index(metadata)
        parent_support = build_parent_support(positive_mask, concept_to_parent, len(parent_names))
        self.num_classes = int(num_classes)
        self.num_concepts = int(num_concepts)
        self.num_parents = len(parent_names)
        self.action_names = action_names
        self.concept_names = concept_names
        self.parent_names = tuple(parent_names)
        # Do not shadow nn.Module.children(), which PyTorch calls while moving
        # the model to a device or recursively applying transforms.
        self.children_by_parent = children
        self.parent_levels = parent_levels
        self.parent_weight = float(parent_weight)
        self.parent_enabled = bool(parent_enabled)
        self.register_buffer("support", positive_mask)
        self.register_buffer("negative_support", negative_mask)
        self.register_buffer("label_confidence", confidence)
        self.register_buffer("parent_support", parent_support)
        self.register_buffer("concept_to_parent", concept_to_parent)

        # Positive parameters are masked by weak support.  Initialising active
        # entries slightly above zero gives the symbolic path useful gradients
        # before the decoder has learned sharply separated concept evidence.
        self.concept_raw = nn.Parameter(torch.full((num_classes, num_concepts), -1.0))
        self.parent_raw = nn.Parameter(torch.full((num_classes, len(parent_names)), -1.0))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    @staticmethod
    def _positive(value: torch.Tensor) -> torch.Tensor:
        return F.softplus(value)

    def _weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        concept_weights = self._positive(self.concept_raw) * self.support
        parent_weights = self._positive(self.parent_raw) * self.parent_support
        return concept_weights, parent_weights

    def forward(
        self, sequence_probs: torch.Tensor, parent_probs: torch.Tensor
    ) -> torch.Tensor:
        if sequence_probs.ndim != 2 or sequence_probs.shape[-1] != self.num_concepts:
            raise ValueError("sequence_probs must have shape [B, num_concepts]")
        if parent_probs.ndim != 2 or parent_probs.shape[-1] != self.num_parents:
            raise ValueError("parent_probs must have shape [B, num_parents]")
        concept_weights, parent_weights = self._weights()
        concept_score = sequence_probs @ concept_weights.t()
        concept_score = concept_score / concept_weights.sum(dim=1).clamp_min(1e-6).unsqueeze(0)
        if self.parent_enabled and self.parent_weight != 0.0:
            parent_score = parent_probs @ parent_weights.t()
            parent_score = parent_score / parent_weights.sum(dim=1).clamp_min(1e-6).unsqueeze(0)
            parent_term = self.parent_weight * parent_score
        else:
            parent_term = torch.zeros_like(concept_score)
        return concept_score + parent_term + self.bias.unsqueeze(0)

    def concept_loss(
        self, sequence_probs: torch.Tensor, action_labels: torch.Tensor,
        positive_weight: float = 1.0, negative_weight: float = 0.0,
    ) -> torch.Tensor:
        positive = self.support[action_labels]
        negative = self.negative_support[action_labels]
        probs = sequence_probs.clamp(1e-6, 1.0 - 1e-6)
        loss = (
            -float(positive_weight) * positive * torch.log(probs)
            -float(negative_weight) * negative * torch.log1p(-probs)
        )
        normalizer = (
            float(positive_weight) * positive
            + float(negative_weight) * negative
        ).sum(dim=1).clamp_min(1.0)
        return (loss.sum(dim=1) / normalizer).mean()

    def positive_concept_loss(
        self, sequence_probs: torch.Tensor, action_labels: torch.Tensor,
        positive_weight: float = 1.0,
    ) -> torch.Tensor:
        return self.concept_loss(sequence_probs, action_labels, positive_weight, 0.0)

    def parent_loss(
        self, parent_probs: torch.Tensor, action_labels: torch.Tensor,
        positive_weight: float = 1.0,
    ) -> torch.Tensor:
        positive = self.parent_support[action_labels]
        probs = parent_probs.clamp(1e-6, 1.0 - 1e-6)
        loss = -float(positive_weight) * positive * torch.log(probs)
        return (loss.sum(dim=1) / positive.sum(dim=1).clamp_min(1.0)).mean()

    def hierarchy_loss(
        self, raw_sequence_probs: torch.Tensor, parent_probs: torch.Tensor,
    ) -> torch.Tensor:
        parent_for_child = parent_probs[:, self.concept_to_parent]
        return F.relu(raw_sequence_probs - parent_for_child).mean()

    def density_loss(
        self, sequence_probs: torch.Tensor, action_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Keep the number of activated concepts near the weak target budget."""
        target_count = self.support[action_labels].sum(dim=1)
        predicted_count = sequence_probs.sum(dim=1)
        scale = target_count.clamp_min(1.0)
        return F.smooth_l1_loss(predicted_count / scale, torch.ones_like(scale))

    def sparsity_loss(self) -> torch.Tensor:
        concept_weights, parent_weights = self._weights()

        def smooth_l0(value: torch.Tensor) -> torch.Tensor:
            return (value / (value + 0.1)).mean()

        parent_term = (
            0.25 * smooth_l0(parent_weights)
            if self.parent_enabled
            else concept_weights.new_zeros(())
        )
        return smooth_l0(concept_weights) + parent_term

    @torch.no_grad()
    def _rule_tensors(self) -> dict[str, torch.Tensor]:
        concept, parent = self._weights()
        return {"concept": concept, "parent": parent, "bias": self.bias}

    @torch.no_grad()
    def rule_statistics(self, threshold: float = 0.5) -> dict[str, float | int]:
        tensors = self._rule_tensors()
        active = torch.cat([tensors["concept"].flatten(), tensors["parent"].flatten()])
        return {
            "active_rule_count": int((active > threshold).sum().item()),
            "total_rule_count": int(active.numel()),
            "concept_rule_count": int((tensors["concept"] > threshold).sum().item()),
            "parent_rule_count": int((tensors["parent"] > threshold).sum().item()),
            "mean_active_concepts_per_action": float(
                (tensors["concept"] > threshold).sum(dim=1).float().mean().item()
            ),
        }

    @torch.no_grad()
    def export_rulebook(self, threshold: float = 0.5) -> dict[str, Any]:
        tensors = self._rule_tensors()
        rules = []
        for action_id, action_name in enumerate(self.action_names):
            concept_ids = torch.nonzero(
                tensors["concept"][action_id] > threshold, as_tuple=False
            ).flatten().tolist()
            parent_ids = torch.nonzero(
                tensors["parent"][action_id] > threshold, as_tuple=False
            ).flatten().tolist()
            rules.append({
                "action_id": action_id,
                "action_name": action_name,
                "concept_ids": concept_ids,
                "concept_names": [self.concept_names[i] for i in concept_ids],
                "parent_ids": parent_ids,
                "parent_names": [self.parent_names[i] for i in parent_ids],
            })
        return {
            "format_version": 2,
            "model": "HCG-NS-SkateFormer-RF",
            "relation_components": [],
            "num_classes": self.num_classes,
            "num_concepts": self.num_concepts,
            "num_parents": self.num_parents,
            "action_names": list(self.action_names),
            "concept_names": list(self.concept_names),
            "parent_names": list(self.parent_names),
            "coefficients": {"parent_weight": self.parent_weight},
            "rules": rules,
            "dense_weights": {
                "concept": tensors["concept"].cpu().tolist(),
                "parent": tensors["parent"].cpu().tolist(),
                "bias": tensors["bias"].cpu().tolist(),
            },
            "rule_statistics": self.rule_statistics(threshold=threshold),
        }

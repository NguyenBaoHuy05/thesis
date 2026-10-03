"""Unary hierarchical symbolic composition for HCG-NS SkateFormer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from model.concept_ontology import (
    build_parent_support,
    load_action_concept_matrix_v2,
    load_ontology_metadata,
    build_parent_index,
)


class HierarchicalSymbolicDecoder(nn.Module):
    """Compose unary concept evidence into action scores.

    The support matrix is a fully labelled binary ontology. Ones are positive
    concept targets and zeros are negative concept targets.
    Parent predictions are auxiliary supervision only and never contribute to
    the action score.
    """

    def __init__(
        self,
        concept_csv: str | Path,
        metadata_csv: str | Path,
        num_classes: int,
        num_concepts: int = 78,
        action_logit_scale: float = 1.0,
        concept_score_mode: str = 'probability',
        concept_idf_power: float = 0.0,
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
        positive_mask = matrix.clone()
        negative_mask = 1.0 - matrix
        parent_names, concept_to_parent, children, parent_levels = build_parent_index(metadata)
        parent_support = build_parent_support(positive_mask, concept_to_parent, len(parent_names))
        parent_negative_support = positive_mask.new_zeros(
            (positive_mask.shape[0], len(parent_names))
        )
        for parent_id in range(len(parent_names)):
            child_ids = torch.nonzero(
                concept_to_parent == parent_id, as_tuple=False
            ).flatten()
            if child_ids.numel():
                # A parent is negative only when every one of its children is
                # explicitly negative.
                parent_negative_support[:, parent_id] = negative_mask[:, child_ids].amin(dim=1)
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
        self.action_logit_scale = float(action_logit_scale)
        self.concept_score_mode = str(concept_score_mode)
        self.concept_idf_power = float(concept_idf_power)
        if self.action_logit_scale <= 0.0:
            raise ValueError("action_logit_scale must be positive")
        if self.concept_score_mode not in {"probability", "logit"}:
            raise ValueError("concept_score_mode must be probability or logit")
        if self.concept_idf_power < 0.0:
            raise ValueError("concept_idf_power must be non-negative")
        self.register_buffer("support", positive_mask)
        self.register_buffer("negative_support", negative_mask)
        self.register_buffer("parent_support", parent_support)
        self.register_buffer("parent_negative_support", parent_negative_support)
        self.register_buffer("concept_to_parent", concept_to_parent)
        concept_frequency = positive_mask.sum(dim=0)
        concept_idf = (((float(self.num_classes) + 1.0) / (concept_frequency + 1.0)).log() + 1.0)
        self.register_buffer("concept_idf", concept_idf.pow(self.concept_idf_power))

        # Positive parameters are masked by weak support.  Initialising active
        # entries slightly above zero gives the symbolic path useful gradients
        # before the decoder has learned sharply separated concept evidence.
        self.concept_raw = nn.Parameter(torch.full((num_classes, num_concepts), -1.0))

    @staticmethod
    def _positive(value: torch.Tensor) -> torch.Tensor:
        return F.softplus(value)

    def _weights(self) -> torch.Tensor:
        return self._positive(self.concept_raw) * self.support

    def _score_evidence(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Map every symbolic branch to the configured evidence domain."""
        if self.concept_score_mode == "logit":
            return torch.logit(probabilities.clamp(1e-4, 1.0 - 1e-4))
        return probabilities

    def forward(self, sequence_probs: torch.Tensor) -> torch.Tensor:
        if sequence_probs.ndim != 2 or sequence_probs.shape[-1] != self.num_concepts:
            raise ValueError("sequence_probs must have shape [B, num_concepts]")
        concept_weights = self._weights()
        concept_evidence = self._score_evidence(sequence_probs)
        score_weights = concept_weights * self.concept_idf.unsqueeze(0)
        concept_score = concept_evidence @ score_weights.t()
        concept_score = concept_score / score_weights.sum(dim=1).clamp_min(1e-6).unsqueeze(0)
        return self.action_logit_scale * concept_score

    def concept_loss(
        self, sequence_probs: torch.Tensor, action_labels: torch.Tensor,
    ) -> torch.Tensor:
        targets = self.support[action_labels]
        probs = sequence_probs.clamp(1e-6, 1.0 - 1e-6)
        return F.binary_cross_entropy(probs, targets)

    def parent_loss(
        self, parent_logits: torch.Tensor, action_labels: torch.Tensor,
        positive_weight: float = 1.0, negative_weight: float = 1.0,
    ) -> torch.Tensor:
        positive = self.parent_support[action_labels]
        negative = self.parent_negative_support[action_labels]
        # Parent supervision is multi-label, so BCE-with-logits is the correct
        # objective. Unknown parent states are masked out rather than treated
        # as negatives.
        positive_loss = F.softplus(-parent_logits) * positive
        negative_loss = F.softplus(parent_logits) * negative
        loss = (
            float(positive_weight) * positive_loss
            + float(negative_weight) * negative_loss
        )
        normalizer = (
            float(positive_weight) * positive
            + float(negative_weight) * negative
        ).sum(dim=1).clamp_min(1.0)
        return (loss.sum(dim=1) / normalizer).mean()

    @torch.no_grad()
    def _rule_tensors(self) -> dict[str, torch.Tensor]:
        return {"concept": self._weights()}

    @torch.no_grad()
    def rule_statistics(self, threshold: float = 0.5) -> dict[str, float | int]:
        tensors = self._rule_tensors()
        active = tensors["concept"].flatten()
        return {
            "active_rule_count": int((active > threshold).sum().item()),
            "total_rule_count": int(active.numel()),
            "concept_rule_count": int((tensors["concept"] > threshold).sum().item()),
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
            rules.append({
                "action_id": action_id,
                "action_name": action_name,
                "concept_ids": concept_ids,
                "concept_names": [self.concept_names[i] for i in concept_ids],
            })
        return {
            "format_version": 4,
            "model": "HCG-NS-SkateFormer-RF",
            "relation_components": [],
            "parent_role": "auxiliary_supervision",
            "num_classes": self.num_classes,
            "num_concepts": self.num_concepts,
            "num_parents": self.num_parents,
            "action_names": list(self.action_names),
            "concept_names": list(self.concept_names),
            "parent_names": list(self.parent_names),
            "coefficients": {
                "action_logit_scale": self.action_logit_scale,
                "concept_score_mode": self.concept_score_mode,
                "concept_idf_power": self.concept_idf_power,
            },
            "rules": rules,
            "dense_weights": {
                "concept": tensors["concept"].cpu().tolist(),
                "concept_idf": self.concept_idf.cpu().tolist(),
            },
            "rule_statistics": self.rule_statistics(threshold=threshold),
        }

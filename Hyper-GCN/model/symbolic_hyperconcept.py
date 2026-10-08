"""Symbolic body hypergraph to action-hyperconcept bridge for Hyper-GCN."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from model.concept_schema import (
    BODY_PART_JOINTS, PART_NAMES, concept_parts, load_concept_tables,
)


def _row_normalize(values):
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _body_part_incidence():
    matrix = torch.zeros(len(PART_NAMES), 25, dtype=torch.float32)
    for index, name in enumerate(PART_NAMES):
        matrix[index, list(BODY_PART_JOINTS[name])] = 1.0
    return _row_normalize(matrix)


def _concept_parts(name, scope, group):
    row = torch.zeros(len(PART_NAMES), dtype=torch.float32)
    ids = {part: index for index, part in enumerate(PART_NAMES)}
    for part in concept_parts(name, scope, group):
        row[ids[part]] = 1.0
    return row


def _load_concepts(concept_csv, metadata_csv, num_classes, num_concepts):
    values, metadata = load_concept_tables(
        concept_csv, metadata_csv, num_classes, num_concepts,
    )
    support = torch.tensor(values, dtype=torch.float32)
    part_map = torch.stack([
        _concept_parts(row["concept_name"], row.get("scope", ""), row.get("group", ""))
        for row in metadata
    ])
    return support, _row_normalize(part_map), _concept_affinity(support), metadata


def _concept_affinity(support):
    # Normalize each concept across classes before measuring co-occurrence.
    # Zero-support concepts have no off-diagonal edges and retain a self-loop.
    columns = F.normalize(support, p=2, dim=0)
    affinity = (columns.t() @ columns).clamp(0.0, 1.0)
    affinity.fill_diagonal_(0.0)
    affinity = affinity + torch.eye(support.shape[1], device=support.device, dtype=support.dtype)
    return _row_normalize(affinity)


class SymbolicHyperConceptBridge(nn.Module):
    """Parallel symbolic branch with learnable residual fusion."""

    def __init__(
        self, feature_dim, concept_csv, metadata_csv, num_classes,
        num_concepts=102, num_people=2, num_points=25, temperature=0.2,
        residual_init=0.0, graph_init=0.0,
        balance_loss=False, pos_weight_max=10.0, pos_weight_source="class",
        action_decoder="none", action_temperature=0.1,
    ):
        super().__init__()
        if int(num_points) != 25:
            raise ValueError("symbolic bridge currently supports NTU's 25 joints only")
        support, concept_part, concept_affinity, metadata = _load_concepts(
            concept_csv, metadata_csv, int(num_classes), int(num_concepts)
        )
        self.feature_dim = int(feature_dim)
        self.num_concepts = int(num_concepts)
        self.num_people = int(num_people)
        self.num_points = int(num_points)
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("symbolic temperature must be finite and positive")
        self.balance_loss = bool(balance_loss)
        self.pos_weight_max = float(pos_weight_max)
        self.pos_weight_source = pos_weight_source
        if not math.isfinite(self.pos_weight_max) or self.pos_weight_max < 1:
            raise ValueError("pos_weight_max must be finite and >= 1")
        if pos_weight_source not in {"class", "train"}:
            raise ValueError("pos_weight_source must be 'class' or 'train'")
        if action_decoder not in {"none", "prototype"}:
            raise ValueError("action_decoder must be 'none' or 'prototype'")
        self.action_decoder = action_decoder
        self.action_temperature = float(action_temperature)
        if not math.isfinite(self.action_temperature) or self.action_temperature <= 0:
            raise ValueError("symbolic action temperature must be finite and positive")
        self.metadata = metadata
        self.concept_names = [row["concept_name"] for row in metadata]
        # Rebuild CSV-derived buffers, rather than restoring stale mappings from weights.
        self.register_buffer("part_incidence", _body_part_incidence(), persistent=False)
        self.register_buffer("concept_part", concept_part, persistent=False)
        self.register_buffer("concept_affinity", concept_affinity, persistent=False)
        self.register_buffer("support", support, persistent=False)
        self.register_buffer("positive_weight", torch.ones(self.num_concepts), persistent=False)
        # Persist training frequencies so test mode does not need the training dataset.
        self.register_buffer("training_class_counts", torch.zeros(int(num_classes)))
        self._positive_weight_ready = not self.balance_loss or pos_weight_source == "class"
        if pos_weight_source == "class":
            self._set_positive_weight(torch.ones(int(num_classes)))
        self.concept_queries = nn.Parameter(torch.empty(self.num_concepts, self.feature_dim))
        self.concept_bias = nn.Parameter(torch.zeros(self.num_concepts))
        self.project_back = nn.Conv2d(self.feature_dim, self.feature_dim, 1, bias=False)
        self.context_norm = nn.BatchNorm2d(self.feature_dim)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init)))
        self.concept_graph_gate = nn.Parameter(torch.tensor(float(graph_init)))
        nn.init.normal_(self.concept_queries, mean=0.0, std=0.02)
        nn.init.kaiming_normal_(self.project_back.weight, mode="fan_out")
        nn.init.ones_(self.context_norm.weight)
        nn.init.zeros_(self.context_norm.bias)

    def forward(self, feature_map):
        expected_nodes = self.num_people * self.num_points
        if feature_map.ndim != 4 or feature_map.shape[-1] != expected_nodes:
            raise ValueError(
                "expected [N,C,T,num_people*num_points], got "
                f"{tuple(feature_map.shape)}"
            )
        batch, channels, timesteps, _ = feature_map.shape
        nodes = feature_map.reshape(
            batch, channels, timesteps, self.num_people, self.num_points
        ).mean(dim=3)
        part_features = torch.einsum(
            "pk,bctk->bctp", self.part_incidence.to(feature_map), nodes
        )
        concept_features = torch.einsum(
            "qp,bctp->bctq", self.concept_part.to(feature_map), part_features
        )
        concept_features = concept_features + torch.tanh(self.concept_graph_gate) * torch.einsum(
            "qr,bctr->bctq", self.concept_affinity.to(feature_map), concept_features
        )
        tokens = F.normalize(concept_features.permute(0, 2, 3, 1), dim=-1)
        queries = F.normalize(self.concept_queries, dim=-1)
        sequence_logits = torch.einsum("btqd,qd->btq", tokens, queries)
        sequence_logits = sequence_logits / max(self.temperature, 1e-6)
        sequence_logits = sequence_logits + self.concept_bias.view(1, 1, -1)
        fused = feature_map
        if self.action_decoder == "none":
            concept_context = torch.einsum(
                "qp,bctq->bctp", self.concept_part.to(feature_map), concept_features
            )
            joint_context = torch.einsum(
                "pk,bctp->bctk", self.part_incidence.to(feature_map), concept_context
            )
            joint_context = self.context_norm(self.project_back(joint_context))
            joint_context = joint_context.unsqueeze(3).expand(
                -1, -1, -1, self.num_people, -1
            ).contiguous().reshape(batch, channels, timesteps, expected_nodes)
            fused = feature_map + torch.tanh(self.residual_gate) * joint_context
        return fused, sequence_logits.mean(dim=1)

    def action_logits(self, concept_logits):
        """Score each action by balanced Bernoulli fit to its fixed concept template."""
        if self.action_decoder != "prototype":
            raise RuntimeError("symbolic action decoder is not enabled")
        if concept_logits.ndim != 2 or concept_logits.shape[1] != self.num_concepts:
            raise ValueError(
                f"expected concept logits [N,{self.num_concepts}], got {tuple(concept_logits.shape)}"
            )
        support = self.support.to(concept_logits)
        positive_count = support.sum(dim=1).clamp_min(1.0)
        negative_count = (1.0 - support).sum(dim=1).clamp_min(1.0)
        log_positive = F.logsigmoid(concept_logits).unsqueeze(1)
        log_negative = F.logsigmoid(-concept_logits).unsqueeze(1)
        positive_fit = (log_positive * support.unsqueeze(0)).sum(dim=2) / positive_count
        negative_fit = (log_negative * (1.0 - support).unsqueeze(0)).sum(dim=2) / negative_count
        # Equal positive/negative mass prevents sparse all-zero matches from dominating.
        return (0.5 * positive_fit + 0.5 * negative_fit) / self.action_temperature

    def trainable_parameters(self):
        """Parameters that can affect the selected inference path."""
        if self.action_decoder == "prototype":
            return (self.concept_queries, self.concept_bias, self.concept_graph_gate)
        return tuple(self.parameters())

    def loss(self, concept_logits, labels):
        if self.balance_loss and not self._positive_weight_ready:
            raise RuntimeError("training class counts are required for balanced concept loss")
        targets = self.support.to(
            device=concept_logits.device, dtype=concept_logits.dtype
        ).index_select(0, labels)
        positive_weight = self.positive_weight.to(concept_logits) if self.balance_loss else None
        return F.binary_cross_entropy_with_logits(
            concept_logits, targets, pos_weight=positive_weight,
        )

    def _set_positive_weight(self, class_counts):
        positives = class_counts.to(self.support) @ self.support
        negatives = class_counts.sum().to(positives) - positives
        weights = (negatives / positives.clamp_min(1.0)).clamp(1.0, self.pos_weight_max)
        # No positive examples: leave negative-only supervision unweighted.
        weights = torch.where(positives > 0, weights, torch.ones_like(weights))
        self.positive_weight.copy_(weights)
        self._positive_weight_ready = True

    def set_training_class_counts(self, class_counts):
        counts = torch.as_tensor(class_counts, device=self.support.device, dtype=torch.float32)
        if (counts.shape != self.training_class_counts.shape or not torch.isfinite(counts).all()
                or (counts < 0).any() or (counts != counts.round()).any() or counts.sum() <= 0):
            raise ValueError("training class counts must be a nonempty, nonnegative integer vector")
        self.training_class_counts.copy_(counts)
        if self.pos_weight_source == "train":
            self._set_positive_weight(counts)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )
        if self.pos_weight_source == "train" and self.training_class_counts.sum() > 0:
            self.set_training_class_counts(self.training_class_counts)

    def audit(self):
        """Serializable schema, mapping and loss settings for each experiment."""
        training_positives = (self.training_class_counts.to(self.support) @ self.support).detach().cpu().tolist()
        return {
            "num_classes": int(self.support.shape[0]),
            "num_concepts": self.num_concepts,
            "target_granularity": "action-class templates, not per-video annotations",
            "affinity": "column cosine similarity + unit self-loops + row normalization",
            "balance_loss": self.balance_loss,
            "pos_weight_source": self.pos_weight_source,
            "pos_weight_max": self.pos_weight_max,
            "pos_weight_ready": self._positive_weight_ready,
            "action_decoder": self.action_decoder,
            "action_temperature": self.action_temperature,
            "training_class_counts": self.training_class_counts.detach().cpu().tolist(),
            "body_parts_zero_based_joints": BODY_PART_JOINTS,
            "concepts": [
                {
                    "name": row["concept_name"], "scope": row["scope"],
                    "parts": concept_parts(row["concept_name"], row["scope"], row["group"]),
                    "positive_classes": int(self.support[:, index].sum().item()),
                    "positive_training_samples": int(training_positives[index]),
                    "pos_weight": float(self.positive_weight[index].item()) if self.balance_loss else 1.0,
                }
                for index, row in enumerate(self.metadata)
            ],
        }


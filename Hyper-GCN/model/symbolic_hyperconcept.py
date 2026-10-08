"""Symbolic body hypergraph to action-hyperconcept bridge for Hyper-GCN."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


PART_NAMES = (
    "head", "torso", "left_arm", "right_arm", "left_hand", "right_hand",
    "left_leg", "right_leg", "left_foot", "right_foot", "full_body",
)


def _row_normalize(values):
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1.0)


def _body_part_incidence():
    groups = {
        "head": [3], "torso": [0, 1, 2, 20],
        "left_arm": [4, 5, 6], "right_arm": [8, 9, 10],
        "left_hand": [7, 21, 22], "right_hand": [11, 23, 24],
        "left_leg": [12, 13, 14], "right_leg": [16, 17, 18],
        "left_foot": [15], "right_foot": [19],
        "full_body": list(range(25)),
    }
    matrix = torch.zeros(len(PART_NAMES), 25, dtype=torch.float32)
    for index, name in enumerate(PART_NAMES):
        matrix[index, groups[name]] = 1.0
    return _row_normalize(matrix)


def _concept_parts(name, scope, group):
    name = name.strip().lower()
    row = torch.zeros(len(PART_NAMES), dtype=torch.float32)
    ids = {part: index for index, part in enumerate(PART_NAMES)}

    def add(*parts):
        for part in parts:
            row[ids[part]] = 1.0

    if name.startswith("head_"):
        add("head")
    elif name.startswith("hand_"):
        add("left_hand", "right_hand")
        if any(token in name for token in ("head_region", "face_region", "mouth_region")):
            add("head")
        if any(token in name for token in ("chest_region", "neck_region", "back_region")):
            add("torso")
    elif name.startswith("arm_"):
        add("left_arm", "right_arm")
        if "cross" in name or "chest" in name:
            add("torso")
    elif name.startswith("upper_limb_"):
        add("left_arm", "right_arm", "left_hand", "right_hand")
        if "head_region" in name:
            add("head")
    elif name.startswith("hip_"):
        add("torso", "left_leg", "right_leg")
    elif name.startswith("leg_"):
        add("left_leg", "right_leg")
    elif name.startswith("foot_"):
        add("left_foot", "right_foot")
    elif scope in {"temporal", "context"} or name.startswith("interaction_"):
        add("full_body")
    elif group == "head":
        add("head")
    else:
        add("full_body")
    return row


def _load_concepts(concept_csv, metadata_csv, num_classes, num_concepts):
    with Path(concept_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0][0] != "action_class":
        raise ValueError("concept CSV must start with action_class")
    csv_names = rows[0][1:]
    if len(csv_names) != int(num_concepts):
        raise ValueError(
            f"concept CSV has {len(csv_names)} concepts, expected {num_concepts}"
        )
    values = []
    for row in rows[1:int(num_classes) + 1]:
        values.append([float(value) for value in row[1:]])
    if len(values) != int(num_classes):
        raise ValueError("concept CSV has fewer action rows than num_classes")
    with Path(metadata_csv).open("r", encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    metadata.sort(key=lambda row: int(row["concept_id"]))
    metadata_names = [row["concept_name"].strip() for row in metadata]
    if csv_names != metadata_names:
        raise ValueError("concept CSV and metadata concept order do not match")
    support = torch.tensor(values, dtype=torch.float32)
    part_map = torch.stack([
        _concept_parts(row["concept_name"], row.get("scope", ""), row.get("group", ""))
        for row in metadata
    ])
    affinity = support.t() @ support
    affinity.fill_diagonal_(1.0)
    return support, _row_normalize(part_map), _row_normalize(affinity)


class SymbolicHyperConceptBridge(nn.Module):
    """Parallel symbolic branch with learnable residual fusion."""

    def __init__(
        self, feature_dim, concept_csv, metadata_csv, num_classes,
        num_concepts=102, num_people=2, num_points=25, temperature=0.2,
        residual_init=0.0, graph_init=0.0,
    ):
        super().__init__()
        if int(num_points) != 25:
            raise ValueError("symbolic bridge currently supports NTU's 25 joints only")
        support, concept_part, concept_affinity = _load_concepts(
            concept_csv, metadata_csv, int(num_classes), int(num_concepts)
        )
        self.feature_dim = int(feature_dim)
        self.num_concepts = int(num_concepts)
        self.num_people = int(num_people)
        self.num_points = int(num_points)
        self.temperature = float(temperature)
        self.register_buffer("part_incidence", _body_part_incidence())
        self.register_buffer("concept_part", concept_part)
        self.register_buffer("concept_affinity", concept_affinity)
        self.register_buffer("support", support)
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

    def loss(self, concept_logits, labels):
        targets = self.support.to(
            device=concept_logits.device, dtype=concept_logits.dtype
        ).index_select(0, labels)
        return F.binary_cross_entropy_with_logits(concept_logits, targets)


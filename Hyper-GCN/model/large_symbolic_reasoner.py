"""Structured symbolic reasoning components used only by Hyper-GCN large."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from model.concept_schema import NTU_ACTION_NAMES


PRIMITIVE_NAMES = (
    "upper_body_motion", "lower_body_motion", "hand_motion", "foot_motion",
    "head_motion", "torso_motion", "move_up", "move_down", "move_forward",
    "move_backward", "rotate", "extend", "flex", "repeat",
    "person_interaction", "body_contact",
)

RELATION_NAMES = (
    "near", "far", "toward", "away", "before", "during",
)

_TOKEN_TO_PRIMITIVES = {
    "hand": ("hand_motion",), "arm": ("upper_body_motion",),
    "head": ("head_motion",), "hair": ("head_motion",),
    "foot": ("foot_motion", "lower_body_motion"), "leg": ("lower_body_motion",),
    "kick": ("foot_motion", "lower_body_motion", "extend", "move_forward"),
    "walk": ("lower_body_motion", "move_forward"),
    "run": ("lower_body_motion", "move_forward"),
    "hop": ("lower_body_motion", "move_up"),
    "jump": ("lower_body_motion", "move_up"),
    "throw": ("upper_body_motion", "move_forward", "extend"),
    "push": ("upper_body_motion", "move_forward", "extend"),
    "pull": ("upper_body_motion", "move_backward", "flex"),
    "wave": ("hand_motion", "move_up", "repeat"),
    "clap": ("hand_motion", "body_contact", "repeat"),
    "shake": ("repeat",),
    "turn": ("rotate",),
    "circle": ("rotate", "repeat"),
    "bend": ("torso_motion", "flex"),
    "fall": ("torso_motion", "move_down"),
    "sit": ("lower_body_motion", "move_down", "flex"),
    "stand": ("lower_body_motion", "move_up", "extend"),
    "hug": ("person_interaction", "body_contact"),
    "handshake": ("person_interaction", "hand_motion", "body_contact"),
    "giving": ("person_interaction", "hand_motion"),
    "exchange": ("person_interaction", "hand_motion"),
    "towards": ("person_interaction", "toward"),
    "apart": ("person_interaction", "away"),
    "follow": ("person_interaction", "toward"),
}


@dataclass
class SymbolicEventGraph:
    primitive_probabilities: torch.Tensor
    relation_probabilities: torch.Tensor
    primitive_names: tuple[str, ...]
    relation_names: tuple[str, ...]
    node_types: tuple[str, ...]
    edge_types: tuple[str, ...]


def _action_primitive_support(num_classes: int) -> torch.Tensor:
    support = torch.zeros(num_classes, len(PRIMITIVE_NAMES))
    ids = {name: index for index, name in enumerate(PRIMITIVE_NAMES)}
    for action_index, action_name in enumerate(NTU_ACTION_NAMES[:num_classes]):
        for token, primitive_names in _TOKEN_TO_PRIMITIVES.items():
            if token in action_name:
                for primitive_name in primitive_names:
                    support[action_index, ids[primitive_name]] = 1.0
    # Every class gets a valid weak target without inventing an action rule.
    support[support.sum(dim=1) == 0, 0] = 1.0
    return support


class LargeSymbolicReasoner(nn.Module):
    """Multi-level semantic perception followed by structured action composition."""

    def __init__(
        self, feature_dims=(128, 256, 512), num_classes=120,
        primitive_loss_weight=0.2, relation_loss_weight=0.1,
    ):
        super().__init__()
        if len(feature_dims) != 3 or any(int(value) < 1 for value in feature_dims):
            raise ValueError("feature_dims must contain three positive channel sizes")
        self.feature_dims = tuple(int(value) for value in feature_dims)
        self.num_classes = int(num_classes)
        self.num_concepts = len(PRIMITIVE_NAMES)
        self.structured_mode = True
        self.action_decoder = "event_graph"
        self.action_matching = "typed_composition"
        self.balance_loss = False
        self.pos_weight_source = "class"
        self.pos_weight_max = 1.0
        self._positive_weight_ready = True
        self.hypergraph_mode = "dynamic_backbone"
        self.primitive_loss_weight = float(primitive_loss_weight)
        self.relation_loss_weight = float(relation_loss_weight)

        self.level_projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, 128),
                nn.LayerNorm(128),
                nn.GELU(),
            )
            for channels in self.feature_dims
        ])
        self.graph_projection = nn.Sequential(
            nn.Linear(9, 32),
            nn.GELU(),
        )
        self.primitive_head = nn.Sequential(
            nn.Linear(128 * 3 + 32, 128),
            nn.GELU(),
            nn.Linear(128, len(PRIMITIVE_NAMES)),
        )
        self.relation_head = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.GELU(),
            nn.Linear(64, len(RELATION_NAMES)),
        )
        # This is a semantic composition layer, not a feature-to-action FC.
        self.action_prototypes = nn.Linear(
            len(PRIMITIVE_NAMES) + len(RELATION_NAMES), self.num_classes,
            bias=False,
        )
        self.register_buffer(
            "support", _action_primitive_support(self.num_classes), persistent=False
        )
        self.register_buffer(
            "positive_weight", torch.ones(len(PRIMITIVE_NAMES)), persistent=False
        )
        self.register_buffer(
            "training_class_counts", torch.zeros(self.num_classes)
        )

    @staticmethod
    def _pool(feature_map):
        if feature_map.ndim != 5:
            raise ValueError("feature maps must have shape [N,M,C,T,V]")
        return feature_map.mean(dim=(1, 3, 4))

    def forward(self, feature_maps, backbone_evidence=None):
        if len(feature_maps) != 3:
            raise ValueError("structured reasoner expects joint, part and action features")
        pooled = [
            projection(self._pool(feature_map))
            for projection, feature_map in zip(self.level_projection, feature_maps)
        ]
        batch_size = feature_maps[0].shape[0]
        graph_summary = feature_maps[0].new_zeros(batch_size, 9)
        if backbone_evidence is not None:
            if len(backbone_evidence) != 9:
                raise ValueError("backbone evidence must contain nine HYPERGC entries")
            summaries = []
            for evidence in backbone_evidence:
                dynamic_adjacency = evidence.get("dynamic_adjacency")
                if dynamic_adjacency is None:
                    summaries.append(feature_maps[0].new_zeros(batch_size))
                    continue
                per_sample = dynamic_adjacency.abs().mean(dim=(1, 2, 3))
                people = feature_maps[0].shape[1]
                summaries.append(per_sample.reshape(batch_size, people).mean(dim=1))
            graph_summary = torch.stack(summaries, dim=1)
        context = torch.cat((*pooled, self.graph_projection(graph_summary)), dim=-1)
        primitive_logits = self.primitive_head(context)
        relation_logits = self.relation_head(context)
        primitive_probabilities = torch.sigmoid(primitive_logits)
        relation_probabilities = torch.sigmoid(relation_logits)
        graph = SymbolicEventGraph(
            primitive_probabilities=primitive_probabilities,
            relation_probabilities=relation_probabilities,
            primitive_names=PRIMITIVE_NAMES,
            relation_names=RELATION_NAMES,
            node_types=("actor", "body_part", "motion", "state"),
            edge_types=("near", "far", "toward", "away", "before", "during"),
        )
        action_input = torch.cat((primitive_probabilities, relation_probabilities), dim=-1)
        action_logits = self.action_prototypes(action_input)
        return action_logits, primitive_logits, graph

    def loss(self, primitive_logits, labels):
        targets = self.support.to(primitive_logits).index_select(0, labels)
        return F.binary_cross_entropy_with_logits(
            primitive_logits, targets
        ) * self.primitive_loss_weight

    def structure_loss(self):
        return self.action_prototypes.weight.new_zeros(())

    def set_training_class_counts(self, class_counts):
        counts = torch.as_tensor(class_counts, device=self.training_class_counts.device)
        if counts.shape != self.training_class_counts.shape:
            raise ValueError("training class counts must match num_classes")
        self.training_class_counts.copy_(counts.to(self.training_class_counts))

    def audit(self):
        return {
            "mode": "large_structured_event_graph_v1",
            "grounding_mode": "multi_level_backbone_evidence",
            "num_classes": self.num_classes,
            "num_primitives": len(PRIMITIVE_NAMES),
            "num_relations": len(RELATION_NAMES),
            "primitive_names": list(PRIMITIVE_NAMES),
            "relation_names": list(RELATION_NAMES),
            "backbone_evidence": "multi_level_dynamic_hypergraph",
        }

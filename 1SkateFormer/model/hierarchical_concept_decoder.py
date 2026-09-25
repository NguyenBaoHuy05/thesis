"""Hierarchical unary concept grounding for HCG-NS SkateFormer."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import math

import torch
from torch import nn

from model.concept_prototypes import ConceptPrototypeDecoder
from model.concept_ontology import build_parent_index


# Joint positions per person after the NTU feeder's ``partition=True``
# reordering (24 joints; spine-shoulder is dropped):
#   0-3 / 4-7   : wrist, hand, hand tip, thumb of each arm
#   8-11 / 12-15: hip, knee, ankle, foot of each leg
#   16, 17      : shoulders      18, 19: elbows
#   20: spine mid  21: neck  22: spine base  23: head
# Parents not listed here (temporal_*, context_participant) attend to all
# tokens.  Can be overridden with ``prototype_args.region_tokens``.
DEFAULT_REGION_JOINTS: dict[str, list[int]] = {
    "region_upper_limb": [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19],
    "region_lower_limb": [8, 9, 10, 11, 12, 13, 14, 15],
    "region_hip_torso": [8, 9, 12, 13, 20, 21, 22],
    "region_head": [16, 17, 21, 23],
}
PARTITIONED_NTU_POINTS = 24


def build_region_token_mask(
    metadata: list[dict[str, str]],
    num_people: int,
    num_points: int | None,
    region_joints: dict[str, Sequence[int]] | None = None,
) -> torch.Tensor:
    """Return a [num_people * num_points, num_concepts] boolean token mask."""
    if region_joints is None:
        if num_points != PARTITIONED_NTU_POINTS:
            raise ValueError(
                "default region pooling assumes the partitioned 24-joint NTU layout "
                f"(num_points=24, feeder partition=True); got num_points={num_points}. "
                "Set prototype_args.region_tokens explicitly for other layouts."
            )
        region_joints = DEFAULT_REGION_JOINTS
    if not num_points:
        raise ValueError("region pooling requires num_points")
    num_tokens = int(num_people) * int(num_points)
    mask = torch.ones(num_tokens, len(metadata), dtype=torch.bool)
    for concept_id, row in enumerate(metadata):
        joints = region_joints.get(row.get("parent_name", ""))
        if joints is None:
            continue
        if any(not 0 <= int(j) < num_points for j in joints):
            raise ValueError(f"region joints for {row['parent_name']} exceed num_points")
        column = torch.zeros(num_tokens, dtype=torch.bool)
        for person in range(int(num_people)):
            for joint in joints:
                column[person * int(num_points) + int(joint)] = True
        mask[:, concept_id] = column
    return mask


class HierarchicalConceptDecoder(nn.Module):
    """Ground concepts and parent groups without pairwise relations.

    The base prototype decoder still produces temporally grounded unary
    evidence. Separate scope heads and parent heads read the final backbone
    representation. The sequence evidence is softly gated by its parent
    probability, never hard thresholded, so uncertain parent predictions do
    not create discontinuous action decisions.
    """

    def __init__(
        self,
        feature_dim: int,
        metadata: list[dict[str, str]],
        num_concepts: int = 78,
        temperature: float = 0.07,
        multi_scale: bool = False,
        pyramid_dims: Sequence[int] | None = None,
        pyramid_levels: Iterable[int] | None = None,
        num_people: int = 1,
        num_points: int | None = None,
        interaction_reasoning: bool = True,
        gate_floor: float = 0.05,
        region_pooling: bool = False,
        region_tokens: dict[str, Sequence[int]] | None = None,
        scope_fusion: str = "fixed",
        scope_scale: float = 0.25,
    ):
        super().__init__()
        if len(metadata) != int(num_concepts):
            raise ValueError("metadata and num_concepts disagree")
        parent_names, concept_to_parent, children, parent_levels = build_parent_index(metadata)
        self.parent_names = tuple(parent_names)
        # Do not call this attribute ``children``: nn.Module already exposes
        # children() and PyTorch uses that method during model.to(...).
        self.children_by_parent = children
        self.parent_levels = parent_levels
        self.num_concepts = int(num_concepts)
        self.num_parents = len(parent_names)
        self.gate_floor = float(gate_floor)
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("gate_floor must be in [0, 1)")
        self.register_buffer("concept_to_parent", concept_to_parent)
        interaction_concepts = [
            int(row["concept_id"])
            for row in metadata
            if row.get("concept_name", "") in {
                "context_multi_person",
                "context_person_person_interaction",
                "context_target_person",
                "interaction_two_person",
            }
        ]
        self.base = ConceptPrototypeDecoder(
            feature_dim=feature_dim,
            num_concepts=num_concepts,
            temperature=temperature,
            multi_scale=multi_scale,
            pyramid_dims=pyramid_dims,
            pyramid_levels=pyramid_levels,
            num_people=num_people,
            num_points=num_points,
            interaction_concepts=interaction_concepts if interaction_reasoning else [],
            token_mask=build_region_token_mask(
                metadata, num_people, num_points, region_tokens
            ) if region_pooling else None,
        )
        # How the time-constant scope logits are added to the per-time
        # prototype logits.  "fixed" (legacy): scope_scale * raw linear logits,
        # which are unbounded while prototype logits are bounded by
        # +-1/temperature.  "bounded": tanh(scope) / temperature scaled by a
        # learnable per-concept gate initialised at scope_scale, so both terms
        # live on the same range regardless of how the temperature evolves.
        if scope_fusion not in {"fixed", "bounded"}:
            raise ValueError("scope_fusion must be 'fixed' or 'bounded'")
        if not 0.0 < float(scope_scale) < 1.0:
            raise ValueError("scope_scale must be in (0, 1)")
        self.scope_fusion = scope_fusion
        self.scope_scale = float(scope_scale)
        if scope_fusion == "bounded":
            gate_logit = math.log(self.scope_scale / (1.0 - self.scope_scale))
            self.scope_gate_raw = nn.Parameter(torch.full((self.num_concepts,), gate_logit))
        parent_level_ids: dict[int, list[int]] = {}
        for parent_id, parent_name in enumerate(self.parent_names):
            parent_level_ids.setdefault(int(parent_levels[parent_name]), []).append(parent_id)
        self.parent_level_ids = parent_level_ids
        self.parent_heads = nn.ModuleDict({
            str(level): nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, len(parent_ids)),
            )
            for level, parent_ids in sorted(parent_level_ids.items())
        })

        scope_ids: dict[str, list[int]] = {}
        for concept_id, row in enumerate(metadata):
            scope_ids.setdefault(row.get("scope", "spatial"), []).append(concept_id)
        self.scope_ids = scope_ids
        self.scope_heads = nn.ModuleDict({
            scope: nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, len(concept_ids)),
            )
            for scope, concept_ids in sorted(scope_ids.items())
        })

    @property
    def temperature(self) -> torch.Tensor:
        return self.base.temperature

    @property
    def centers(self) -> torch.Tensor:
        return self.base.centers

    def normalized_centers(self) -> torch.Tensor:
        return self.base.normalized_centers()

    def _last_feature_map(
        self, feature_output: torch.Tensor | Sequence[torch.Tensor]
    ) -> torch.Tensor:
        return feature_output[-1] if isinstance(feature_output, (list, tuple)) else feature_output

    def forward(
        self, feature_output: torch.Tensor | Sequence[torch.Tensor]
    ) -> dict[str, Any]:
        base = self.base(feature_output)
        feature_map = self._last_feature_map(feature_output)
        pooled = feature_map.mean(dim=(2, 3))
        parent_logits = pooled.new_zeros((pooled.shape[0], self.num_parents))
        for level, parent_ids in self.parent_level_ids.items():
            values = self.parent_heads[str(level)](pooled)
            ids = torch.tensor(parent_ids, dtype=torch.long, device=pooled.device)
            parent_logits[:, ids] = values
        parent_probs = torch.sigmoid(parent_logits)
        parent_for_child = parent_probs[:, self.concept_to_parent]
        gate = self.gate_floor + (1.0 - self.gate_floor) * parent_for_child
        scope_logits = pooled.new_zeros((pooled.shape[0], self.num_concepts))
        for scope, concept_ids in self.scope_ids.items():
            values = self.scope_heads[scope](pooled)
            ids = torch.tensor(concept_ids, dtype=torch.long, device=pooled.device)
            scope_logits[:, ids] = values
        if self.scope_fusion == "bounded":
            scope_term = (
                torch.sigmoid(self.scope_gate_raw) * torch.tanh(scope_logits) / self.temperature
            )
        else:
            scope_term = self.scope_scale * scope_logits
        time_logits = base["time_logits"] + scope_term.unsqueeze(1)
        time_probs = torch.sigmoid(time_logits)
        temporal_stats = self.base._temporal_statistics(time_probs, time_logits)
        raw_sequence_probs = temporal_stats["sequence_probs"]
        effective_sequence_probs = raw_sequence_probs * gate
        return {
            **base,
            "time_logits": time_logits,
            "time_probs": time_probs,
            **temporal_stats,
            "raw_sequence_probs": raw_sequence_probs,
            "sequence_probs": effective_sequence_probs,
            "parent_logits": parent_logits,
            "parent_probs": parent_probs,
            "parent_for_child": parent_for_child,
            "soft_gate": gate,
            "scope_logits": scope_logits,
            "scope_probs": torch.sigmoid(scope_logits),
            "scope_term": scope_term,
        }

    def center_grounding_loss(
        self, cosine_similarity: torch.Tensor, action_labels: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        return self.base.center_grounding_loss(cosine_similarity, action_labels, support)

    def center_separation_loss(self, margin: float = 0.15) -> torch.Tensor:
        return self.base.center_separation_loss(margin=margin)

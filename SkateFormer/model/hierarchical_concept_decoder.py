"""Hierarchical unary concept grounding for HCG-NS SkateFormer."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from model.concept_prototypes import ConceptPrototypeDecoder
from model.concept_ontology import build_parent_index


class OntologyMaskedConceptRefiner(nn.Module):
    """Refine sequence logits without mixing unrelated ontology groups.

    Each parent and its children form an independent residual MLP block.  This
    is equivalent to a block-diagonal mask over concepts: evidence can be
    refined among siblings, but it cannot leak between unrelated body-part or
    temporal groups.  The per-concept residual scale starts at zero, making the
    module an exact identity at initialization.
    """

    def __init__(
        self,
        num_concepts: int,
        concept_to_parent: torch.Tensor,
        num_parents: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("refiner hidden_dim must be positive")
        self.num_concepts = int(num_concepts)
        self.num_parents = int(num_parents)
        self.hidden_dim = int(hidden_dim)
        self.blocks = nn.ModuleDict()
        for parent_id in range(self.num_parents):
            concept_ids = torch.nonzero(
                concept_to_parent == parent_id, as_tuple=False
            ).flatten()
            if concept_ids.numel() == 0:
                continue
            self.register_buffer(
                f"group_ids_{parent_id}", concept_ids, persistent=False
            )
            width = int(concept_ids.numel())
            self.blocks[str(parent_id)] = nn.Sequential(
                nn.Linear(width, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, width),
            )
        self.alpha = nn.Parameter(torch.zeros(self.num_concepts))

    @property
    def residual_scale(self) -> torch.Tensor:
        return torch.tanh(self.alpha)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.num_concepts:
            raise ValueError("concept logits must have shape [B, num_concepts]")
        delta = torch.zeros_like(logits)
        for parent_id, block in self.blocks.items():
            concept_ids = getattr(self, f"group_ids_{parent_id}")
            group_delta = block(logits.index_select(1, concept_ids))
            delta = delta.index_copy(1, concept_ids, group_delta)
        return logits + self.residual_scale.unsqueeze(0) * delta


class HierarchicalConceptDecoder(nn.Module):
    """Ground concepts and parent groups without pairwise relations.

    Scope metadata routes spatial/context concepts through a pose projector and
    temporal concepts through a motion-difference projector.  Adaptive MIL
    pools the resulting per-time evidence, then an ontology-masked residual
    refiner models sibling co-occurrence. Parent predictions remain auxiliary
    supervision and never modify concept probabilities at inference time.
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
        temporal_pooling: str = "adaptive_mil",
        scale_fusion: str = "shared",
        num_people: int = 1,
        num_points: int | None = None,
        interaction_reasoning: bool = True,
        parent_enabled: bool = True,
        refiner_enabled: bool = True,
        refiner_hidden_dim: int = 128,
        temporal_gru_enabled: bool = False,
        temporal_gru_hidden_dim: int = 64,
        temporal_gru_layers: int = 1,
        temporal_gru_dropout: float = 0.0,
        temporal_gru_residual_init: float = 0.0,
        center_loss_mode: str = "legacy",
        center_positive_margin: float = 0.35,
        center_gap_margin: float = 0.30,
        center_gap_topk: int = 5,
        center_negative_margin: float = 0.10,
        center_negative_weight: float = 0.5,
        center_negative_topk: int = 10,
        center_separation_margin: float = 0.15,
        center_separation_topk: int = 5,
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
        self.parent_enabled = bool(parent_enabled)
        self.refiner_enabled = bool(refiner_enabled)
        self.register_buffer("concept_to_parent", concept_to_parent)
        self.register_buffer(
            "parent_child_count",
            torch.bincount(concept_to_parent, minlength=self.num_parents).float(),
            persistent=False,
        )
        interaction_concepts = [
            int(row["concept_id"])
            for row in metadata
            if (
                row.get("scope", "").strip().lower() == "context"
                or row.get("concept_name", "") in {
                    "context_multi_person",
                    "context_person_person_interaction",
                    "context_target_person",
                    "interaction_two_person",
                }
            )
        ]
        scope_ids: dict[str, list[int]] = {}
        for concept_id, row in enumerate(metadata):
            scope = row.get("scope", "spatial").strip().lower() or "spatial"
            scope_ids.setdefault(scope, []).append(concept_id)
        self.scope_ids = {name: tuple(ids) for name, ids in scope_ids.items()}
        temporal_concepts = self.scope_ids.get("temporal", ())
        self.base = ConceptPrototypeDecoder(
            feature_dim=feature_dim,
            num_concepts=num_concepts,
            temperature=temperature,
            multi_scale=multi_scale,
            pyramid_dims=pyramid_dims,
            pyramid_levels=pyramid_levels,
            temporal_pooling=temporal_pooling,
            scale_fusion=scale_fusion,
            num_people=num_people,
            num_points=num_points,
            temporal_concepts=temporal_concepts,
            temporal_gru_enabled=temporal_gru_enabled,
            temporal_gru_hidden_dim=temporal_gru_hidden_dim,
            temporal_gru_layers=temporal_gru_layers,
            temporal_gru_dropout=temporal_gru_dropout,
            temporal_gru_residual_init=temporal_gru_residual_init,
            interaction_concepts=(
                interaction_concepts if interaction_reasoning else []
            ),
            center_loss_mode=center_loss_mode,
            center_positive_margin=center_positive_margin,
            center_gap_margin=center_gap_margin,
            center_gap_topk=center_gap_topk,
            center_negative_margin=center_negative_margin,
            center_negative_weight=center_negative_weight,
            center_negative_topk=center_negative_topk,
            center_separation_margin=center_separation_margin,
            center_separation_topk=center_separation_topk,
        )
        self.concept_refiner: nn.Module
        if self.refiner_enabled:
            self.concept_refiner = OntologyMaskedConceptRefiner(
                num_concepts=self.num_concepts,
                concept_to_parent=concept_to_parent,
                num_parents=self.num_parents,
                hidden_dim=refiner_hidden_dim,
            )
        else:
            self.concept_refiner = nn.Identity()
        parent_level_ids: dict[int, list[int]] = {}
        for parent_id, parent_name in enumerate(self.parent_names):
            parent_level_ids.setdefault(int(parent_levels[parent_name]), []).append(parent_id)
        self.parent_level_ids = parent_level_ids
        self.parent_heads = nn.ModuleDict(
            {
                str(level): nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, len(parent_ids)),
                )
                for level, parent_ids in sorted(parent_level_ids.items())
            }
            if self.parent_enabled else {}
        )

    @property
    def multi_scale(self) -> bool:
        return self.base.multi_scale

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
        feature_map = self._last_feature_map(feature_output)
        pooled = feature_map.mean(dim=(2, 3))
        parent_logits = pooled.new_zeros((pooled.shape[0], self.num_parents))
        if self.parent_enabled:
            for level, parent_ids in self.parent_level_ids.items():
                values = self.parent_heads[str(level)](pooled)
                ids = torch.tensor(parent_ids, dtype=torch.long, device=pooled.device)
                parent_logits[:, ids] = values
        parent_probs = (
            torch.sigmoid(parent_logits)
            if self.parent_enabled
            else torch.zeros_like(parent_logits)
        )
        parent_for_child = parent_probs[:, self.concept_to_parent]
        base = self.base(feature_output)
        raw_sequence_logits = base["sequence_logits"]
        sequence_logits = self.concept_refiner(raw_sequence_logits)
        sequence_probs = torch.sigmoid(sequence_logits)
        return {
            **base,
            "raw_sequence_logits": raw_sequence_logits,
            "raw_sequence_probs": base["sequence_probs"],
            "sequence_logits": sequence_logits,
            "sequence_probs": sequence_probs,
            "parent_logits": parent_logits,
            "parent_probs": parent_probs,
            "parent_for_child": parent_for_child,
        }

    def center_grounding_loss(
        self, cosine_similarity: torch.Tensor | None, action_labels: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        if cosine_similarity is None:
            return support.new_zeros(())
        return self.base.center_grounding_loss(cosine_similarity, action_labels, support)

    def center_separation_loss(
        self, support: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.base.center_separation_loss(support=support)

    def parent_graph_loss(
        self, sequence_probs: torch.Tensor, parent_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Use the fixed ontology graph to regularize child concepts.

        Each concept has exactly one parent edge.  Parent probabilities act as
        detached teacher targets, so this loss only updates the concept path;
        the independently supervised parent head cannot reduce the loss by
        following an over-confident child.  Violations are measured in logit
        space and averaged per parent before the final mean so large parent
        groups do not dominate smaller groups.
        """
        if not self.parent_enabled:
            return sequence_probs.new_zeros(())
        if sequence_probs.ndim != 2 or sequence_probs.shape[1] != self.num_concepts:
            raise ValueError("sequence_probs must have shape [B, num_concepts]")
        if parent_probs.ndim != 2 or parent_probs.shape[1] != self.num_parents:
            raise ValueError("parent_probs must have shape [B, num_parents]")

        child_evidence = torch.logit(sequence_probs.clamp(1e-4, 1.0 - 1e-4))
        parent_evidence = torch.logit(parent_probs.clamp(1e-4, 1.0 - 1e-4)).detach()
        parent_for_child = parent_evidence[:, self.concept_to_parent]
        edge_penalty = F.relu(child_evidence - parent_for_child).square()

        per_parent = edge_penalty.new_zeros(
            (edge_penalty.shape[0], self.num_parents)
        )
        parent_index = self.concept_to_parent.unsqueeze(0).expand_as(edge_penalty)
        per_parent = per_parent.scatter_add(1, parent_index, edge_penalty)
        per_parent = per_parent / self.parent_child_count.clamp_min(1.0).unsqueeze(0)
        return per_parent.mean()

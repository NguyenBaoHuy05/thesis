"""Non-prototype concept heads for controlled HCG-NS ablations."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from model.concept_prototypes import AdaptiveTemporalMILPool, TemporalConceptGRU


# SkateFormer receives the NTU joints in a fixed 24-token reordered layout.
# These indices recover anatomical regions from that layout (the shoulder and
# elbow tokens live outside the two distal-arm partitions). They refer to the
# reordered representation, not the raw NTU joint order.
NTU_GROUP_JOINTS: dict[str, tuple[int, ...]] = {
    "upper_limb": (*range(0, 8), 16, 17, 18, 19),
    "lower_limb": tuple(range(8, 16)),
    # Left/right hips plus spine-mid and spine-base.
    "hip_torso": (8, 12, 20, 22),
    # Partition positions 21 and 23 are neck/head in the reordered skeleton.
    "head": (21, 23),
}


class ConceptHeadDecoder(nn.Module):
    """Predict concepts with ordinary linear heads instead of prototypes.

    ``direct`` uses one classifier for the complete concept vocabulary and is
    the closest non-prototype control. ``grouped`` creates an independent head
    for every ontology group and masks anatomical groups to their corresponding
    NTU joint partitions. Both variants retain pose/motion routing, multi-scale
    fusion, temporal GRU context and temporal MIL pooling.
    """

    def __init__(
        self,
        feature_dim: int,
        metadata: list[dict[str, str]],
        decoder_type: str,
        num_concepts: int,
        multi_scale: bool = False,
        pyramid_dims: Sequence[int] | None = None,
        pyramid_levels: Iterable[int] | None = None,
        temporal_pooling: str = "adaptive_mil",
        scale_fusion: str = "shared",
        num_people: int = 1,
        num_points: int | None = None,
        temporal_concepts: Iterable[int] | None = None,
        temporal_gru_enabled: bool = False,
        temporal_gru_hidden_dim: int = 64,
        temporal_gru_layers: int = 1,
        temporal_gru_dropout: float = 0.0,
        temporal_gru_residual_init: float = 0.0,
        interaction_concepts: Iterable[int] | None = None,
    ):
        super().__init__()
        if decoder_type not in {"direct", "grouped"}:
            raise ValueError("head decoder_type must be 'direct' or 'grouped'")
        if len(metadata) != int(num_concepts):
            raise ValueError("metadata and num_concepts disagree")
        self.decoder_type = decoder_type
        self.feature_dim = int(feature_dim)
        self.num_concepts = int(num_concepts)
        self.multi_scale = bool(multi_scale)
        self.temporal_pooling = str(temporal_pooling)
        if self.temporal_pooling not in {"legacy_noisy_or", "adaptive_mil"}:
            raise ValueError(
                "temporal_pooling must be 'legacy_noisy_or' or 'adaptive_mil'"
            )
        self.scale_fusion = "single" if not self.multi_scale else str(scale_fusion)
        if self.multi_scale and self.scale_fusion not in {"shared", "per_concept"}:
            raise ValueError(
                "scale_fusion must be 'shared' or 'per_concept' when multi_scale=true"
            )
        self.adaptive_mil_pool = (
            AdaptiveTemporalMILPool(self.num_concepts)
            if self.temporal_pooling == "adaptive_mil"
            else None
        )
        self.num_people = int(num_people)
        self.num_points = int(num_points) if num_points is not None else None

        temporal_ids = tuple(sorted(set(int(i) for i in (temporal_concepts or []))))
        temporal_mask = torch.zeros(self.num_concepts, dtype=torch.bool)
        if temporal_ids:
            temporal_mask[list(temporal_ids)] = True
        self.temporal_concepts = temporal_ids
        self.register_buffer("temporal_mask", temporal_mask, persistent=False)
        self.temporal_gru = (
            TemporalConceptGRU(
                num_temporal_concepts=len(temporal_ids),
                hidden_dim=temporal_gru_hidden_dim,
                num_layers=temporal_gru_layers,
                dropout=temporal_gru_dropout,
                residual_init=temporal_gru_residual_init,
            )
            if temporal_gru_enabled
            else None
        )
        if temporal_gru_enabled and not temporal_ids:
            raise ValueError(
                "temporal_gru_enabled requires at least one temporal concept"
            )
        self.interaction_concepts = tuple(
            sorted(set(int(i) for i in (interaction_concepts or [])))
        )

        group_ids: dict[str, list[int]] = {}
        for concept_id, row in enumerate(metadata):
            group = row.get("group", "").strip().lower()
            if not group:
                raise ValueError(f"concept {concept_id} has no ontology group")
            group_ids.setdefault(group, []).append(concept_id)
        self.group_ids = {name: tuple(ids) for name, ids in group_ids.items()}

        if self.decoder_type == "direct":
            self.concept_heads = nn.ModuleDict({
                # No bias keeps the parameterization aligned with prototype
                # centers; adaptive MIL already supplies a per-concept bias.
                "all": nn.Linear(
                    self.feature_dim, self.num_concepts, bias=False
                ),
            })
        else:
            self.concept_heads = nn.ModuleDict({
                name: nn.Linear(self.feature_dim, len(ids), bias=False)
                for name, ids in sorted(self.group_ids.items())
            })

        if self.multi_scale:
            dims = tuple(int(dim) for dim in (pyramid_dims or (feature_dim,)))
            levels = tuple(int(level) for level in (pyramid_levels or range(len(dims))))
            if len(dims) != len(levels) or not dims:
                raise ValueError(
                    "pyramid_dims and pyramid_levels must have the same non-zero length"
                )
            self.pyramid_levels = levels
            self.pose_scale_projectors = nn.ModuleList(
                [self._make_projector(dim, self.feature_dim) for dim in dims]
            )
            self.motion_scale_projectors = nn.ModuleList(
                [self._make_projector(dim, self.feature_dim, bias=False) for dim in dims]
            )
            scale_shape = (
                (len(levels), self.num_concepts)
                if self.scale_fusion == "per_concept"
                else (len(levels),)
            )
            self.scale_logits = nn.Parameter(torch.zeros(scale_shape))
        else:
            self.pose_projector = self._make_projector(feature_dim, self.feature_dim)
            self.motion_projector = self._make_projector(
                feature_dim, self.feature_dim, bias=False
            )

        self.interaction_projector = None
        if self.interaction_concepts and self.num_people >= 2:
            self.interaction_projector = nn.Sequential(
                nn.LayerNorm(4 * self.feature_dim),
                nn.Linear(4 * self.feature_dim, self.feature_dim),
            )

    @staticmethod
    def _make_projector(
        input_dim: int, output_dim: int, bias: bool = True
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim, bias=bias),
        )

    @property
    def scale_weights(self) -> torch.Tensor | None:
        if not self.multi_scale:
            return None
        return torch.softmax(self.scale_logits, dim=0)

    def _classify(self, features: torch.Tensor) -> torch.Tensor:
        """Apply direct or ontology-grouped heads on the final feature axis."""
        if self.decoder_type == "direct":
            return self.concept_heads["all"](features)
        logits = features.new_zeros((*features.shape[:-1], self.num_concepts))
        for name, concept_ids in self.group_ids.items():
            ids = torch.as_tensor(concept_ids, dtype=torch.long, device=features.device)
            logits = logits.index_copy(-1, ids, self.concept_heads[name](features))
        return logits

    @staticmethod
    def _motion_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError(
                f"feature_map must be [B,C,T,V], got {tuple(feature_map.shape)}"
            )
        if feature_map.shape[2] <= 1:
            return torch.zeros_like(feature_map)
        differences = feature_map[:, :, 1:] - feature_map[:, :, :-1]
        return torch.cat((differences[:, :, :1], differences), dim=2)

    def _group_token_mask(
        self, group: str, token_count: int, device: torch.device
    ) -> torch.Tensor:
        """Return a mask over flattened person-major NTU joint tokens."""
        local_ids = NTU_GROUP_JOINTS.get(group)
        if local_ids is None or self.num_points != 24:
            return torch.ones(token_count, dtype=torch.bool, device=device)
        selected = [
            person * self.num_points + joint
            for person in range(self.num_people)
            for joint in local_ids
            if person * self.num_points + joint < token_count
        ]
        mask = torch.zeros(token_count, dtype=torch.bool, device=device)
        if selected:
            mask[selected] = True
        else:
            mask.fill_(True)
        return mask

    def _pool_joint_logits(self, token_logits: torch.Tensor) -> torch.Tensor:
        """Use concept evidence to select joints, with group masks when enabled."""
        if self.decoder_type == "direct":
            attention = torch.softmax(token_logits, dim=2)
            return (attention * token_logits).sum(dim=2)

        time_logits = token_logits.new_zeros(
            (token_logits.shape[0], token_logits.shape[1], self.num_concepts)
        )
        for name, concept_ids in self.group_ids.items():
            ids = torch.as_tensor(
                concept_ids, dtype=torch.long, device=token_logits.device
            )
            values = token_logits.index_select(3, ids)
            joint_mask = self._group_token_mask(
                name, token_logits.shape[2], token_logits.device
            ).view(1, 1, -1, 1)
            attention = torch.softmax(
                values.masked_fill(~joint_mask, torch.finfo(values.dtype).min),
                dim=2,
            )
            pooled = (attention * values).sum(dim=2)
            time_logits = time_logits.index_copy(2, ids, pooled)
        return time_logits

    def _project_tokens(
        self, feature_map: torch.Tensor, projector: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_map.ndim != 4:
            raise ValueError(
                f"feature_map must be [B,C,T,V], got {tuple(feature_map.shape)}"
            )
        tokens = feature_map.permute(0, 2, 3, 1).contiguous()
        features = projector(tokens)
        return features, self._pool_joint_logits(self._classify(features))

    def _project_scope_routed_tokens(
        self,
        feature_map: torch.Tensor,
        pose_projector: nn.Module,
        motion_projector: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pose_features, pose_logits = self._project_tokens(feature_map, pose_projector)
        _, motion_logits = self._project_tokens(
            self._motion_feature_map(feature_map), motion_projector
        )
        time_logits = torch.where(
            self.temporal_mask.view(1, 1, -1), motion_logits, pose_logits
        )
        return pose_features, time_logits

    def _interaction_logits(self, pose_features: torch.Tensor) -> torch.Tensor | None:
        if self.interaction_projector is None or self.num_points is None:
            return None
        batch, timesteps, tokens, channels = pose_features.shape
        if tokens != self.num_people * self.num_points or self.num_people < 2:
            return None
        people = pose_features.view(
            batch, timesteps, self.num_people, self.num_points, channels
        )
        person_repr = people.mean(dim=3)
        left, right = person_repr[:, :, 0], person_repr[:, :, 1]
        pair = torch.cat([left, right, (left - right).abs(), left * right], dim=-1)
        return self._classify(self.interaction_projector(pair))

    def _apply_temporal_gru(self, time_logits: torch.Tensor) -> torch.Tensor:
        if self.temporal_gru is None:
            return time_logits
        temporal_ids = torch.as_tensor(
            self.temporal_concepts, dtype=torch.long, device=time_logits.device
        )
        contextualized = self.temporal_gru(time_logits.index_select(2, temporal_ids))
        return time_logits.index_copy(2, temporal_ids, contextualized)

    def _temporal_statistics(
        self, time_probs: torch.Tensor, time_logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        safe_probs = time_probs.clamp(1e-5, 1.0 - 1e-5)
        timesteps = time_probs.shape[1]
        time_index = torch.linspace(
            0.0, 1.0, timesteps, device=time_probs.device, dtype=time_probs.dtype
        ).view(1, -1, 1)
        mass = safe_probs.sum(dim=1).clamp_min(1e-6)
        if self.adaptive_mil_pool is not None:
            sequence_logits = self.adaptive_mil_pool.pool_logits(time_logits)
            sequence_probs = torch.sigmoid(sequence_logits)
        else:
            sequence_probs = 1.0 - torch.exp(torch.log1p(-safe_probs).sum(dim=1))
            sequence_logits = torch.logit(sequence_probs.clamp(1e-6, 1.0 - 1e-6))
        peak_attention = torch.softmax(time_logits, dim=1)
        return {
            "sequence_logits": sequence_logits,
            "sequence_probs": sequence_probs,
            "expected_time": (safe_probs * time_index).sum(dim=1) / mass,
            "peak_time": (peak_attention * time_index).sum(dim=1),
            "duration": safe_probs.mean(dim=1),
            "rising_activity": F.relu(
                safe_probs[:, 1:] - safe_probs[:, :-1]
            ).sum(dim=1),
            "falling_activity": F.relu(
                safe_probs[:, :-1] - safe_probs[:, 1:]
            ).sum(dim=1),
        }

    def forward(
        self, feature_map: torch.Tensor | Sequence[torch.Tensor]
    ) -> dict[str, torch.Tensor | None]:
        if self.multi_scale:
            if not isinstance(feature_map, (list, tuple)):
                raise ValueError("multi-scale decoder expects a feature-map pyramid")
            selected = []
            for level, pose_projector, motion_projector in zip(
                self.pyramid_levels,
                self.pose_scale_projectors,
                self.motion_scale_projectors,
            ):
                if level >= len(feature_map) or level < -len(feature_map):
                    raise ValueError(f"pyramid level {level} is unavailable")
                selected.append(self._project_scope_routed_tokens(
                    feature_map[level], pose_projector, motion_projector
                ))
            target_steps = max(item[1].shape[1] for item in selected)
            fused_logits = []
            for _, logits in selected:
                if logits.shape[1] != target_steps:
                    logits = F.interpolate(
                        logits.transpose(1, 2), size=target_steps,
                        mode="linear", align_corners=False,
                    ).transpose(1, 2)
                fused_logits.append(logits)
            scale_weights = self.scale_weights
            if self.scale_fusion == "per_concept":
                time_logits = sum(
                    weight.view(1, 1, -1) * logits
                    for weight, logits in zip(scale_weights, fused_logits)
                )
            else:
                time_logits = sum(
                    weight * logits
                    for weight, logits in zip(scale_weights, fused_logits)
                )
            final_pose_features = selected[-1][0]
        else:
            if isinstance(feature_map, (list, tuple)):
                feature_map = feature_map[-1]
            final_pose_features, time_logits = self._project_scope_routed_tokens(
                feature_map, self.pose_projector, self.motion_projector
            )

        interaction_logits = self._interaction_logits(final_pose_features)
        if interaction_logits is not None and self.interaction_concepts:
            if interaction_logits.shape[1] != time_logits.shape[1]:
                interaction_logits = F.interpolate(
                    interaction_logits.transpose(1, 2), size=time_logits.shape[1],
                    mode="linear", align_corners=False,
                ).transpose(1, 2)
            interaction_ids = torch.as_tensor(
                self.interaction_concepts,
                dtype=torch.long,
                device=time_logits.device,
            )
            time_logits = time_logits.index_copy(
                2, interaction_ids,
                interaction_logits.index_select(2, interaction_ids),
            )

        time_logits = self._apply_temporal_gru(time_logits)
        time_probs = torch.sigmoid(time_logits)
        stats = self._temporal_statistics(time_probs, time_logits)
        return {
            "time_logits": time_logits,
            "time_probs": time_probs,
            **stats,
            "scale_weights": self.scale_weights,
            "similarity": None,
            "cosine_similarity": None,
            "centers": None,
        }

"""Hard-grounded concept tokens from Hyper-GCN joint/person feature maps."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class GroundedConceptEncoder(nn.Module):
    """Route each named concept through its allowed physical joints only."""

    def __init__(
        self,
        feature_dim,
        concept_dim,
        concept_joint_mask,
        interaction_mask,
        num_people=2,
        num_points=25,
        temporal_bins=4,
        dropout=0.1,
    ):
        super().__init__()
        joint_mask = torch.as_tensor(concept_joint_mask, dtype=torch.bool)
        interaction_mask = torch.as_tensor(interaction_mask, dtype=torch.bool)
        if joint_mask.ndim != 2 or joint_mask.shape[1] != int(num_points):
            raise ValueError(
                f"concept_joint_mask must be [K,{int(num_points)}], got {tuple(joint_mask.shape)}"
            )
        if not joint_mask.any(dim=1).all():
            raise ValueError("every concept must have at least one grounded physical joint")
        if interaction_mask.shape != (joint_mask.shape[0],):
            raise ValueError("interaction_mask must have one value per concept")
        if int(num_people) < 1:
            raise ValueError("num_people must be positive")
        if int(temporal_bins) < 1:
            raise ValueError("temporal_bins must be positive")
        if not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1:
            raise ValueError("grounding dropout must be finite and in [0, 1)")

        self.feature_dim = int(feature_dim)
        self.concept_dim = int(concept_dim)
        self.num_concepts = int(joint_mask.shape[0])
        self.num_people = int(num_people)
        self.num_points = int(num_points)
        self.temporal_bins = int(temporal_bins)

        normalized_mask = joint_mask.float()
        normalized_mask = normalized_mask / normalized_mask.sum(dim=1, keepdim=True)
        self.register_buffer("concept_joint_mask", joint_mask, persistent=False)
        self.register_buffer("normalized_joint_mask", normalized_mask, persistent=False)
        self.register_buffer("interaction_mask", interaction_mask, persistent=False)

        self.channel_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.concept_dim),
            nn.LayerNorm(self.concept_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.temporal_embedding = nn.Parameter(
            torch.empty(self.temporal_bins, self.concept_dim)
        )
        self.temporal_queries = nn.Parameter(
            torch.empty(self.num_concepts, self.concept_dim)
        )
        self.person_queries = nn.Parameter(
            torch.empty(self.num_concepts, self.concept_dim)
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(3 * self.concept_dim, self.concept_dim),
            nn.LayerNorm(self.concept_dim),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(self.concept_dim)

        nn.init.normal_(self.temporal_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.temporal_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.person_queries, mean=0.0, std=0.02)

    def _validate_input(self, feature_map, valid_person_mask):
        expected = (self.num_people, self.feature_dim, self.num_points)
        if feature_map.ndim != 5:
            raise ValueError(
                "grounded encoder expects [N,M,C,T,V], got "
                f"{tuple(feature_map.shape)}"
            )
        if (feature_map.shape[1], feature_map.shape[2], feature_map.shape[4]) != expected:
            raise ValueError(
                "grounded encoder expected M/C/V={} but got {}".format(
                    expected,
                    (feature_map.shape[1], feature_map.shape[2], feature_map.shape[4]),
                )
            )
        if valid_person_mask is None:
            valid_person_mask = feature_map.detach().abs().sum(dim=(2, 3, 4)) > 0
        else:
            valid_person_mask = torch.as_tensor(
                valid_person_mask, device=feature_map.device, dtype=torch.bool
            )
        if valid_person_mask.shape != feature_map.shape[:2]:
            raise ValueError(
                f"valid_person_mask must be {tuple(feature_map.shape[:2])}, "
                f"got {tuple(valid_person_mask.shape)}"
            )
        # Degenerate empty samples still need a finite softmax. Treat person 0 as
        # a zero-valued fallback; real NTU samples always contain a valid person.
        no_valid_person = ~valid_person_mask.any(dim=1)
        if no_valid_person.any():
            valid_person_mask = valid_person_mask.clone()
            valid_person_mask[no_valid_person, 0] = True
        return valid_person_mask

    def forward(self, feature_map, valid_person_mask=None):
        valid_person_mask = self._validate_input(feature_map, valid_person_mask)
        batch, people, _, timesteps, _ = feature_map.shape

        # Project channels before expanding to K concepts to keep memory bounded.
        projected = self.channel_projection(
            feature_map.permute(0, 1, 3, 4, 2).contiguous()
        )  # [N,M,T,V,D]
        grounded = torch.einsum(
            "bmtvd,kv->bmktd", projected, self.normalized_joint_mask.to(projected)
        )  # [N,M,K,T,D]

        pooled = F.adaptive_avg_pool1d(
            grounded.permute(0, 1, 2, 4, 3).reshape(
                batch * people * self.num_concepts, self.concept_dim, timesteps
            ),
            self.temporal_bins,
        )
        pooled = pooled.reshape(
            batch, people, self.num_concepts, self.concept_dim, self.temporal_bins
        ).permute(0, 1, 2, 4, 3).contiguous()
        temporal_tokens = pooled + self.temporal_embedding.view(
            1, 1, 1, self.temporal_bins, self.concept_dim
        )
        temporal_scores = torch.einsum(
            "bmksd,kd->bmks",
            F.normalize(temporal_tokens, dim=-1),
            F.normalize(self.temporal_queries, dim=-1),
        )
        temporal_weights = temporal_scores.softmax(dim=-1)
        person_tokens = torch.sum(
            temporal_weights.unsqueeze(-1) * temporal_tokens, dim=3
        )  # [N,M,K,D]

        person_scores = torch.einsum(
            "bmkd,kd->bmk",
            F.normalize(person_tokens, dim=-1),
            F.normalize(self.person_queries, dim=-1),
        )
        person_scores = person_scores.masked_fill(
            ~valid_person_mask.unsqueeze(-1), torch.finfo(person_scores.dtype).min
        )
        person_weights = person_scores.softmax(dim=1)
        single_person_tokens = torch.sum(
            person_weights.unsqueeze(-1) * person_tokens, dim=1
        )

        if people >= 2:
            first, second = person_tokens[:, 0], person_tokens[:, 1]
            pair_features = torch.cat(
                ((first + second) * 0.5, (first - second).abs(), first * second),
                dim=-1,
            )
            pair_tokens = self.pair_projection(pair_features)
            pair_valid = valid_person_mask.sum(dim=1) >= 2
            pair_tokens = torch.where(
                pair_valid.view(batch, 1, 1), pair_tokens, single_person_tokens
            )
        else:
            pair_tokens = single_person_tokens

        concept_tokens = torch.where(
            self.interaction_mask.view(1, self.num_concepts, 1),
            pair_tokens,
            single_person_tokens,
        )
        concept_tokens = self.output_norm(concept_tokens)
        auxiliary = {
            "temporal_weights": temporal_weights,
            "person_weights": person_weights.permute(0, 2, 1).contiguous(),
            "valid_person_mask": valid_person_mask,
            "token_norm": concept_tokens.norm(dim=-1),
        }
        return concept_tokens, auxiliary

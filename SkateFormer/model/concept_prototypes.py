"""Input-conditioned concept prototype grounding for SkateFormer."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveTemporalMILPool(nn.Module):
    """Pool per-timestep concept logits with a learned mean/LSE mixture.

    Every concept uses the same pooling rule, while ``beta``, ``mix`` and
    ``bias`` are learned independently for each concept.  The bounded beta
    parameter keeps the log-sum-exp branch numerically stable and makes the
    module permutation invariant over timesteps when no mask is supplied.
    """

    def __init__(
        self,
        num_concepts: int,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        initial_beta: float = 1.0,
    ):
        super().__init__()
        if int(num_concepts) <= 0:
            raise ValueError("num_concepts must be positive")
        if not 0.0 < float(beta_min) < float(beta_max):
            raise ValueError("beta_min and beta_max must satisfy 0 < min < max")
        if not float(beta_min) < float(initial_beta) < float(beta_max):
            raise ValueError("initial_beta must lie strictly inside the beta bounds")
        self.num_concepts = int(num_concepts)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        beta_fraction = (float(initial_beta) - self.beta_min) / (
            self.beta_max - self.beta_min
        )
        beta_raw = math.log(beta_fraction / (1.0 - beta_fraction))
        self.beta_raw = nn.Parameter(torch.full((self.num_concepts,), beta_raw))
        self.mix_raw = nn.Parameter(torch.zeros(self.num_concepts))
        self.bias = nn.Parameter(torch.zeros(self.num_concepts))

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(
            self.beta_raw
        )

    @property
    def mix(self) -> torch.Tensor:
        return torch.sigmoid(self.mix_raw)

    def pool_logits(self, time_logits: torch.Tensor) -> torch.Tensor:
        if time_logits.ndim != 3:
            raise ValueError(
                f"time_logits must have shape [B,T,K], got {tuple(time_logits.shape)}"
            )
        if time_logits.shape[-1] != self.num_concepts:
            raise ValueError(
                f"time_logits has {time_logits.shape[-1]} concepts, "
                f"expected {self.num_concepts}"
            )
        timesteps = int(time_logits.shape[1])
        if timesteps <= 0:
            raise ValueError("time_logits must contain at least one timestep")
        beta = self.beta.to(dtype=time_logits.dtype, device=time_logits.device)
        mix = self.mix.to(dtype=time_logits.dtype, device=time_logits.device)
        bias = self.bias.to(dtype=time_logits.dtype, device=time_logits.device)
        mean_logits = time_logits.mean(dim=1)
        scaled = time_logits * beta.view(1, 1, -1)
        lse_logits = (
            torch.logsumexp(scaled, dim=1) - math.log(float(timesteps))
        ) / beta.view(1, -1)
        return (
            mix.view(1, -1) * lse_logits
            + (1.0 - mix).view(1, -1) * mean_logits
            + bias.view(1, -1)
        )

    def forward(self, time_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.pool_logits(time_logits))


class ConceptPrototypeDecoder(nn.Module):
    """Convert SkateFormer feature maps into temporally grounded concepts.

    The default path is checkpoint-compatible with the original decoder. An
    optional multi-scale path fuses the last two backbone stages so temporal
    concepts retain a higher-resolution signal. Interaction concepts can use a
    small person-pair branch while ordinary concepts continue to use token
    cosine similarity.
    """

    def __init__(
        self,
        feature_dim: int,
        num_concepts: int = 74,
        temperature: float = 0.07,
        multi_scale: bool = False,
        pyramid_dims: Sequence[int] | None = None,
        pyramid_levels: Iterable[int] | None = None,
        temporal_pooling: str = "legacy_noisy_or",
        scale_fusion: str = "shared",
        num_people: int = 1,
        num_points: int | None = None,
        interaction_concepts: Iterable[int] | None = None,
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
        self.feature_dim = feature_dim
        self.num_concepts = num_concepts
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
            AdaptiveTemporalMILPool(num_concepts)
            if self.temporal_pooling == "adaptive_mil"
            else None
        )
        self.num_people = int(num_people)
        self.num_points = int(num_points) if num_points is not None else None
        self.interaction_concepts = tuple(
            sorted(set(int(i) for i in (interaction_concepts or [])))
        )
        self.center_loss_mode = str(center_loss_mode)
        if self.center_loss_mode not in {
            "legacy", "margin_hard_negative", "pairwise_gap_hard_negative"
        }:
            raise ValueError(
                "center_loss_mode must be 'legacy', 'margin_hard_negative', "
                "or 'pairwise_gap_hard_negative'"
            )
        self.center_positive_margin = float(center_positive_margin)
        self.center_gap_margin = float(center_gap_margin)
        self.center_gap_topk = int(center_gap_topk)
        self.center_negative_margin = float(center_negative_margin)
        self.center_negative_weight = float(center_negative_weight)
        self.center_negative_topk = int(center_negative_topk)
        self.center_separation_margin = float(center_separation_margin)
        self.center_separation_topk = int(center_separation_topk)
        if not -1.0 <= self.center_negative_margin <= 1.0:
            raise ValueError("center_negative_margin must be in [-1, 1]")
        if (
            self.center_loss_mode == "margin_hard_negative"
            and not self.center_negative_margin < self.center_positive_margin <= 1.0
        ):
            raise ValueError(
                "absolute center margins must satisfy negative < positive <= 1"
            )
        if not 0.0 <= self.center_gap_margin <= 2.0:
            raise ValueError("center_gap_margin must be in [0, 2]")
        if self.center_negative_weight < 0.0:
            raise ValueError("center_negative_weight must be non-negative")
        if (
            self.center_gap_topk <= 0
            or self.center_negative_topk <= 0
            or self.center_separation_topk <= 0
        ):
            raise ValueError("center hard-negative top-k values must be positive")
        if not -1.0 <= self.center_separation_margin <= 1.0:
            raise ValueError("center_separation_margin must be in [-1, 1]")

        if self.multi_scale:
            dims = tuple(int(dim) for dim in (pyramid_dims or (feature_dim,)))
            levels = tuple(
                int(level)
                for level in (pyramid_levels or range(len(dims)))
            )
            if len(dims) != len(levels) or not dims:
                raise ValueError(
                    "pyramid_dims and pyramid_levels must have the same non-zero length"
                )
            self.pyramid_levels = levels
            self.scale_projectors = nn.ModuleList(
                [self._make_projector(dim, feature_dim) for dim in dims]
            )
            scale_shape = (
                (len(levels), num_concepts)
                if self.scale_fusion == "per_concept"
                else (len(levels),)
            )
            self.scale_logits = nn.Parameter(torch.zeros(scale_shape))
        else:
            self.projector = self._make_projector(feature_dim, feature_dim)

        self.centers = nn.Parameter(torch.empty(num_concepts, feature_dim))
        nn.init.normal_(self.centers, std=0.02)
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

        self.interaction_projector = None
        if self.interaction_concepts and self.num_people >= 2:
            self.interaction_projector = nn.Sequential(
                nn.LayerNorm(4 * feature_dim),
                nn.Linear(4 * feature_dim, feature_dim),
            )

    @staticmethod
    def _make_projector(input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )

    @property
    def temperature(self) -> torch.Tensor:
        # Keep cosine logits numerically useful while allowing the decoder to
        # learn sharper or softer concept assignments.
        return self.log_temperature.exp().clamp(0.02, 0.5)

    def normalized_centers(self) -> torch.Tensor:
        return F.normalize(self.centers, dim=-1)

    @property
    def scale_weights(self) -> torch.Tensor | None:
        if not self.multi_scale:
            return None
        return torch.softmax(self.scale_logits, dim=0)

    def _project_tokens(
        self, feature_map: torch.Tensor, projector: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if feature_map.ndim != 4:
            raise ValueError(
                f"feature_map must be [B,C,T,V], got {tuple(feature_map.shape)}"
            )
        tokens = feature_map.permute(0, 2, 3, 1).contiguous()  # [B,T,V,C]
        z = F.normalize(projector(tokens), dim=-1)
        centers = self.normalized_centers()
        cosine_similarity = torch.einsum("btvc,kc->btvk", z, centers)
        similarity = cosine_similarity / self.temperature

        # Concept-specific soft selection over joints/persons at each time.
        joint_attention = torch.softmax(similarity, dim=2)
        time_logits = (joint_attention * similarity).sum(dim=2)
        return z, cosine_similarity, time_logits

    def _interaction_logits(
        self, z: torch.Tensor, centers: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.interaction_projector is None or self.num_points is None:
            return None
        batch, timesteps, tokens, channels = z.shape
        expected_tokens = self.num_people * self.num_points
        if tokens != expected_tokens or self.num_people < 2:
            return None
        people = z.view(batch, timesteps, self.num_people, self.num_points, channels)
        person_repr = people.mean(dim=3)
        left, right = person_repr[:, :, 0], person_repr[:, :, 1]
        pair_features = torch.cat(
            [left, right, (left - right).abs(), left * right], dim=-1
        )
        pair_z = F.normalize(self.interaction_projector(pair_features), dim=-1)
        pair_cosine = torch.einsum("btc,kc->btk", pair_z, centers)
        return pair_cosine, pair_cosine / self.temperature

    def _temporal_statistics(
        self, time_probs: torch.Tensor, time_logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        safe_probs = time_probs.clamp(1e-5, 1.0 - 1e-5)
        timesteps = time_probs.shape[1]
        time_index = torch.linspace(
            0.0,
            1.0,
            timesteps,
            device=time_probs.device,
            dtype=time_probs.dtype,
        ).view(1, -1, 1)
        mass = safe_probs.sum(dim=1).clamp_min(1e-6)
        if self.adaptive_mil_pool is not None:
            sequence_probs = self.adaptive_mil_pool(time_logits)
        else:
            sequence_probs = 1.0 - torch.exp(
                torch.log1p(-safe_probs).sum(dim=1)
            )
        expected_time = (safe_probs * time_index).sum(dim=1) / mass
        peak_attention = torch.softmax(time_logits, dim=1)
        peak_time = (peak_attention * time_index).sum(dim=1)
        duration = safe_probs.mean(dim=1)
        rising = F.relu(safe_probs[:, 1:] - safe_probs[:, :-1]).sum(dim=1)
        falling = F.relu(safe_probs[:, :-1] - safe_probs[:, 1:]).sum(dim=1)
        return {
            "sequence_probs": sequence_probs,
            "expected_time": expected_time,
            "peak_time": peak_time,
            "duration": duration,
            "rising_activity": rising,
            "falling_activity": falling,
        }

    def forward(
        self, feature_map: torch.Tensor | Sequence[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        centers = self.normalized_centers()

        if self.multi_scale:
            if not isinstance(feature_map, (list, tuple)):
                raise ValueError("multi-scale decoder expects a feature-map pyramid")
            selected = []
            for level, projector in zip(self.pyramid_levels, self.scale_projectors):
                if level >= len(feature_map) or level < -len(feature_map):
                    raise ValueError(f"pyramid level {level} is unavailable")
                selected.append(self._project_tokens(feature_map[level], projector))
            target_steps = max(item[2].shape[1] for item in selected)
            fused_logits = []
            for _, _, logits in selected:
                if logits.shape[1] != target_steps:
                    logits = F.interpolate(
                        logits.transpose(1, 2),
                        size=target_steps,
                        mode="linear",
                        align_corners=False,
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
            final_z, final_cosine, _ = selected[-1]
        else:
            if isinstance(feature_map, (list, tuple)):
                feature_map = feature_map[-1]
            final_z, final_cosine, time_logits = self._project_tokens(
                feature_map, self.projector
            )

        # Interaction concepts use a person-pair evidence signal. The signal is
        # inserted only for configured interaction concept ids; all other
        # concepts remain token-grounded.
        interaction = self._interaction_logits(final_z, centers)
        grounding_cosine = final_cosine
        if interaction is not None and self.interaction_concepts:
            pair_cosine, interaction_logits = interaction
            if interaction_logits.shape[1] != time_logits.shape[1]:
                interaction_logits = F.interpolate(
                    interaction_logits.transpose(1, 2),
                    size=time_logits.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            time_logits = time_logits.clone()
            interaction_ids = list(self.interaction_concepts)
            time_logits[:, :, interaction_ids] = interaction_logits[:, :, interaction_ids]

            interaction_cosine = pair_cosine.unsqueeze(2).expand(
                -1, -1, final_cosine.shape[2], -1
            )
            grounding_cosine = grounding_cosine.clone()
            grounding_cosine[:, :, :, interaction_ids] = interaction_cosine[:, :, :, interaction_ids]

        time_probs = torch.sigmoid(time_logits)
        stats = self._temporal_statistics(time_probs, time_logits)
        return {
            "time_logits": time_logits,
            "time_probs": time_probs,
            "sequence_probs": stats["sequence_probs"],
            "expected_time": stats["expected_time"],
            "peak_time": stats["peak_time"],
            "duration": stats["duration"],
            "rising_activity": stats["rising_activity"],
            "falling_activity": stats["falling_activity"],
            "scale_weights": self.scale_weights,
            "similarity": final_cosine / self.temperature,
            "cosine_similarity": grounding_cosine,
            "centers": centers,
        }

    def center_grounding_loss(
        self,
        cosine_similarity: torch.Tensor,
        action_labels: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """Ground positive concepts and reject the most similar negatives.

        The legacy objective pulls every positive concept toward cosine 1. The
        absolute-margin objective stops at a fixed positive cosine. The
        pairwise-gap objective instead requires each positive to outrank the
        mean of the top-k negatives by a configurable cosine margin.
        """
        if cosine_similarity.ndim != 4:
            raise ValueError("cosine_similarity must have shape [B,T,V,K]")
        positive = support[action_labels].to(cosine_similarity.dtype)
        best_token_similarity = cosine_similarity.amax(dim=(1, 2))
        positive_normalizer = positive.sum(dim=1).clamp_min(1.0)
        if self.center_loss_mode == "legacy":
            loss = ((1.0 - best_token_similarity) * positive).sum(dim=1)
            return (loss / positive_normalizer).mean()

        negative_mask = positive <= 0.0
        negative_candidates = best_token_similarity.masked_fill(
            ~negative_mask, float("-inf")
        )
        selected_topk = min(
            max(self.center_gap_topk, self.center_negative_topk),
            self.num_concepts,
        )
        selected_negative_similarity = negative_candidates.topk(
            selected_topk, dim=1
        ).values
        if self.center_loss_mode == "pairwise_gap_hard_negative":
            gap_topk = min(self.center_gap_topk, selected_topk)
            gap_negative_similarity = selected_negative_similarity[:, :gap_topk]
            valid_gap_negative = torch.isfinite(gap_negative_similarity)
            has_negative = valid_gap_negative.any(dim=1, keepdim=True)
            topk_negative_mean = (
                gap_negative_similarity.masked_fill(~valid_gap_negative, 0.0).sum(
                    dim=1, keepdim=True
                )
                / valid_gap_negative.sum(dim=1, keepdim=True).clamp_min(1)
            )
            positive_gap = best_token_similarity - topk_negative_mean
            positive_loss = (
                F.relu(self.center_gap_margin - positive_gap)
                * positive
                * has_negative.to(positive.dtype)
            ).sum(dim=1) / positive_normalizer
        else:
            positive_loss = (
                F.relu(self.center_positive_margin - best_token_similarity)
                * positive
            ).sum(dim=1) / positive_normalizer

        negative_topk = min(self.center_negative_topk, selected_topk)
        hard_negative_similarity = selected_negative_similarity[:, :negative_topk]
        valid_negative = torch.isfinite(hard_negative_similarity)
        negative_loss = (
            F.relu(hard_negative_similarity - self.center_negative_margin)
            .masked_fill(~valid_negative, 0.0)
            .sum(dim=1)
            / valid_negative.sum(dim=1).clamp_min(1)
        )
        return (
            positive_loss + self.center_negative_weight * negative_loss
        ).mean()

    def center_separation_loss(self, support: torch.Tensor | None = None) -> torch.Tensor:
        centers = self.normalized_centers()
        similarity = centers @ centers.t()
        eye = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
        if self.center_loss_mode == "legacy":
            off_diagonal = similarity.masked_fill(eye, -1.0)
            return F.relu(off_diagonal - self.center_separation_margin).mean()

        if support is None:
            raise ValueError("support is required for hard-negative center separation")
        support = support.to(device=similarity.device, dtype=similarity.dtype)
        if support.ndim != 2 or support.shape[1] != self.num_concepts:
            raise ValueError("support must have shape [num_actions, num_concepts]")

        # Centers are negatives only when their concepts never co-occur in an
        # annotated action. This avoids repelling semantically compatible
        # concepts while focusing the loss on each center's nearest negatives.
        cooccurs = (support.t() @ support) > 0
        negative_pair = ~cooccurs
        negative_pair.masked_fill_(eye, False)
        candidates = similarity.masked_fill(~negative_pair, float("-inf"))
        topk = min(self.center_separation_topk, max(self.num_concepts - 1, 1))
        hard_negative_similarity = candidates.topk(topk, dim=1).values
        valid_negative = torch.isfinite(hard_negative_similarity)
        if not bool(valid_negative.any()):
            return similarity.sum() * 0.0
        per_center = (
            F.relu(hard_negative_similarity - self.center_separation_margin)
            .masked_fill(~valid_negative, 0.0)
            .sum(dim=1)
            / valid_negative.sum(dim=1).clamp_min(1)
        )
        return per_center.mean()

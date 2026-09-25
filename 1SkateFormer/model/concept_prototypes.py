"""Input-conditioned concept prototype grounding for SkateFormer."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


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
        num_people: int = 1,
        num_points: int | None = None,
        interaction_concepts: Iterable[int] | None = None,
        token_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_concepts = num_concepts
        self.multi_scale = bool(multi_scale)
        self.num_people = int(num_people)
        self.num_points = int(num_points) if num_points is not None else None
        self.interaction_concepts = tuple(
            sorted(set(int(i) for i in (interaction_concepts or [])))
        )

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
            self.scale_logits = nn.Parameter(torch.zeros(len(levels)))
        else:
            self.projector = self._make_projector(feature_dim, feature_dim)

        self.centers = nn.Parameter(torch.empty(num_concepts, feature_dim))
        nn.init.normal_(self.centers, std=0.02)
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

        # Optional [V, K] boolean mask restricting which tokens each concept
        # may attend to (region pooling).  Non-persistent so checkpoints stay
        # interchangeable between region-pooled and global-pooled runs.
        if token_mask is not None:
            token_mask = torch.as_tensor(token_mask, dtype=torch.bool)
            if token_mask.ndim != 2 or token_mask.shape[1] != num_concepts:
                raise ValueError("token_mask must have shape [num_tokens, num_concepts]")
            if not token_mask.any(dim=0).all():
                raise ValueError("token_mask leaves a concept without any token")
            self.register_buffer("token_mask", token_mask, persistent=False)
        else:
            self.token_mask = None

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
        attention_logits = similarity
        if self.token_mask is not None:
            if self.token_mask.shape[0] != similarity.shape[2]:
                raise ValueError(
                    f"token_mask covers {self.token_mask.shape[0]} tokens, "
                    f"feature map has {similarity.shape[2]}"
                )
            attention_logits = similarity.masked_fill(
                ~self.token_mask, torch.finfo(similarity.dtype).min
            )
            # Grounding losses must also ignore out-of-region tokens.
            cosine_similarity = cosine_similarity.masked_fill(~self.token_mask, -1.0)
        joint_attention = torch.softmax(attention_logits, dim=2)
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

    @staticmethod
    def _temporal_statistics(
        time_probs: torch.Tensor, time_logits: torch.Tensor
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
            scale_weights = torch.softmax(self.scale_logits, dim=0)
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
        """Pull supported centers toward an observed token or pair evidence."""
        if cosine_similarity.ndim != 4:
            raise ValueError("cosine_similarity must have shape [B,T,V,K]")
        positive = support[action_labels].to(cosine_similarity.dtype)
        best_token_similarity = cosine_similarity.amax(dim=(1, 2))
        loss = ((1.0 - best_token_similarity) * positive).sum(dim=1)
        normalizer = positive.sum(dim=1).clamp_min(1.0)
        return (loss / normalizer).mean()

    def center_separation_loss(self, margin: float = 0.15) -> torch.Tensor:
        centers = self.normalized_centers()
        similarity = centers @ centers.t()
        eye = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
        off_diagonal = similarity.masked_fill(eye, -1.0)
        return F.relu(off_diagonal - margin).mean()

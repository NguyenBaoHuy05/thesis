"""HCG-NS-SkateFormer-RF: relation-free hierarchical neuro-symbolic model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from model.SkateFormer import SkateFormer_
from model.concept_ontology import load_ontology_metadata
from model.hierarchical_concept_decoder import HierarchicalConceptDecoder
from model.hierarchical_symbolic_decoder import HierarchicalSymbolicDecoder


class HCGNSSkateFormer(nn.Module):
    """SkateFormer backbone + temporal unary concepts + hierarchy + composition."""

    def __init__(
        self,
        backbone_args: dict[str, Any],
        concept_csv: str | Path,
        metadata_csv: str | Path,
        num_classes: int,
        num_concepts: int = 78,
        residual_action_head: bool = False,
        residual_action_weight: float = 0.1,
        prototype_args: dict[str, Any] | None = None,
        action_logit_scale: float = 1.0,
        concept_score_mode: str = 'probability',
        concept_idf_power: float = 0.0,
    ):
        super().__init__()
        args = dict(backbone_args)
        args["num_classes"] = int(num_classes)
        self.backbone = SkateFormer_(**args)
        feature_dim = int(args.get("channels", (96, 192, 192, 192))[-1])
        self.feature_dim = feature_dim
        self.num_classes = int(num_classes)
        self.num_concepts = int(num_concepts)
        metadata = load_ontology_metadata(metadata_csv, num_concepts=num_concepts)
        prototype_args = dict(prototype_args or {})
        removed_scope_options = {
            "concept_classifier", "scope_enabled", "scope_fusion", "scope_scale",
        }.intersection(prototype_args)
        if removed_scope_options:
            raise ValueError(
                "scope-head options were removed; scope metadata now routes pose and "
                "motion projectors: " + ", ".join(sorted(removed_scope_options))
            )
        parent_enabled = bool(prototype_args.get("parent_enabled", True))
        all_pyramid_dims = tuple(args.get("channels", (96, 192, 192, 192)))
        pyramid_levels = tuple(prototype_args.get("pyramid_levels", (2, 3)))
        try:
            pyramid_dims = tuple(all_pyramid_dims[level] for level in pyramid_levels)
        except IndexError as exc:
            raise ValueError(
                f"prototype pyramid levels {pyramid_levels} exceed backbone stages"
            ) from exc
        self.concept_decoder = HierarchicalConceptDecoder(
            feature_dim=feature_dim,
            metadata=metadata,
            num_concepts=num_concepts,
            temperature=float(prototype_args.get("temperature", 0.07)),
            multi_scale=bool(prototype_args.get("multi_scale", False)),
            pyramid_dims=pyramid_dims,
            pyramid_levels=pyramid_levels,
            temporal_pooling=str(
                prototype_args.get("temporal_pooling", "adaptive_mil")
            ),
            scale_fusion=str(prototype_args.get("scale_fusion", "shared")),
            num_people=int(args.get("num_people", 1)),
            num_points=int(args.get("num_points", 0)) or None,
            interaction_reasoning=bool(prototype_args.get("interaction_reasoning", True)),
            parent_enabled=parent_enabled,
            refiner_enabled=bool(prototype_args.get("refiner_enabled", True)),
            refiner_hidden_dim=int(prototype_args.get("refiner_hidden_dim", 128)),
            temporal_gru_enabled=bool(
                prototype_args.get("temporal_gru_enabled", False)
            ),
            temporal_gru_hidden_dim=int(
                prototype_args.get("temporal_gru_hidden_dim", 64)
            ),
            temporal_gru_layers=int(
                prototype_args.get("temporal_gru_layers", 1)
            ),
            temporal_gru_dropout=float(
                prototype_args.get("temporal_gru_dropout", 0.0)
            ),
            temporal_gru_residual_init=float(
                prototype_args.get("temporal_gru_residual_init", 0.0)
            ),
            center_loss_mode=str(prototype_args.get("center_loss_mode", "legacy")),
            center_positive_margin=float(
                prototype_args.get("center_positive_margin", 0.35)
            ),
            center_gap_margin=float(
                prototype_args.get("center_gap_margin", 0.30)
            ),
            center_gap_topk=int(
                prototype_args.get("center_gap_topk", 5)
            ),
            center_negative_margin=float(
                prototype_args.get("center_negative_margin", 0.10)
            ),
            center_negative_weight=float(
                prototype_args.get("center_negative_weight", 0.5)
            ),
            center_negative_topk=int(
                prototype_args.get("center_negative_topk", 10)
            ),
            center_separation_margin=float(
                prototype_args.get("center_separation_margin", 0.15)
            ),
            center_separation_topk=int(
                prototype_args.get("center_separation_topk", 5)
            ),
        )
        self.symbolic = HierarchicalSymbolicDecoder(
            concept_csv=concept_csv,
            metadata_csv=metadata_csv,
            num_classes=num_classes,
            num_concepts=num_concepts,
            action_logit_scale=float(action_logit_scale),
            concept_score_mode=str(concept_score_mode),
            concept_idf_power=float(concept_idf_power),
        )
        self.residual_action_head_enabled = bool(residual_action_head)
        self.residual_action_weight = float(residual_action_weight)
        if self.residual_action_head_enabled:
            self.residual_action_head = nn.Linear(feature_dim, num_classes)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if not any(parameter.requires_grad for parameter in module.parameters()):
                    module.eval()
        return self

    def unfreeze_last_stage(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.stages[-1].parameters():
            parameter.requires_grad = True

    def forward(self, x: torch.Tensor, index_t: torch.Tensor | None = None) -> dict[str, Any]:
        feature_output = self.backbone.forward_features_raw(
            x, index_t, return_pyramid=self.concept_decoder.multi_scale
        )
        feature_map = feature_output[-1] if isinstance(feature_output, (list, tuple)) else feature_output
        concepts = self.concept_decoder(feature_output)
        action_scores = self.symbolic(concepts["sequence_probs"])
        residual_scores = None
        if self.residual_action_head_enabled:
            residual_scores = self.residual_action_head(feature_map.mean(dim=(2, 3)))
            action_scores = action_scores + self.residual_action_weight * residual_scores
        return {
            "feature_map": feature_map,
            "action_scores": action_scores,
            "time_logits": concepts["time_logits"],
            "time_probs": concepts["time_probs"],
            "raw_sequence_logits": concepts["raw_sequence_logits"],
            "raw_sequence_probs": concepts["raw_sequence_probs"],
            "sequence_logits": concepts["sequence_logits"],
            "sequence_probs": concepts["sequence_probs"],
            "expected_time": concepts["expected_time"],
            "peak_time": concepts["peak_time"],
            "duration": concepts["duration"],
            "rising_activity": concepts["rising_activity"],
            "falling_activity": concepts["falling_activity"],
            "parent_logits": concepts["parent_logits"],
            "parent_probs": concepts["parent_probs"],
            "parent_for_child": concepts["parent_for_child"],
            "scale_weights": concepts["scale_weights"],
            "cosine_similarity": concepts["cosine_similarity"],
            "centers": concepts["centers"],
            "residual_action_scores": residual_scores,
        }

    def compute_loss(
        self, output: dict[str, Any], action_labels: torch.Tensor,
        weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        weights = {
            "parent": 0.25,
            "concept": 0.25,
            "hierarchy": 0.10,
            "graph": 0.0,
            "center": 0.05,
            **(weights or {}),
        }
        action_loss = F.cross_entropy(output["action_scores"], action_labels)
        concept_loss = self.symbolic.concept_loss(
            output["sequence_logits"], action_labels
        )
        if self.concept_decoder.parent_enabled:
            parent_loss = self.symbolic.parent_loss(
                output["parent_logits"], action_labels
            )
            graph_loss = self.concept_decoder.parent_graph_loss(
                output["sequence_probs"], output["parent_probs"]
            )
        else:
            parent_loss = output["sequence_probs"].new_zeros(())
            graph_loss = output["sequence_probs"].new_zeros(())
        grounding_loss = self.concept_decoder.center_grounding_loss(
            output["cosine_similarity"], action_labels, self.symbolic.support
        )
        separation_loss = self.concept_decoder.center_separation_loss(
            self.symbolic.support
        )
        center_loss = grounding_loss + separation_loss

        # Logging-only diagnostics. Prototype-only values are zero in the
        # direct scope-head ablation, which has no centers or token cosine.
        with torch.no_grad():
            concept_targets = self.symbolic.support[action_labels].to(
                output["sequence_probs"].dtype
            )
            negative_targets = 1.0 - concept_targets
            if output["cosine_similarity"] is None:
                positive_best_cosine = output["sequence_probs"].new_zeros(())
                negative_best_cosine = output["sequence_probs"].new_zeros(())
                cosine_gap = output["sequence_probs"].new_zeros(())
                positive_margin_violation = output["sequence_probs"].new_zeros(())
                gap_margin_violation = output["sequence_probs"].new_zeros(())
                hard_negative_topk_mean = output["sequence_probs"].new_zeros(())
                positive_topk_gap_mean = output["sequence_probs"].new_zeros(())
                negative_margin_violation = output["sequence_probs"].new_zeros(())
            else:
                best_token_cosine = output["cosine_similarity"].amax(dim=(1, 2))
                positive_best_cosine = (
                    (best_token_cosine * concept_targets).sum()
                    / concept_targets.sum().clamp_min(1.0)
                )
                negative_best_cosine = (
                    (best_token_cosine * negative_targets).sum()
                    / negative_targets.sum().clamp_min(1.0)
                )
                cosine_gap = positive_best_cosine - negative_best_cosine
                base = self.concept_decoder.base
                if base is None:  # pragma: no cover - cosine implies a prototype base
                    positive_margin_violation = best_token_cosine.new_zeros(())
                    gap_margin_violation = best_token_cosine.new_zeros(())
                    hard_negative_topk_mean = best_token_cosine.new_zeros(())
                    positive_topk_gap_mean = best_token_cosine.new_zeros(())
                    negative_margin_violation = best_token_cosine.new_zeros(())
                else:
                    if base.center_loss_mode == "pairwise_gap_hard_negative":
                        negative_candidates = best_token_cosine.masked_fill(
                            concept_targets > 0.0, float("-inf")
                        )
                        gap_topk = min(
                            base.center_gap_topk, best_token_cosine.shape[1]
                        )
                        topk_negative = negative_candidates.topk(
                            gap_topk, dim=1
                        ).values
                        valid_topk_negative = torch.isfinite(topk_negative)
                        has_negative = valid_topk_negative.any(dim=1, keepdim=True)
                        topk_negative_mean = (
                            topk_negative.masked_fill(
                                ~valid_topk_negative, 0.0
                            ).sum(dim=1, keepdim=True)
                            / valid_topk_negative.sum(
                                dim=1, keepdim=True
                            ).clamp_min(1)
                        )
                        positive_gap = best_token_cosine - topk_negative_mean
                        hard_negative_topk_mean = (
                            topk_negative_mean
                            * has_negative.to(topk_negative_mean.dtype)
                        ).sum() / has_negative.sum().clamp_min(1)
                        positive_topk_gap_mean = (
                            positive_gap
                            * concept_targets
                            * has_negative.to(best_token_cosine.dtype)
                        ).sum() / concept_targets.sum().clamp_min(1.0)
                        gap_margin_violation = (
                            (
                                positive_gap < base.center_gap_margin
                            ).to(best_token_cosine.dtype)
                            * concept_targets
                            * has_negative.to(best_token_cosine.dtype)
                        ).sum() / concept_targets.sum().clamp_min(1.0)
                        positive_margin_violation = best_token_cosine.new_zeros(())
                    else:
                        positive_margin_violation = (
                            (
                                best_token_cosine < base.center_positive_margin
                            ).to(best_token_cosine.dtype) * concept_targets
                        ).sum() / concept_targets.sum().clamp_min(1.0)
                        gap_margin_violation = best_token_cosine.new_zeros(())
                        hard_negative_topk_mean = best_token_cosine.new_zeros(())
                        positive_topk_gap_mean = best_token_cosine.new_zeros(())
                    negative_margin_violation = (
                        (
                            best_token_cosine > base.center_negative_margin
                        ).to(best_token_cosine.dtype) * negative_targets
                    ).sum() / negative_targets.sum().clamp_min(1.0)

            sequence_probs = output["sequence_probs"]
            positive_concept_probability = (
                (sequence_probs * concept_targets).sum()
                / concept_targets.sum().clamp_min(1.0)
            )
            negative_concept_probability = (
                (sequence_probs * negative_targets).sum()
                / negative_targets.sum().clamp_min(1.0)
            )
            positive_concept_recall = (
                ((sequence_probs >= 0.5).to(sequence_probs.dtype) * concept_targets).sum()
                / concept_targets.sum().clamp_min(1.0)
            )
            negative_concept_violation_rate = (
                ((sequence_probs >= 0.5).to(sequence_probs.dtype) * negative_targets).sum()
                / negative_targets.sum().clamp_min(1.0)
            )

        total = (
            action_loss
            + weights["concept"] * concept_loss
            + weights["parent"] * parent_loss
            + weights["graph"] * graph_loss
            + weights["center"] * center_loss
        )
        return {
            "total": total,
            "action": action_loss,
            "concept": concept_loss,
            "parent": parent_loss,
            "graph": graph_loss,
            "center": center_loss,
            "grounding": grounding_loss,
            "separation": separation_loss,
            "weighted_concept": float(weights["concept"]) * concept_loss.detach(),
            "weighted_parent": float(weights["parent"]) * parent_loss.detach(),
            "weighted_graph": float(weights["graph"]) * graph_loss.detach(),
            "weighted_center": float(weights["center"]) * center_loss.detach(),
            "center_positive_best_cosine": positive_best_cosine,
            "center_negative_best_cosine": negative_best_cosine,
            "center_positive_negative_gap": cosine_gap,
            "grounding_positive_margin_violation_rate": positive_margin_violation,
            "grounding_gap_margin_violation_rate": gap_margin_violation,
            "grounding_hard_negative_topk_mean_cosine": hard_negative_topk_mean,
            "grounding_positive_topk_gap_mean": positive_topk_gap_mean,
            "grounding_negative_margin_violation_rate": negative_margin_violation,
            "concept_positive_probability": positive_concept_probability,
            "concept_negative_probability": negative_concept_probability,
            "concept_positive_recall_at_0_5": positive_concept_recall,
            "concept_negative_violation_at_0_5": negative_concept_violation_rate,
        }

    @torch.no_grad()
    def training_diagnostics(self) -> dict[str, Any]:
        """Return scalar decoder diagnostics suitable for per-epoch JSON logs."""
        base = self.concept_decoder.base
        raw_centers = base.centers.detach()
        center_norms = raw_centers.norm(dim=-1)
        centers = base.normalized_centers().detach()
        center_similarity = centers @ centers.t()
        off_diagonal_mask = ~torch.eye(
            centers.shape[0], dtype=torch.bool, device=centers.device
        )
        pairwise = center_similarity[off_diagonal_mask]
        nearest = center_similarity.masked_fill(
            ~off_diagonal_mask, float("-inf")
        ).amax(dim=1)
        closest_flat_index = center_similarity.masked_fill(
            ~off_diagonal_mask, float("-inf")
        ).argmax()
        closest_left_id = int(
            (closest_flat_index // centers.shape[0]).detach().cpu().item()
        )
        closest_right_id = int(
            (closest_flat_index % centers.shape[0]).detach().cpu().item()
        )

        diagnostics = {
            "center_loss_mode": base.center_loss_mode,
            "center_positive_margin": base.center_positive_margin,
            "center_gap_margin": base.center_gap_margin,
            "center_gap_topk": float(base.center_gap_topk),
            "center_negative_margin": base.center_negative_margin,
            "center_separation_margin": base.center_separation_margin,
            "temperature": float(base.temperature.detach().cpu()),
            "center_raw_norm_mean": float(center_norms.mean().cpu()),
            "center_raw_norm_std": float(center_norms.std(unbiased=False).cpu()),
            "center_raw_norm_min": float(center_norms.min().cpu()),
            "center_raw_norm_max": float(center_norms.max().cpu()),
            "center_pair_cosine_mean": float(pairwise.mean().cpu()),
            "center_pair_cosine_std": float(pairwise.std(unbiased=False).cpu()),
            "center_pair_cosine_min": float(pairwise.min().cpu()),
            "center_pair_cosine_max": float(pairwise.max().cpu()),
            "center_nearest_cosine_mean": float(nearest.mean().cpu()),
            "center_nearest_cosine_max": float(nearest.max().cpu()),
            "center_margin_violation_rate": float(
                (pairwise > base.center_separation_margin).float().mean().cpu()
            ),
            "center_closest_pair_left_id": closest_left_id,
            "center_closest_pair_right_id": closest_right_id,
            "center_closest_pair_left_name": self.symbolic.concept_names[closest_left_id],
            "center_closest_pair_right_name": self.symbolic.concept_names[closest_right_id],
        }

        support = self.symbolic.support.to(
            device=center_similarity.device, dtype=center_similarity.dtype
        )
        negative_pair_mask = (support.t() @ support) <= 0
        negative_pair_mask &= off_diagonal_mask
        negative_pairwise = center_similarity[negative_pair_mask]
        if negative_pairwise.numel() > 0:
            diagnostics.update({
                "center_negative_pair_cosine_mean": float(
                    negative_pairwise.mean().cpu()
                ),
                "center_negative_pair_margin_violation_rate": float(
                    (
                        negative_pairwise > base.center_separation_margin
                    ).float().mean().cpu()
                ),
            })

        scale_weights = base.scale_weights
        if scale_weights is not None:
            scale_weights = scale_weights.detach()
            if scale_weights.ndim == 1:
                per_concept_weights = scale_weights[:, None].expand(
                    -1, base.num_concepts
                )
            else:
                per_concept_weights = scale_weights
            dominant_levels = per_concept_weights.argmax(dim=0)
            if per_concept_weights.shape[0] > 1:
                top_two = per_concept_weights.topk(k=2, dim=0).values
                unique_preference = (top_two[0] - top_two[1]) > 1e-6
                diagnostics["scale_tied_concept_count"] = float(
                    (~unique_preference).sum().cpu()
                )
            else:
                unique_preference = torch.ones(
                    base.num_concepts, dtype=torch.bool,
                    device=per_concept_weights.device,
                )
            for scale_index, level in enumerate(base.pyramid_levels):
                values = per_concept_weights[scale_index]
                prefix = f"scale_level_{level}_weight"
                diagnostics.update({
                    f"{prefix}_mean": float(values.mean().cpu()),
                    f"{prefix}_std": float(values.std(unbiased=False).cpu()),
                    f"{prefix}_min": float(values.min().cpu()),
                    f"{prefix}_max": float(values.max().cpu()),
                    f"{prefix}_min_concept_id": int(values.argmin().cpu()),
                    f"{prefix}_max_concept_id": int(values.argmax().cpu()),
                    f"scale_level_{level}_dominant_concept_count": float(
                        (
                            (dominant_levels == scale_index) & unique_preference
                        ).sum().cpu()
                    ),
                })
            if per_concept_weights.shape[0] > 1:
                entropy = -(
                    per_concept_weights
                    * per_concept_weights.clamp_min(1e-12).log()
                ).sum(dim=0)
                entropy = entropy / math.log(float(per_concept_weights.shape[0]))
                diagnostics.update({
                    "scale_weight_normalized_entropy_mean": float(entropy.mean().cpu()),
                    "scale_weight_normalized_entropy_min": float(entropy.min().cpu()),
                })

        mil_pool = base.adaptive_mil_pool
        if mil_pool is not None:
            for name, values in {
                "mil_beta": mil_pool.beta.detach(),
                "mil_mix": mil_pool.mix.detach(),
                "mil_bias": mil_pool.bias.detach(),
            }.items():
                diagnostics.update({
                    f"{name}_mean": float(values.mean().cpu()),
                    f"{name}_std": float(values.std(unbiased=False).cpu()),
                    f"{name}_min": float(values.min().cpu()),
                    f"{name}_max": float(values.max().cpu()),
                })

        temporal_gru = base.temporal_gru
        if temporal_gru is not None:
            residual_scale = temporal_gru.residual_scale.detach()
            diagnostics.update({
                "temporal_gru_residual_scale_mean": float(
                    residual_scale.mean().cpu()
                ),
                "temporal_gru_residual_scale_std": float(
                    residual_scale.std(unbiased=False).cpu()
                ),
                "temporal_gru_residual_scale_min": float(
                    residual_scale.min().cpu()
                ),
                "temporal_gru_residual_scale_max": float(
                    residual_scale.max().cpu()
                ),
            })

        refiner = self.concept_decoder.concept_refiner
        if hasattr(refiner, "residual_scale"):
            residual_scale = refiner.residual_scale.detach()
            diagnostics.update({
                "refiner_residual_scale_mean": float(residual_scale.mean().cpu()),
                "refiner_residual_scale_std": float(
                    residual_scale.std(unbiased=False).cpu()
                ),
                "refiner_residual_scale_min": float(residual_scale.min().cpu()),
                "refiner_residual_scale_max": float(residual_scale.max().cpu()),
            })
        return diagnostics

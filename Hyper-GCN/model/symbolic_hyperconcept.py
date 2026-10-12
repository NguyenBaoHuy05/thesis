"""Symbolic body hypergraph to action-hyperconcept bridge for Hyper-GCN."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from model.concept_schema import (
    BODY_PART_JOINTS, PART_NAMES, concept_joints, concept_parts,
    load_concept_tables,
)
from model.grounded_concept_encoder import GroundedConceptEncoder


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


def _concept_joint_mask(metadata, num_points=25):
    mask = torch.zeros(len(metadata), int(num_points), dtype=torch.bool)
    for index, row in enumerate(metadata):
        joints = concept_joints(
            row["concept_name"], row.get("scope", ""), row.get("group", ""),
            num_points=num_points,
        )
        mask[index, list(joints)] = True
    if not mask.any(dim=1).all():
        raise ValueError("every concept must map to at least one physical joint")
    return mask


def _interaction_mask(metadata):
    return torch.tensor([
        row.get("group", "").strip().lower() == "participant"
        or row["concept_name"].strip().lower().startswith("interaction_")
        for row in metadata
    ], dtype=torch.bool)


def _concept_affinity(support):
    # Normalize each concept across classes before measuring co-occurrence.
    # Zero-support concepts have no off-diagonal edges and retain a self-loop.
    columns = F.normalize(support, p=2, dim=0)
    affinity = (columns.t() @ columns).clamp(0.0, 1.0)
    affinity.fill_diagonal_(0.0)
    affinity = affinity + torch.eye(support.shape[1], device=support.device, dtype=support.dtype)
    return _row_normalize(affinity)


def _semantic_hypergraph(concept_part, metadata):
    """Create named body/group/scope hyperedges without using action labels."""
    groups = tuple(dict.fromkeys(row["group"].strip().lower() for row in metadata))
    scopes = tuple(dict.fromkeys(row["scope"].strip().lower() for row in metadata))
    edge_names = (
        tuple("body:" + name for name in PART_NAMES)
        + tuple("group:" + name for name in groups)
        + tuple("scope:" + name for name in scopes)
    )
    incidence = torch.zeros(len(metadata), len(edge_names), dtype=torch.float32)
    incidence[:, :len(PART_NAMES)] = (concept_part > 0).to(incidence)
    group_offset = len(PART_NAMES)
    scope_offset = group_offset + len(groups)
    group_ids = {name: index for index, name in enumerate(groups)}
    scope_ids = {name: index for index, name in enumerate(scopes)}
    for concept_index, row in enumerate(metadata):
        incidence[concept_index, group_offset + group_ids[row["group"].strip().lower()]] = 1
        incidence[concept_index, scope_offset + scope_ids[row["scope"].strip().lower()]] = 1
    if (incidence.sum(dim=0) == 0).any() or (incidence.sum(dim=1) == 0).any():
        raise ValueError("semantic hypergraph contains an empty concept or hyperedge")
    return incidence, edge_names


class SymbolicHyperConceptBridge(nn.Module):
    """Parallel symbolic branch for concept prediction."""

    def __init__(
        self, feature_dim, concept_csv, metadata_csv, num_classes,
        num_concepts=102, num_people=2, num_points=25, temperature=0.2,
        graph_init=0.0,
        balance_loss=False, pos_weight_max=10.0, pos_weight_source="class",
        action_decoder="none", action_temperature=0.1,
        hypergraph_mode="static_graph",
        concept_dim=64, temporal_bins=4, grounding_dropout=0.1,
        action_matching="bernoulli", action_missing_weight=1.0,
        action_extra_weight=0.15, action_idf_weighting=True,
        action_idf_min=0.5, action_idf_max=2.0,
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
        if action_matching not in {"bernoulli", "cosine", "tversky"}:
            raise ValueError("action_matching must be 'bernoulli', 'cosine' or 'tversky'")
        if hypergraph_mode not in {"static_graph", "learnable_v1"}:
            raise ValueError("hypergraph_mode must be 'static_graph' or 'learnable_v1'")
        self.hypergraph_mode = hypergraph_mode
        self.action_decoder = action_decoder
        self.action_matching = action_matching
        self.action_temperature = float(action_temperature)
        if not math.isfinite(self.action_temperature) or self.action_temperature <= 0:
            raise ValueError("symbolic action temperature must be finite and positive")
        self.action_missing_weight = float(action_missing_weight)
        self.action_extra_weight = float(action_extra_weight)
        if (not math.isfinite(self.action_missing_weight)
                or not math.isfinite(self.action_extra_weight)
                or self.action_missing_weight < 0 or self.action_extra_weight < 0):
            raise ValueError("Tversky missing/extra weights must be finite and nonnegative")
        self.action_idf_weighting = bool(action_idf_weighting)
        self.action_idf_min = float(action_idf_min)
        self.action_idf_max = float(action_idf_max)
        if (not math.isfinite(self.action_idf_min)
                or not math.isfinite(self.action_idf_max)
                or self.action_idf_min <= 0
                or self.action_idf_max < self.action_idf_min):
            raise ValueError("IDF limits must be finite, positive and ordered")
        self.metadata = metadata
        self.concept_names = [row["concept_name"] for row in metadata]
        self.concept_dim = int(concept_dim if action_decoder == "prototype" else feature_dim)
        if self.concept_dim < 1:
            raise ValueError("concept_dim must be positive")
        hypergraph_prior, hyperedge_names = _semantic_hypergraph(concept_part, metadata)
        self.hyperedge_names = hyperedge_names
        # Rebuild CSV-derived buffers, rather than restoring stale mappings from weights.
        self.register_buffer("part_incidence", _body_part_incidence(), persistent=False)
        self.register_buffer("concept_part", concept_part, persistent=False)
        self.register_buffer("concept_affinity", concept_affinity, persistent=False)
        self.register_buffer("hypergraph_prior", hypergraph_prior, persistent=False)
        soft_prior = torch.where(
            hypergraph_prior.bool(), torch.full_like(hypergraph_prior, 0.95),
            torch.full_like(hypergraph_prior, 0.01),
        )
        self.register_buffer("hypergraph_prior_probability", soft_prior, persistent=False)
        self.register_buffer("support", support, persistent=False)
        frequency = support.sum(dim=0)
        idf = torch.log((float(support.shape[0]) + 1.0) / (frequency + 1.0))
        idf = idf.clamp(self.action_idf_min, self.action_idf_max)
        if not self.action_idf_weighting:
            idf = torch.ones_like(idf)
        self.register_buffer("action_idf", idf, persistent=False)
        self.register_buffer("positive_weight", torch.ones(self.num_concepts), persistent=False)
        # Persist training frequencies so test mode does not need the training dataset.
        self.register_buffer("training_class_counts", torch.zeros(int(num_classes)))
        self._positive_weight_ready = not self.balance_loss or pos_weight_source == "class"
        if pos_weight_source == "class":
            self._set_positive_weight(torch.ones(int(num_classes)))
        joint_mask = _concept_joint_mask(metadata, num_points=self.num_points)
        interaction_mask = _interaction_mask(metadata)
        self.grounded_encoder = None
        if self.action_decoder == "prototype":
            self.grounded_encoder = GroundedConceptEncoder(
                feature_dim=self.feature_dim,
                concept_dim=self.concept_dim,
                concept_joint_mask=joint_mask,
                interaction_mask=interaction_mask,
                num_people=self.num_people,
                num_points=self.num_points,
                temporal_bins=temporal_bins,
                dropout=grounding_dropout,
            )
        self.concept_queries = nn.Parameter(torch.empty(self.num_concepts, self.concept_dim))
        self.concept_bias = nn.Parameter(torch.zeros(self.num_concepts))
        self.concept_graph_gate = nn.Parameter(torch.tensor(float(graph_init)))
        if self.hypergraph_mode == "learnable_v1":
            self.hypergraph_delta = nn.Parameter(torch.zeros_like(hypergraph_prior))
            initial_edge_weight = math.log(math.expm1(1.0))
            self.hyperedge_weight_raw = nn.Parameter(
                torch.full((hypergraph_prior.shape[1],), initial_edge_weight)
            )
            self.hypergraph_gate = nn.Parameter(torch.tensor(float(graph_init)))
        else:
            self.register_parameter("hypergraph_delta", None)
            self.register_parameter("hyperedge_weight_raw", None)
            self.register_parameter("hypergraph_gate", None)
        nn.init.normal_(self.concept_queries, mean=0.0, std=0.02)
        self._grounding_accumulator = None

    def _legacy_blackbox_forward(self, feature_map):
        """Predict concepts without modifying the backbone feature map."""
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
        return feature_map, sequence_logits.mean(dim=1)

    def _propagate_concept_tokens(self, concept_tokens):
        if self.hypergraph_mode == "learnable_v1":
            incidence = self.learned_hypergraph().to(concept_tokens)
            edge_weight = F.softplus(self.hyperedge_weight_raw).to(concept_tokens)
            edge_weight = edge_weight / edge_weight.mean().clamp_min(1e-12)
            edge_degree = incidence.sum(dim=0).clamp_min(1e-12)
            edge_features = torch.einsum(
                "qe,bqd->bed", incidence, concept_tokens
            ) / edge_degree.view(1, -1, 1)
            edge_features = edge_features * edge_weight.view(1, -1, 1)
            node_degree = torch.einsum(
                "qe,e->q", incidence, edge_weight
            ).clamp_min(1e-12)
            update = torch.einsum(
                "qe,bed->bqd", incidence, edge_features
            ) / node_degree.view(1, -1, 1)
            return concept_tokens + torch.tanh(self.hypergraph_gate) * update
        update = torch.einsum(
            "qr,brd->bqd", self.concept_affinity.to(concept_tokens), concept_tokens
        )
        return concept_tokens + torch.tanh(self.concept_graph_gate) * update

    def forward(self, feature_map, valid_person_mask=None):
        expected_nodes = self.num_people * self.num_points
        input_was_flat = feature_map.ndim == 4
        if input_was_flat:
            if feature_map.shape[-1] != expected_nodes:
                raise ValueError(
                    "expected [N,C,T,num_people*num_points], got "
                    f"{tuple(feature_map.shape)}"
                )
            batch, channels, timesteps, _ = feature_map.shape
            grounded_input = feature_map.reshape(
                batch, channels, timesteps, self.num_people, self.num_points
            ).permute(0, 3, 1, 2, 4).contiguous()
        elif feature_map.ndim == 5:
            batch, people, channels, timesteps, points = feature_map.shape
            if (people, channels, points) != (
                    self.num_people, self.feature_dim, self.num_points):
                raise ValueError(
                    "expected [N,M,C,T,V] with M/C/V={}, got {}".format(
                        (self.num_people, self.feature_dim, self.num_points),
                        tuple(feature_map.shape),
                    )
                )
            grounded_input = feature_map
        else:
            raise ValueError(
                "symbolic bridge expects [N,M,C,T,V] or legacy [N,C,T,M*V], got "
                f"{tuple(feature_map.shape)}"
            )

        if self.action_decoder == "none":
            flat = grounded_input.permute(0, 2, 3, 1, 4).reshape(
                batch, channels, timesteps, expected_nodes
            )
            fused, concept_logits = self._legacy_blackbox_forward(flat)
            if input_was_flat:
                return fused, concept_logits
            fused = fused.reshape(
                batch, channels, timesteps, self.num_people, self.num_points
            ).permute(0, 3, 1, 2, 4).contiguous()
            return fused, concept_logits

        concept_tokens, auxiliary = self.grounded_encoder(
            grounded_input, valid_person_mask=valid_person_mask
        )
        concept_tokens = self._propagate_concept_tokens(concept_tokens)
        logits = torch.einsum(
            "bqd,qd->bq",
            F.normalize(concept_tokens, dim=-1),
            F.normalize(self.concept_queries, dim=-1),
        )
        logits = logits / max(self.temperature, 1e-6)
        logits = logits + self.concept_bias.view(1, -1)
        self._accumulate_grounding(auxiliary)
        return feature_map, logits

    def reset_grounding_statistics(self):
        if self.grounded_encoder is None:
            raise RuntimeError("grounded concept encoder is not enabled")
        bins = self.grounded_encoder.temporal_bins
        self._grounding_accumulator = {
            "samples": 0,
            "temporal": torch.zeros(self.num_concepts, bins),
            "person": torch.zeros(self.num_concepts, self.num_people),
            "token_norm": torch.zeros(self.num_concepts),
            "valid_people": torch.zeros(self.num_people),
        }

    def _accumulate_grounding(self, auxiliary):
        if self._grounding_accumulator is None:
            return
        temporal = auxiliary["temporal_weights"].detach().float().cpu()
        person = auxiliary["person_weights"].detach().float().cpu()
        token_norm = auxiliary["token_norm"].detach().float().cpu()
        valid_people = auxiliary["valid_person_mask"].detach().float().cpu()
        # Weight temporal summaries by the selected person for non-interaction
        # concepts. Pair concepts remain interpretable through their person stats.
        temporal = temporal.permute(0, 2, 1, 3)
        temporal = (temporal * person.unsqueeze(-1)).sum(dim=2)
        self._grounding_accumulator["samples"] += int(temporal.shape[0])
        self._grounding_accumulator["temporal"] += temporal.sum(dim=0)
        self._grounding_accumulator["person"] += person.sum(dim=0)
        self._grounding_accumulator["token_norm"] += token_norm.sum(dim=0)
        self._grounding_accumulator["valid_people"] += valid_people.sum(dim=0)

    def grounding_snapshot(self):
        accumulator = self._grounding_accumulator
        if not accumulator or accumulator["samples"] <= 0:
            raise RuntimeError("grounding statistics are empty; reset and run evaluation first")
        samples = float(accumulator["samples"])
        temporal = accumulator["temporal"] / samples
        person = accumulator["person"] / samples
        token_norm = accumulator["token_norm"] / samples
        joint_mask = self.grounded_encoder.concept_joint_mask.detach().cpu()
        interaction = self.grounded_encoder.interaction_mask.detach().cpu()
        temporal_entropy = -(
            temporal.clamp_min(1e-12) * temporal.clamp_min(1e-12).log()
        ).sum(dim=1)
        person_entropy = -(
            person.clamp_min(1e-12) * person.clamp_min(1e-12).log()
        ).sum(dim=1)
        return {
            "mode": "hard_joint_temporal_person_v2",
            "num_samples": int(samples),
            "num_concepts": self.num_concepts,
            "concept_dim": self.concept_dim,
            "temporal_bins": self.grounded_encoder.temporal_bins,
            "mean_temporal_entropy": float(temporal_entropy.mean().item()),
            "mean_person_entropy": float(person_entropy.mean().item()),
            "mean_token_norm": float(token_norm.mean().item()),
            "valid_person_rate": (
                accumulator["valid_people"] / samples
            ).tolist(),
            "concepts": [
                {
                    "name": self.concept_names[index],
                    "scope": self.metadata[index]["scope"],
                    "group": self.metadata[index]["group"],
                    "parts": list(concept_parts(
                        self.metadata[index]["concept_name"],
                        self.metadata[index]["scope"],
                        self.metadata[index]["group"],
                    )),
                    "joints": joint_mask[index].nonzero().flatten().tolist(),
                    "interaction": bool(interaction[index].item()),
                    "top_temporal_bin": int(temporal[index].argmax().item()),
                    "temporal_weights": temporal[index].tolist(),
                    "top_person": int(person[index].argmax().item()),
                    "person_weights": person[index].tolist(),
                    "token_norm": float(token_norm[index].item()),
                }
                for index in range(self.num_concepts)
            ],
        }

    def action_logits(self, concept_logits):
        """Score actions from concepts without a learned black-box classifier."""
        if self.action_decoder != "prototype":
            raise RuntimeError("symbolic action decoder is not enabled")
        if concept_logits.ndim != 2 or concept_logits.shape[1] != self.num_concepts:
            raise ValueError(
                f"expected concept logits [N,{self.num_concepts}], got {tuple(concept_logits.shape)}"
            )
        support = self.support.to(concept_logits)
        if self.action_matching == "bernoulli":
            positive_count = support.sum(dim=1).clamp_min(1.0)
            negative_count = (1.0 - support).sum(dim=1).clamp_min(1.0)
            log_positive = F.logsigmoid(concept_logits).unsqueeze(1)
            log_negative = F.logsigmoid(-concept_logits).unsqueeze(1)
            positive_fit = (
                log_positive * support.unsqueeze(0)
            ).sum(dim=2) / positive_count
            negative_fit = (
                log_negative * (1.0 - support).unsqueeze(0)
            ).sum(dim=2) / negative_count
            return (0.5 * positive_fit + 0.5 * negative_fit) / self.action_temperature

        probabilities = concept_logits.sigmoid()
        weights = self.action_idf.to(concept_logits)
        weighted_probabilities = probabilities * weights.view(1, -1)
        weighted_support = support * weights.view(1, -1)
        if self.action_matching == "cosine":
            numerator = torch.einsum("bc,ac->ba", weighted_probabilities, weighted_support)
            prediction_norm = weighted_probabilities.norm(dim=1, keepdim=True)
            prototype_norm = weighted_support.norm(dim=1).view(1, -1)
            similarity = numerator / (prediction_norm * prototype_norm).clamp_min(1e-8)
            return similarity / self.action_temperature

        probabilities = probabilities.unsqueeze(1)
        support = support.unsqueeze(0)
        weights = weights.view(1, 1, -1)
        intersection = (weights * support * probabilities).sum(dim=2)
        missing = (weights * support * (1.0 - probabilities)).sum(dim=2)
        extra = (weights * (1.0 - support) * probabilities).sum(dim=2)
        denominator = (
            intersection
            + self.action_missing_weight * missing
            + self.action_extra_weight * extra
        ).clamp_min(1e-8)
        similarity = (intersection / denominator).clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(similarity) / self.action_temperature

    def learned_hypergraph(self):
        if self.hypergraph_mode != "learnable_v1":
            raise RuntimeError("learnable hypergraph V1 is not enabled")
        prior_logits = torch.logit(self.hypergraph_prior_probability)
        return torch.sigmoid(prior_logits + self.hypergraph_delta)

    def structure_loss(self):
        """Keep learned topology sparse, near its semantic prior and non-collapsed."""
        if self.hypergraph_mode != "learnable_v1":
            return self.concept_graph_gate.square() * 0.0
        incidence = self.learned_hypergraph()
        prior_anchor = F.mse_loss(incidence, self.hypergraph_prior_probability)
        absent = self.hypergraph_prior == 0
        sparsity = incidence.masked_select(absent).mean()
        binarization = (incidence * (1.0 - incidence)).mean()
        coverage = F.relu(0.5 - incidence.sum(dim=1)).mean()
        edge_weight = F.softplus(self.hyperedge_weight_raw)
        edge_weight = edge_weight / edge_weight.mean().clamp_min(1e-12)
        edge_weight_anchor = (edge_weight - 1.0).square().mean()
        return prior_anchor + 0.1 * (sparsity + binarization + edge_weight_anchor) + coverage

    def hypergraph_snapshot(self, top_k=10):
        """JSON-safe learned topology summary for monitoring and explanations."""
        if self.hypergraph_mode != "learnable_v1":
            raise RuntimeError("learnable hypergraph V1 is not enabled")
        incidence = self.learned_hypergraph().detach().cpu()
        prior = self.hypergraph_prior.detach().cpu()
        edge_weight = F.softplus(self.hyperedge_weight_raw.detach().cpu())
        edge_weight = edge_weight / edge_weight.mean().clamp_min(1e-12)
        selected = incidence >= 0.5
        return {
            "mode": self.hypergraph_mode,
            "num_concepts": self.num_concepts,
            "num_hyperedges": len(self.hyperedge_names),
            "structure_loss": float(self.structure_loss().detach().cpu()),
            "mean_membership": float(incidence.mean().item()),
            "mean_prior_membership": float(incidence[prior.bool()].mean().item()),
            "mean_nonprior_membership": float(incidence[~prior.bool()].mean().item()),
            "changed_memberships_at_0.5": int((selected != prior.bool()).sum().item()),
            "hyperedges": [
                {
                    "name": name,
                    "weight": float(edge_weight[edge].item()),
                    "prior_concepts": int(prior[:, edge].sum().item()),
                    "selected_concepts_at_0.5": int(selected[:, edge].sum().item()),
                    "top_concepts": [
                        {
                            "name": self.concept_names[index],
                            "membership": float(incidence[index, edge].item()),
                            "in_prior": bool(prior[index, edge].item()),
                        }
                        for index in incidence[:, edge].topk(min(int(top_k), self.num_concepts)).indices.tolist()
                    ],
                }
                for edge, name in enumerate(self.hyperedge_names)
            ],
        }

    def trainable_parameters(self):
        """Parameters that can affect the selected inference path."""
        if self.action_decoder == "prototype":
            parameters = list(self.grounded_encoder.parameters())
            parameters.extend((self.concept_queries, self.concept_bias))
            if self.hypergraph_mode == "learnable_v1":
                parameters.extend((
                    self.hypergraph_delta, self.hyperedge_weight_raw,
                    self.hypergraph_gate,
                ))
            else:
                parameters.append(self.concept_graph_gate)
            return tuple(parameters)
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
        result = {
            "num_classes": int(self.support.shape[0]),
            "num_concepts": self.num_concepts,
            "target_granularity": "action-class templates, not per-video annotations",
            "affinity": (
                "learned body/group/scope hypergraph incidence"
                if self.hypergraph_mode == "learnable_v1"
                else "column cosine similarity + unit self-loops + row normalization"
            ),
            "balance_loss": self.balance_loss,
            "pos_weight_source": self.pos_weight_source,
            "pos_weight_max": self.pos_weight_max,
            "pos_weight_ready": self._positive_weight_ready,
            "action_decoder": self.action_decoder,
            "action_matching": self.action_matching,
            "action_temperature": self.action_temperature,
            "action_missing_weight": self.action_missing_weight,
            "action_extra_weight": self.action_extra_weight,
            "action_idf_weighting": self.action_idf_weighting,
            "action_idf_min": float(self.action_idf.min().item()),
            "action_idf_max": float(self.action_idf.max().item()),
            "hypergraph_mode": self.hypergraph_mode,
            "grounding_mode": (
                "hard_joint_temporal_person_v2"
                if self.grounded_encoder is not None else "legacy_body_part_v1"
            ),
            "concept_dim": self.concept_dim,
            "temporal_bins": (
                self.grounded_encoder.temporal_bins
                if self.grounded_encoder is not None else None
            ),
            "training_class_counts": self.training_class_counts.detach().cpu().tolist(),
            "body_parts_zero_based_joints": BODY_PART_JOINTS,
            "concepts": [
                {
                    "name": row["concept_name"], "scope": row["scope"],
                    "parts": concept_parts(row["concept_name"], row["scope"], row["group"]),
                    "joints": concept_joints(
                        row["concept_name"], row["scope"], row["group"]
                    ),
                    "interaction": bool(
                        row.get("group", "").strip().lower() == "participant"
                        or row["concept_name"].strip().lower().startswith("interaction_")
                    ),
                    "positive_classes": int(self.support[:, index].sum().item()),
                    "positive_training_samples": int(training_positives[index]),
                    "pos_weight": float(self.positive_weight[index].item()) if self.balance_loss else 1.0,
                }
                for index, row in enumerate(self.metadata)
            ],
        }
        if self.hypergraph_mode == "learnable_v1":
            result["hypergraph"] = self.hypergraph_snapshot()
        return result

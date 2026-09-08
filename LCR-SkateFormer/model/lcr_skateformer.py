"""LCR-SkateFormer: Looping Concept Reasoning SkateFormer.

SkateFormer serves as the perception backbone to extract initial concept logits (z^0).
An iterative knowledge-graph-guided reasoning engine (IterativeKGReasoner) then refines
the concept representations over K loops (z^t -> z^{t+1} = z^t + beta * Delta z^t)
in concept/logit space, generating progressive action evidence and verifiable reasoning traces.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SkateFormer import SkateFormer_


def _logit(probability: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Stable logit function mapping probabilities in (0, 1) to (-inf, +inf)."""
    probability = probability.clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)


class KGUnionLayer(nn.Module):
    """Differentiable conjunction (AND) / disjunction (OR) rule bank.

    Weights W are stored in logit space and constrained to [0, 1] via sigmoid.
    The rule bank can be initialized with domain-specific Knowledge Graph priors.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        prior: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if prior is None:
            prior = torch.full((input_dim, output_dim), 0.05)
            diag_len = min(input_dim, output_dim)
            prior[torch.arange(diag_len), torch.arange(diag_len)] = 0.9

        prior = prior.float().clamp(1e-4, 1.0 - 1e-4)
        self.register_buffer("prior", prior.clone())

        noise = 0.02 * torch.randn_like(prior)
        self.and_weight = nn.Parameter(_logit((prior + noise).clamp(1e-4, 1.0 - 1e-4)))
        self.or_weight = nn.Parameter(_logit((prior + noise).clamp(1e-4, 1.0 - 1e-4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying AND and OR soft operations.

        Args:
            x: Input literals [B, input_dim] in [0, 1].

        Returns:
            Concatenated [conjunction, disjunction] tensor of shape [B, 2 * output_dim].
        """
        and_weight = torch.sigmoid(self.and_weight)  # [input_dim, output_dim]
        or_weight = torch.sigmoid(self.or_weight)    # [input_dim, output_dim]

        # Soft conjunction (AND): prod(1 - w_i * (1 - x_i))
        conjunction = torch.prod(
            1.0 - and_weight.unsqueeze(0) * (1.0 - x.unsqueeze(-1)), dim=1
        )
        # Soft disjunction (OR): 1 - prod(1 - w_i * x_i)
        disjunction = 1.0 - torch.prod(
            1.0 - or_weight.unsqueeze(0) * x.unsqueeze(-1), dim=1
        )
        return torch.cat([conjunction, disjunction], dim=1)

    def weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the bounded AND/OR rule weights."""
        return torch.sigmoid(self.and_weight), torch.sigmoid(self.or_weight)

    def prior_retention_loss(self) -> torch.Tensor:
        """Discourage learned rule weights from drifting too far from KG initialization."""
        and_weight, or_weight = self.weights()
        return 0.5 * (
            (and_weight - self.prior).abs().mean()
            + (or_weight - self.prior).abs().mean()
        )

    def sparsity_loss(self) -> torch.Tensor:
        """Penalize redundant active rule connections."""
        and_weight, or_weight = self.weights()
        return 0.5 * (and_weight.mean() + or_weight.mean())


class KGPrimedReasoner(nn.Module):
    """Concept-to-action logical reasoner with KG-primed rule banks.

    Supports both 'linear' action classification and 'prototype' schema compatibility.
    """

    def __init__(
        self,
        n_concepts: int,
        n_actions: int,
        concept_names: List[str],
        l1_dim: int = 128,
        l2_dim: int = 128,
        kg_init_mode: str = "kg_primed",
        kg_prior_map: Optional[Dict[str, List]] = None,
        kg_prior_strength: float = 1.0,
        action_mode: str = "linear",
        action_prototypes: Optional[Union[torch.Tensor, List]] = None,
        prototype_and_weight: float = 0.35,
    ):
        super().__init__()
        self.n_concepts = int(n_concepts)
        self.n_actions = int(n_actions)
        self.concept_names = list(concept_names)
        if len(self.concept_names) != self.n_concepts:
            raise ValueError(
                f"Expected {self.n_concepts} concept names, got {len(self.concept_names)}"
            )

        self.kg_init_mode = str(kg_init_mode).lower()
        self.kg_prior_map = kg_prior_map or {"require": [], "alongside": [], "excludes": []}
        self.kg_prior_strength = float(kg_prior_strength)
        self.action_mode = str(action_mode).lower()
        self.prototype_and_weight = float(prototype_and_weight)
        self.l1_dim = int(l1_dim)
        self.l2_dim = int(l2_dim)

        if self.action_mode == "prototype":
            if action_prototypes is None:
                raise ValueError("action_prototypes are required when action_mode='prototype'")
            prototypes = torch.as_tensor(action_prototypes, dtype=torch.float32)
            if prototypes.shape[0] > self.n_actions:
                prototypes = prototypes[: self.n_actions]
            if prototypes.shape != (self.n_actions, self.n_concepts):
                raise ValueError(
                    f"action_prototypes shape must be ({self.n_actions}, {self.n_concepts}), got {tuple(prototypes.shape)}"
                )
            prototypes = (prototypes >= 0.5).float()
            self.register_buffer("action_prototypes", prototypes)
            action_prior = self._build_action_prior(prototypes)
            self.action_rule = KGUnionLayer(
                2 * self.n_concepts, self.n_actions, prior=action_prior
            )
            self.logic_dim = 2 * self.n_actions
        else:
            first_prior = None
            if self.kg_init_mode == "kg_primed":
                first_prior = self._build_first_layer_prior(2 * self.n_concepts, self.l1_dim)

            self.first = KGUnionLayer(2 * self.n_concepts, self.l1_dim, prior=first_prior)
            self.second = KGUnionLayer(2 * self.l1_dim, self.l2_dim)
            self.action = nn.Linear(2 * self.l2_dim, self.n_actions)
            self.logic_dim = 2 * self.l1_dim + 2 * self.l2_dim

    def _resolve(self, name: Union[str, int]) -> Optional[int]:
        if isinstance(name, int):
            return name if 0 <= name < self.n_concepts else None
        key = str(name).strip().lower()
        for idx, cname in enumerate(self.concept_names):
            if str(cname).strip().lower() == key:
                return idx
        return None

    def _build_first_layer_prior(self, input_dim: int, output_dim: int) -> torch.Tensor:
        prior = torch.full((input_dim, output_dim), 0.03)
        relations = {
            "require": self.kg_prior_map.get("require", []),
            "alongside": self.kg_prior_map.get("alongside", []),
            "excludes": self.kg_prior_map.get("excludes", []),
        }

        for rule_idx in range(output_dim):
            seed = rule_idx % self.n_concepts
            prior[seed, rule_idx] = 0.95
            for rel_name, pairs in relations.items():
                val = {
                    "require": 0.90,
                    "alongside": 0.70,
                    "excludes": 0.01,
                }[rel_name] * self.kg_prior_strength
                val = max(0.01, min(0.98, val))
                for pair in pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    a = self._resolve(pair[0])
                    b = self._resolve(pair[1])
                    if a is None or b is None:
                        continue
                    if seed == a:
                        prior[b, rule_idx] = val
                    elif seed == b:
                        prior[a, rule_idx] = val
        return prior

    def _build_action_prior(self, prototypes: torch.Tensor) -> torch.Tensor:
        prior = torch.full(
            (2 * self.n_concepts, self.n_actions), 1e-4, dtype=prototypes.dtype
        )
        prior[: self.n_concepts] = prototypes.t() * 0.95
        for action_idx in range(self.n_actions):
            pos_concepts = set(torch.nonzero(prototypes[action_idx] > 0.5).flatten().tolist())
            for rel_name, pairs in self.kg_prior_map.items():
                if rel_name not in {"require", "alongside", "excludes"}:
                    continue
                rel_val = {
                    "require": 0.90,
                    "alongside": 0.70,
                    "excludes": 0.90,
                }[rel_name] * self.kg_prior_strength
                rel_val = max(0.01, min(0.98, rel_val))
                for pair in pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    a = self._resolve(pair[0])
                    b = self._resolve(pair[1])
                    if a is None or b is None:
                        continue
                    if rel_name == "require":
                        if b in pos_concepts:
                            prior[a, action_idx] = max(prior[a, action_idx], rel_val)
                    elif rel_name == "alongside":
                        if a in pos_concepts:
                            prior[b, action_idx] = max(prior[b, action_idx], rel_val)
                        if b in pos_concepts:
                            prior[a, action_idx] = max(prior[a, action_idx], rel_val)
                    else:
                        if a in pos_concepts:
                            prior[self.n_concepts + b, action_idx] = max(
                                prior[self.n_concepts + b, action_idx], rel_val
                            )
                        if b in pos_concepts:
                            prior[self.n_concepts + a, action_idx] = max(
                                prior[self.n_concepts + a, action_idx], rel_val
                            )
        return prior.clamp(1e-4, 0.98)

    def kg_loss(self, concept_logits: torch.Tensor) -> torch.Tensor:
        """Soft consistency penalty for domain KG relations."""
        probs = torch.sigmoid(concept_logits)
        penalties = []
        for rel_name in ("require", "alongside", "excludes"):
            for pair in self.kg_prior_map.get(rel_name, []):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                a = self._resolve(pair[0])
                b = self._resolve(pair[1])
                if a is None or b is None:
                    continue
                pa, pb = probs[:, a], probs[:, b]
                if rel_name == "require":
                    # pb requires pa: penalty if pb > pa
                    penalties.append(F.relu(pb - pa))
                elif rel_name == "alongside":
                    # Co-occurrence expected: penalty if low joint probability
                    penalties.append(F.relu(0.5 - pa * pb))
                else:
                    # Mutual exclusion: penalty if both are high
                    penalties.append(pa * pb)
        if not penalties:
            return concept_logits.new_zeros(())
        return torch.stack([p.mean() for p in penalties]).mean()

    def rule_prior_loss(self) -> torch.Tensor:
        if self.action_mode == "prototype":
            return self.action_rule.prior_retention_loss()
        return 0.5 * (self.first.prior_retention_loss() + self.second.prior_retention_loss())

    def rule_sparsity_loss(self) -> torch.Tensor:
        if self.action_mode == "prototype":
            return self.action_rule.sparsity_loss()
        return 0.5 * (self.first.sparsity_loss() + self.second.sparsity_loss())

    def forward(
        self,
        concept_logits: torch.Tensor,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward pass of reasoner.

        Args:
            concept_logits: [B, n_concepts]
            return_trace: whether to return intermediate rule tensors

        Returns:
            action_logits: [B, n_actions]
            logic_state: [B, logic_dim]
            trace_dict: dict of rule activations
        """
        concepts = torch.sigmoid(concept_logits)
        literals = torch.cat([concepts, 1.0 - concepts], dim=-1)  # [B, 2 * n_concepts]

        if self.action_mode == "prototype":
            rule_output = self.action_rule(literals)  # [B, 2 * n_actions]
            and_score, or_score = rule_output.chunk(2, dim=-1)
            evidence = (
                self.prototype_and_weight * and_score
                + (1.0 - self.prototype_and_weight) * or_score
            ).clamp(1e-4, 1.0 - 1e-4)
            action_logits = _logit(evidence)
            logic_state = rule_output
            trace = {
                "and_score": and_score,
                "or_score": or_score,
                "evidence": evidence,
                "literals": literals,
            }
        else:
            hidden_1 = self.first(literals)  # [B, 2 * l1_dim]
            hidden_2 = self.second(hidden_1)  # [B, 2 * l2_dim]
            action_logits = self.action(hidden_2)
            logic_state = torch.cat([hidden_1, hidden_2], dim=-1)
            trace = {
                "hidden_1": hidden_1,
                "hidden_2": hidden_2,
                "literals": literals,
            }

        return action_logits, logic_state, trace


class ConceptUpdateCell(nn.Module):
    """Concept Refinement Cell.

    Maps current concept logit state z^t and logical evidence r^t (plus optional action evidence)
    to a residual logit update Delta z^t.
    Notice: Visual feature maps are NOT fed into this cell to strictly protect
    the concept bottleneck from black-box shortcutting.
    """

    def __init__(
        self,
        num_concepts: int,
        logic_dim: int,
        action_dim: Optional[int] = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.logic_dim = int(logic_dim)
        self.has_action_feedback = action_dim is not None and action_dim > 0

        # Project logic evidence to concept space
        self.logic_projection = nn.Sequential(
            nn.Linear(logic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
        )

        in_dim = num_concepts * 2
        if self.has_action_feedback:
            self.action_projection = nn.Sequential(
                nn.Linear(action_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_concepts),
            )
            in_dim += num_concepts

        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        logic_state: torch.Tensor,
        action_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute residual logit delta Delta z^t.

        Args:
            z_t: Current concept logits [B, num_concepts]
            logic_state: Rule bank logic evidence [B, logic_dim]
            action_logits: Optional action logits [B, num_actions]

        Returns:
            delta_z: [B, num_concepts]
        """
        r_proj = self.logic_projection(logic_state)
        inputs = [z_t, r_proj]
        if self.has_action_feedback and action_logits is not None:
            act_proj = self.action_projection(action_logits)
            inputs.append(act_proj)
        update_in = torch.cat(inputs, dim=-1)
        return self.update_mlp(update_in)


class IterativeKGReasoner(nn.Module):
    """KG-guided Looping Concept Reasoning Engine (LCR).

    Iterates over K reasoning loops to refine concept logits and generate
    consistent action evidence step-by-step.
    """

    def __init__(
        self,
        num_concepts: int,
        num_actions: int,
        concept_names: List[str],
        num_loops: int = 2,
        shared_weights: bool = True,
        update_gate_init: float = 0.1,
        max_gate: float = 0.5,
        use_action_feedback: bool = False,
        l1_dim: int = 128,
        l2_dim: int = 128,
        kg_init_mode: str = "kg_primed",
        kg_prior_map: Optional[Dict[str, List]] = None,
        kg_prior_strength: float = 1.0,
        action_mode: str = "linear",
        action_prototypes: Optional[Union[torch.Tensor, List]] = None,
        prototype_and_weight: float = 0.35,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_actions = int(num_actions)
        self.num_loops = max(1, int(num_loops))
        self.shared_weights = bool(shared_weights)
        self.max_gate = float(max_gate)
        self.use_action_feedback = bool(use_action_feedback)
        self.action_mode = str(action_mode).lower()

        # Learnable gating parameter initialized to yield update_gate_init
        # beta = sigmoid(raw_beta) * max_gate
        init_gate = min(max(update_gate_init, 1e-4), max_gate - 1e-4)
        raw_init = _logit(torch.tensor(init_gate / self.max_gate, dtype=torch.float32))
        self.raw_beta = nn.Parameter(raw_init.clone().detach())

        def _make_reasoner():
            return KGPrimedReasoner(
                n_concepts=self.num_concepts,
                n_actions=self.num_actions,
                concept_names=concept_names,
                l1_dim=l1_dim,
                l2_dim=l2_dim,
                kg_init_mode=kg_init_mode,
                kg_prior_map=kg_prior_map,
                kg_prior_strength=kg_prior_strength,
                action_mode=action_mode,
                action_prototypes=action_prototypes,
                prototype_and_weight=prototype_and_weight,
            )

        def _make_update_cell(logic_dim: int):
            act_dim = self.num_actions if self.use_action_feedback else None
            return ConceptUpdateCell(
                num_concepts=self.num_concepts,
                logic_dim=logic_dim,
                action_dim=act_dim,
                hidden_dim=256,
            )

        if self.shared_weights:
            self.base_reasoner = _make_reasoner()
            self.update_cell = _make_update_cell(self.base_reasoner.logic_dim)
        else:
            self.reasoners = nn.ModuleList([_make_reasoner() for _ in range(self.num_loops + 1)])
            self.update_cells = nn.ModuleList([
                _make_update_cell(self.reasoners[i].logic_dim) for i in range(self.num_loops)
            ])

    def get_beta(self) -> torch.Tensor:
        """Effective bounded refinement gate beta in [0, max_gate]."""
        return torch.sigmoid(self.raw_beta) * self.max_gate

    def kg_loss(self, concept_logits: torch.Tensor) -> torch.Tensor:
        if self.shared_weights:
            return self.base_reasoner.kg_loss(concept_logits)
        return self.reasoners[0].kg_loss(concept_logits)

    def rule_prior_loss(self) -> torch.Tensor:
        if self.shared_weights:
            return self.base_reasoner.rule_prior_loss()
        return torch.stack([r.rule_prior_loss() for r in self.reasoners]).mean()

    def rule_sparsity_loss(self) -> torch.Tensor:
        if self.shared_weights:
            return self.base_reasoner.rule_sparsity_loss()
        return torch.stack([r.rule_sparsity_loss() for r in self.reasoners]).mean()

    def forward(
        self,
        concept_logits: torch.Tensor,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[Dict]]:
        """Forward pass executing K reasoning and refinement loops.

        Args:
            concept_logits: Initial concept logits z^0 [B, num_concepts]
            return_trace: Whether to record complete reasoning traces

        Returns:
            final_action_logits: y^K [B, num_actions]
            final_concept_logits: z^K [B, num_concepts]
            all_action_logits: list of action logits [y^0, y^1, ..., y^{K-1}]
            trace_history: list of dicts detailing each step
        """
        z_t = concept_logits
        all_action_logits = []
        trace_history = []
        beta = self.get_beta()

        for step in range(self.num_loops):
            reasoner = self.base_reasoner if self.shared_weights else self.reasoners[step]
            update_cell = self.update_cell if self.shared_weights else self.update_cells[step]

            y_t, logic_t, trace_t = reasoner(z_t, return_trace=return_trace)
            all_action_logits.append(y_t)

            delta_t = update_cell(
                z_t,
                logic_t,
                action_logits=y_t if self.use_action_feedback else None,
            )
            z_next = z_t + beta * delta_t

            if return_trace:
                trace_history.append({
                    "step": step,
                    "concept_logits": z_t.detach().cpu() if not self.training else z_t,
                    "concept_probs": torch.sigmoid(z_t).detach().cpu() if not self.training else torch.sigmoid(z_t),
                    "action_logits": y_t.detach().cpu() if not self.training else y_t,
                    "delta_z": delta_t.detach().cpu() if not self.training else delta_t,
                    "beta": beta.item(),
                    "rule_trace": trace_t,
                })

            z_t = z_next

        # Final reasoning step on refined concept logits z^K
        final_reasoner = self.base_reasoner if self.shared_weights else self.reasoners[self.num_loops]
        y_final, final_logic, final_rule_trace = final_reasoner(z_t, return_trace=return_trace)

        if return_trace:
            trace_history.append({
                "step": self.num_loops,
                "concept_logits": z_t.detach().cpu() if not self.training else z_t,
                "concept_probs": torch.sigmoid(z_t).detach().cpu() if not self.training else torch.sigmoid(z_t),
                "action_logits": y_final.detach().cpu() if not self.training else y_final,
                "delta_z": None,
                "beta": beta.item(),
                "rule_trace": final_rule_trace,
            })

        return y_final, z_t, all_action_logits, trace_history


class LCRSkateFormer(nn.Module):
    """Looping Concept Reasoning SkateFormer (LCR-SkateFormer).

    Integrates the SkateFormer perception backbone with the IterativeKGReasoner
    to perform multi-loop concept refinement and action reasoning.
    """

    def __init__(
        self,
        num_classes: int = 120,
        num_concepts: int = 72,
        spatial_concepts: int = 52,
        temporal_concepts: int = 20,
        num_people: int = 2,
        num_points: int = 24,
        num_frames: int = 64,
        backbone_weights: Optional[str] = None,
        concept_names: Optional[List[str]] = None,
        # Looping reasoning args
        num_loops: int = 2,
        shared_weights: bool = True,
        update_gate_init: float = 0.1,
        max_gate: float = 0.5,
        use_action_feedback: bool = False,
        # Reasoner args
        l1_dim: int = 128,
        l2_dim: int = 128,
        kg_init_mode: str = "kg_primed",
        kg_prior_map: Optional[Dict[str, List]] = None,
        kg_prior_strength: float = 1.0,
        action_mode: str = "linear",
        action_prototypes: Optional[Union[torch.Tensor, List]] = None,
        prototype_and_weight: float = 0.35,
        # Loss weights
        concept_loss_weight: float = 1.0,
        early_concept_loss_weight: float = 0.5,
        aux_action_loss_weight: float = 0.3,
        stability_loss_weight: float = 0.05,
        kg_loss_weight: float = 0.01,
        prior_retention_weight: float = 0.01,
        rule_sparsity_weight: float = 0.001,
        concept_negative_weight: float = 0.15,
        concept_positive_weight: float = 1.0,
        asymmetric_gamma_neg: float = 2.0,
        part_diversity_weight: float = 0.01,
        return_aux: bool = True,
        **kwargs,
    ):
        super().__init__()
        if int(spatial_concepts) + int(temporal_concepts) != int(num_concepts):
            raise ValueError("spatial_concepts + temporal_concepts must equal num_concepts")

        self.num_classes = int(num_classes)
        self.num_concepts = int(num_concepts)
        self.spatial_concepts = int(spatial_concepts)
        self.temporal_concepts = int(temporal_concepts)
        self.num_loops = int(num_loops)

        # Loss weights
        self.concept_loss_weight = float(concept_loss_weight)
        self.early_concept_loss_weight = float(early_concept_loss_weight)
        self.aux_action_loss_weight = float(aux_action_loss_weight)
        self.stability_loss_weight = float(stability_loss_weight)
        self.kg_loss_weight = float(kg_loss_weight)
        self.prior_retention_weight = float(prior_retention_weight)
        self.rule_sparsity_weight = float(rule_sparsity_weight)
        self.concept_negative_weight = float(concept_negative_weight)
        self.concept_positive_weight = float(concept_positive_weight)
        self.asymmetric_gamma_neg = float(asymmetric_gamma_neg)
        self.part_diversity_weight = float(part_diversity_weight)
        self.return_aux = bool(return_aux)
        self.accepts_concepts = True

        # Backbone
        self.backbone = SkateFormer_(
            num_classes=num_classes,
            num_people=num_people,
            num_points=num_points,
            num_frames=num_frames,
            **kwargs,
        )
        self._load_backbone_weights(backbone_weights)
        feature_dim = self.backbone.head.in_features

        # Concept Heads
        self.spatial_pool_score = nn.Conv1d(feature_dim, 1, kernel_size=1)
        self.temporal_pool_score = nn.Conv1d(feature_dim, 1, kernel_size=1)
        self.spatial_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.spatial_concepts),
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.temporal_concepts),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        if concept_names is None:
            concept_names = [f"concept_{i}" for i in range(self.num_concepts)]
        self.concept_names = list(concept_names)

        # Iterative KG Reasoner Engine
        self.reasoner = IterativeKGReasoner(
            num_concepts=self.num_concepts,
            num_actions=self.num_classes,
            concept_names=self.concept_names,
            num_loops=self.num_loops,
            shared_weights=shared_weights,
            update_gate_init=update_gate_init,
            max_gate=max_gate,
            use_action_feedback=use_action_feedback,
            l1_dim=l1_dim,
            l2_dim=l2_dim,
            kg_init_mode=kg_init_mode,
            kg_prior_map=kg_prior_map,
            kg_prior_strength=kg_prior_strength,
            action_mode=action_mode,
            action_prototypes=action_prototypes,
            prototype_and_weight=prototype_and_weight,
        )

    def _load_backbone_weights(self, checkpoint_path: Optional[str]):
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            print(f"[Warning] Backbone checkpoint not found: {path}. Initializing randomly.")
            return
        checkpoint = torch.load(str(path), map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError("Unsupported SkateFormer checkpoint format")

        backbone_state = self.backbone.state_dict()
        mapped = OrderedDict()
        for key, value in checkpoint.items():
            clean_key = key.replace("module.", "", 1)
            if clean_key.startswith("backbone."):
                clean_key = clean_key[len("backbone."):]
            if clean_key.startswith("head."):
                continue
            if clean_key in backbone_state:
                mapped[clean_key] = value
        missing, unexpected = self.backbone.load_state_dict(mapped, strict=False)
        print(
            f"Loaded {len(mapped)} SkateFormer backbone tensors from {path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def _pool_features(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_tokens = feature_map.mean(dim=2)
        spatial_attention = torch.softmax(self.spatial_pool_score(spatial_tokens), dim=-1)
        spatial = (spatial_tokens * spatial_attention).sum(dim=-1)

        temporal_tokens = feature_map.mean(dim=3)
        temporal_attention = torch.softmax(self.temporal_pool_score(temporal_tokens), dim=-1)
        temporal = (temporal_tokens * temporal_attention).sum(dim=-1)
        return spatial, temporal

    @staticmethod
    def _part_diversity_loss(feature_map: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, tokens = feature_map.shape
        if tokens % 6 != 0:
            return feature_map.new_zeros(())
        parts = feature_map.view(batch, channels, frames, 6, tokens // 6).mean(dim=(2, 4))
        parts = F.normalize(parts.transpose(1, 2), dim=-1)
        gram = torch.matmul(parts, parts.transpose(-1, -2))
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        return ((gram - eye) ** 2).mean()

    def _concept_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Asymmetric, confidence-aware loss for weak concept supervision."""
        targets = targets.float().clamp(0.0, 1.0)
        probabilities = torch.sigmoid(logits)
        positive_loss = F.softplus(-logits) * targets
        negative_loss = (
            F.softplus(logits)
            * probabilities.pow(self.asymmetric_gamma_neg)
            * (1.0 - targets)
        )
        weights = (
            targets * self.concept_positive_weight
            + (1.0 - targets) * self.concept_negative_weight
        )
        if target_mask is not None:
            weights = weights * target_mask.float().to(weights.device)
        if target_confidence is not None:
            weights = weights * target_confidence.float().to(weights.device)
        loss = positive_loss + negative_loss
        denominator = weights.sum().clamp_min(1.0)
        return (loss * weights).sum() / denominator

    def forward(
        self,
        input: torch.Tensor,
        index_t: Optional[torch.Tensor] = None,
        concept_targets: Optional[torch.Tensor] = None,
        concept_target_mask: Optional[torch.Tensor] = None,
        concept_target_confidence: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> Union[torch.Tensor, Dict]:
        feature_map = self.backbone.forward_features_from_input(input, index_t=index_t)
        spatial_feat, temporal_feat = self._pool_features(feature_map)

        spatial_logits = self.spatial_head(spatial_feat)
        temporal_logits = self.temporal_head(temporal_feat)
        z0 = torch.cat([spatial_logits, temporal_logits], dim=1)

        # Looping KG reasoning
        y_final, z_final, all_actions, trace_history = self.reasoner(
            z0, return_trace=return_trace
        )

        if not self.return_aux and concept_targets is None and not self.training and not return_trace:
            return y_final

        # Calculate stability loss between successive concept states
        c0 = torch.sigmoid(z0)
        c_final = torch.sigmoid(z_final)
        stability_loss = (c_final - c0).abs().mean()

        output = {
            "action_logits": y_final,
            "concept_logits": z_final,
            "initial_concept_logits": z0,
            "spatial_logits": spatial_logits,
            "temporal_logits": temporal_logits,
            "all_action_logits": all_actions,
            "beta": self.reasoner.get_beta(),
            "fused_feature": self.fusion(torch.cat([spatial_feat, temporal_feat], dim=1)),
            "kg_loss": 0.5 * (self.reasoner.kg_loss(z0) + self.reasoner.kg_loss(z_final)),
            "prior_retention_loss": self.reasoner.rule_prior_loss(),
            "rule_sparsity_loss": self.reasoner.rule_sparsity_loss(),
            "part_diversity_loss": self._part_diversity_loss(feature_map),
            "stability_loss": stability_loss,
            "reasoning_trace": trace_history,
        }

        # Concept Supervision Losses
        if concept_targets is not None:
            targets = concept_targets.float().to(z0.device)
            if targets.shape[-1] != self.num_concepts:
                raise ValueError(
                    f"Expected concept targets with {self.num_concepts} labels, got {targets.shape[-1]}"
                )

            loss_c0 = self._concept_loss(
                z0,
                targets,
                target_mask=concept_target_mask,
                target_confidence=concept_target_confidence,
            )
            loss_cK = self._concept_loss(
                z_final,
                targets,
                target_mask=concept_target_mask,
                target_confidence=concept_target_confidence,
            )
            output["early_concept_loss"] = loss_c0
            output["final_concept_loss"] = loss_cK
            output["concept_loss"] = loss_cK

            # Auxiliary action loss if action_labels are provided in forward
            aux_act_loss = z0.new_zeros(())
            if action_labels is not None and len(all_actions) > 0:
                for act_t in all_actions:
                    aux_act_loss = aux_act_loss + F.cross_entropy(act_t, action_labels)
                aux_act_loss = aux_act_loss / len(all_actions)
            output["aux_action_loss"] = aux_act_loss

            output["aux_loss"] = (
                self.concept_loss_weight * loss_cK
                + self.early_concept_loss_weight * loss_c0
                + self.aux_action_loss_weight * aux_act_loss
                + self.stability_loss_weight * stability_loss
                + self.kg_loss_weight * output["kg_loss"]
                + self.prior_retention_weight * output["prior_retention_loss"]
                + self.rule_sparsity_weight * output["rule_sparsity_loss"]
                + self.part_diversity_weight * output["part_diversity_loss"]
            )

        return output


def LCR_SkateFormer(**kwargs):
    """Factory for LCR-SkateFormer."""
    return LCRSkateFormer(**kwargs)


def REASON_SkateFormer(**kwargs):
    """Backwards compatibility alias for REASON-SkateFormer."""
    return LCRSkateFormer(**kwargs)

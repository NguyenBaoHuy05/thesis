"""SFB-ECR-SkateFormer: Concept-Guided Soft Feedback Attention + Logical Reasoner.

Extends ECR-SkateFormer by extracting intermediate concept representations from early
backbone stages and injecting confidence-gated soft feedback (channel modulation and
spatial-temporal token bias) into downstream perception blocks.
"""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SkateFormer import SkateFormer_
from .soft_concept_feedback import SoftConceptFeedback


def _logit(probability, eps=1e-4):
    probability = probability.clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)


class KGUnionLayer(nn.Module):
    """Differentiable conjunction/disjunction rule bank."""

    def __init__(self, input_dim, output_dim, prior=None):
        super().__init__()
        if prior is None:
            prior = torch.full((input_dim, output_dim), 0.05)
            prior[torch.arange(min(input_dim, output_dim)),
                  torch.arange(min(input_dim, output_dim))] = 0.9
        prior = prior.float().clamp(1e-4, 1.0 - 1e-4)
        self.register_buffer("prior", prior.clone())
        noise = 0.02 * torch.randn_like(prior)
        self.and_weight = nn.Parameter(_logit((prior + noise).clamp(1e-4, 1 - 1e-4)))
        self.or_weight = nn.Parameter(_logit((prior + noise).clamp(1e-4, 1 - 1e-4)))

    def forward(self, x):
        and_weight = torch.sigmoid(self.and_weight)
        or_weight = torch.sigmoid(self.or_weight)

        conjunction = torch.prod(
            1.0 - and_weight.unsqueeze(0) * (1.0 - x.unsqueeze(-1)), dim=1
        )
        disjunction = 1.0 - torch.prod(
            1.0 - or_weight.unsqueeze(0) * x.unsqueeze(-1), dim=1
        )
        return torch.cat([conjunction, disjunction], dim=1)

    def weights(self):
        return torch.sigmoid(self.and_weight), torch.sigmoid(self.or_weight)

    def prior_retention_loss(self):
        and_weight, or_weight = self.weights()
        return (
            (and_weight - self.prior).abs().mean()
            + (or_weight - self.prior).abs().mean()
        ) * 0.5

    def sparsity_loss(self):
        and_weight, or_weight = self.weights()
        return 0.5 * (and_weight.mean() + or_weight.mean())


class KGPrimedReasoner(nn.Module):
    """Concept-to-action logical reasoner with first-layer KG priming."""

    def __init__(
        self,
        n_concepts,
        n_actions,
        concept_names,
        l1_dim=128,
        l2_dim=128,
        kg_init_mode="kg_primed",
        kg_prior_map=None,
        kg_prior_strength=1.0,
        action_mode="linear",
        action_prototypes=None,
        prototype_and_weight=0.35,
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

        if self.action_mode == "prototype":
            if action_prototypes is None:
                raise ValueError("action_prototypes are required when action_mode='prototype'")
            prototypes = torch.as_tensor(action_prototypes, dtype=torch.float32)
            if prototypes.shape[0] > self.n_actions:
                prototypes = prototypes[:self.n_actions]
            if prototypes.shape != (self.n_actions, self.n_concepts):
                raise ValueError(
                    f"action_prototypes must have shape ({self.n_actions}, {self.n_concepts}), got {tuple(prototypes.shape)}"
                )
            prototypes = (prototypes >= 0.5).float()
            self.register_buffer("action_prototypes", prototypes)
            action_prior = self._build_action_prior(prototypes)
            self.action_rule = KGUnionLayer(
                2 * self.n_concepts, self.n_actions, prior=action_prior
            )
        elif self.action_mode == "direct_mlp":
            self.direct_mlp = nn.Sequential(
                nn.Linear(self.n_concepts, l1_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(l1_dim, n_actions)
            )
        else:
            first_prior = None
            if self.kg_init_mode == "kg_primed":
                first_prior = self._build_first_layer_prior(2 * self.n_concepts, l1_dim)

            self.first = KGUnionLayer(2 * self.n_concepts, l1_dim, prior=first_prior)
            self.second = KGUnionLayer(2 * l1_dim, l2_dim)
            self.action = nn.Linear(2 * l2_dim, n_actions)

    def _resolve(self, name):
        if isinstance(name, int):
            return name if 0 <= name < self.n_concepts else None
        key = str(name).strip().lower()
        for index, concept_name in enumerate(self.concept_names):
            if str(concept_name).strip().lower() == key:
                return index
        return None

    def _build_first_layer_prior(self, input_dim, output_dim):
        prior = torch.full((input_dim, output_dim), 0.03)
        relations = {
            "require": self.kg_prior_map.get("require", []),
            "alongside": self.kg_prior_map.get("alongside", []),
            "excludes": self.kg_prior_map.get("excludes", []),
        }

        for rule_index in range(output_dim):
            seed = rule_index % self.n_concepts
            prior[seed, rule_index] = 0.95
            for relation_name, pairs in relations.items():
                value = {
                    "require": 0.90,
                    "alongside": 0.70,
                    "excludes": 0.01,
                }[relation_name] * self.kg_prior_strength
                value = max(0.01, min(0.98, value))
                for pair in pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    a = self._resolve(pair[0])
                    b = self._resolve(pair[1])
                    if a is None or b is None:
                        continue
                    if seed == a:
                        prior[b, rule_index] = value
                    elif seed == b:
                        prior[a, rule_index] = value
        return prior

    def _build_action_prior(self, prototypes):
        prior = torch.full(
            (2 * self.n_concepts, self.n_actions), 1e-4, dtype=prototypes.dtype
        )
        prior[: self.n_concepts] = prototypes.t() * 0.95
        for action_index in range(self.n_actions):
            positive = set(torch.nonzero(prototypes[action_index] > 0.5).flatten().tolist())
            for relation_name, pairs in self.kg_prior_map.items():
                if relation_name not in {"require", "alongside", "excludes"}:
                    continue
                relation_value = {
                    "require": 0.90,
                    "alongside": 0.70,
                    "excludes": 0.90,
                }[relation_name] * self.kg_prior_strength
                relation_value = max(0.01, min(0.98, relation_value))
                for pair in pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    a = self._resolve(pair[0])
                    b = self._resolve(pair[1])
                    if a is None or b is None:
                        continue
                    if relation_name == "require":
                        if b in positive:
                            prior[a, action_index] = max(prior[a, action_index], relation_value)
                    elif relation_name == "alongside":
                        if a in positive:
                            prior[b, action_index] = max(prior[b, action_index], relation_value)
                        if b in positive:
                            prior[a, action_index] = max(prior[a, action_index], relation_value)
                    else:
                        if a in positive:
                            prior[self.n_concepts + b, action_index] = max(
                                prior[self.n_concepts + b, action_index], relation_value
                            )
                        if b in positive:
                            prior[self.n_concepts + a, action_index] = max(
                                prior[self.n_concepts + a, action_index], relation_value
                            )
        return prior.clamp(1e-4, 0.98)

    def kg_loss(self, concept_logits):
        probabilities = torch.sigmoid(concept_logits)
        penalties = []
        for relation_name in ("require", "alongside", "excludes"):
            for pair in self.kg_prior_map.get(relation_name, []):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                a = self._resolve(pair[0])
                b = self._resolve(pair[1])
                if a is None or b is None:
                    continue
                pa, pb = probabilities[:, a], probabilities[:, b]
                if relation_name == "require":
                    penalties.append(F.relu(pb - pa))
                elif relation_name == "alongside":
                    penalties.append(F.relu(0.5 - pa * pb))
                else:
                    penalties.append(pa * pb)
        if not penalties:
            return concept_logits.new_zeros(())
        return torch.stack([penalty.mean() for penalty in penalties]).mean()

    def rule_prior_loss(self):
        if self.action_mode == "prototype":
            return self.action_rule.prior_retention_loss()
        elif self.action_mode == "direct_mlp":
            return torch.tensor(0.0, device=self.direct_mlp[0].weight.device)
        return 0.5 * (self.first.prior_retention_loss() + self.second.prior_retention_loss())

    def rule_sparsity_loss(self):
        if self.action_mode == "prototype":
            return self.action_rule.sparsity_loss()
        elif self.action_mode == "direct_mlp":
            return torch.tensor(0.0, device=self.direct_mlp[0].weight.device)
        return 0.5 * (self.first.sparsity_loss() + self.second.sparsity_loss())

    def forward(self, concept_logits, return_trace=False):
        concepts = torch.sigmoid(concept_logits)
        if self.action_mode == "direct_mlp":
            action_logits = self.direct_mlp(concepts)
            if return_trace:
                return action_logits, {"concepts": concepts}
            return action_logits

        literals = torch.cat([concepts, 1.0 - concepts], dim=1)
        if self.action_mode == "prototype":
            rule_output = self.action_rule(literals)
            and_score, or_score = rule_output.chunk(2, dim=1)
            evidence = (
                self.prototype_and_weight * and_score
                + (1.0 - self.prototype_and_weight) * or_score
            ).clamp(1e-4, 1.0 - 1e-4)
            action_logits = _logit(evidence)
            trace = {
                "and_score": and_score,
                "or_score": or_score,
                "evidence": evidence,
            }
        else:
            hidden_1 = self.first(literals)
            hidden_2 = self.second(hidden_1)
            action_logits = self.action(hidden_2)
            trace = {"hidden_1": hidden_1, "hidden_2": hidden_2}
        if return_trace:
            return action_logits, trace
        return action_logits


class SFB_ECR_SkateFormer(nn.Module):
    """ECR-SkateFormer with Concept-Guided Soft Feedback Attention."""

    def __init__(
        self,
        num_classes=60,
        num_concepts=72,
        spatial_concepts=52,
        temporal_concepts=20,
        num_people=2,
        num_points=24,
        num_frames=64,
        backbone_weights=None,
        concept_names=None,
        l1_dim=128,
        l2_dim=128,
        kg_init_mode="kg_primed",
        kg_prior_map=None,
        kg_prior_strength=1.0,
        action_mode="prototype",
        action_prototypes=None,
        prototype_and_weight=0.35,
        # Feedback Attention Hyperparameters
        feedback_mode="token_soft_bias",
        feedback_beta=0.2,
        feedback_lambda=0.1,
        feedback_confidence_gating=True,
        early_stages=2,
        # Loss Weights
        concept_loss_weight=1.0,
        early_concept_loss_weight=0.5,
        feedback_consistency_weight=0.02,
        kg_loss_weight=0.01,
        prior_retention_weight=0.01,
        rule_sparsity_weight=0.001,
        concept_negative_weight=0.15,
        concept_positive_weight=1.0,
        asymmetric_gamma_neg=2.0,
        part_diversity_weight=0.01,
        return_aux=True,
        **kwargs,
    ):
        super().__init__()
        if int(spatial_concepts) + int(temporal_concepts) != int(num_concepts):
            raise ValueError("spatial_concepts + temporal_concepts must equal num_concepts")

        self.num_classes = int(num_classes)
        self.num_concepts = int(num_concepts)
        self.spatial_concepts = int(spatial_concepts)
        self.temporal_concepts = int(temporal_concepts)
        self.early_stages = int(early_stages)

        # Loss weights
        self.concept_loss_weight = float(concept_loss_weight)
        self.early_concept_loss_weight = float(early_concept_loss_weight)
        self.feedback_consistency_weight = float(feedback_consistency_weight)
        self.kg_loss_weight = float(kg_loss_weight)
        self.prior_retention_weight = float(prior_retention_weight)
        self.rule_sparsity_weight = float(rule_sparsity_weight)
        self.concept_negative_weight = float(concept_negative_weight)
        self.concept_positive_weight = float(concept_positive_weight)
        self.asymmetric_gamma_neg = float(asymmetric_gamma_neg)
        self.part_diversity_weight = float(part_diversity_weight)
        self.return_aux = bool(return_aux)
        self.accepts_concepts = True

        # 1. Perception Backbone
        self.backbone = SkateFormer_(
            num_classes=num_classes,
            num_people=num_people,
            num_points=num_points,
            num_frames=num_frames,
            **kwargs,
        )
        self._load_backbone_weights(backbone_weights)

        # Backbone features
        early_dim = self.backbone.stages[self.early_stages - 1].blocks[-1].transformer.norm_1.normalized_shape[0]
        late_dim = self.backbone.head.in_features


        # 2. Soft Concept Feedback Module
        self.feedback_module = SoftConceptFeedback(
            in_channels=early_dim,
            num_concepts=self.num_concepts,
            spatial_concepts=self.spatial_concepts,
            temporal_concepts=self.temporal_concepts,
            beta=feedback_beta,
            lambda_fb=feedback_lambda,
            mode=feedback_mode,
            confidence_gating=feedback_confidence_gating,
        )

        # 3. Final Pooling & Concept Decoders
        self.spatial_pool_score = nn.Conv1d(late_dim, 1, kernel_size=1)
        self.temporal_pool_score = nn.Conv1d(late_dim, 1, kernel_size=1)
        self.spatial_head = nn.Sequential(
            nn.Linear(late_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, self.spatial_concepts)
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(late_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, self.temporal_concepts)
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * late_dim, 512), nn.LayerNorm(512), nn.GELU()
        )

        # 4. Symbolic Logical Reasoner
        if action_prototypes is None and action_mode == "prototype":
            csv_candidates = [
                Path(__file__).resolve().parent.parent / "concepts" / "ntu_rgbd_60_120.csv",
                Path("./concepts/ntu_rgbd_60_120.csv"),
                Path("concepts/ntu_rgbd_60_120.csv"),
            ]
            for p in csv_candidates:
                if p.exists():
                    import csv
                    with open(p, 'r', newline='', encoding='utf-8-sig') as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        c_names = [c for c in rows[0].keys() if c != 'action_class']
                        c_names = [c for i, c in enumerate(c_names) if i not in (50, 65)]
                        vals = []
                        for row in rows[:num_classes]:
                            vals.append([float(row[c]) for c in c_names])
                        action_prototypes = torch.tensor(vals, dtype=torch.float32)
                        if concept_names is None:
                            concept_names = c_names
                        break
            if action_prototypes is None:
                action_prototypes = torch.zeros((num_classes, self.num_concepts))
                for i in range(num_classes):
                    action_prototypes[i, i % self.num_concepts] = 1.0

        if concept_names is None:
            concept_names = [f"concept_{i}" for i in range(self.num_concepts)]
        self.concept_names = list(concept_names)
        self.reasoner = KGPrimedReasoner(
            self.num_concepts,
            num_classes,
            self.concept_names,
            l1_dim=l1_dim,
            l2_dim=l2_dim,
            kg_init_mode=kg_init_mode,
            kg_prior_map=kg_prior_map,
            kg_prior_strength=kg_prior_strength,
            action_mode=action_mode,
            action_prototypes=action_prototypes,
            prototype_and_weight=prototype_and_weight,
        )


    def _load_backbone_weights(self, checkpoint_path):
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            print(f"[Warning] SkateFormer checkpoint not found at {path}. Starting with random init.")
            return
        checkpoint = torch.load(str(path), map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

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
        print(f"Loaded {len(mapped)} backbone tensors; missing={len(missing)}, unexpected={len(unexpected)}")

    def set_feedback_lambda(self, lambda_val):
        """Set lambda_fb on feedback module."""
        self.feedback_module.set_lambda_fb(lambda_val)

    def set_feedback_mode(self, mode_str):
        """Switch ablation mode dynamically."""
        self.feedback_module.set_mode(mode_str)

    def _pool_features(self, feature_map):
        spatial_tokens = feature_map.mean(dim=2)  # [B, C, N]
        spatial_attention = torch.softmax(self.spatial_pool_score(spatial_tokens), dim=-1)
        spatial = (spatial_tokens * spatial_attention).sum(dim=-1)

        temporal_tokens = feature_map.mean(dim=3)  # [B, C, T]
        temporal_attention = torch.softmax(self.temporal_pool_score(temporal_tokens), dim=-1)
        temporal = (temporal_tokens * temporal_attention).sum(dim=-1)
        return spatial, temporal

    @staticmethod
    def _part_diversity_loss(feature_map):
        batch, channels, frames, tokens = feature_map.shape
        if tokens % 6 != 0:
            return feature_map.new_zeros(())
        parts = feature_map.view(batch, channels, frames, 6, tokens // 6).mean(dim=(2, 4))
        parts = F.normalize(parts.transpose(1, 2), dim=-1)
        similarity = torch.bmm(parts, parts.transpose(1, 2))
        identity = torch.eye(6, device=similarity.device).unsqueeze(0)
        return (similarity - identity).pow(2).mean()

    def asymmetric_concept_loss(self, logits, targets):
        """Confidence-aware asymmetric loss for imbalanced concept targets."""
        probabilities = torch.sigmoid(logits)
        pos_loss = targets * torch.log(probabilities.clamp(min=1e-6))
        neg_p = (1.0 - probabilities).clamp(min=1e-6)
        neg_loss = (
            (1.0 - targets)
            * torch.pow(probabilities, self.asymmetric_gamma_neg)
            * torch.log(neg_p)
        )
        loss = -(self.concept_positive_weight * pos_loss + self.concept_negative_weight * neg_loss)
        return loss.mean()

    def compute_feedback_consistency_loss(self, early_concepts, late_concepts):
        """Encourages consistency between early and late concepts without rigid cloning."""
        return F.mse_loss(early_concepts, late_concepts.detach())

    def forward(self, input, index_t=None, return_aux=None):
        if return_aux is None:
            return_aux = self.return_aux

        # 1. Forward through early stages + soft feedback + late stages
        F_early, F_late, fb_aux = self.backbone.forward_features_with_intermediate(
            input,
            feedback_fn=self.feedback_module,
            early_stages=self.early_stages,
            index_t=index_t
        )

        # 2. Extract final spatial & temporal concepts from F_late
        spatial_feat, temporal_feat = self._pool_features(F_late)
        spatial_concept_logits = self.spatial_head(spatial_feat)
        temporal_concept_logits = self.temporal_head(temporal_feat)
        concept_logits = torch.cat([spatial_concept_logits, temporal_concept_logits], dim=-1)
        late_concepts = torch.sigmoid(concept_logits)

        # 3. Reasoner: Concept-to-Action Logical Inference
        action_logits, logic_trace = self.reasoner(concept_logits, return_trace=True)

        if not return_aux:
            return action_logits

        # 4. Compile Auxiliary Outputs & Losses
        aux = {
            "action_logits": action_logits,
            "concept_logits": concept_logits,
            "late_concepts": late_concepts,
            "spatial_concept_logits": spatial_concept_logits,
            "temporal_concept_logits": temporal_concept_logits,
            "early_concept_logits": fb_aux.get("early_concept_logits"),
            "early_concepts": fb_aux.get("early_concepts"),
            "feedback_bias": fb_aux.get("feedback_bias"),
            "feedback_gate": fb_aux.get("feedback_gate"),
            "feedback_mode": fb_aux.get("feedback_mode", self.feedback_module.mode),
            "lambda_fb": fb_aux.get("lambda_fb", self.feedback_module.lambda_fb),
            "logic_trace": logic_trace,
            "kg_loss": self.reasoner.kg_loss(concept_logits),
            "prior_loss": self.reasoner.rule_prior_loss(),
            "sparsity_loss": self.reasoner.rule_sparsity_loss(),
            "part_diversity_loss": self._part_diversity_loss(F_late),
            "feedback_consistency_loss": (
                self.compute_feedback_consistency_loss(fb_aux["early_concepts"], late_concepts)
                if fb_aux.get("early_concepts") is not None else concept_logits.new_zeros(())
            ),
        }
        return action_logits, aux

    def compute_loss(self, outputs, action_targets, concept_targets=None):
        """Multi-task loss computation for end-to-end training."""
        if isinstance(outputs, tuple):
            action_logits, aux = outputs
        else:
            action_logits = outputs
            aux = {}

        # 1. Action Cross-Entropy
        action_loss = F.cross_entropy(action_logits, action_targets)

        # 2. Concept Losses
        late_concept_loss = action_logits.new_zeros(())
        early_concept_loss = action_logits.new_zeros(())
        if concept_targets is not None:
            if "concept_logits" in aux and aux["concept_logits"] is not None:
                late_concept_loss = self.asymmetric_concept_loss(aux["concept_logits"], concept_targets)
            if "early_concept_logits" in aux and aux["early_concept_logits"] is not None:
                early_concept_loss = self.asymmetric_concept_loss(aux["early_concept_logits"], concept_targets)

        # 3. Regularization & Auxiliary Losses
        kg_loss = aux.get("kg_loss", action_logits.new_zeros(()))
        prior_loss = aux.get("prior_loss", action_logits.new_zeros(()))
        sparsity_loss = aux.get("sparsity_loss", action_logits.new_zeros(()))
        part_loss = aux.get("part_diversity_loss", action_logits.new_zeros(()))
        consistency_loss = aux.get("feedback_consistency_loss", action_logits.new_zeros(()))

        total_loss = (
            action_loss
            + self.concept_loss_weight * late_concept_loss
            + self.early_concept_loss_weight * early_concept_loss
            + self.feedback_consistency_weight * consistency_loss
            + self.kg_loss_weight * kg_loss
            + self.prior_retention_weight * prior_loss
            + self.rule_sparsity_weight * sparsity_loss
            + self.part_diversity_weight * part_loss
        )

        loss_breakdown = {
            "total_loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "late_concept_loss": late_concept_loss.item() if torch.is_tensor(late_concept_loss) else 0.0,
            "early_concept_loss": early_concept_loss.item() if torch.is_tensor(early_concept_loss) else 0.0,
            "consistency_loss": consistency_loss.item() if torch.is_tensor(consistency_loss) else 0.0,
            "kg_loss": kg_loss.item() if torch.is_tensor(kg_loss) else 0.0,
            "prior_loss": prior_loss.item() if torch.is_tensor(prior_loss) else 0.0,
            "sparsity_loss": sparsity_loss.item() if torch.is_tensor(sparsity_loss) else 0.0,
            "part_loss": part_loss.item() if torch.is_tensor(part_loss) else 0.0,
        }
        return total_loss, loss_breakdown


# Aliases for configuration flexibility
REASON_SkateFormer = SFB_ECR_SkateFormer
ReasonSkateFormer = SFB_ECR_SkateFormer

"""REASON-SkateFormer.

SkateFormer is used as the perception backbone.  The output is split into
spatial and temporal concept predictions, followed by a KG-primed logical
reasoner.  This module deliberately keeps the KG after the concept decoder:
the visual backbone remains independent from the symbolic prior.
"""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SkateFormer import SkateFormer_


def _logit(probability, eps=1e-4):
    probability = probability.clamp(eps, 1.0 - eps)
    return torch.log(probability) - torch.log1p(-probability)


class KGUnionLayer(nn.Module):
    """Differentiable conjunction/disjunction rule bank.

    ``W`` is stored in logit space and constrained to [0, 1] at use time.
    The first rule bank can be initialized from concept-level KG relations.
    Later rule banks learn over intermediate logical nodes and are therefore
    intentionally initialized without concept-name assumptions.
    """

    def __init__(self, input_dim, output_dim, prior=None):
        super().__init__()
        if prior is None:
            prior = torch.full((input_dim, output_dim), 0.05)
            prior[torch.arange(min(input_dim, output_dim)),
                  torch.arange(min(input_dim, output_dim))] = 0.9
        noise = 0.02 * torch.randn_like(prior)
        self.and_weight = nn.Parameter(_logit((prior + noise).clamp(1e-4, 1 - 1e-4)))
        self.or_weight = nn.Parameter(_logit((prior + noise).clamp(1e-4, 1 - 1e-4)))

    def forward(self, x):
        and_weight = torch.sigmoid(self.and_weight)
        or_weight = torch.sigmoid(self.or_weight)

        # A selected input contributes x to an AND rule and x to an OR rule;
        # an input with a near-zero rule weight has negligible influence.
        conjunction = torch.prod(
            1.0 - and_weight.unsqueeze(0) * (1.0 - x.unsqueeze(-1)), dim=1
        )
        disjunction = 1.0 - torch.prod(
            1.0 - or_weight.unsqueeze(0) * x.unsqueeze(-1), dim=1
        )
        return torch.cat([conjunction, disjunction], dim=1)


class KGPrimedReasoner(nn.Module):
    """Concept-to-action logical reasoner with first-layer KG priming.

    KG pairs are interpreted symmetrically during initialization, matching the
    current REASON configuration.  Only the first rule bank sees raw concept
    indices; subsequent banks operate on learned logical nodes.
    """

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

        # Each rule is seeded by one positive concept.  The NOT half of the
        # input is left available to the learned logic instead of being given
        # a misleading positive KG edge.
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

    def kg_loss(self, concept_logits):
        """Optional soft consistency penalty for the configured KG edges."""
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
                    # The second concept requires the first one.
                    penalties.append(F.relu(pb - pa))
                elif relation_name == "alongside":
                    penalties.append(F.relu(0.5 - pa * pb))
                else:
                    penalties.append(pa * pb)
        if not penalties:
            return concept_logits.new_zeros(())
        return torch.stack([penalty.mean() for penalty in penalties]).mean()

    def forward(self, concept_logits):
        # The binary representation exposes both the concept and its NOT
        # literal to the first logical layer, as in the original REASON CRL.
        concepts = torch.sigmoid(concept_logits)
        literals = torch.cat([concepts, 1.0 - concepts], dim=1)
        hidden_1 = self.first(literals)
        hidden_2 = self.second(hidden_1)
        return self.action(hidden_2)


class ReasonSkateFormer(nn.Module):
    """SkateFormer perception + REASON concept/logic heads."""

    def __init__(
        self,
        num_classes=120,
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
        concept_loss_weight=1.0,
        kg_loss_weight=0.01,
        part_diversity_weight=0.01,
        return_aux=True,
        **kwargs,
    ):
        super().__init__()
        if int(spatial_concepts) + int(temporal_concepts) != int(num_concepts):
            raise ValueError("spatial_concepts + temporal_concepts must equal num_concepts")
        self.num_concepts = int(num_concepts)
        self.spatial_concepts = int(spatial_concepts)
        self.temporal_concepts = int(temporal_concepts)
        self.concept_loss_weight = float(concept_loss_weight)
        self.kg_loss_weight = float(kg_loss_weight)
        self.part_diversity_weight = float(part_diversity_weight)
        self.return_aux = bool(return_aux)
        self.accepts_concepts = True

        self.backbone = SkateFormer_(
            num_classes=num_classes,
            num_people=num_people,
            num_points=num_points,
            num_frames=num_frames,
            **kwargs,
        )
        self._load_backbone_weights(backbone_weights)
        # The original local SkateFormer implementation predates the
        # ``num_features`` convenience attribute used by the HF port.
        feature_dim = self.backbone.head.in_features

        # Learned pooling preserves the distinction between pose-centric and
        # dynamics-centric concepts while still using the full Skate feature map.
        self.spatial_pool_score = nn.Conv1d(feature_dim, 1, kernel_size=1)
        self.temporal_pool_score = nn.Conv1d(feature_dim, 1, kernel_size=1)
        self.spatial_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, self.spatial_concepts)
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, self.temporal_concepts)
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * feature_dim, 512), nn.LayerNorm(512), nn.GELU()
        )

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
        )

    def _load_backbone_weights(self, checkpoint_path):
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"SkateFormer backbone checkpoint not found: {path}")
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
            # The action classifier is dataset-specific (e.g. 60 vs 120
            # classes).  REASON replaces it with its concept/logic heads.
            if clean_key.startswith("head."):
                continue
            if clean_key in backbone_state:
                mapped[clean_key] = value
        missing, unexpected = self.backbone.load_state_dict(mapped, strict=False)
        loaded = len(mapped)
        print(
            f"Loaded {loaded} SkateFormer backbone tensors from {path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if loaded == 0:
            raise RuntimeError(f"No compatible tensors found in checkpoint: {path}")

    def _pool_features(self, feature_map):
        # feature_map: [B, C, T, V*M], with V*M = 24*2 for NTU.
        spatial_tokens = feature_map.mean(dim=2)
        spatial_attention = torch.softmax(
            self.spatial_pool_score(spatial_tokens), dim=-1
        )
        spatial = (spatial_tokens * spatial_attention).sum(dim=-1)

        temporal_tokens = feature_map.mean(dim=3)
        temporal_attention = torch.softmax(
            self.temporal_pool_score(temporal_tokens), dim=-1
        )
        temporal = (temporal_tokens * temporal_attention).sum(dim=-1)
        return spatial, temporal

    @staticmethod
    def _part_diversity_loss(feature_map):
        # The partitioned NTU input is six groups of four joints per person.
        # After flattening people, each group contains eight tokens.
        batch, channels, frames, tokens = feature_map.shape
        if tokens % 6 != 0:
            return feature_map.new_zeros(())
        parts = feature_map.view(batch, channels, frames, 6, tokens // 6).mean(dim=(2, 4))
        parts = F.normalize(parts.transpose(1, 2), dim=-1)
        gram = torch.matmul(parts, parts.transpose(-1, -2))
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        return ((gram - eye) ** 2).mean()

    def forward(self, input, index_t=None, concept_targets=None):
        feature_map = self.backbone.forward_features_from_input(input, index_t=index_t)
        spatial_feat, temporal_feat = self._pool_features(feature_map)
        spatial_logits = self.spatial_head(spatial_feat)
        temporal_logits = self.temporal_head(temporal_feat)
        concept_logits = torch.cat([spatial_logits, temporal_logits], dim=1)
        action_logits = self.reasoner(concept_logits)

        if not self.return_aux and concept_targets is None:
            return action_logits

        output = {
            "action_logits": action_logits,
            "concept_logits": concept_logits,
            "spatial_logits": spatial_logits,
            "temporal_logits": temporal_logits,
            "fused_feature": self.fusion(torch.cat([spatial_feat, temporal_feat], dim=1)),
            "kg_loss": self.reasoner.kg_loss(concept_logits),
            "part_diversity_loss": self._part_diversity_loss(feature_map),
        }
        if concept_targets is not None:
            targets = concept_targets.float().to(concept_logits.device)
            if targets.shape[-1] != self.num_concepts:
                raise ValueError(
                    f"Expected concept targets with {self.num_concepts} labels, got {targets.shape[-1]}"
                )
            output["concept_loss"] = F.binary_cross_entropy_with_logits(
                concept_logits, targets
            )
            output["aux_loss"] = (
                self.concept_loss_weight * output["concept_loss"]
                + self.kg_loss_weight * output["kg_loss"]
                + self.part_diversity_weight * output["part_diversity_loss"]
            )
        return output


def REASON_SkateFormer(**kwargs):
    """Config-friendly factory used by ``main.py``."""
    return ReasonSkateFormer(**kwargs)

"""Soft Concept Feedback Attention Module for SkateFormer.

Extracts early spatial and temporal concept representations from intermediate
backbone features and injects soft feedback bias (channel modulation + spatial-temporal
token bias) into downstream backbone layers with learnable confidence gating.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftConceptFeedback(nn.Module):
    """Concept-Guided Soft Feedback Attention Module.

    Supports multiple ablation modes:
    - 'token_soft_bias' / 'feature_modulation': Full proposed model (Channel Mod + Spatial-Temporal Soft Token Bias + Gate)
    - 'channel_modulation': Channel modulation only
    - 'early_head_only': Early concept extraction for auxiliary loss, but no feedback into backbone
    - 'no_confidence_gate': Spatial-temporal soft bias without confidence gating (gate=1.0)
    - 'random_concept_proj': Feedback driven by random projection / noise to test semantic necessity
    - 'disabled' / 'none': Passthrough without feedback
    """

    def __init__(
        self,
        in_channels=192,
        num_concepts=72,
        spatial_concepts=52,
        temporal_concepts=20,
        beta=0.2,
        lambda_fb=0.1,
        mode="token_soft_bias",
        confidence_gating=True,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_concepts = num_concepts
        self.spatial_concepts = spatial_concepts
        self.temporal_concepts = temporal_concepts
        self.beta = float(beta)
        self.lambda_fb = float(lambda_fb)
        self.mode = str(mode).lower()
        self.confidence_gating = bool(confidence_gating)

        # 1. Early Spatial & Temporal Pooling Scores
        self.spatial_pool_score = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.temporal_pool_score = nn.Conv1d(in_channels, 1, kernel_size=1)

        # 2. Early Concept Prediction Heads
        self.early_spatial_head = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.spatial_concepts)
        )
        self.early_temporal_head = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.temporal_concepts)
        )

        # 3. Channel Modulation Generator
        self.mlp_channel = nn.Sequential(
            nn.Linear(num_concepts, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels)
        )

        # 4. Spatial-Temporal Token Bias Generator
        self.conv_token = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        )
        self.mlp_concept = nn.Sequential(
            nn.Linear(num_concepts, in_channels // 4),
            nn.GELU(),
            nn.Linear(in_channels // 4, 1)
        )

        # 5. Confidence Gating
        self.confidence_fc = nn.Sequential(
            nn.Linear(num_concepts, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize token and channel mod layers with small weights to ensure stable warmup
        for m in self.conv_token.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.mlp_channel.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.mlp_concept.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_lambda_fb(self, lambda_val):
        """Dynamically adjust lambda_fb (e.g. during warmup / annealing)."""
        self.lambda_fb = float(lambda_val)

    def set_mode(self, mode_str):
        self.mode = str(mode_str).lower()

    def extract_early_concepts(self, F_early):
        """Extract spatial and temporal early concept predictions from F_early.
        Args:
            F_early: [B, C, T, N]
        Returns:
            early_concept_logits: [B, num_concepts]
            c_early: [B, num_concepts] in [0, 1]
            a_spa: [B, 1, N]
            a_tem: [B, 1, T]
        """
        # Spatial pooling across time frames
        spatial_tokens = F_early.mean(dim=2)  # [B, C, N]
        a_spa = torch.softmax(self.spatial_pool_score(spatial_tokens), dim=-1)  # [B, 1, N]
        f_spa = (spatial_tokens * a_spa).sum(dim=-1)  # [B, C]

        # Temporal pooling across joints/people
        temporal_tokens = F_early.mean(dim=3)  # [B, C, T]
        a_tem = torch.softmax(self.temporal_pool_score(temporal_tokens), dim=-1)  # [B, 1, T]
        f_tem = (temporal_tokens * a_tem).sum(dim=-1)  # [B, C]

        # Heads
        z_spa_early = self.early_spatial_head(f_spa)  # [B, spatial_concepts]
        z_tem_early = self.early_temporal_head(f_tem)  # [B, temporal_concepts]
        early_concept_logits = torch.cat([z_spa_early, z_tem_early], dim=-1)  # [B, num_concepts]
        c_early = torch.sigmoid(early_concept_logits)

        return early_concept_logits, c_early, a_spa, a_tem

    def forward(self, F_early):
        """
        Args:
            F_early: Intermediate backbone feature map of shape [B, C, T, N]
        Returns:
            F_mod: Modulated feature map [B, C, T, N]
            aux: Dict containing early logits, concepts, biases, gates and attention maps
        """
        early_concept_logits, c_early, a_spa, a_tem = self.extract_early_concepts(F_early)

        aux = {
            "early_concept_logits": early_concept_logits,
            "early_concepts": c_early,
            "spatial_attn": a_spa,
            "temporal_attn": a_tem,
            "feedback_bias": None,
            "feedback_gate": None,
            "feedback_mode": self.mode,
            "lambda_fb": self.lambda_fb,
        }

        # 1. Disabled or Early-Head-Only: No feature alteration
        if self.mode in ("disabled", "none", "early_head_only"):
            return F_early, aux

        # 2. Random Concept Projection (Ablation A6)
        if self.mode == "random_concept_proj":
            c_input = torch.randn_like(c_early)
        else:
            c_input = c_early

        # 3. Channel Modulation
        g_chan = torch.sigmoid(self.mlp_channel(c_input)).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 4. Mode: Channel Modulation Only (Ablation A3)
        if self.mode == "channel_modulation":
            F_mod = F_early * (1.0 + self.beta * g_chan)
            return F_mod, aux

        # 5. Token & Feature Soft Bias (Ablations A4, A5, A6)
        b_token = self.conv_token(F_early)  # [B, 1, T, N]
        b_concept = self.mlp_concept(c_input).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        bias = torch.tanh(b_token + b_concept)  # [B, 1, T, N]

        # 6. Confidence Gating
        if self.mode == "no_confidence_gate" or not self.confidence_gating:
            gate = torch.ones((F_early.shape[0], 1, 1, 1), device=F_early.device, dtype=F_early.dtype)
        else:
            gate = self.confidence_fc(c_early.detach()).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]

        # 7. Modulate Feature Map: Residual Channel Scaling + Gated Soft Spatial-Temporal Token Bias
        F_mod = F_early * (1.0 + self.beta * g_chan) + (self.lambda_fb * gate) * bias * F_early

        aux["feedback_bias"] = bias
        aux["feedback_gate"] = gate

        return F_mod, aux

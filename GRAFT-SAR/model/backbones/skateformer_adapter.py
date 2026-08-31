"""
SkateFormer Backbone Adapter
=============================
Wraps SkateFormer (Skeletal-Temporal Transformer) into the standardized
BaseBackboneAdapter interface.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.backbones.base_adapter import BaseBackboneAdapter
from model.SkateFormer import SkateFormer_


DEFAULT_SKATEFORMER_CFG = dict(
    in_channels=3,
    num_people=2,
    num_points=24,
    num_frames=64,
    kernel_size=7,
    num_heads=32,
    attn_drop=0.0,
    head_drop=0.0,
    rel=True,
    drop_path=0.0,
    type_1_size=(8, 8),
    type_2_size=(8, 12),
    type_3_size=(8, 8),
    type_4_size=(8, 12),
    mlp_ratio=4.0,
    index_t=True,
)


class SkateFormerAdapter(BaseBackboneAdapter):
    """
    Adapter for SkateFormer Transformer Backbone.
    Feature Dimension: 192
    """

    def __init__(self, **kwargs):
        super().__init__()
        cfg = DEFAULT_SKATEFORMER_CFG.copy()
        cfg.update(kwargs)
        # Handle parameter aliases gracefully
        if "drop_path_rate" in cfg:
            cfg["drop_path"] = cfg.pop("drop_path_rate")
        if "attn_drop_rate" in cfg:
            cfg["attn_drop"] = cfg.pop("attn_drop_rate")
        if "drop_rate" in cfg:
            cfg["drop"] = cfg.pop("drop_rate")

        self.num_classes = cfg.get("num_classes", 60)
        self.backbone = SkateFormer_(**cfg)
        self._feat_dim = cfg.get("channels", (96, 192, 192, 192))[-1]

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def extract_feat(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extracts pooled feature (B, 192).
        Mirrors SkateFormer.forward() through stem and transformer stages.
        """
        tokens = self.extract_tokens(x, index_t=index_t)
        return tokens.mean(dim=1)

    def extract_tokens(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract spatio-temporal tokens before global pooling.

        Returns a tensor of shape ``(B, T' * V', C)`` for concept decoders.
        """
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        B, C, T, V, M = x.shape
        expected_c = self.backbone.stem[0].in_channels
        if C != expected_c:
            raise ValueError(
                "SkateFormer input channel mismatch: "
                f"received C={C}, but the backbone was built with "
                f"in_channels={expected_c}. Set model_args.backbone_cfg.in_channels "
                "to match the feeder output."
            )
        output = x.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)  # [B, C, T, M * V]

        for layer in self.backbone.stem:
            output = layer(output)

        if self.backbone.index_t:
            if index_t is None:
                index_t = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
            te = torch.zeros(B, T, self.backbone.embed_dim, device=x.device)
            div_term = torch.exp(
                torch.arange(0, self.backbone.embed_dim, 2, dtype=torch.float, device=x.device)
                * -(math.log(10000.0) / self.backbone.embed_dim)
            )
            te[:, :, 0::2] = torch.sin(index_t.float().unsqueeze(-1) * div_term)
            te[:, :, 1::2] = torch.cos(index_t.float().unsqueeze(-1) * div_term)
            output = output + torch.einsum(
                "btc,cv->bctv", te, self.backbone.joint_person_embedding
            )
        else:
            output = output + self.backbone.joint_person_temporal_embedding

        output = self.backbone.forward_features(output)  # (B, C, T', V')
        tokens = output.flatten(2).transpose(1, 2).contiguous()
        return tokens

    def forward(self, x: torch.Tensor, index_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(-1)
        if index_t is None and self.backbone.index_t:
            B, _, T, _, _ = x.shape
            index_t = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return self.backbone(x, index_t)

    def load_pretrained(self, checkpoint_path: str, device: torch.device = torch.device("cpu")) -> Tuple[list, list]:
        raw = torch.load(checkpoint_path, map_location=device)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        state = {k.replace("module.", ""): v for k, v in state.items()}

        # If saved under 'backbone.', strip it
        if any(k.startswith("backbone.") for k in state):
            state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}

        model_state = self.backbone.state_dict()
        compatible_state = {
            key: value
            for key, value in state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        skipped = [
            key for key, value in state.items()
            if key in model_state and model_state[key].shape != value.shape
        ]
        missing, unexpected = self.backbone.load_state_dict(compatible_state, strict=False)
        # Expose shape-mismatched checkpoint keys as unexpected so the
        # caller's diagnostic reports that they were intentionally skipped.
        return missing, unexpected + skipped

    def get_classifier_weights(self) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if hasattr(self.backbone, "head") and isinstance(self.backbone.head, nn.Linear):
            return self.backbone.head.weight, self.backbone.head.bias
        return None

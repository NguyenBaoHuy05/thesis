"""
concept_model.py
----------------
Wrapper combining:
  1. Frozen SkateFormer backbone (pretrained).
  2. Trainable ConceptDecoder outputting 74 concept logits.
  3. SymbolicDecoder translating concept probabilities → action logits.

Forward returns both concept logits and action logits so that a combined loss
(masked concept BCE + action CE) can be computed in the training loop.

Usage example:
    model = ConceptModel(
        backbone_ckpt="path/to/skateformer.pt",
        num_classes=60,
        num_concepts=74,
        backbone_kwargs=dict(num_people=2, num_points=24, kernel_size=7,
                              num_heads=32, attn_drop=0.5, ...),
        decoder_kwargs=dict(d_model=256, num_heads=8, num_layers=2),
    )
    concept_logits, action_logits = model(data, index_t)
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

from model.SkateFormer import SkateFormer_
from model.concept_decoder import ConceptDecoder
from model.symbolic_decoder import SymbolicDecoder


class ConceptModel(nn.Module):
    """
    Frozen SkateFormer backbone + trainable ConceptDecoder + SymbolicDecoder.

    Args:
        backbone_ckpt:   Path to SkateFormer pretrained weights (.pt).
                         If None, backbone is randomly initialised (for debugging).
        num_classes:     Number of action classes (60 or 120).
        num_concepts:    Number of binary concepts (default 74).
        num_people:      Number of people in the skeleton sequence (default 2).
        backbone_kwargs: Extra keyword arguments forwarded to SkateFormer_.
        decoder_kwargs:  Extra keyword arguments forwarded to ConceptDecoder.
        freeze_backbone: Whether to freeze backbone parameters (default True).
        backbone_dim:    Channel dimension of the last backbone stage output.
                         For SkateFormer_ this is 192 (channels[-1]).
        use_symbolic:    Whether to attach a SymbolicDecoder (default True).
        template_matrix: Tensor [num_classes, num_concepts] with 0/1 (concept-to-class
                         mapping) for initialising SymbolicDecoder.
    """

    def __init__(
        self,
        backbone_ckpt: Optional[str] = None,
        num_classes: int = 60,
        num_concepts: int = 74,
        backbone_dim: int = 192,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        freeze_backbone: bool = True,
        use_symbolic: bool = True,
        template_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.freeze_backbone = freeze_backbone

        # ---- Backbone ----
        bkw = backbone_kwargs or {}
        # SkateFormer_ default num_classes=60; set correctly
        bkw.setdefault("num_classes", num_classes)
        self.backbone = SkateFormer_(**bkw)

        if backbone_ckpt is not None:
            self._load_backbone(backbone_ckpt)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        # ---- Concept Decoder ----
        dkw = dict(decoder_kwargs or {})
        # 'mode', 'hidden_dim', 'dropout' (in SymbolicDecoder sense) are NOT
        # ConceptDecoder kwargs — strip them before forwarding.
        symbolic_mode   = dkw.pop("mode", "learned")
        symbolic_hidden = dkw.pop("hidden_dim", 0)
        dkw["backbone_dim"] = backbone_dim
        dkw["num_concepts"] = num_concepts
        self.decoder = ConceptDecoder(**dkw)

        # ---- Symbolic Decoder (concept probs → action logits) ----
        self.use_symbolic = use_symbolic
        if use_symbolic:
            self.symbolic = SymbolicDecoder(
                num_concepts=num_concepts,
                num_classes=num_classes,
                template_matrix=template_matrix,
                mode=symbolic_mode,
                hidden_dim=symbolic_hidden,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_backbone(self, ckpt_path: str):
        """Load pretrained SkateFormer weights, ignoring action head mismatches."""
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        raw = torch.load(ckpt_path, map_location="cpu")

        # Handle nested state dicts (sometimes saved as {"model": ...})
        if isinstance(raw, dict) and "model" in raw:
            raw = raw["model"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]

        # Strip "module." prefix from DataParallel checkpoints
        weights = OrderedDict(
            [(k.replace("module.", ""), v) for k, v in raw.items()]
        )

        state = self.backbone.state_dict()
        # Only keep keys that match (ignore head mismatch or extra keys)
        filtered = {k: v for k, v in weights.items() if k in state and v.shape == state[k].shape}
        missing = [k for k in state.keys() if k not in filtered]
        unexpected = [k for k in weights.keys() if k not in state]

        state.update(filtered)
        self.backbone.load_state_dict(state)

        print(
            f"[ConceptModel] Loaded backbone from {ckpt_path}\n"
            f"  Matched:    {len(filtered)}/{len(state)} keys\n"
            f"  Missing:    {len(missing)} keys\n"
            f"  Unexpected: {len(unexpected)} keys"
        )
        if missing:
            print(f"  First missing keys: {missing[:5]}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def extract_features(
        self, data: torch.Tensor, index_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Run backbone stem + stages; return feature map [B, C, T', V'].

        The SkateFormer forward does:
          stem → positional embedding → forward_features → forward_head
        We replicate stem + forward_features WITHOUT forward_head to get
        the spatial-temporal feature map.
        """
        B, C, T, V, M = data.shape
        x = data.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)  # [B, C, T, M*V]

        for layer in self.backbone.stem:
            x = layer(x)

        if self.backbone.index_t:
            embed_dim = self.backbone.embed_dim
            te = torch.zeros(B, T, embed_dim, device=x.device, dtype=x.dtype)
            div_term = torch.exp(
                torch.arange(0, embed_dim, 2, dtype=torch.float, device=x.device)
                * -(math.log(10000.0) / embed_dim)
            )
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            x = x + torch.einsum(
                "b t c, c v -> b c t v", te, self.backbone.joint_person_embedding
            )
        else:
            x = x + self.backbone.joint_person_temporal_embedding

        # feature map [B, channels[-1], T', V']
        x = self.backbone.forward_features(x)
        return x

    def forward(
        self,
        data: torch.Tensor,    # [B, 3, T, V, M]
        index_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            concept_logits:  [B, num_concepts]  (raw; sigmoid for probs)
            action_logits:   [B, num_classes]   (from symbolic decoder) or None
        """
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.extract_features(data, index_t)
        else:
            features = self.extract_features(data, index_t)

        concept_logits = self.decoder(features)          # [B, num_concepts]

        action_logits = None
        if self.use_symbolic:
            concept_probs = torch.sigmoid(concept_logits)
            action_logits = self.symbolic(concept_probs)   # [B, num_classes]

        return concept_logits, action_logits

    def forward_action_only(
        self,
        data: torch.Tensor,
        index_t: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: run full backbone (original forward) for baseline comparison."""
        return self.backbone(data, index_t)

    def train(self, mode: bool = True):
        """Override train() to keep backbone in eval mode when frozen."""
        super().train(mode)
        if self.freeze_backbone and mode:
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Parameter groups for optimiser
    # ------------------------------------------------------------------

    def decoder_parameters(self):
        """Yield only the trainable decoder + symbolic parameters."""
        params = list(self.decoder.parameters())
        if self.use_symbolic:
            params += list(self.symbolic.parameters())
        return params

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

"""
CLIP Semantic Alignment and Multi-Positive Contrastive Loss
===========================================================
Aligns skeleton concept representations with natural language prompt embeddings
via Low-Rank Adaptation (LoRA) and Multi-Positive Contrastive Learning.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import clip
from model.loralib.utils import apply_lora, mark_only_lora_as_trainable


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@dataclass
class MultiPositiveClipConfig:
    init_logit_scale: float = 2.6592  # np.log(1 / 0.07) ~ 2.6592
    logit_scale_min: float = -4.0
    logit_scale_max: float = 4.6      # max temperature scale ~ 100
    l2_normalize: bool = True
    eps: float = 1e-6
    reduction: str = "mean"
    symmetric: bool = True
    label_smoothing: float = 0.0


class MultiPositiveClipLoss(nn.Module):
    """
    Robust Multi-Positive CLIP Loss supporting many-to-many concept-prompt alignments.
    """

    def __init__(self, cfg: MultiPositiveClipConfig = MultiPositiveClipConfig()):
        super().__init__()
        self.cfg = cfg
        self.logit_scale = nn.Parameter(torch.tensor(cfg.init_logit_scale, dtype=torch.float32))

    def forward(
        self,
        image_feats: torch.Tensor,      # (B, D)
        text_feats: torch.Tensor,       # (C, D)
        pos_mask: torch.Tensor,         # (B, C) binary ground-truth mask
    ) -> Tuple[torch.Tensor, dict]:
        assert image_feats.dim() == 2 and text_feats.dim() == 2
        assert pos_mask.size(0) == image_feats.size(0) and pos_mask.size(1) == text_feats.size(0)

        if self.cfg.l2_normalize:
            image_feats = _l2_normalize(image_feats, self.cfg.eps)
            text_feats = _l2_normalize(text_feats, self.cfg.eps)

        scale = self.logit_scale.clamp(self.cfg.logit_scale_min, self.cfg.logit_scale_max).exp()
        sim = scale * torch.matmul(image_feats, text_feats.t())  # (B, C)

        # Image -> Text loss
        # loss_i = logsumexp(all) - logsumexp(positives)
        # Avoid log(0) if row has no positives
        row_has_pos = pos_mask.sum(dim=-1) > 0
        if not row_has_pos.any():
            return torch.tensor(0.0, device=image_feats.device, requires_grad=True), {"loss_i2t": 0.0, "loss_t2i": 0.0}

        logsumexp_all = torch.logsumexp(sim, dim=-1)
        sim_pos = torch.where(pos_mask > 0, sim, torch.full_like(sim, -1e9))
        logsumexp_pos = torch.logsumexp(sim_pos, dim=-1)

        loss_i2t = (logsumexp_all[row_has_pos] - logsumexp_pos[row_has_pos]).mean()

        if self.cfg.symmetric:
            col_has_pos = pos_mask.sum(dim=0) > 0
            if col_has_pos.any():
                sim_t = sim.t()
                pos_mask_t = pos_mask.t()
                logsumexp_all_t = torch.logsumexp(sim_t, dim=-1)
                sim_pos_t = torch.where(pos_mask_t > 0, sim_t, torch.full_like(sim_t, -1e9))
                logsumexp_pos_t = torch.logsumexp(sim_pos_t, dim=-1)
                loss_t2i = (logsumexp_all_t[col_has_pos] - logsumexp_pos_t[col_has_pos]).mean()
            else:
                loss_t2i = loss_i2t
            total_loss = 0.5 * (loss_i2t + loss_t2i)
        else:
            total_loss = loss_i2t

        return total_loss, {"loss_i2t": loss_i2t.item(), "loss_t2i": total_loss.item(), "scale": scale.item()}


class LoRATextEncoder(nn.Module):
    """
    CLIP Text Encoder with optional LoRA Adaptation.
    """

    def __init__(
        self,
        clip_backbone: str = "ViT-B/32",
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        use_lora: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.use_lora = use_lora

        # Load CLIP model
        base_clip, _ = clip.load(clip_backbone, device=device)
        del base_clip.visual  # We only need text branch
        self.clip_model = base_clip

        if use_lora and lora_r > 0:
            from model.loralib.lora_args import LoraArgs
            lora_args = LoraArgs(
                r=lora_r,
                alpha=lora_alpha,
                dropout_rate=lora_dropout,
                encoder="text",
                position="all",
                params=["q", "k", "v"],
            )
            apply_lora(lora_args, self.clip_model)
            mark_only_lora_as_trainable(self.clip_model)
        else:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        # ``clip.load`` may return a FP16 model on CUDA.  The LoRA modules and
        # the text activations are initialized/produced in FP32, which causes
        # Linear layers inside the text transformer to fail with
        # "mat1 and mat2 must have the same dtype".  Keep the text branch in
        # FP32; this is also more stable for text embeddings and LoRA training.
        self.clip_model.float()

    def encode_prompts(self, prompt_list: List[str], device: torch.device) -> torch.Tensor:
        """
        Tokenizes and encodes a list of prompt strings into text embeddings (N, D).
        """
        tokens = clip.tokenize(prompt_list).to(device)
        text_features = self.clip_model.encode_text(tokens)
        return _l2_normalize(text_features.float())

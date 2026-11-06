"""LightConceptGrounding: Semantic grounding nhẹ cho concept (G2 trong
`Architecture/10_CLIR-SkateFormer_v2_Proposal.md`).

Khác với REASON/GRAFT-SAR (LoRA fine-tune toàn bộ CLIP text encoder + prompt
theo từng batch), module này:
  - Dùng CLIP **đóng băng hoàn toàn** (không LoRA, không backprop qua text tower).
  - Encode text anchor cho từng concept **đúng 1 lần** lúc khởi tạo model, lưu vào
    buffer — không forward CLIP lại trong training loop.
  - Chỉ học một projection nhỏ (Linear + LayerNorm) từ concept feature (đã có sẵn
    từ BodyPartGroundedConceptHeads) sang không gian CLIP.

An toàn khi không có internet / không tải được checkpoint CLIP: `available=False`,
mọi lời gọi loss trả về 0 thay vì crash — giữ nguyên tắc "graceful fallback" như
`_load_backbone_weights` của CLIR-SkateFormer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_concept_prompt(concept_name: str) -> str:
    """Sinh câu mô tả mặc định từ tên concept dạng snake_case, vd:
    'hand_lift_to_face' -> 'a person is doing hand lift to face'."""
    words = concept_name.replace("_", " ")
    return f"a person is doing {words}"


class LightConceptGrounding(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        concept_names: List[str],
        clip_backbone: str = "ViT-B/32",
        clip_dim: int = 512,
        device: Optional[torch.device] = None,
        concept_prompts: Optional[List[str]] = None,
    ):
        super().__init__()
        self.concept_names = list(concept_names)
        self.num_concepts = len(self.concept_names)
        self.clip_dim = int(clip_dim)

        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, clip_dim),
            nn.LayerNorm(clip_dim),
        )

        self.register_buffer("text_anchors", torch.zeros(self.num_concepts, clip_dim))
        self.available = False

        prompts = concept_prompts or [default_concept_prompt(c) for c in self.concept_names]
        self._try_build_text_anchors(prompts, clip_backbone, device)

    @torch.no_grad()
    def _try_build_text_anchors(self, prompts: List[str], clip_backbone: str, device):
        try:
            from model.clip import clip  # vendored package trong model/clip/ (khong phai pip package 'clip')
        except Exception as e:
            print(f"[LightConceptGrounding] Khong import duoc goi 'clip': {e}. Grounding se bi TAT.")
            return
        try:
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            clip_model, _ = clip.load(clip_backbone, device=device)
            clip_model.eval()
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            text_features = clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1).cpu()
            self.text_anchors.copy_(text_features)
            self.available = True
            del clip_model
            print(f"[LightConceptGrounding] Da encode {len(prompts)} concept text anchors qua CLIP '{clip_backbone}' (dong bang).")
        except Exception as e:
            print(f"[LightConceptGrounding] Khong tai duoc CLIP checkpoint ({e}). Grounding se bi TAT — "
                  f"model van chay binh thuong, chi thieu L_ground.")

    def project(self, feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.feat_proj(feat), dim=-1)

    def compute_loss(
        self,
        group_feats: Dict[str, torch.Tensor],
        concept_targets: torch.Tensor,
        concept_slices: Dict[str, Tuple[int, int]],
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Multi-positive InfoNCE giữa skeleton feature từng nhóm và text anchor của
        đúng các concept thuộc nhóm đó.

        Args:
            group_feats: {group_name: [B, feature_dim]}
            concept_targets: [B, num_concepts] nhãn nhị phân
            concept_slices: {group_name: (start, end)} chỉ số concept của nhóm trong 72 concept
        """
        if not self.available or not group_feats:
            ref = next(iter(group_feats.values())) if group_feats else concept_targets
            return ref.new_zeros(())

        device = concept_targets.device
        total_loss = concept_targets.new_zeros(())
        count = 0
        for g, feat in group_feats.items():
            if g not in concept_slices:
                continue
            start, end = concept_slices[g]
            if end <= start:
                continue
            pos_mask = concept_targets[:, start:end].to(device)  # [B, C_g]
            if pos_mask.sum() == 0:
                continue

            skel_embed = self.project(feat)                              # [B, clip_dim]
            text_embed = self.text_anchors[start:end].to(device)         # [C_g, clip_dim]
            logits = skel_embed @ text_embed.t() / temperature           # [B, C_g]
            log_prob = F.log_softmax(logits, dim=-1)

            pos_count = pos_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            loss_g = -(log_prob * pos_mask).sum(dim=-1) / pos_count.squeeze(-1)  # [B]
            total_loss = total_loss + loss_g.mean()
            count += 1

        if count == 0:
            return concept_targets.new_zeros(())
        return total_loss / count

    def extra_repr(self) -> str:
        return f"num_concepts={self.num_concepts}, clip_dim={self.clip_dim}, available={self.available}"

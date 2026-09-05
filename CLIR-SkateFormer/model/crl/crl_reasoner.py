"""CRLReasoner: Concept Reasoning Logic Reasoner.

Module này gói gọn toàn bộ pipeline suy luận logic 1 lượt:
  z^t ∈ R^(B × 72)
    → LiteralLayer → l^t ∈ R^(B × 144)
    → ConjunctionLayer (L1 AND, Part-grouped) → r_and ∈ R^(B × 256)
    → [ConjunctionLayer2 (L2a AND2) ‖ DisjunctionLayer (L2b OR1)]
    → r_and2 ∈ R^(B × 128), r_or1 ∈ R^(B × 128)
    → ActionHead [Linear: 256 → 120] → y^t ∈ R^(B × 120)

    Logic Evidence (cho ConceptUpdateCell):
      r^t = [r_and; r_and2; r_or1] ∈ R^(B × 512)

Trọng số được chia sẻ (shared) giữa các vòng lặp trong IterativeLogicReasoner.

Thiết kế:
  - ActionHead KHÔNG nhận feature bypass từ backbone
    (đảm bảo tính giải thích tuyệt đối — strict logic mode)
  - Logic Evidence kết hợp cả ba tầng: L1 (body-part features),
    L2a (coordinated patterns), L2b (alternative patterns)
"""

from typing import Tuple

import torch
import torch.nn as nn

from .literal_layer import LiteralLayer
from .conjunction_layer import ConjunctionLayer, ConjunctionLayer2
from .disjunction_layer import DisjunctionLayer

# ─── Kích thước mặc định ────────────────────────────────────────────────────
_N_CONCEPTS  = 72   # số concepts (72 sau khi loại 2 null concepts)
_N_ACTIONS   = 120  # số lớp hành động NTU RGB+D 120
_R1          = 256  # L1 AND rules
_R2A         = 128  # L2a AND2 rules
_R2B         = 128  # L2b OR1 rules
_LOGIC_DIM   = _R1 + _R2A + _R2B  # = 512


class CRLReasoner(nn.Module):
    """Concept Reasoning Logic Reasoner (1-pass).

    Được dùng với shared weights trong IterativeLogicReasoner:
    mỗi vòng lặp gọi cùng 1 instance CRLReasoner này.

    Args:
        n_concepts:  Số concepts (mặc định 72).
        n_actions:   Số lớp hành động (mặc định 120).
        r1:          Số L1 AND rules (mặc định 256).
        r2a:         Số L2a AND2 rules (mặc định 128).
        r2b:         Số L2b OR1 rules (mặc định 128).
        literal_mode: 'soft' hoặc 'ste' (có thể thay đổi runtime).
        temperature: Nhiệt độ STE cho LiteralLayer.
    """

    def __init__(
        self,
        n_concepts:   int   = _N_CONCEPTS,
        n_actions:    int   = _N_ACTIONS,
        r1:           int   = _R1,
        r2a:          int   = _R2A,
        r2b:          int   = _R2B,
        literal_mode: str   = "ste",
        temperature:  float = 2.0,
    ):
        super().__init__()
        self.n_concepts  = int(n_concepts)
        self.n_actions   = int(n_actions)
        self.r1  = int(r1)
        self.r2a = int(r2a)
        self.r2b = int(r2b)
        self.logic_dim = r1 + r2a + r2b  # = 512 mặc định

        # ── Các sub-module ──────────────────────────────────────────────────
        self.literal_layer = LiteralLayer(
            n_concepts=n_concepts,
            mode=literal_mode,
            temperature=temperature,
        )

        self.l1_and   = ConjunctionLayer(
            n_literals=2 * n_concepts,  # 148
            n_rules=r1,                 # 256
            part_grouped=True,
        )

        self.l2a_and2 = ConjunctionLayer2(
            n_rules_l1=r1,    # 256
            n_rules_l2a=r2a,  # 128
        )

        self.l2b_or1  = DisjunctionLayer(
            n_rules_l1=r1,    # 256
            n_rules_l2b=r2b,  # 128
        )

        # ActionHead: KHÔNG có feature bypass — strict logic mode
        self.action_head = nn.Linear(r2a + r2b, n_actions)  # 256 → 120

    # ── Properties để điều chỉnh LiteralLayer ────────────────────────────────
    @property
    def literal_mode(self) -> str:
        return self.literal_layer.mode

    @literal_mode.setter
    def literal_mode(self, value: str):
        self.literal_layer.mode = value

    @property
    def temperature(self) -> float:
        return self.literal_layer.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.literal_layer.temperature = float(value)

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Suy luận logic 1 lượt.

        Args:
            z: Concept logits [B, n_concepts]  (LUôN là số thực liên tục)

        Returns:
            r_evidence: Logic Evidence [B, logic_dim]   (= [r_and; r_and2; r_or1])
            y:          Action logits  [B, n_actions]
        """
        # Bước 1: Tạo literals (chỉ binarize ở đây, z vẫn liên tục)
        l = self.literal_layer(z)              # [B, 148]

        # Bước 2: L1 AND rules (part-grouped)
        r_and = self.l1_and(l)                 # [B, 256]

        # Bước 3: L2 song song
        r_and2 = self.l2a_and2(r_and)         # [B, 128]
        r_or1  = self.l2b_or1(r_and)          # [B, 128]

        # Bước 4: ActionHead (strict logic mode)
        l2_cat = torch.cat([r_and2, r_or1], dim=-1)  # [B, 256]
        y = self.action_head(l2_cat)                   # [B, n_actions]

        # Logic Evidence: [r_and; r_and2; r_or1] = [B, 256+128+128] = [B, 512]
        r_evidence = torch.cat([r_and, r_and2, r_or1], dim=-1)

        return r_evidence, y

    # ── Loss helpers ─────────────────────────────────────────────────────────
    def rule_sparsity_loss(self) -> torch.Tensor:
        """Rule Sparsity Loss theo kế hoạch (Mục 8.2 CLIR plan).

        L_sparse = mean(w_and1) + mean(w_and2) + mean(w_or1)
        """
        return (
            self.l1_and.sparsity_loss()
            + self.l2a_and2.sparsity_loss()
            + self.l2b_or1.sparsity_loss()
        ) / 3.0

    def extra_repr(self) -> str:
        return (
            f"n_concepts={self.n_concepts}, n_actions={self.n_actions}, "
            f"r1={self.r1}, r2a={self.r2a}, r2b={self.r2b}, "
            f"logic_dim={self.logic_dim}"
        )

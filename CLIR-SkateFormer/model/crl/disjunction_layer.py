"""DisjunctionLayer: Tầng OR rules cho CRL pipeline (L2b).

Module này thực hiện tuyển (disjunction) của các AND rules từ L1:
  r_and ∈ R^(B × 256)  →  r_or1 ∈ R^(B × 128)

Ý nghĩa: Mô tả các hành động có nhiều cách thực hiện khác nhau (alternative patterns).
Ví dụ: (arm_raise ∧ hand_lift) ∨ (hand_grasp ∧ arm_retract)

Công thức Probabilistic OR (De Morgan + log-space):
  r_or1_k = 1 - exp( Σ_j w_jk · log(1 - r_and_j + ε) )
           = 1 - ∏_j (1 - r_and_j + ε)^w_jk

  Trọng số w_jk = sigmoid(θ_jk) ∈ [0, 1]

Lưu ý:
  - Nếu tất cả w ≈ 0: r_or1 ≈ 0 (không kích hoạt)
  - Nếu ít nhất một AND rule j mạnh (r_and_j ≈ 1) và w_jk lớn:
    log(1 - r_and_j) → -∞  → r_or1_k → 1.0
"""

from typing import Optional

import torch
import torch.nn as nn

from .conjunction_layer import GradGraft, Product, _EPS


class DisjunctionLayer(nn.Module):
    """L2b OR1 layer: Tuyển của các AND rules (Alternative Action Patterns).

    Cải tiến theo 6 cơ chế REASON:
      1. W ∈ [0, 1] trực tiếp (khởi tạo 0.5 * rand, clip sau mỗi optimizer step)
      2. Łukasiewicz Product T-norm: 1 - Product.apply(1 - r_and * W)
      3. GradGraft: forward binary, backward continuous
      4. L1 Sparsity Penalty: l1_norm() và sparsity_loss()

    Input:  r_and  ∈ R^(B × n_rules_l1)  — L1 conjunction rule activations
    Output: r_or1  ∈ R^(B × n_rules_l2b) — L2b disjunction rule activations
    """

    def __init__(
        self,
        n_rules_l1: int = 256,
        n_rules_l2b: int = 128,
        init_range: float = 0.5,
        grad_graft: bool = False,
        eps: float = _EPS,
    ):
        super().__init__()
        self.n_rules_l1 = int(n_rules_l1)
        self.n_rules_l2b = int(n_rules_l2b)
        self.grad_graft = bool(grad_graft)
        self.eps = float(eps)

        # Trọng số W ∈ [0, 1] trực tiếp
        self.W = nn.Parameter(float(init_range) * torch.rand(self.n_rules_l1, self.n_rules_l2b))

    @property
    def weight_logit(self) -> torch.Tensor:
        """Hỗ trợ tương thích ngược cho code cũ truy cập weight_logit."""
        return torch.logit(self.W.data.clamp(1e-4, 1.0 - 1e-4))

    def _effective_weights(self) -> torch.Tensor:
        """Trọng số thực tế W ∈ [0, 1]."""
        return self.W.clamp(0.0, 1.0)

    def clip(self):
        """Clip trọng số vào đoạn [0, 1] sau optimizer step (REASON Mechanism 3)."""
        self.W.data.clamp_(0.0, 1.0)

    def l1_norm(self) -> torch.Tensor:
        """Tổng L1 norm cho sparsity penalty (REASON Mechanism 5)."""
        return torch.sum(self._effective_weights())

    def sparsity_loss(self) -> torch.Tensor:
        """Mean sparsity loss."""
        return self._effective_weights().mean()

    def continuous_forward(self, r_and: torch.Tensor) -> torch.Tensor:
        """Łukasiewicz Product T-norm tuyển liên tục: 1 - Product(1 - r_and * W)."""
        w = self._effective_weights()                       # [n_rules_l1, n_rules_l2b]
        X = 1.0 - r_and.unsqueeze(-1) * w                   # [B, n_rules_l1, n_rules_l2b]
        return 1.0 - Product.apply(X)                       # [B, n_rules_l2b]

    @torch.no_grad()
    def binarized_forward(self, r_and: torch.Tensor) -> torch.Tensor:
        """Binarized rule execution với W > 0.5."""
        Wb = (self._effective_weights() > 0.5).float()
        return 1.0 - torch.prod(1.0 - r_and.unsqueeze(-1) * Wb, dim=1)

    def forward(self, r_and: torch.Tensor, continuous: Optional[bool] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            r_and: L1 conjunction activations [B, n_rules_l1]
            continuous: Nếu True, dùng continuous forward. Nếu None, dùng not self.grad_graft.
        """
        res_tilde = self.continuous_forward(r_and)
        use_cont = continuous if continuous is not None else (not self.grad_graft)
        if use_cont:
            return res_tilde
        res_bar = self.binarized_forward(r_and)
        return GradGraft.apply(res_bar, res_tilde)

    def extra_repr(self) -> str:
        return f"n_rules_l1={self.n_rules_l1}, n_rules_l2b={self.n_rules_l2b}, grad_graft={self.grad_graft}"


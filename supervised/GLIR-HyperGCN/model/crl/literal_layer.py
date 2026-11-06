"""LiteralLayer: Chuyển đổi concept logits thành literals nhị phân.

Module này thực hiện bước đầu tiên trong luận chứng logic (CRL pipeline):
  z^t ∈ R^(B × 72)  →  l^t ∈ R^(B × 144)

Hai chế độ:
  - 'soft': Dùng sigmoid liên tục (Phase 1 warm-up, gradient mượt)
  - 'ste':  Dùng Straight-Through Estimator nhị phân (Phase 2 rule sparsification)

Nguyên tắc bất biến:
  z^t (concept logits) LUÔN là số thực liên tục.
  Binarization CHỈ xảy ra khi tạo l^t để truyền vào CRL.
"""

import torch
import torch.nn as nn


class BinarizeSTE(torch.autograd.Function):
    """Straight-Through Estimator (STE) cho binarization.

    Forward:  hard = (z > 0).float()  [nhị phân {0, 1}]
    Backward: gradient đi thẳng qua sigmoid liên tục (không bị chặn bởi hard step)
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, temperature: float) -> torch.Tensor:
        # Lưu sigmoid liên tục để dùng trong backward
        soft = torch.sigmoid(temperature * z)
        ctx.save_for_backward(soft)
        ctx.temperature = temperature
        # Output nhị phân
        hard = (z > 0.0).float()
        # STE trick: hard + soft - soft.detach() → forward = hard, backward ≈ d(soft)/dz
        return hard + soft - soft.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (soft,) = ctx.saved_tensors
        # Gradient của sigmoid: soft * (1 - soft) * temperature
        grad_z = grad_output * soft * (1.0 - soft) * ctx.temperature
        return grad_z, None  # None cho temperature (không học được)


def binarize_ste(z: torch.Tensor, temperature: float = 5.0) -> torch.Tensor:
    """Hàm tiện ích gọi BinarizeSTE."""
    return BinarizeSTE.apply(z, temperature)


class LiteralLayer(nn.Module):
    """Chuyển đổi concept logits z^t thành 144 literals.

    Output gồm 72 literal dương (p_i) nối với 72 literal phủ định (1 - p_i).

    Args:
        n_concepts: Số lượng concepts (mặc định 72).
        mode: 'soft' hoặc 'ste'.
        temperature: Nhiệt độ cho STE (annealing dần từ 5.0 → 10.0 ở Phase 2).
    """

    def __init__(self, n_concepts: int = 72, mode: str = "soft", temperature: float = 5.0):
        super().__init__()
        self.n_concepts = int(n_concepts)
        self._mode = mode
        self.temperature = float(temperature)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        assert value in ("soft", "ste"), f"mode phải là 'soft' hoặc 'ste', nhận được: {value!r}"
        self._mode = value

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Tính literals từ concept logits.

        Args:
            z: Concept logits [B, n_concepts]  (LUôN là số thực liên tục)

        Returns:
            l: Literals [B, 2 * n_concepts]  (cả dương lẫn phủ định)
        """
        if self._mode == "soft":
            p = torch.sigmoid(z)                   # [B, 74] ∈ (0, 1)
            l = torch.cat([p, 1.0 - p], dim=-1)   # [B, 148]
        else:  # 'ste'
            c_ste = binarize_ste(z, self.temperature)   # ≈ {0, 1}, grad qua sigmoid
            l = torch.cat([c_ste, 1.0 - c_ste], dim=-1)  # [B, 148]
        return l

    def extra_repr(self) -> str:
        return f"n_concepts={self.n_concepts}, mode={self._mode!r}, temperature={self.temperature}"

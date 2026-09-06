"""LiteralLayer: Chuyển đổi concept logits thành literals nhị phân.

Module này thực hiện bước đầu tiên trong luận chứng logic (CRL pipeline):
  z^t ∈ R^(B × 72)  →  l^t ∈ R^(B × 144)

Hai chế độ:
  - 'ste':  Dùng Straight-Through Estimator nhị phân (mặc định: Pure STE, temperature annealing 2.0 → 10.0)
  - 'soft': Dùng sigmoid liên tục (hỗ trợ warm-up / phân tích liên tục nếu cần)

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


def binarize_ste(z: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """Hàm tiện ích gọi BinarizeSTE."""
    return BinarizeSTE.apply(z, temperature)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (tương tự REASON / PyTorch RMSNorm).

    Chuẩn hóa concept logits z^t để kháng vanishing/exploding gradient trước logic layer.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-8, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = (int(normalized_shape),)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        rms = x.pow(2).mean(dim=dims, keepdim=True).add(self.eps).sqrt()
        y = x / rms
        if self.weight is not None:
            y = y * self.weight
        return y

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class ScaledSoftsign(nn.Module):
    """Softsign with adjustable bounds - gentlest saturation, best gradient preservation (từ REASON).

    scaled_softsign(x) = scale * x / (beta + |x|), output ∈ (-scale, scale).
    Không bão hòa nhanh như sigmoid/tanh, gradient mượt hơn ở vùng xa 0.
    """

    def __init__(self, scale: float = 3.0, beta: float = 1.0):
        super().__init__()
        self.scale = float(scale)
        self.beta = float(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x / (self.beta + torch.abs(x))

    def extra_repr(self) -> str:
        return f"scale={self.scale}, beta={self.beta}"


class LiteralLayer(nn.Module):
    """Chuyển đổi concept logits z^t thành literals nhị phân hoặc liên tục.

    Quy trình tiền xử lý theo REASON:
      z^t ∈ R^(B × n_concepts)
        → RMSNorm(n_concepts)
        → ScaledSoftsign() ∈ (-3, 3)
        → STE binarization hoặc Sigmoid
        → Literals l = [p, 1 - p] ∈ R^(B × 2*n_concepts)

    Args:
        n_concepts: Số lượng concepts (mặc định 72).
        mode: 'ste' (mặc định cho Pure STE) hoặc 'soft'.
        temperature: Nhiệt độ cho STE (annealing từ 2.0 → 10.0 xuyên suốt training).
        scale: Scale factor cho ScaledSoftsign (mặc định 3.0).
        beta: Beta parameter cho ScaledSoftsign (mặc định 1.0).
    """

    def __init__(
        self,
        n_concepts: int = 72,
        mode: str = "ste",
        temperature: float = 2.0,
        scale: float = 3.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.n_concepts = int(n_concepts)
        self._mode = mode
        self.temperature = float(temperature)

        # REASON Mechanism 1: Preprocessing Input
        self.norm = RMSNorm(self.n_concepts)
        self.scaled_softsign = ScaledSoftsign(scale=scale, beta=beta)

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
            z: Concept logits [B, n_concepts] (LUÔN là số thực liên tục)

        Returns:
            l: Literals [B, 2 * n_concepts] (cả dương lẫn phủ định)
        """
        # Bước 1: RMSNorm + ScaledSoftsign tiền xử lý logits
        z_prep = self.norm(z)
        z_prep = self.scaled_softsign(z_prep)

        # Bước 2: Binarize hoặc Soft Sigmoid
        if self._mode == "soft":
            p = torch.sigmoid(z_prep)                  # [B, n_concepts] ∈ (0, 1)
            l = torch.cat([p, 1.0 - p], dim=-1)       # [B, 2 * n_concepts]
        else:  # 'ste'
            c_ste = binarize_ste(z_prep, self.temperature)   # ≈ {0, 1}, grad qua sigmoid
            l = torch.cat([c_ste, 1.0 - c_ste], dim=-1)      # [B, 2 * n_concepts]
        return l

    def extra_repr(self) -> str:
        return f"n_concepts={self.n_concepts}, mode={self._mode!r}, temperature={self.temperature}"


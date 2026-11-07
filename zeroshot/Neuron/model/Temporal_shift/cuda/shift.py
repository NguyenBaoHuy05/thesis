"""Shift operation of Shift-GCN (Cheng et al., CVPR 2020).

Each channel c is moved by a learnable fractional offset (ypos[c] along time H, xpos[c] along
joints W) using bilinear interpolation with zero padding; the output keeps every `stride`-th row.

Two backends:
- 'torch' (default): pure PyTorch (gather + bilinear weights). No CUDA build is needed. Autograd
  gives the input gradient and the plain bilinear gradient for xpos/ypos.
- 'cuda': the original compiled `shift_cuda` kernel. It is used only when SHIFT_BACKEND=cuda and the
  extension is importable. Its backward also applies `applyShiftConstraint` to the position
  gradients, which the torch backend does not reproduce. This only matters when the Shift
  positions themselves are trained; in this thesis the Shift-GCN encoder is frozen.
"""
import os

import torch
from torch.autograd import Function
from torch.nn import Module, Parameter

BACKEND = os.environ.get('SHIFT_BACKEND', 'torch')
if BACKEND == 'cuda':
    import shift_cuda


def shift_torch(input, xpos, ypos, stride=1):
    """Same forward as shift_cuda_forward_kernel. input: (N, C, H, W); xpos, ypos: (C,)."""
    if stride != 1:
        ypos = ypos + 0.5          # same as ShiftFunction.forward of the CUDA version
    N, C, H, W = input.shape
    Ho = H // stride
    dev = input.device

    x1f, y1f = torch.floor(xpos), torch.floor(ypos)
    dx = (xpos - x1f).to(input.dtype).view(1, C, 1, 1)
    dy = (ypos - y1f).to(input.dtype).view(1, C, 1, 1)
    x1, y1 = x1f.long(), y1f.long()

    h = torch.arange(Ho, device=dev) * stride      # row on the input map
    w = torch.arange(W, device=dev)

    def sample(hi, wi):                               # hi: (C, Ho), wi: (C, W)
        valid = ((hi >= 0) & (hi < H)).view(1, C, Ho, 1) & ((wi >= 0) & (wi < W)).view(1, C, 1, W)
        rows = torch.gather(input, 2, hi.clamp(0, H - 1).view(1, C, Ho, 1).expand(N, C, Ho, W))
        vals = torch.gather(rows, 3, wi.clamp(0, W - 1).view(1, C, 1, W).expand(N, C, Ho, W))
        return vals * valid

    hy1 = h.view(1, Ho) + y1.view(C, 1)
    wx1 = w.view(1, W) + x1.view(C, 1)
    q11 = sample(hy1, wx1)
    q21 = sample(hy1, wx1 + 1)
    q12 = sample(hy1 + 1, wx1)
    q22 = sample(hy1 + 1, wx1 + 1)
    return q11 * (1 - dx) * (1 - dy) + q21 * dx * (1 - dy) + q12 * (1 - dx) * dy + q22 * dx * dy


class ShiftFunction(Function):
    """Original CUDA autograd function (only with SHIFT_BACKEND=cuda)."""

    @staticmethod
    def forward(ctx, input, xpos, ypos, stride=1):
        if stride != 1:
            ypos = ypos + 0.5
        input = input.contiguous()
        output = shift_cuda.forward(input, xpos, ypos, stride)
        ctx.save_for_backward(input, output, xpos, ypos)
        ctx.stride = stride
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        input, output, xpos, ypos = ctx.saved_tensors
        grad_input, grad_xpos, grad_ypos = shift_cuda.backward(grad_output, input, output, xpos, ypos, ctx.stride)
        return grad_input, grad_xpos, grad_ypos, None


class Shift(Module):

    def __init__(self, channel, stride, init_scale=3):
        super(Shift, self).__init__()
        self.stride = stride
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xpos = Parameter(torch.zeros(channel, device=dev))
        self.ypos = Parameter(torch.zeros(channel, device=dev))
        self.xpos.data.uniform_(-1e-8, 1e-8)
        self.ypos.data.uniform_(-init_scale, init_scale)

    def forward(self, input):
        if BACKEND == 'cuda':
            return ShiftFunction.apply(input, self.xpos, self.ypos, self.stride)
        return shift_torch(input, self.xpos, self.ypos, self.stride)

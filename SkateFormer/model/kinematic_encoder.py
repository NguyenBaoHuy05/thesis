"""Lightweight motion stream for HCG-NS concept grounding."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class KinematicEncoder(nn.Module):
    """Encode per-joint velocity/acceleration into backbone-aligned tokens.

    The backbone sees joint coordinates and downsamples time 64 -> 8, which
    smooths out fast, small hand motion (tapping, pressing).  This branch
    computes finite differences at the full input frame rate, runs a small
    per-joint temporal convolution and only then pools to the backbone's
    temporal resolution.  Token order matches ``SkateFormer.forward_features_raw``
    (person-major: token = person * num_points + joint).

    The output projection is zero-initialised, so a freshly added branch
    leaves the pretrained feature map unchanged at the start of training.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, out_dim: int = 192):
        super().__init__()
        # velocity (C) + acceleration (C) + speed magnitude (1)
        feature_channels = 2 * int(in_channels) + 1
        self.input_norm = nn.BatchNorm2d(feature_channels)
        self.temporal = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, kernel_size=(5, 1), padding=(2, 0)),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 1), padding=(2, 0)),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _difference(x: torch.Tensor) -> torch.Tensor:
        # Keep T by repeating a zero difference at the first frame.
        return F.pad(x[:, :, 1:] - x[:, :, :-1], (0, 0, 1, 0))

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected raw skeleton [B,C,T,V,M], got {tuple(x.shape)}")
        batch, channels, timesteps, points, people = x.shape
        tokens = x.permute(0, 1, 2, 4, 3).reshape(batch, channels, timesteps, people * points)
        if tokens.shape[-1] != target_size[1]:
            raise ValueError(
                f"motion tokens ({tokens.shape[-1]}) do not match backbone tokens ({target_size[1]})"
            )
        velocity = self._difference(tokens)
        acceleration = self._difference(velocity)
        speed = velocity.norm(dim=1, keepdim=True)
        features = self.input_norm(torch.cat([velocity, acceleration, speed], dim=1))
        hidden = self.temporal(features)
        hidden = F.adaptive_avg_pool2d(hidden, target_size)  # [B,H,T',V']
        return self.output(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

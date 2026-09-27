"""Shared spatial encoder: ``X [B,T,1,H,W] -> H_m [B,T,d,h_d,w_d]`` (FR-ENC-01).

The time dimension is never folded (interface constraint of the specification):
the encoder is applied frame-wise with shared weights.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .blocks import ConvBlock, build_act, build_norm

__all__ = ["SpatialEncoder"]


class SpatialEncoder(nn.Module):
    """2D convolutional encoder shared by both branches.

    Args:
        in_channels: input frame channels (1 for radar/VIL).
        base_channels: width of the first stage.
        channels: channel widths of the successive stages.
        downsample: total spatial downsampling factor (1, 2, 4, ...).
        norm / act: normalisation and activation type.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32,
                 channels: Sequence[int] = (64, 64), downsample: int = 2,
                 norm: str = "group", act: str = "gelu") -> None:
        super().__init__()
        if downsample < 1 or downsample & (downsample - 1):
            raise ValueError(f"downsample must be a power of two, got {downsample}")
        self.downsample = int(downsample)
        stages = int(round(torch.log2(torch.tensor(float(self.downsample))).item())) if self.downsample > 1 else 0

        layers: List[nn.Module] = []
        c_in = int(in_channels)
        c_out = int(base_channels)
        for i in range(max(stages, 1)):
            stride = 2 if i < stages else 1
            layers.append(ConvBlock(c_in, c_out, 3, stride=stride, norm=norm, act=act))
            layers.append(ConvBlock(c_out, c_out, 3, stride=1, norm=norm, act=act))
            c_in = c_out
        for width in channels:
            layers.append(ConvBlock(c_in, int(width), 3, stride=1, norm=norm, act=act))
            layers.append(ConvBlock(int(width), int(width), 3, stride=1, norm=norm, act=act))
            c_in = int(width)
        self.net = nn.Sequential(*layers)
        self.out_channels = int(c_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B,T,1,H,W]`` (or ``[B*T,1,H,W]``) -> ``[B,T,d,h_d,w_d]``."""
        if x.ndim == 5:
            b, t = x.shape[0], x.shape[1]
            flat = x.reshape(b * t, *x.shape[2:])
            out = self.net(flat)
            return out.reshape(b, t, *out.shape[1:])
        return self.net(x)

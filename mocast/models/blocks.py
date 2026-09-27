"""Shared building blocks (norms, conv stacks, SimVP-style group convolutions)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "build_norm",
    "build_act",
    "ConvBlock",
    "GroupConvBlock",
    "resize_tensor",
]


def build_norm(norm: str, channels: int) -> nn.Module:
    """``group`` (default) / ``layer`` (channels-last) / ``batch`` / ``instance`` / ``none``."""
    norm = (norm or "none").lower()
    if norm == "group":
        groups = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else 1)
        return nn.GroupNorm(groups, channels)
    if norm == "layer":
        return nn.GroupNorm(1, channels)
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm '{norm}'")


def build_act(act: str) -> nn.Module:
    act = (act or "gelu").lower()
    if act == "gelu":
        return nn.GELU()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "silu" or act == "swish":
        return nn.SiLU(inplace=True)
    if act == "tanh":
        return nn.Tanh()
    if act == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation '{act}'")


class ConvBlock(nn.Module):
    """``Conv2d -> Norm -> Act`` (padding keeps the spatial size)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, norm: str = "group", act: str = "gelu",
                 groups: int = 1, bias: bool = True) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.norm = build_norm(norm, out_channels)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class GroupConvBlock(nn.Module):
    """SimVP / PhyDNet style bottleneck: 1x1 -> depthwise k x k -> 1x1."""

    def __init__(self, channels: int, kernel_size: int = 3, hidden: Optional[int] = None,
                 norm: str = "group", act: str = "gelu") -> None:
        super().__init__()
        hidden = int(hidden or channels)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size, padding=kernel_size // 2,
                                   groups=hidden)
        self.expand = nn.Conv2d(hidden, channels, 1)
        self.norm = build_norm(norm, channels)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        y = self.depthwise(y)
        y = self.expand(y)
        return self.act(self.norm(x + y)) if y.shape == x.shape else self.act(self.norm(y))


def resize_tensor(x: torch.Tensor, size: Tuple[int, int], mode: str = "bilinear",
                  align_corners: bool = False) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    kwargs = {}
    if mode in ("bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = align_corners
    return F.interpolate(x, size=tuple(size), mode=mode, **kwargs)

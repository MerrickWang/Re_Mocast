"""Motion-guided experts (FR-MSM-01, FR-MSM-02).

Each expert owns a receptive field (kernel / dilation) and modulates the
source-sink candidate ``E_s`` with three groups of scale/shift parameters - one
per motion kind (mean, fluctuation, total) - exactly as in Eq. (10) of the paper.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from ..blocks import ConvBlock, build_act, build_norm, resize_tensor
from .modulation import MotionModulation

__all__ = ["SourceSinkStem", "MotionGuidedExpert"]


class SourceSinkStem(nn.Module):
    """Multi-scale source-sink candidates from the precipitation frames.

    The stem downsamples the frames onto the latent grid (``downsample`` matches
    the encoder, so that motion and source-sink features are aligned in space and
    time - interface constraint of section 5).
    """

    def __init__(self, in_channels: int = 1, channels: int = 64,
                 kernels: Sequence[int] = (3, 5, 7), dilations: Sequence[int] = (1, 1, 1),
                 stride: int = 2, norm: str = "group", act: str = "gelu",
                 num_blocks: int = 1) -> None:
        super().__init__()
        self.kernels = [int(k) for k in kernels]
        self.dilations = [int(d) for d in dilations]
        self.channels = int(channels)
        self.stride = int(stride)
        if self.stride < 1 or (self.stride & (self.stride - 1)):
            raise ValueError(f"stem stride must be a power of two, got {self.stride}")
        stages = int(round(torch.log2(torch.tensor(float(self.stride))).item())) \
            if self.stride > 1 else 0
        layers: List[nn.Module] = []
        c_in = int(in_channels)
        for i in range(stages):
            c_out = self.channels if i == stages - 1 else max(self.channels // 2, 8)
            layers.append(ConvBlock(c_in, c_out, 3, stride=2, norm=norm, act=act))
            c_in = c_out
        if stages == 0:
            layers.append(ConvBlock(c_in, self.channels, 3, norm=norm, act=act))
        self.stem = nn.Sequential(*layers)
        self.experts = nn.ModuleList([
            nn.Sequential(*[
                ConvBlock(self.channels, self.channels, k, dilation=d, norm=norm, act=act)
                for _ in range(max(int(num_blocks), 1))
            ])
            for k, d in zip(self.kernels, self.dilations)
        ])

    @property
    def num_scales(self) -> int:
        return len(self.kernels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """``x``: ``[B,C,H,W]`` or ``[B,T,C,H,W]`` -> list of 4D/5D candidates."""
        if x.ndim == 5:
            b, t = x.shape[0], x.shape[1]
            feats = self.forward(x.reshape(b * t, *x.shape[2:]))
            return [f.reshape(b, t, *f.shape[1:]) for f in feats]
        features = self.stem(x)
        return [expert(features) for expert in self.experts]


class MotionGuidedExpert(nn.Module):
    """``E'_s = Linear( pool_k( gamma_k(M_k) * Norm(E_s) + beta_k(M_k) ) )``."""

    def __init__(self, channels: int, motion_channels: int, num_motion: int = 3,
                 hidden: int = 64, norm: str = "group", act: str = "gelu",
                 modulation_hidden: int = 32, cfg_init_zero: bool = False) -> None:
        super().__init__()
        self.num_motion = int(num_motion)
        self.channels = int(channels)
        self.norm = build_norm(norm, channels)
        # one modulation head per motion kind (mean / fluctuation / total).
        # NOTE: `init_zero=False` keeps every parameter on the gradient path from the
        # first step (UT-07); the adaLN-zero variant is available via cfg.
        self.modulation = nn.ModuleList([
            MotionModulation(motion_channels, hidden=modulation_hidden, out_channels=2,
                             act=act, init_zero=bool(cfg_init_zero))
            for _ in range(self.num_motion)
        ])
        self.proj = ConvBlock(channels * self.num_motion, channels, 1, norm=norm, act=act)

    @staticmethod
    def _align(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
        return resize_tensor(x, (int(size[-2]), int(size[-1])), mode="bilinear",
                             align_corners=False)

    def forward(self, candidate: torch.Tensor, motions: Sequence[torch.Tensor]) -> torch.Tensor:
        """``candidate``: ``[B,C,h,w]`` (or ``[B,T,C,h,w]``), one motion per kind."""
        if len(motions) != self.num_motion:
            raise ValueError(f"expected {self.num_motion} motion groups, got {len(motions)}")
        squeeze_time = candidate.ndim == 5
        if squeeze_time:
            b, t = candidate.shape[0], candidate.shape[1]
            candidate = candidate.reshape(b * t, *candidate.shape[2:])
            motions = [m.reshape(b * t, *m.shape[2:]) for m in motions]
        base = self.norm(candidate)
        modulations: List[torch.Tensor] = []
        for net, motion in zip(self.modulation, motions):
            aligned = self._align(motion, candidate.shape) if tuple(motion.shape[-2:]) != tuple(
                candidate.shape[-2:]) else motion
            gamma, beta = net(aligned)
            modulations.append(gamma * base + beta)
        out = self.proj(torch.cat(modulations, dim=1))
        if squeeze_time:
            out = out.reshape(b, t, *out.shape[1:])
        return out

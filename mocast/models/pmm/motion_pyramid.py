"""Multi-scale motion feature extraction (FR-PMM-08 / section 6.3).

The physical decomposition happens *first* (mean / fluctuation / total), the
multi-scale convolutions afterwards only project those semantics onto larger
receptive fields.  ``separate_heads`` (default) keeps one head per motion kind so
that the ablation "mean+fluctuation without decomposition" stays meaningful;
``joint_heads`` implements the cheaper shared variant.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from ..blocks import ConvBlock, build_act

__all__ = ["MotionPyramid"]


class MotionPyramid(nn.Module):
    """Extract ``S`` scale-specific motion representations ``M^s``."""

    def __init__(self, in_channels: int = 2, channels: int = 64,
                 kernels: Sequence[int] = (3, 5, 7), dilations: Sequence[int] = (1, 1, 1),
                 mode: str = "separate", num_motion: int = 3, norm: str = "group",
                 act: str = "gelu") -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.num_motion = int(num_motion)
        self.channels = int(channels)
        self.kernels = [int(k) for k in kernels]
        self.dilations = [int(d) for d in dilations]
        if len(self.kernels) != len(self.dilations):
            raise ValueError("kernels and dilations must have the same length")
        if self.mode == "separate":
            per_head = max(self.channels // self.num_motion, 8)
            self.heads = nn.ModuleList()
            for k, d in zip(self.kernels, self.dilations):
                scale_heads = nn.ModuleList([
                    ConvBlock(in_channels + 0, per_head, k, dilation=d, norm=norm, act=act)
                    for _ in range(self.num_motion)
                ])
                self.heads.append(scale_heads)
            self.out_channels = per_head * self.num_motion
        elif self.mode == "joint":
            self.joint = nn.ModuleList([
                ConvBlock(in_channels * self.num_motion, self.channels, k, dilation=d,
                          norm=norm, act=act)
                for k, d in zip(self.kernels, self.dilations)
            ])
            self.out_channels = self.channels
        else:
            raise ValueError(f"Unknown pyramid mode '{self.mode}'")
        self.act = build_act(act)

    @property
    def num_scales(self) -> int:
        return len(self.kernels)

    def forward(self, motions: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """``motions``: ``(M_mean, M_fluctuation, M_total)`` each ``[B,T,2,h,w]``."""
        if len(motions) != self.num_motion:
            raise ValueError(f"expected {self.num_motion} motion tensors, got {len(motions)}")
        b, t = motions[0].shape[0], motions[0].shape[1]
        flat = [m.reshape(b * t, *m.shape[2:]) for m in motions]
        outputs: List[torch.Tensor] = []
        if self.mode == "joint":
            joined = torch.cat(flat, dim=1)
            outputs = [conv(joined) for conv in self.joint]
        else:
            for scale, heads in zip(self.kernels, self.heads):
                features = [head(motion) for head, motion in zip(heads, flat)]
                outputs.append(torch.cat(features, dim=1))
        return [out.reshape(b, t, *out.shape[1:]) for out in outputs]

"""Motion conditioned scale/shift modulation utilities (FR-MSM-02)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..blocks import build_act

__all__ = ["MotionModulation"]


class MotionModulation(nn.Module):
    """Generate ``(gamma, beta)`` from motion features.

    Kept as a standalone module so that the modulation structure can be ablated
    or replaced (risk R3) without touching the expert implementation.  The final
    convolution is zero-initialised so that training starts from an identity
    modulation.
    """

    def __init__(self, motion_channels: int, hidden: int = 32, out_channels: int = 2,
                 act: str = "gelu", init_zero: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(motion_channels, hidden, 3, padding=1), build_act(act),
            nn.Conv2d(hidden, hidden, 3, padding=1), build_act(act),
            nn.Conv2d(hidden, out_channels, 1),
        )
        if init_zero:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, motion: torch.Tensor) -> Sequence[torch.Tensor]:
        """``[B,C,h,w]`` (or ``[B,T,C,h,w]``) -> ``(gamma, beta)`` broadcastable over channels."""
        return self.net(motion).chunk(2, dim=-3)

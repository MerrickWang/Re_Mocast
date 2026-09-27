"""Motion-adaptive gating network (FR-MSM-03 / FR-MSM-04).

``W_gate = softmax_s(logits_s)`` with ``sum_s W_gate = 1`` at every position -
asserted by ``UT-04``.  ``mode="uniform"`` implements ablation A5 (average
fusion), ``mode="sigmoid"`` is kept as a diagnostic alternative.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import build_act

__all__ = ["MotionAdaptiveGating"]


class MotionAdaptiveGating(nn.Module):
    """Produce per-position expert selection weights from multi-scale motions."""

    def __init__(self, motion_channels_per_scale: int, num_experts: int, hidden: int = 32,
                 mode: str = "softmax", act: str = "gelu") -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.mode = str(mode).lower()
        if self.mode in ("uniform", "mean", "average"):
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Conv2d(motion_channels_per_scale * self.num_experts, hidden, 3, padding=1),
                build_act(act),
                nn.Conv2d(hidden, hidden, 3, padding=1), build_act(act),
                nn.Conv2d(hidden, self.num_experts, 1),
            )

    def forward(self, motion_features: Sequence[torch.Tensor],
                size: Sequence[int]) -> torch.Tensor:
        """``motion_features``: list of ``[...,C,h,w]`` -> weights ``[...,S,h,w]``."""
        lead = motion_features[0].shape[:-3]
        h, w = int(size[-2]), int(size[-1])
        if self.mode in ("uniform", "mean", "average"):
            shape = (*lead, self.num_experts, h, w)
            return torch.full(shape, 1.0 / self.num_experts,
                              dtype=motion_features[0].dtype, device=motion_features[0].device)
        stack_shape = (*lead, sum(m.shape[-3] for m in motion_features), h, w)
        stacked = torch.cat([m.reshape(*m.shape[:-3], -1, h, w)
                             if tuple(m.shape[-2:]) == (h, w)
                             else F.interpolate(m.reshape(-1, *m.shape[-3:]), size=(h, w),
                                                mode="bilinear", align_corners=False).reshape(
                                 *m.shape[:-3], m.shape[-3], h, w)
                             for m in motion_features], dim=-3)
        flat = stacked.reshape(-1, stack_shape[-3], h, w)
        logits = self.net(flat).reshape(*lead, self.num_experts, h, w)
        if self.mode == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-3)

"""Classical baselines required by the implementation plan (P1, section 8.3).

* :class:`PersistenceBaseline` - repeat the last frame (the standard sanity floor).
* :class:`BlockMatchingFlow` - dependency free dense motion estimation by local
  block matching (the "optical-flow style" baseline of the specification).
* :class:`OpticalFlowBaseline` - semi-Lagrangian advection with the estimated
  flow, i.e. the pure-advection reference used to judge whether learning the
  source-sink term is worth it (ablation A6 comparison).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .advection import Advection, warp

__all__ = ["PersistenceBaseline", "BlockMatchingFlow", "OpticalFlowBaseline"]


class PersistenceBaseline(nn.Module):
    """``Y_hat^{t+k} = X^T`` for every lead time."""

    def __init__(self, output_len: int = 20) -> None:
        super().__init__()
        self.output_len = int(output_len)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        pred = x[:, -1:].repeat(1, self.output_len, 1, 1, 1)
        return {"pred": pred, "motion": torch.zeros(pred.shape[0], self.output_len, 2,
                                                    *pred.shape[-2:], device=pred.device)}


class BlockMatchingFlow(nn.Module):
    """Dense flow by exhaustive local block matching (integer pixel precision)."""

    def __init__(self, search_radius: int = 4, block_size: int = 8,
                 padding_mode: str = "border", smooth_kernel: int = 3) -> None:
        super().__init__()
        self.search_radius = int(search_radius)
        self.block_size = int(block_size)
        self.padding_mode = str(padding_mode)
        self.smooth_kernel = int(smooth_kernel)

    def forward(self, previous: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
        """``[B,1,H,W]`` x2 -> flow ``[B,2,H,W]`` with the *content* displacement."""
        b, c, h, w = previous.shape
        bs = self.block_size
        if h % bs or w % bs:
            pad_h, pad_w = (-h) % bs, (-w) % bs
            previous = F.pad(previous, (0, pad_w, 0, pad_h), mode=self.padding_mode)
            following = F.pad(following, (0, pad_w, 0, pad_h), mode=self.padding_mode)
        _, _, hh, ww = previous.shape

        targets = F.unfold(following, kernel_size=bs, stride=bs)      # [B, bs*bs, L]
        best_cost = torch.full((b, targets.shape[-1]), float("inf"), device=previous.device)
        best_dx = torch.zeros_like(best_cost)
        best_dy = torch.zeros_like(best_cost)
        radius = self.search_radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                flow = torch.zeros(b, 2, hh, ww, device=previous.device, dtype=previous.dtype)
                flow[:, 0] = float(dx)
                flow[:, 1] = float(dy)
                shifted = warp(previous, flow, unit="pixel", semantics="displacement",
                               padding_mode=self.padding_mode, align_corners=True)
                candidates = F.unfold(shifted, kernel_size=bs, stride=bs)
                cost = (candidates - targets).abs().mean(dim=1)        # [B,L]
                improved = cost < best_cost
                best_cost = torch.where(improved, cost, best_cost)
                best_dx = torch.where(improved, torch.full_like(best_dx, float(dx)), best_dx)
                best_dy = torch.where(improved, torch.full_like(best_dy, float(dy)), best_dy)

        grid_h, grid_w = hh // bs, ww // bs
        block_flow = torch.cat([
            best_dx.reshape(b, 1, grid_h, grid_w),
            best_dy.reshape(b, 1, grid_h, grid_w),
        ], dim=1)                                                       # [B,2,gh,gw]
        flow = F.interpolate(block_flow, size=(hh, ww), mode="nearest")
        if self.smooth_kernel > 1:
            k = self.smooth_kernel
            flow = F.avg_pool2d(F.pad(flow, (k // 2, k // 2, k // 2, k // 2), mode="replicate"),
                                kernel_size=k, stride=1)
        return flow[..., :h, :w]


class OpticalFlowBaseline(nn.Module):
    """Advect the last input frame with a persistent block-matching flow."""

    def __init__(self, output_len: int = 20, search_radius: int = 4, block_size: int = 8,
                 smooth_kernel: int = 3, align_corners: bool = True) -> None:
        super().__init__()
        self.output_len = int(output_len)
        self.flow_estimator = BlockMatchingFlow(search_radius, block_size,
                                                smooth_kernel=smooth_kernel)
        self.advection = Advection(unit="pixel", semantics="displacement",
                                   align_corners=align_corners)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """``x``: ``[B,T,1,H,W]`` -> persistent-flow advection of the last frame."""
        flow = self.flow_estimator(x[:, -2], x[:, -1])                 # [B,2,H,W]
        current = x[:, -1]
        outputs = []
        for _ in range(self.output_len):
            current = self.advection(current, flow)
            outputs.append(current)
        pred = torch.stack(outputs, dim=1)
        return {"pred": pred, "motion": flow[:, None].repeat(1, self.output_len, 1, 1, 1)}

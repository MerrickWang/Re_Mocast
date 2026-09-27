"""Differentiable advection (FR-ADV-01).

``X_hat^{t+1} = Adv(X_hat^t, M^{t->t+1})`` is implemented with
``torch.nn.functional.grid_sample``.

Conventions are made explicit and *must* be recorded in the run config
(risk R5):

``unit``
    ``"pixel"`` (default) - motion is a displacement in pixels of the target
    grid; ``"normalized"`` - motion already lives in grid units ([-1, 1]).
``semantics``
    ``"displacement"`` (default) - the flow is the displacement of the
    precipitation pattern from ``t`` to ``t+1``, so the warp samples at
    ``p - flow``; ``"backward_flow"`` - the flow is the sampling offset itself
    (``p + flow``, the formula printed in the reproduction document ``grid = x +
    2*dx/(W-1)``).
``align_corners``
    controls the pixel <-> grid conversion (``2*dx/(W-1)`` vs ``2*dx/W``).

``UT-05`` pins the sign with a synthetic translation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["base_grid", "warp", "Advection", "motion_magnitude"]


def base_grid(height: int, width: int, align_corners: bool = True,
              device: Optional[torch.device] = None,
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Identity sampling grid ``[1,H,W,2]`` with (x, y) in ``[-1, 1]``."""
    if align_corners:
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    else:
        xs = (2.0 * torch.arange(width, device=device, dtype=dtype) + 1.0) / width - 1.0
        ys = (2.0 * torch.arange(height, device=device, dtype=dtype) + 1.0) / height - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)[None]


def _flow_to_grid(flow: torch.Tensor, height: int, width: int, unit: str = "pixel",
                  semantics: str = "displacement", align_corners: bool = True) -> torch.Tensor:
    """Convert a motion field ``[...,2,H,W]`` into a ``grid_sample`` grid."""
    if flow.shape[1] != 2:
        raise ValueError(f"motion must have 2 channels (dx,dy), got {flow.shape[1]}")
    unit = (unit or "pixel").lower()
    if unit in ("pixel", "pixels", "px"):
        if align_corners:
            scale_x = 2.0 / max(width - 1, 1)
            scale_y = 2.0 / max(height - 1, 1)
        else:
            scale_x = 2.0 / width
            scale_y = 2.0 / height
        flow_n = torch.stack([flow[:, 0] * scale_x, flow[:, 1] * scale_y], dim=1)
    elif unit in ("normalized", "norm", "grid"):
        flow_n = flow
    else:
        raise ValueError(f"Unknown motion unit '{unit}'")

    semantics = (semantics or "displacement").lower()
    if semantics in ("displacement", "forward", "content"):
        sign = -1.0
    elif semantics in ("backward_flow", "sampling", "backward"):
        sign = 1.0
    else:
        raise ValueError(f"Unknown motion semantics '{semantics}'")

    flow_n = flow_n.permute(0, 2, 3, 1)                     # [B,H,W,2]
    grid = base_grid(height, width, align_corners, flow.device, flow.dtype)
    return grid + sign * flow_n


def warp(image: torch.Tensor, flow: torch.Tensor, unit: str = "pixel",
         semantics: str = "displacement", mode: str = "bilinear",
         padding_mode: str = "border", align_corners: bool = True) -> torch.Tensor:
    """Sample ``image`` according to ``flow`` (``[B,2,H,W]``)."""
    if image.ndim != 4 or flow.ndim != 4:
        raise ValueError("warp expects image [B,C,H,W] and flow [B,2,H,W]")
    if tuple(image.shape[-2:]) != tuple(flow.shape[-2:]):
        raise ValueError(
            f"warp requires matching shapes, got image {tuple(image.shape[-2:])} "
            f"and flow {tuple(flow.shape[-2:])}"
        )
    b, c, h, w = image.shape
    # Construct coordinates in FP32, before grid_sample's autocast promotion.
    # A BF16 identity grid at 128x128 already displaces some pixels; repeating
    # that warp compounds interpolation error even when motion is zero.
    dtype = torch.float64 if image.dtype == torch.float64 or flow.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=image.device.type, enabled=False):
        grid = _flow_to_grid(flow.to(dtype), h, w, unit, semantics, align_corners)
        return F.grid_sample(image.to(dtype), grid, mode=mode, padding_mode=padding_mode,
                             align_corners=align_corners)


class Advection(nn.Module):
    """Configurable advection operator (also used as the pure-advection baseline)."""

    def __init__(self, unit: str = "pixel", semantics: str = "displacement",
                 mode: str = "bilinear", padding_mode: str = "border",
                 align_corners: bool = True) -> None:
        super().__init__()
        self.unit = str(unit)
        self.semantics = str(semantics)
        self.mode = str(mode)
        self.padding_mode = str(padding_mode)
        self.align_corners = bool(align_corners)

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> "Advection":
        cfg = dict(cfg or {})
        return cls(
            unit=str(cfg.get("unit", "pixel")),
            semantics=str(cfg.get("semantics", "displacement")),
            mode=str(cfg.get("mode", "bilinear")),
            padding_mode=str(cfg.get("padding_mode", "border")),
            align_corners=bool(cfg.get("align_corners", True)),
        )

    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return warp(image, flow, self.unit, self.semantics, self.mode,
                    self.padding_mode, self.align_corners)

    def extra_repr(self) -> str:  # pragma: no cover - debug helper
        return (f"unit={self.unit}, semantics={self.semantics}, mode={self.mode}, "
                f"padding_mode={self.padding_mode}, align_corners={self.align_corners}")


def motion_magnitude(flow: torch.Tensor, unit: str = "pixel",
                     align_corners: bool = True) -> torch.Tensor:
    """Speed map in pixels per frame (used by the visualisation tool)."""
    if unit in ("normalized", "norm", "grid"):
        return torch.sqrt((flow**2).sum(dim=1, keepdim=True))
    h, w = flow.shape[-2:]
    sx = (w - 1) / 2.0 if align_corners else w / 2.0
    sy = (h - 1) / 2.0 if align_corners else h / 2.0
    return torch.sqrt((flow[:, 0:1] * sx / max(sx, 1e-6))**2 + (flow[:, 1:2] * sy / max(sy, 1e-6))**2)

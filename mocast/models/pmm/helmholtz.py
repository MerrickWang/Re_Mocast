"""Helmholtz based mean motion (FR-PMM-03 / FR-PMM-04).

``M_mean = grad(phi) + curl(A)`` with two *fixed* discrete differential operators
(central differences by default, Sobel kernels as an alternative).  Freezing the
operators is what makes the decomposition a structural prior instead of an extra
loss term (section 2.1 of the reproduction document).

Channel order is always ``(dx, dy)`` = ``(v_x, v_y)``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "spatial_gradient",
    "spatial_divergence",
    "curl_2d",
    "HelmholtzMeanMotion",
    "GradientHead",
    "make_kernels",
]


def make_kernels(kernel: str = "central", dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(k_x, k_y)`` finite-difference kernels (cross-correlation form)."""
    kernel = (kernel or "central").lower()
    if kernel in ("central", "central_diff", "cd"):
        k = torch.tensor([[0.0, 0.0, 0.0],
                          [-0.5, 0.0, 0.5],
                          [0.0, 0.0, 0.0]], dtype=dtype, device=device)
    elif kernel in ("sobel", "sobel5"):
        k = torch.tensor([[-1.0, 0.0, 1.0],
                          [-2.0, 0.0, 2.0],
                          [-1.0, 0.0, 1.0]], dtype=dtype, device=device) / 8.0
    elif kernel in ("forward", "fd"):
        k = torch.tensor([[0.0, 0.0, 0.0],
                          [0.0, -1.0, 1.0],
                          [0.0, 0.0, 0.0]], dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown derivative kernel '{kernel}'")
    k_x = k                                    # d/dx  (variation along width)
    k_y = k.transpose(0, 1).contiguous()       # d/dy  (variation along height)
    return k_x, k_y


def _pad(x: torch.Tensor, padding: str) -> torch.Tensor:
    padding = (padding or "replicate").lower()
    if padding in ("replicate", "edge", "reflect"):
        mode = "replicate" if padding in ("replicate", "edge") else "reflect"
        return F.pad(x, (1, 1, 1, 1), mode=mode)
    if padding == "zeros":
        return F.pad(x, (1, 1, 1, 1))
    if padding == "circular":
        return F.pad(x, (1, 1, 1, 1), mode="circular")
    raise ValueError(f"Unknown padding '{padding}'")


def _apply(x: torch.Tensor, kernel_2d: torch.Tensor, padding: str) -> torch.Tensor:
    b, c, h, w = x.shape
    weight = kernel_2d.to(dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    xp = _pad(x, padding)
    out = F.conv2d(xp.reshape(b * c, 1, h + 2, w + 2), weight)
    return out.reshape(b, c, h, w)


def spatial_gradient(field: torch.Tensor, kernel: str = "central",
                     padding: str = "replicate") -> torch.Tensor:
    """``[B,1,H,W] -> [B,2,H,W]`` with channel order ``(d/dx, d/dy)``."""
    if field.shape[1] != 1:
        raise ValueError(f"spatial_gradient expects a single channel, got {field.shape[1]}")
    k_x, k_y = make_kernels(kernel, field.dtype, field.device)
    dx = _apply(field, k_x, padding)
    dy = _apply(field, k_y, padding)
    return torch.cat([dx, dy], dim=1)


def spatial_divergence(vector: torch.Tensor, kernel: str = "central",
                       padding: str = "replicate") -> torch.Tensor:
    """``[B,2,H,W] -> [B,1,H,W]``: ``div(v) = d v_x / dx + d v_y / dy``."""
    if vector.shape[1] != 2:
        raise ValueError(f"spatial_divergence expects 2 channels, got {vector.shape[1]}")
    k_x, k_y = make_kernels(kernel, vector.dtype, vector.device)
    return _apply(vector[:, 0:1], k_x, padding) + _apply(vector[:, 1:2], k_y, padding)


def curl_2d(potential: torch.Tensor, kernel: str = "central",
            padding: str = "replicate") -> torch.Tensor:
    """Scalar potential -> divergence-free 2D field ``(dA/dy, -dA/dx)``."""
    if potential.shape[1] != 1:
        raise ValueError(f"curl_2d expects a single channel, got {potential.shape[1]}")
    k_x, k_y = make_kernels(kernel, potential.dtype, potential.device)
    vx = _apply(potential, k_y, padding)
    vy = -_apply(potential, k_x, padding)
    return torch.cat([vx, vy], dim=1)


class _PotentialNet(nn.Module):
    """Small convolutional network mapping the motion cue to a scalar potential."""

    def __init__(self, in_channels: int, hidden: int = 64, depth: int = 2,
                 norm: str = "group", act: str = "gelu") -> None:
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(depth):
            layers.append(nn.Conv2d(c_in, hidden, 3, padding=1))
            layers.append(nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden))
            layers.append(nn.GELU())
            c_in = hidden
        layers.append(nn.Conv2d(c_in, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HelmholtzMeanMotion(nn.Module):
    """Mean motion from scalar / vector potentials (learnable) and fixed operators.

    Args:
        in_channels: number of cue channels (``2*K*K``).
        hidden: width of the potential networks.
        kernel / padding: discrete operator configuration (recorded in the config).
        motion_scale: converts the potential gradients into pixel displacements
            (``H / h_d`` by default, i.e. the latent -> pixel factor).
        use_helmholtz: ``False`` implements ablation A1 (learnable motion head).
    """

    def __init__(self, in_channels: int, hidden: int = 64, depth: int = 2,
                 kernel: str = "central", padding: str = "replicate",
                 motion_scale: float = 2.0, use_helmholtz: bool = True,
                 norm: str = "group") -> None:
        super().__init__()
        self.kernel = kernel
        self.padding = padding
        self.motion_scale = float(motion_scale)
        self.use_helmholtz = bool(use_helmholtz)
        if self.use_helmholtz:
            self.to_phi = _PotentialNet(in_channels, hidden, depth, norm=norm)
            self.to_psi = _PotentialNet(in_channels, hidden, depth, norm=norm)
        else:
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, 2, 1),
            )

    def forward(self, cue: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``cue``: ``[B,T-1,C,h,w]`` -> dict with ``mean`` ``[B,T-1,2,h,w]``."""
        b, t = cue.shape[0], cue.shape[1]
        flat = cue.reshape(b * t, *cue.shape[2:])
        if self.use_helmholtz:
            phi = self.to_phi(flat)
            psi = self.to_psi(flat)
            v_grad = spatial_gradient(phi, self.kernel, self.padding)
            v_curl = curl_2d(psi, self.kernel, self.padding)
            mean = v_grad + v_curl
        else:
            phi = psi = None
            v_grad = v_curl = None
            mean = self.head(flat)
        mean = mean * self.motion_scale
        out = {"mean": mean.reshape(b, t, 2, *mean.shape[-2:])}
        if phi is not None:
            out["phi"] = phi.reshape(b, t, *phi.shape[-2:])
            out["psi"] = psi.reshape(b, t, *psi.shape[-2:])
            out["curl_free"] = (v_grad * self.motion_scale).reshape(b, t, 2, *mean.shape[-2:])
            out["divergence_free"] = (v_curl * self.motion_scale).reshape(b, t, 2, *mean.shape[-2:])
        return out


class GradientHead(nn.Module):
    """Ablation A3: single learnable motion head, no Reynolds decomposition."""

    def __init__(self, in_channels: int, hidden: int = 64, out_scale: float = 2.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 2, 1),
        )
        self.out_scale = float(out_scale)

    def forward(self, cue: torch.Tensor) -> torch.Tensor:
        b, t = cue.shape[0], cue.shape[1]
        return self.net(cue.reshape(b * t, *cue.shape[2:])).reshape(
            b, t, 2, *cue.shape[-2:]) * self.out_scale

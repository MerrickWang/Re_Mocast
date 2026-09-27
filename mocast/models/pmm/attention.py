"""Local block attention producing patch level motion cues (FR-PMM-01, FR-PMM-02).

Equation (5)-(6) of the paper::

    alpha^{t->t+1}_i = softmax_j( Q^t_i (K^{t+1}_{N(i)})^T / scale )
    D~^{t->t+1}_i   = sum_j alpha_ij (D_{N(i),j} - D_i)

``D`` are the *normalised relative coordinates* of the patch centres, so the cue
is a set of ``2*K*K`` displacement channels (one 2-vector per neighbour in the
``K x K`` search window).  The cue stays piecewise constant inside a patch and is
broadcast back to the latent grid; the potentials (and the gradient/curl
operators) afterwards act as the smoothness prior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LocalPatchAttention"]


class LocalPatchAttention(nn.Module):
    """Patch-wise spatiotemporal attention restricted to a local search window.

    ``patch_stride`` defaults to ``patch_size`` (non-overlapping tiling, as in the
    paper).  A smaller stride enables overlapped patches; the per-patch cues are
    then scattered back with coverage-normalised folding.
    """

    def __init__(self, dim: int, patch_size: int = 4, window: int = 3, heads: int = 4,
                 attn_dim: int = 64, scale: str = "sqrt", dropout: float = 0.0,
                 coord_range: float = 1.0, patch_stride: Optional[int] = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride) if patch_stride else int(patch_size)
        if self.patch_stride > self.patch_size:
            raise ValueError("patch_stride must be <= patch_size (overlap or exact tiling)")
        self.window = int(window)
        self.heads = max(int(heads), 1)
        self.attn_dim = int(attn_dim)
        self.scale_mode = str(scale).lower()
        self.dropout = float(dropout)
        self.coord_range = float(coord_range)

        embed = self.dim * self.patch_size * self.patch_size
        self.query = nn.Linear(embed, self.attn_dim * self.heads)
        self.key = nn.Linear(embed, self.attn_dim * self.heads)
        self._cue_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    # ------------------------------------------------------------------ utils
    def _relative_coords(self, grid_h: int, grid_w: int, device: torch.device,
                         dtype: torch.dtype) -> torch.Tensor:
        """``[2, K*K, L]`` relative neighbour coordinates (``dx``/``dy`` per patch centre)."""
        key = (grid_h, grid_w)
        cached = self._cue_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        k = self.window
        r = k // 2

        def norm(idx: torch.Tensor, size: int) -> torch.Tensor:
            if size <= 1:
                return torch.zeros_like(idx, dtype=torch.float32)
            return 2.0 * idx.to(torch.float32) / (size - 1) - 1.0

        centre_y = torch.arange(grid_h, dtype=torch.float32)
        centre_x = torch.arange(grid_w, dtype=torch.float32)
        cy = norm(centre_y, grid_h)[:, None].expand(grid_h, grid_w)
        cx = norm(centre_x, grid_w)[None, :].expand(grid_h, grid_w)
        dx_list, dy_list = [], []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny = torch.clamp(centre_y + dy, 0, grid_h - 1)
                nx = torch.clamp(centre_x + dx, 0, grid_w - 1)
                ny_n = norm(ny, grid_h)[:, None].expand(grid_h, grid_w)
                nx_n = norm(nx, grid_w)[None, :].expand(grid_h, grid_w)
                dx_list.append((nx_n - cx).reshape(-1))
                dy_list.append((ny_n - cy).reshape(-1))
        coords = torch.stack([torch.stack(dx_list), torch.stack(dy_list)], dim=0)  # [2, K*K, L]
        coords = coords * self.coord_range
        coords = coords.to(device=device, dtype=dtype)
        self._cue_cache[key] = coords
        return coords

    # ---------------------------------------------------------------- forward
    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``h``: ``[B,T,d,h_d,w_d]`` -> motion cue ``[B,T-1,2*K*K,h_d,w_d]``."""
        if h.ndim != 5:
            raise ValueError(f"LocalPatchAttention expects [B,T,d,H,W], got {tuple(h.shape)}")
        b, t, c, hh, ww = h.shape
        p, k, stride = self.patch_size, self.window, self.patch_stride
        if (hh - p) % stride or (ww - p) % stride or hh < p or ww < p:
            pad_h = (-(hh - p)) % stride
            pad_w = (-(ww - p)) % stride
            h = F.pad(h, (0, pad_w, 0, pad_h), mode="replicate")
        _, _, _, hh_p, ww_p = h.shape
        grid_h, grid_w = (hh_p - p) // stride + 1, (ww_p - p) // stride + 1
        n_patches = grid_h * grid_w

        flat = h.reshape(b * t, c, hh_p, ww_p)
        patches = F.unfold(flat, kernel_size=p, stride=stride)     # [B*T, c*p*p, L]
        patches = patches.transpose(1, 2)                          # [B*T, L, c*p*p]
        queries = self.query(patches).reshape(b, t, n_patches, self.heads, self.attn_dim)

        # neighbour patches inside the K x K window at t+1
        grid = patches.reshape(b, t, grid_h, grid_w, c * p * p).permute(0, 1, 4, 2, 3)
        grid = grid.reshape(b * t, c * p * p, grid_h, grid_w)
        neighbours = F.unfold(grid, kernel_size=k, padding=k // 2, stride=1)  # [B*T, C*K*K, L]
        neighbours = neighbours.transpose(1, 2).reshape(b, t, n_patches, k * k, c * p * p)
        keys = self.key(neighbours).reshape(b, t, n_patches, k * k, self.heads, self.attn_dim)
        keys = keys.permute(0, 1, 2, 4, 3, 5)                      # [b,t,L,heads,K*K,de]

        q = queries[:, : t - 1]                                    # [b,T-1,L,heads,de]
        kk = keys[:, 1:]                                           # [b,T-1,L,heads,K*K,de]
        logits = torch.einsum("btlhd,btlhjd->btlhj", q, kk)
        if self.scale_mode in ("sqrt", "sqrt_d", "sqrtd"):
            logits = logits / (self.attn_dim ** 0.5)
        elif self.scale_mode in ("d", "dim"):
            logits = logits / float(self.attn_dim)
        elif self.scale_mode in ("none", "1"):
            pass
        else:
            raise ValueError(f"Unknown attention scale '{self.scale_mode}'")
        attn = torch.softmax(logits, dim=-1)                       # [b,T-1,L,heads,K*K]
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, self.dropout)
        attn_mean = attn.mean(dim=3)                               # head reduction (documented)

        coords = self._relative_coords(grid_h, grid_w, h.device, h.dtype)  # [2,K*K,L]
        # cue[d, j, l] = alpha[l, j] * relcoords[d, j, l]
        weighted = attn_mean[:, :, None, :, :] * coords.permute(0, 2, 1)[None, None]
        cue_patches = weighted.permute(0, 1, 2, 4, 3).reshape(
            b * (t - 1), -1, n_patches)                            # [B*(T-1), 2*K*K, L]

        # scatter the patch-level cues back onto the (possibly overlapping) latent grid
        expanded = cue_patches.repeat_interleave(p * p, dim=1)
        cue = F.fold(expanded, output_size=(hh_p, ww_p), kernel_size=p, stride=stride)
        coverage = F.fold(torch.ones_like(cue_patches[:, :1]).repeat_interleave(p * p, dim=1),
                          output_size=(hh_p, ww_p), kernel_size=p, stride=stride)
        cue = cue / coverage.clamp_min(1.0)
        cue = cue.reshape(b, t - 1, -1, hh_p, ww_p)
        cue = cue[..., :hh, :ww]
        return {
            "cue": cue.contiguous(),
            "attention": attn_mean.reshape(b, t - 1, grid_h, grid_w, k * k),
            "grid": (grid_h, grid_w),
        }

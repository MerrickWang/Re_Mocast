"""Temporal embedding and prediction (FR-TEMP-01 / FR-PRED-01).

Following the paper, each branch gets its own embedding network (bottleneck 1x1
convolution followed by group convolutions) and its own predictor.  The predictor
is implemented as two sequential CNN blocks: the first maps the temporal
dimension to the prediction horizon, the second maps the latent spatial grid to
the target resolution.

``mode="autoregressive"`` implements the alternative listed as *待确认* in the
specification (one step at a time with a rolling window of embeddings).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, GroupConvBlock, build_act, build_norm, resize_tensor

__all__ = ["TemporalEmbedding", "TemporalPredictor"]


class TemporalEmbedding(nn.Module):
    """Bottleneck temporal embedding: 1x1 conv + group convolutions per frame."""

    def __init__(self, in_channels: int, embed_dim: int, num_blocks: int = 2,
                 kernel_size: int = 3, norm: str = "group", act: str = "gelu") -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, 1)
        self.blocks = nn.ModuleList([
            GroupConvBlock(embed_dim, kernel_size=kernel_size, norm=norm, act=act)
            for _ in range(max(int(num_blocks), 1))
        ])
        self.norm = build_norm(norm, embed_dim)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B,T,C,h,w]`` -> ``[B,T,embed_dim,h,w]``."""
        if x.ndim != 5:
            raise ValueError(f"TemporalEmbedding expects [B,T,C,H,W], got {tuple(x.shape)}")
        b, t = x.shape[0], x.shape[1]
        flat = self.proj(x.reshape(b * t, *x.shape[2:]))
        for block in self.blocks:
            flat = block(flat)
        flat = self.act(self.norm(flat))
        return flat.reshape(b, t, *flat.shape[1:])


class TemporalPredictor(nn.Module):
    """Extrapolate ``in_steps`` embeddings to ``out_steps`` steps at pixel resolution."""

    def __init__(self, embed_dim: int, in_steps: int, out_steps: int, out_channels: int,
                 latent_size: Optional[Sequence[int]] = None,
                 target_size: Optional[Sequence[int]] = None, hidden: Optional[int] = None,
                 spatial_blocks: int = 2, mode: str = "oneshot", norm: str = "group",
                 act: str = "gelu", align_corners: bool = False) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.in_steps = int(in_steps)
        self.out_steps = int(out_steps)
        self.out_channels = int(out_channels)
        self.mode = str(mode).lower()
        self.align_corners = bool(align_corners)

        if self.mode == "oneshot":
            self.temporal_mix = nn.Conv2d(self.embed_dim * self.in_steps,
                                          self.embed_dim * self.out_steps, 1)
        elif self.mode == "autoregressive":
            self.window = max(self.in_steps, 2)
            self.temporal_mix = nn.Conv2d(self.embed_dim * self.window, self.embed_dim, 1)
        else:
            raise ValueError(f"Unknown prediction mode '{self.mode}'")

        self.spatial_blocks = nn.ModuleList([
            GroupConvBlock(self.embed_dim, 3, norm=norm, act=act)
            for _ in range(max(int(spatial_blocks), 1))
        ])
        self.levels = 0
        if latent_size is not None and target_size is not None:
            ratio_h = int(target_size[0]) // int(latent_size[0])
            ratio_w = int(target_size[1]) // int(latent_size[1])
            self.levels = max(int(round(torch.log2(torch.tensor(float(max(ratio_h, 1)))).item())), 0)
            self.upsample = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1), build_norm(norm, self.embed_dim),
                    build_act(act),
                )
                for _ in range(self.levels)
            ])
        else:
            self.upsample = nn.ModuleList()
        self.head = nn.Conv2d(self.embed_dim, self.out_channels, 1)

    # ------------------------------------------------------------------ utils
    def _decode(self, features: torch.Tensor) -> torch.Tensor:
        """``[B*P, embed, h, w] -> [B*P, out_channels, H, W]``."""
        x = features
        for upsample in self.upsample:
            x = resize_tensor(x, (x.shape[-2] * 2, x.shape[-1] * 2), mode="bilinear",
                              align_corners=self.align_corners)
            x = upsample(x)
        return self.head(x)

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.spatial_blocks:
            x = block(x)
        return x

    # ---------------------------------------------------------------- forward
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """``emb``: ``[B,T_in,embed,h,w]`` -> ``[B,P,out_channels,H,W]``."""
        if emb.ndim != 5:
            raise ValueError(f"TemporalPredictor expects [B,T,C,H,W], got {tuple(emb.shape)}")
        b, t, c, h, w = emb.shape
        if self.mode == "oneshot":
            # block 1: temporal dimension -> prediction horizon
            x = self.temporal_mix(emb.transpose(1, 2).reshape(b, t * c, h, w))
            x = x.reshape(b * self.out_steps, self.embed_dim, h, w)
            # block 2: spatial dimension -> target resolution
            decoded = self._decode(self._spatial(x))
            return decoded.reshape(b, self.out_steps, self.out_channels, *decoded.shape[-2:])

        # rolling autoregressive variant
        window: List[torch.Tensor] = [emb[:, i] for i in range(t)]
        while len(window) < self.window:
            window.insert(0, window[0])
        outputs: List[torch.Tensor] = []
        for _ in range(self.out_steps):
            stacked = torch.cat(window[-self.window:], dim=1)
            mixed = self.temporal_mix(stacked)
            mixed = self._spatial(mixed)
            decoded = self._decode(mixed)
            outputs.append(decoded)
            window.append(mixed)
        stacked_out = torch.stack(outputs, dim=1)
        return stacked_out

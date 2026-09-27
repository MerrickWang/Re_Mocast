"""MoCast+ : residual diffusion post-processing (FR-DIFF-01, P8/stage two).

Following the hybrid paradigm of DiffCast, MoCast+ keeps the deterministic
backbone ``mu = MoCast(X)`` and models the *residual* ``r = Y - mu`` with a
Gaussian diffusion process (DDPM) whose denoiser is a UNet augmented with
temporal attention and a ConvGRU.

Everything uncertainty related stays configurable and decoupled from the
deterministic stage (risk R9): the deterministic results are always available
through ``forward()["base_pred"]``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, build_act, build_norm
from .mocast import MoCast
from ..utils.config import Config

__all__ = ["MoCastPlus", "GaussianDiffusion", "ResidualUNet", "ConvGRUCell", "TemporalAttention"]


# --------------------------------------------------------------------------- #
#  building blocks
# --------------------------------------------------------------------------- #
class ConvGRUCell(nn.Module):
    """Convolutional GRU used to propagate context over the lead time."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.reset = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.update = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size,
                                   padding=padding)
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_channels, *x.shape[-2:],
                                 dtype=x.dtype, device=x.device)
        combined = torch.cat([x, hidden], dim=1)
        r = torch.sigmoid(self.reset(combined))
        z = torch.sigmoid(self.update(combined))
        candidate = torch.tanh(self.candidate(torch.cat([x, r * hidden], dim=1)))
        return (1 - z) * hidden + z * candidate


class TemporalAttention(nn.Module):
    """Multi-head attention across the prediction horizon at low resolution."""

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.channels = channels
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B,P,C,h,w]`` -> same shape (attention across the P axis)."""
        return self.forward_grouped(x)

    def forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B,P,C,h,w]`` -> same shape."""
        if x.ndim != 5:
            raise ValueError(f"TemporalAttention expects [B,P,C,h,w], got {tuple(x.shape)}")
        b, p, c, h, w = x.shape
        tokens = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, p, c)
        attended, _ = self.attn(tokens, tokens, tokens)
        attended = self.norm(tokens + attended)
        return attended.reshape(b, h, w, p, c).permute(0, 3, 4, 1, 2).contiguous()


class _FiLMBlock(nn.Module):
    """Residual conv block with diffusion-step conditioning."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 norm: str = "group", act: str = "gelu") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = build_norm(norm, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = build_norm(norm, out_channels)
        self.act = build_act(act)
        self.cond = nn.Linear(cond_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        scale, shift = self.cond(cond).chunk(2, dim=1)
        h = self.norm2(self.conv2(h)) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.act(h) + self.skip(x)


class ResidualUNet(nn.Module):
    """UNet denoiser over ``[B,P,1,H,W]`` residuals conditioned on the base forecast."""

    def __init__(self, in_channels: int = 1, cond_frames: int = 5, base_channels: int = 64,
                 channel_mults: Sequence[int] = (1, 2, 4), num_blocks: int = 2,
                 cond_dim: int = 128, temporal_attention: bool = True, conv_gru: bool = True,
                 num_heads: int = 4, norm: str = "group", act: str = "gelu") -> None:
        super().__init__()
        self.cond_frames = int(cond_frames)
        self.base_channels = int(base_channels)
        self.cond_dim = int(cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, cond_dim), build_act(act),
            nn.Linear(cond_dim, cond_dim),
        )
        stem_in = in_channels * 4  # noisy residual + base forecast + mask + context frame
        self.stem = nn.Conv2d(stem_in, base_channels, 3, padding=1)

        widths = [base_channels * int(m) for m in channel_mults]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        c_prev = base_channels
        for i, width in enumerate(widths):
            blocks = nn.ModuleList([
                _FiLMBlock(c_prev if j == 0 else width, width, cond_dim, norm=norm, act=act)
                for j in range(max(int(num_blocks), 1))
            ])
            self.downs.append(blocks)
            c_prev = width
        self.gru = ConvGRUCell(c_prev, c_prev) if conv_gru else None
        self.temporal_attention = TemporalAttention(c_prev, num_heads) if temporal_attention else None
        for i, width in enumerate(reversed(widths)):
            in_channels_up = c_prev + width
            blocks = nn.ModuleList([
                _FiLMBlock(in_channels_up if j == 0 else width, width, cond_dim, norm=norm, act=act)
                for j in range(max(int(num_blocks), 1))
            ])
            self.ups.append(blocks)
            c_prev = width
        self.out = nn.Conv2d(c_prev, in_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # ------------------------------------------------------------------ utils
    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.base_channels // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device,
                                                             dtype=torch.float32) / half)
        args = timesteps.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if embedding.shape[1] < self.base_channels:
            embedding = F.pad(embedding, (0, self.base_channels - embedding.shape[1]))
        return self.time_mlp(embedding)

    # ---------------------------------------------------------------- forward
    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, base_pred: torch.Tensor,
                context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``noisy``/``base_pred``: ``[B,P,1,H,W]``, ``context``: ``[B,T,1,H,W]``."""
        b, p, c, h, w = noisy.shape
        cond = self._time_embedding(timesteps)                       # [B,cond_dim]
        if mask is None:
            mask = torch.ones_like(noisy)
        context = torch.cat([context, context[:, -1:].repeat(1, p - context.shape[1], 1, 1, 1)], dim=1) \
            if context.shape[1] < p else context[:, :p]
        x = torch.cat([noisy, base_pred, mask, context], dim=2)      # [B,P,C',H,W]
        x = x.reshape(b * p, -1, h, w)
        cond_bp = cond.repeat_interleave(p, dim=0)
        h1 = self.stem(x)
        skips: List[torch.Tensor] = []
        for blocks in self.downs:
            for block in blocks:
                h1 = block(h1, cond_bp)
            skips.append(h1)
            h1 = F.avg_pool2d(h1, 2)
        if self.gru is not None:  # temporal context, ConvGRU over the P axis
            seq = h1.reshape(b, p, *h1.shape[1:]).permute(0, 2, 1, 3, 4)
            states = []
            hidden = None
            for i in range(p):
                hidden = self.gru(seq[:, :, i], hidden)
                states.append(hidden)
            h1 = torch.stack(states, dim=2).permute(0, 2, 1, 3, 4).reshape(b * p, *h1.shape[1:])
        if self.temporal_attention is not None:
            h1 = self.temporal_attention.forward_grouped(
                h1.reshape(b, p, *h1.shape[1:])).reshape(b * p, *h1.shape[1:])
        for i, blocks in enumerate(self.ups):
            h1 = F.interpolate(h1, scale_factor=2.0, mode="nearest")
            skip = skips[len(skips) - 1 - i]
            h1 = torch.cat([h1, skip], dim=1)
            for block in blocks:
                h1 = block(h1, cond_bp)
        out = self.out(h1)
        return out.reshape(b, p, c, h, w)


class GaussianDiffusion(nn.Module):
    """DDPM / DDIM process over residual frames (``r = Y - mu``)."""

    def __init__(self, num_steps: int = 1000, schedule: str = "linear", beta_start: float = 1e-4,
                 beta_end: float = 2e-2, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.schedule = str(schedule).lower()
        self.residual_scale = float(residual_scale)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        betas = self._make_betas()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())

    def _make_betas(self) -> torch.Tensor:
        if self.schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_steps)
        if self.schedule == "cosine":
            steps = torch.arange(self.num_steps + 1, dtype=torch.float32) / self.num_steps
            f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return betas.clamp(1e-8, 0.999)
        raise ValueError(f"Unknown diffusion schedule '{self.schedule}'")

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        return sqrt_acp * x0 + sqrt_omacp * noise

    def training_loss(self, model: nn.Module, residual: torch.Tensor, base_pred: torch.Tensor,
                      context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = residual.shape[0]
        t = torch.randint(0, self.num_steps, (b,), device=residual.device)
        noise = torch.randn_like(residual)
        noisy = self.q_sample(residual / self.residual_scale, t, noise)
        predicted = model(noisy, t, base_pred / self.residual_scale, context, mask)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, base_pred: torch.Tensor, context: torch.Tensor,
               steps: Optional[int] = None, eta: float = 0.0,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """DDIM (``eta=0``) or DDPM (``eta=1``) sampling on top of ``base_pred``."""
        b, p, c, h, w = base_pred.shape
        device = base_pred.device
        use_steps = int(steps or self.num_steps)
        use_steps = max(1, min(use_steps, self.num_steps))
        indices = torch.linspace(0, self.num_steps - 1, use_steps, device=device).round().long()
        indices = torch.flip(indices, dims=[0])
        x = torch.randn(b, p, c, h, w, device=device)
        base_scaled = base_pred / self.residual_scale
        for i, t_cur in enumerate(indices):
            t_batch = t_cur.expand(b).long()
            eps = model(x, t_batch, base_scaled, context, mask)
            acp = self.alphas_cumprod[t_cur]
            x0 = (x - (1 - acp).sqrt() * eps) / acp.sqrt()
            if i == len(indices) - 1:
                x = x0
                break
            next_t = indices[i + 1]
            acp_next = self.alphas_cumprod[next_t]
            sigma = eta * ((1 - acp_next) / (1 - acp) * (1 - acp / acp_next)).sqrt()
            direction = (1 - acp_next - sigma**2).clamp_min(0).sqrt() * eps
            x = acp_next.sqrt() * x0 + direction
            if sigma > 0:
                x = x + sigma * torch.randn_like(x)
        return x * self.residual_scale


class MoCastPlus(nn.Module):
    """Deterministic MoCast backbone + residual diffusion (Eq. 12 + FR-DIFF-01)."""

    def __init__(self, backbone_cfg: Dict[str, Any], diffusion_cfg: Optional[Dict[str, Any]] = None,
                 unet_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.backbone = MoCast(backbone_cfg)
        diff = Config(diffusion_cfg or {})
        unet = Config(unet_cfg or {})
        self.diffusion = GaussianDiffusion(
            num_steps=int(diff.get("num_steps", 1000)),
            schedule=str(diff.get("schedule", "linear")),
            residual_scale=float(diff.get("residual_scale", 1.0)),
        )
        self.denoiser = ResidualUNet(
            in_channels=1,
            cond_frames=int(self.backbone.input_len),
            base_channels=int(unet.get("base_channels", 64)),
            channel_mults=unet.get("channel_mults", (1, 2, 4)),
            num_blocks=int(unet.get("num_blocks", 2)),
            cond_dim=int(unet.get("cond_dim", 128)),
            temporal_attention=bool(unet.get("temporal_attention", True)),
            conv_gru=bool(unet.get("conv_gru", True)),
            num_heads=int(unet.get("num_heads", 4)),
        )
        self.warmup_backbone = bool(diff.get("warmup_backbone", False))

    # ------------------------------------------------------------------ api
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        out = self.backbone(x, y=y)
        result = {"base_pred": out["pred"], "backbone": out}
        if y is not None:
            residual = y[:, self.backbone.input_len:] - out["pred"]
            result["diffusion_loss"] = self.diffusion.training_loss(
                self.denoiser, residual, out["pred"], x)
        return result

    @torch.no_grad()
    def sample(self, x: torch.Tensor, num_samples: int = 1, steps: Optional[int] = None,
               eta: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(base_pred [B,P,1,H,W], samples [B,N,P,1,H,W])``."""
        out = self.backbone(x)

        def expand(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.repeat_interleave(num_samples, dim=0)

        base = expand(out["pred"])
        context = expand(x)
        sampled = self.diffusion.sample(self.denoiser, base, context, steps=steps, eta=eta)
        predictions = base + sampled
        b = out["pred"].shape[0]
        predictions = predictions.reshape(b, num_samples, *predictions.shape[1:])
        return out["pred"], predictions

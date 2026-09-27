"""Wavelet based fluctuating motion (FR-PMM-05 / FR-PMM-06 / FR-PMM-07).

The discrete wavelet transform is implemented directly with ``conv2d`` /
``conv_transpose2d`` on fixed orthonormal filter banks: the synthesis is the
*exact adjoint* of the analysis (circular boundary handling), which makes
``UT-01`` (reconstruction error below tolerance) a machine-precision property
instead of a numerical accident, and keeps the transform fully on-device with no
extra dependency.

Enhancement follows Eq. (8)::

    H'^t_h = sigmoid(f_h(grad X^t)) * H^t_h
    H^{t->t+1}_h = g_h( H'^t_h || H'^{t+1}_h )

and the fluctuation is ``M_f = f_f(IDWT(LL*, H_h, H_v, H_d))``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import build_act, build_norm
from .helmholtz import spatial_gradient

__all__ = [
    "wavelet_filters",
    "filters_2d",
    "dwt2",
    "idwt2",
    "WaveletFluctuationBlock",
    "DWT2D",
    "IDWT2D",
]

_MATH_SQRT2 = 2.0 ** 0.5


def wavelet_filters(basis: str = "haar") -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(low_pass, high_pass)`` analysis filters (orthonormal)."""
    basis = (basis or "haar").lower()
    if basis == "haar":
        s = 1.0 / _MATH_SQRT2
        low = torch.tensor([s, s])
        high = torch.tensor([s, -s])
    elif basis in ("db2", "daubechies2"):
        s3 = 3.0 ** 0.5
        denom = 4.0 * _MATH_SQRT2
        low = torch.tensor([(1 + s3) / denom, (3 + s3) / denom,
                            (3 - s3) / denom, (1 - s3) / denom])
        high = torch.tensor([low[3], -low[2], low[1], -low[0]])
    else:
        raise ValueError(f"Unsupported wavelet basis '{basis}' (use 'haar' or 'db2')")
    return low, high


def filters_2d(basis: str = "haar") -> torch.Tensor:
    """``[4,1,F,F]`` 2D filters ordered ``(LL, H_h, H_v, H_d)``.

    ``H_h`` is high-pass along the *width* (horizontal variation), ``H_v`` is
    high-pass along the *height*, ``H_d`` suppresses both directions.
    """
    low, high = wavelet_filters(basis)
    def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.outer(a, b)

    filters = torch.stack([
        outer(low, low),    # LL
        outer(low, high),   # H_h : high-pass along width
        outer(high, low),   # H_v : high-pass along height
        outer(high, high),  # H_d : diagonal
    ])
    return filters[:, None, :, :]


def _analysis_pad(size: int, filter_len: int) -> int:
    return filter_len - 2 + (size % 2)


def dwt2(x: torch.Tensor, filters: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """2D analysis. ``x``: ``[B,C,H,W]`` -> ``(LL, H_h, H_v, H_d)`` at half size."""
    b, c, h, w = x.shape
    f = filters.shape[-1]
    ph, pw = _analysis_pad(h, f), _analysis_pad(w, f)
    xp = F.pad(x, (0, pw, 0, ph), mode="circular")
    weight = filters.to(dtype=x.dtype, device=x.device)
    out = F.conv2d(xp.reshape(b * c, 1, h + ph, w + pw), weight, stride=2)
    out = out.reshape(b, c, 4, out.shape[-2], out.shape[-1])
    return tuple(out[:, :, i] for i in range(4))  # type: ignore[return-value]


def idwt2(ll: torch.Tensor, h_h: torch.Tensor, h_v: torch.Tensor, h_d: torch.Tensor,
          filters: torch.Tensor, out_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """2D synthesis (exact adjoint of :func:`dwt2`)."""
    b, c, hs, ws = ll.shape
    f = filters.shape[-1]
    stacked = torch.stack([ll, h_h, h_v, h_d], dim=2)            # [B,C,4,hs,ws]
    sub = stacked.reshape(b * c, 4, hs, ws)
    weight = filters.to(dtype=ll.dtype, device=ll.device)
    y = F.conv_transpose2d(sub, weight, stride=2)                # [B*C,1,H+ph,W+pw]
    y = y.reshape(b, c, y.shape[-2], y.shape[-1])
    out_h = out_size[0] if out_size else hs * 2
    out_w = out_size[1] if out_size else ws * 2
    ph, pw = _analysis_pad(out_h, f), _analysis_pad(out_w, f)
    if ph:
        y = y.clone()
        y[..., :ph, :] += y[..., out_h:out_h + ph, :]
    if pw:
        y = y.clone()
        y[..., :, :pw] += y[..., :, out_w:out_w + pw]
    return y[..., :out_h, :out_w]


class DWT2D(nn.Module):
    """Module wrapper around :func:`dwt2` (filter bank as a buffer)."""

    def __init__(self, basis: str = "haar") -> None:
        super().__init__()
        self.basis = str(basis).lower()
        self.register_buffer("filters", filters_2d(self.basis), persistent=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return dwt2(x, self.filters)


class IDWT2D(nn.Module):
    """Module wrapper around :func:`idwt2`."""

    def __init__(self, basis: str = "haar") -> None:
        super().__init__()
        self.basis = str(basis).lower()
        self.register_buffer("filters", filters_2d(self.basis), persistent=True)

    def forward(self, ll: torch.Tensor, h_h: torch.Tensor, h_v: torch.Tensor,
                h_d: torch.Tensor, out_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        return idwt2(ll, h_h, h_v, h_d, self.filters, out_size)


def _match_size(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    if x.shape[-2] % size[0] == 0 and x.shape[-1] % size[1] == 0:
        k = (x.shape[-2] // size[0], x.shape[-1] // size[1])
        return F.avg_pool2d(x, kernel_size=k, stride=k)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class _DirectionalEnhancer(nn.Module):
    """``sigmoid`` gated re-weighting network for one wavelet direction."""

    def __init__(self, in_channels: int = 2, hidden: int = 32, mode: str = "sigmoid") -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, grad: torch.Tensor) -> torch.Tensor:
        raw = self.net(grad)
        if self.mode in ("sigmoid", "gate"):
            return torch.sigmoid(raw)
        if self.mode in ("softplus", "sp"):
            return F.softplus(raw) / 4.0
        if self.mode in ("residual", "one_plus"):
            return 1.0 + torch.tanh(raw)
        raise ValueError(f"Unknown enhancement mode '{self.mode}'")


class WaveletFluctuationBlock(nn.Module):
    """Fluctuating motion component (Eq. 8-9 of the paper).

    Args:
        dim: latent feature channels produced by the encoder.
        hidden: width of the mixing / decoding convs.
        basis: ``haar`` (default) or ``db2`` (ablation A7).
        ll_mode: ``zero`` (default, keeps only fluctuations), ``keep`` or
            ``learnable`` - the open question #4 in the specification.
        enhance: ``sigmoid`` / ``softplus`` / ``residual`` weighting.
        grad_kernel: derivative kernel used for the enhancement gradients.
    """

    def __init__(self, dim: int, hidden: int = 64, basis: str = "haar", ll_mode: str = "zero",
                 enhance: str = "sigmoid", grad_kernel: str = "central",
                 padding: str = "replicate", motion_scale: float = 2.0, act: str = "gelu",
                 norm: str = "group") -> None:
        super().__init__()
        self.basis = str(basis).lower()
        self.ll_mode = str(ll_mode).lower()
        self.grad_kernel = str(grad_kernel).lower()
        self.padding = str(padding).lower()
        self.motion_scale = float(motion_scale)
        self.dim = int(dim)

        self.register_buffer("filters", filters_2d(self.basis), persistent=True)
        self.enhance_h = _DirectionalEnhancer(2, hidden, enhance)
        self.enhance_v = _DirectionalEnhancer(2, hidden, enhance)
        self.enhance_d = _DirectionalEnhancer(2, hidden, enhance)
        self.mix_h = nn.Conv2d(2 * dim, dim, 3, padding=1)
        self.mix_v = nn.Conv2d(2 * dim, dim, 3, padding=1)
        self.mix_d = nn.Conv2d(2 * dim, dim, 3, padding=1)
        self.mix_ll = nn.Conv2d(2 * dim, dim, 3, padding=1) if self.ll_mode == "learnable" else None
        self.norm = build_norm(norm, dim)
        self.act = build_act(act)
        self.to_motion = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1), build_norm(norm, hidden), build_act(act),
            nn.Conv2d(hidden, 2, 1),
        )

    # ------------------------------------------------------------------ utils
    def _enhanced_subbands(self, h: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per-frame enhanced high-frequency subbands (``[B,d,hs,ws]`` each)."""
        b, t = h.shape[0], h.shape[1]
        ll, hh, hv, hd = dwt2(h.reshape(b * t, *h.shape[2:]), self.filters)
        grad = spatial_gradient(x.reshape(b * t, *x.shape[2:])[:, :1],
                                self.grad_kernel, self.padding)
        grad = _match_size(grad, (hh.shape[-2], hh.shape[-1]))
        w_h = self.enhance_h(grad)
        w_v = self.enhance_v(grad)
        w_d = self.enhance_d(grad)
        return {"ll": ll, "h": w_h * hh, "v": w_v * hv, "d": w_d * hd}

    # ---------------------------------------------------------------- forward
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``h``: latent ``[B,T,d,h_d,w_d]``, ``x``: frames ``[B,T,1,H,W]``."""
        if h.ndim != 5 or x.ndim != 5:
            raise ValueError("WaveletFluctuationBlock expects [B,T,...] tensors")
        b, t, _, hh_, ww_ = h.shape
        sub = self._enhanced_subbands(h, x)                                # [B*T,d,hs,ws]
        def pair(key: str) -> torch.Tensor:
            first = sub[key].reshape(b, t, *sub[key].shape[1:])[:, : t - 1]
            second = sub[key].reshape(b, t, *sub[key].shape[1:])[:, 1:]
            return torch.cat([first, second], dim=2)                       # [B,T-1,2d,hs,ws]

        n = b * (t - 1)
        mixed_h = self.act(self.norm(self.mix_h(pair("h").reshape(n, -1, *sub["h"].shape[-2:]))))
        mixed_v = self.act(self.norm(self.mix_v(pair("v").reshape(n, -1, *sub["v"].shape[-2:]))))
        mixed_d = self.act(self.norm(self.mix_d(pair("d").reshape(n, -1, *sub["d"].shape[-2:]))))

        if self.ll_mode == "zero":
            ll_pair = torch.zeros_like(mixed_h)
        elif self.ll_mode == "keep":
            ll_pair = pair("ll").reshape(n, -1, *sub["ll"].shape[-2:])
            ll_pair = 0.5 * (ll_pair[:, : self.dim] + ll_pair[:, self.dim:])
        elif self.ll_mode == "learnable":
            assert self.mix_ll is not None
            ll_pair = self.act(self.norm(self.mix_ll(pair("ll").reshape(n, -1, *sub["ll"].shape[-2:]))))
        else:
            raise ValueError(f"Unknown ll_mode '{self.ll_mode}'")

        fluctuation = idwt2(ll_pair, mixed_h, mixed_v, mixed_d, self.filters,
                            out_size=(hh_, ww_))                            # [B*(T-1),d,h,w]
        motion = self.to_motion(fluctuation) * self.motion_scale           # [B*(T-1),2,h,w]
        out = {
            "fluctuation": fluctuation.reshape(b, t - 1, *fluctuation.shape[1:]).contiguous()
            if fluctuation.ndim == 4 else fluctuation,
            "motion": motion.reshape(b, t - 1, 2, *motion.shape[-2:]).contiguous(),
        }
        return out

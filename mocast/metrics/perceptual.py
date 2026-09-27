"""Perceptual metrics: SSIM and LPIPS (paper: "Learned perceptual image patch similarity").

LPIPS uses the official ``lpips`` package when installed; otherwise the module
falls back to a VGG-16 feature distance with uniform layer weights, and reports
which backend was used so that numbers stay traceable.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["ssim", "SSIMMetric", "LPIPS", "PerceptualMetrics"]


def _to_tensor(array: Any) -> "Any":
    import torch

    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(np.ascontiguousarray(array))


def _gaussian_window(window_size: int, sigma: float, device: Any, dtype: Any) -> "Any":
    import torch

    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.view(1, 1, window_size, window_size)


def _fold_leading(array: "Any") -> "Any":
    """Collapse ``[B,P,C,H,W]`` (and friends) into ``[N,C,H,W]`` for image metrics."""
    import torch

    tensor = _to_tensor(array).float()
    if tensor.ndim == 2:
        return tensor[None, None]
    if tensor.ndim == 3:
        return tensor[:, None]
    if tensor.ndim > 4:
        return tensor.reshape(-1, *tensor.shape[-3:])
    return tensor


def ssim(pred: Any, target: Any, data_range: float = 1.0, window_size: int = 11,
         sigma: float = 1.5) -> float:
    """Mean SSIM over every frame of the input tensors (``[..., H, W]`` or ``[N,C,H,W]``)."""
    import torch
    import torch.nn.functional as F

    pred = _fold_leading(pred)
    target = _fold_leading(target)
    if pred.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
    b, c, h, w = pred.shape
    win = _gaussian_window(window_size, sigma, pred.device, pred.dtype).expand(c, 1, -1, -1)
    pad = window_size // 2
    mu1 = F.conv2d(pred, win, padding=pad, groups=c)
    mu2 = F.conv2d(target, win, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, win, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, win, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, win, padding=pad, groups=c) - mu12
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu12 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    score = (numerator / denominator.clamp_min(1e-12)).mean()
    del b, h, w
    return float(score)


class SSIMMetric:
    """Configurable SSIM (window size / data range follow the dataset units)."""

    def __init__(self, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> None:
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)

    def __call__(self, pred: Any, target: Any) -> float:
        return ssim(pred, target, data_range=self.data_range,
                    window_size=self.window_size, sigma=self.sigma)


class LPIPS:
    """LPIPS wrapper with a graceful torchvision fallback."""

    def __init__(self, net: str = "alex", device: str = "cpu", data_range: float = 1.0) -> None:
        self.net_name = net
        self.device = device
        self.data_range = float(data_range)
        self.backend = "none"
        self._model = None
        self._torchvision_model = None
        try:  # pragma: no cover - depends on optional dependency
            import lpips as lpips_pkg

            self._model = lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()
            self.backend = f"lpips:{net}"
        except Exception:  # pragma: no cover - fallback path
            try:
                import torch
                import torchvision

                weights = torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES
                model = torchvision.models.vgg16(weights=weights).features.to(device).eval()
                self._torchvision_model = model
                self._layers = [3, 8, 15, 22, 29]
                self.backend = "vgg16_feature_distance"
            except Exception as exc:
                warnings.warn(f"LPIPS unavailable ({exc}); perceptual distance disabled")

    @property
    def available(self) -> bool:
        return self._model is not None or self._torchvision_model is not None

    def __call__(self, pred: Any, target: Any) -> float:
        import torch

        if not self.available:
            return float("nan")
        pred = _fold_leading(pred) / self.data_range
        target = _fold_leading(target) / self.data_range
        with torch.no_grad():
            if self._model is not None:  # pragma: no cover - optional dependency
                p = pred.repeat(1, 3, 1, 1).to(self.device) * 2 - 1
                t = target.repeat(1, 3, 1, 1).to(self.device) * 2 - 1
                value = self._model(p, t).mean().item()
                return float(value)
            p = pred.repeat(1, 3, 1, 1).to(self.device)
            t = target.repeat(1, 3, 1, 1).to(self.device)
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            p, t = (p - mean) / std, (t - mean) / std
            total = 0.0
            x, y = p, t
            idx = 0
            for layer, module in enumerate(self._torchvision_model):  # type: ignore[union-attr]
                x, y = module(x), module(y)
                if layer in self._layers:
                    diff = (x - y) ** 2
                    total += float(diff.mean().item())
                    idx += 1
                if layer >= max(self._layers):
                    break
            return float(total / max(idx, 1))

    def state_dict(self) -> Dict[str, Any]:  # pragma: no cover - diagnostics
        return {"backend": self.backend, "net": self.net_name, "available": self.available}


class PerceptualMetrics:
    """SSIM + LPIPS bundle used by the evaluator."""

    def __init__(self, data_range: float = 1.0, device: str = "cpu", enable_lpips: bool = True,
                 lpips_net: str = "alex", ssim_window: int = 11) -> None:
        self.ssim = SSIMMetric(data_range=data_range, window_size=ssim_window)
        self.lpips = LPIPS(net=lpips_net, device=device, data_range=data_range) if enable_lpips else None

    def __call__(self, pred: Any, target: Any) -> Tuple[float, float]:
        s = self.ssim(pred, target)
        l = self.lpips(pred, target) if self.lpips is not None and self.lpips.available else float("nan")
        return s, l

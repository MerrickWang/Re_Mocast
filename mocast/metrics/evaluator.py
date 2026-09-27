"""Evaluation driver: streams predictions through all metrics (section 8.1)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..utils.misc import AverageMeter, ensure_dir, save_json
from .csi import ScoreAccumulator
from .perceptual import LPIPS, SSIMMetric

__all__ = ["Evaluator", "evaluate_predictions"]


class Evaluator:
    """Accumulates spatial / perceptual metrics over a whole split.

    All inputs are expected in **physical units** (de-normalised), so that the
    dataset thresholds keep their meaning (FR-DATA-03).
    """

    def __init__(self, thresholds: Sequence[float], pool_sizes: Sequence[int] = (4, 16),
                 perceptual: bool = True, data_range: Optional[float] = None,
                 device: str = "cpu", per_lead_time: bool = False, max_lead_time: int = 20,
                 enable_lpips: bool = True, lpips_net: str = "alex") -> None:
        self.thresholds = [float(t) for t in thresholds]
        self.perceptual_enabled = bool(perceptual)
        self.data_range = float(data_range) if data_range else 1.0
        self.scores = ScoreAccumulator(self.thresholds, pool_sizes, per_lead_time=per_lead_time,
                                       max_lead_time=max_lead_time)
        self.mse = AverageMeter("mse")
        self.mae = AverageMeter("mae")
        self.ssim_meter = AverageMeter("ssim")
        self.lpips_meter = AverageMeter("lpips")
        self.ssim_metric = SSIMMetric(data_range=self.data_range) if perceptual else None
        self.lpips_metric = (
            LPIPS(net=lpips_net, device=device, data_range=self.data_range)
            if (perceptual and enable_lpips) else None
        )
        self.n_samples = 0

    # ------------------------------------------------------------------ update
    def update(self, pred: Any, target: Any) -> None:
        """``pred`` / ``target``: ``[B, P, 1, H, W]`` tensors in physical units."""
        import torch

        if isinstance(pred, torch.Tensor):
            pred = pred.detach().float().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().float().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if pred.shape != target.shape:
            raise ValueError(f"Evaluator shape mismatch {pred.shape} vs {target.shape}")
        self.scores.update(pred, target)
        diff = pred - target
        self.mse.update(float((diff**2).mean()), n=1)
        self.mae.update(float(np.abs(diff).mean()), n=1)
        if self.perceptual_enabled:
            pred_t = torch.from_numpy(pred.astype(np.float32))
            target_t = torch.from_numpy(target.astype(np.float32))
            if self.ssim_metric is not None:
                try:
                    value = self.ssim_metric(pred_t, target_t)
                except Exception:  # pragma: no cover - defensive
                    value = float("nan")
                self.ssim_meter.update(value, n=1)
            if self.lpips_metric is not None and self.lpips_metric.available:
                try:
                    value = self.lpips_metric(pred_t, target_t)
                except Exception:  # pragma: no cover - defensive
                    value = float("nan")
                self.lpips_meter.update(value, n=1)
        self.n_samples += int(pred.shape[0])

    # ----------------------------------------------------------------- compute
    def compute(self) -> Dict[str, Any]:
        out = self.scores.compute()
        out.update({
            "mse": float(self.mse.avg),
            "mae": float(self.mae.avg),
            "ssim": float(self.ssim_meter.avg) if self.ssim_meter.count else float("nan"),
            "lpips": float(self.lpips_meter.avg) if self.lpips_meter.count else float("nan"),
            "n_samples": int(self.n_samples),
        })
        if self.lpips_metric is not None:
            out["lpips_backend"] = self.lpips_metric.backend
        return out

    def reset(self) -> None:
        self.scores.reset()
        for meter in (self.mse, self.mae, self.ssim_meter, self.lpips_meter):
            meter.reset()
        self.n_samples = 0

    def save(self, path: str) -> str:
        ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
        return save_json(path, self.compute())


def evaluate_predictions(predictions: Sequence[Any], targets: Sequence[Any], thresholds: Sequence[float],
                         **kwargs: Any) -> Dict[str, Any]:
    """Convenience helper for metric unit tests / offline comparisons."""
    evaluator = Evaluator(thresholds, perceptual=kwargs.pop("perceptual", False),
                          pool_sizes=kwargs.pop("pool_sizes", (4, 16)),
                          data_range=kwargs.pop("data_range", 1.0))
    for pred, target in zip(predictions, targets):
        evaluator.update(pred, target)
    return evaluator.compute()

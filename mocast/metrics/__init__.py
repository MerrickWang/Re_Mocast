"""Metrics required by the reproduction document (section 8.1)."""

from .csi import (
    DEFAULT_EPS,
    ScoreAccumulator,
    csi,
    csi_from_counts,
    csi_hss_scores,
    hss,
    hss_from_counts,
    pooled_csi,
)
from .evaluator import Evaluator, evaluate_predictions
from .perceptual import LPIPS, PerceptualMetrics, SSIMMetric, ssim
from .perceptual import SSIMMetric as SSIM

#: paper metric names -> direction of improvement (for early stopping / reporting)
METRIC_DIRECTIONS = {
    "csi": "max",
    "hss": "max",
    "csi_p4": "max",
    "csi_p16": "max",
    "ssim": "max",
    "lpips": "min",
    "mse": "min",
    "mae": "min",
}

__all__ = [
    "ScoreAccumulator",
    "Evaluator",
    "evaluate_predictions",
    "csi",
    "hss",
    "pooled_csi",
    "csi_hss_scores",
    "csi_from_counts",
    "hss_from_counts",
    "ssim",
    "SSIM",
    "LPIPS",
    "PerceptualMetrics",
    "DEFAULT_EPS",
    "METRIC_DIRECTIONS",
]

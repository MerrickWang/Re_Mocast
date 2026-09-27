"""Metrics (section 8.1) and the physical threshold mapping (FR-DATA-03)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mocast.datasets.base import Normalizer
from mocast.metrics import Evaluator, ScoreAccumulator, csi, hss, pooled_csi, ssim
from mocast.metrics.csi import csi_hss_scores


def test_perfect_prediction_scores_one() -> None:
    target = np.zeros((4, 1, 32, 32))
    target[:, 0, 8:16, 8:16] = 100.0
    pred = target.copy()
    assert csi(pred, target, 16) == pytest.approx(1.0)
    assert hss(pred, target, 16) == pytest.approx(1.0)
    assert pooled_csi(pred, target, 16, 4) == pytest.approx(1.0)


def test_no_rain_predicted_scores_zero() -> None:
    target = np.zeros((2, 1, 16, 16))
    target[:, 0, 4:8, 4:8] = 50.0
    pred = np.zeros_like(target)
    assert csi(pred, target, 16) == pytest.approx(0.0)
    assert hss(pred, target, 16) <= 0.0


def test_csi_pooling_is_monotone() -> None:
    rng = np.random.default_rng(0)
    target = (rng.random((4, 1, 64, 64)) > 0.8).astype(np.float32) * 30.0
    pred = (rng.random((4, 1, 64, 64)) > 0.8).astype(np.float32) * 30.0
    base = csi(pred, target, 16)
    p4 = pooled_csi(pred, target, 16, 4)
    p16 = pooled_csi(pred, target, 16, 16)
    assert p4 >= base - 1e-9
    assert p16 >= p4 - 1e-9


def test_csi_hss_scores_matches_direct_computation() -> None:
    rng = np.random.default_rng(1)
    target = (rng.random((2, 1, 32, 32)) > 0.7).astype(np.float32) * 40.0
    pred = (rng.random((2, 1, 32, 32)) > 0.7).astype(np.float32) * 40.0
    scores = csi_hss_scores(pred, target, [16, 32], pool_sizes=(4,))
    assert scores["csi"] == pytest.approx(np.mean([csi(pred, target, 16), csi(pred, target, 32)]))
    assert set(scores["per_threshold"].keys()) == {"16", "32"}


def test_score_accumulator_equals_batch_computation() -> None:
    rng = np.random.default_rng(2)
    pred = rng.random((3, 4, 1, 32, 32)) * 60.0
    target = rng.random((3, 4, 1, 32, 32)) * 60.0
    acc = ScoreAccumulator([16, 32], pool_sizes=(4, 16))
    acc.update(pred, target)
    result = acc.compute()
    expected = np.mean([csi(pred, target, t) for t in (16, 32)])
    assert result["csi"] == pytest.approx(expected, rel=1e-6)
    assert result["n_frames"] == pred.shape[-4] * pred.shape[0]


def test_score_accumulator_streaming_matches_single_shot() -> None:
    rng = np.random.default_rng(3)
    pred = rng.random((4, 2, 1, 16, 16)) * 40.0
    target = rng.random((4, 2, 1, 16, 16)) * 40.0
    acc = ScoreAccumulator([16])
    for i in range(4):
        acc.update(pred[i:i + 1], target[i:i + 1])
    streamed = acc.compute()["csi"]
    single = ScoreAccumulator([16])
    single.update(pred, target)
    assert streamed == pytest.approx(single.compute()["csi"], rel=1e-6)


def test_per_lead_time_reporting() -> None:
    acc = ScoreAccumulator([16], per_lead_time=True, max_lead_time=4)
    rng = np.random.default_rng(4)
    acc.update(rng.random((1, 4, 1, 16, 16)) * 40, rng.random((1, 4, 1, 16, 16)) * 40)
    result = acc.compute()
    assert set(result["per_leadtime"].keys()) == {"1", "2", "3", "4"}


def test_ssim_properties() -> None:
    x = torch.rand(2, 1, 32, 32)
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-4)
    assert ssim(x, torch.zeros_like(x)) < 0.5


def test_ssim_accepts_the_full_prediction_tensor() -> None:
    """SSIM/LPIPS are per-frame metrics: ``[B,P,1,H,W]`` must be folded, not rejected."""
    x = torch.rand(2, 4, 1, 32, 32)
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-4)
    flat = x.reshape(-1, 1, 32, 32)
    assert ssim(x, x + 0.1) == pytest.approx(ssim(flat, flat + 0.1), rel=1e-5)


def test_lpips_metric_is_optional() -> None:
    from mocast.metrics.perceptual import LPIPS

    metric = LPIPS(net="alex", device="cpu")
    if metric.available:  # the official package (or the VGG fallback) is installed
        value = metric(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32))
        assert np.isfinite(value) and value >= 0
    assert metric.state_dict()["backend"] != ""


def test_evaluator_produces_expected_keys() -> None:
    thresholds = [16, 32]
    evaluator = Evaluator(thresholds, perceptual=False, data_range=255.0)
    pred = torch.rand(1, 3, 1, 32, 32) * 255
    target = torch.rand(1, 3, 1, 32, 32) * 255
    evaluator.update(pred, target)
    metrics = evaluator.compute()
    for key in ("csi", "hss", "csi_p4", "csi_p16", "mse", "mae"):
        assert key in metrics
    assert metrics["n_samples"] == 1


def test_evaluator_perceptual_metrics_are_finite() -> None:
    evaluator = Evaluator([16], perceptual=True, data_range=1.0, enable_lpips=False)
    evaluator.update(torch.rand(1, 2, 1, 32, 32), torch.rand(1, 2, 1, 32, 32))
    metrics = evaluator.compute()
    assert np.isfinite(metrics["ssim"])


def test_normalizer_threshold_round_trip() -> None:
    """FR-DATA-03: thresholds keep their physical meaning after normalisation."""
    normalizer = Normalizer({"mode": "minmax", "vmin": 0.0, "vmax": 255.0})
    thresholds = [16, 74, 133, 160, 181, 219]
    frame = np.array(thresholds, dtype=np.float32).reshape(1, 1, 1, 6)
    normalized = normalizer.normalize(frame)
    restored = normalizer.denormalize(normalized)
    assert np.allclose(restored, frame, atol=1e-4)
    for thr in thresholds:
        assert abs(normalizer.to_physical(normalizer.to_normalized_threshold(thr)) - thr) < 1e-3


def test_zscore_normalizer_round_trip() -> None:
    normalizer = Normalizer({"mode": "zscore", "mean": 25.0, "std": 8.0})
    values = np.array([0.0, 12.5, 40.0], dtype=np.float32)
    assert np.allclose(normalizer.denormalize(normalizer.normalize(values)), values, atol=1e-5)

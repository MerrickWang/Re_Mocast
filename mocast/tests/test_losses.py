"""Loss functions: MSE, motion trend-consistency (Eq. 13) and the total objective."""

from __future__ import annotations

import os

import pytest
import torch

from mocast.losses import MoCastLoss, MotionTrendConsistencyLoss, PrecipMSELoss, build_motion_mask


def test_precip_mse_matches_torch() -> None:
    pred = torch.randn(2, 3, 1, 8, 8)
    target = torch.randn(2, 3, 1, 8, 8)
    assert torch.allclose(PrecipMSELoss()(pred, target),
                          torch.nn.functional.mse_loss(pred, target))


def test_motion_mask_thresholding() -> None:
    frames = torch.zeros(1, 4, 1, 8, 8)
    frames[0, 1, 0, :4, :4] = 100.0
    mask = build_motion_mask(frames, threshold=50.0)
    assert mask.shape == (1, 3, 1, 8, 8)
    assert mask[0, 0].sum() == 16          # pair (t=0,t=1) sees the rainy frame
    assert mask[0, 1].sum() == 16          # pair (t=1,t=2) as well
    assert mask[0, 2].sum() == 0           # pair (t=2,t=3) no longer


def test_motion_mask_downsampling_to_latent_grid() -> None:
    frames = torch.zeros(1, 3, 1, 32, 32)
    frames[0, 0, 0, 10, 10] = 10.0
    mask = build_motion_mask(frames, threshold=1.0, latent_size=(16, 16))
    assert mask.shape == (1, 2, 1, 16, 16)
    assert mask[0, 0].sum() == 0.25  # average-pool coverage of one pixel in a 2x2 cell
    assert mask[0, 1].sum() == 0


def test_constant_motion_has_zero_loss() -> None:
    loss = MotionTrendConsistencyLoss(threshold=0.0, mode="mse_mask", detach_target=False)
    motion = torch.ones(1, 4, 2, 8, 8)
    mask = torch.ones(1, 4, 8, 8)
    out = loss(motion, mask=mask)
    assert out["loss"].item() == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("steps", [3, 5])
def test_motion_loss_matches_eq13_vector_norm(steps):
    motion = torch.arange(steps, dtype=torch.float32)[None, :, None, None, None].expand(1, steps, 2, 4, 4)
    mask = torch.full((1, steps, 4, 4), 0.25)
    loss = MotionTrendConsistencyLoss()(motion, mask=mask)["loss"]
    # Every difference is (1,1): squared vector norm is 2, independent of T.
    assert loss.item() == pytest.approx(2.0)


def test_mask_averages_each_frame_before_temporal_max():
    frames = torch.zeros(1, 2, 1, 5, 5)
    frames[0, 0, 0, 2, 1] = 9
    frames[0, 1, 0, 2, 3] = 9
    mask = build_motion_mask(frames, threshold=1.5, smooth_window=3)
    assert mask[0, 0, 0, 2, 2] == 0  # max(avg)=1, whereas avg(max)=2


def test_varying_motion_has_positive_loss() -> None:
    loss = MotionTrendConsistencyLoss(threshold=0.0, detach_target=False)
    motion = torch.zeros(1, 4, 2, 8, 8)
    motion[:, 1:] = 1.0
    mask = torch.ones(1, 4, 8, 8)
    out = loss(motion, mask=mask)
    assert out["loss"].item() > 0


def test_mask_restricts_the_loss() -> None:
    loss = MotionTrendConsistencyLoss(detach_target=False)
    motion = torch.zeros(1, 4, 2, 8, 8)
    motion[:, 1:] = 5.0
    empty = torch.zeros(1, 4, 8, 8)
    out = loss(motion, mask=empty)
    assert out["loss"].item() == pytest.approx(0.0, abs=1e-7)


def test_detach_target_stops_gradient_through_later_steps() -> None:
    loss = MotionTrendConsistencyLoss(detach_target=True)
    motion = torch.randn(1, 4, 2, 8, 8, requires_grad=True)
    mask = torch.ones(1, 4, 8, 8)
    out = loss(motion, mask=mask)
    out["loss"].backward()
    assert motion.grad is not None
    # every step but the last one acts as a *source* of a difference term
    assert motion.grad[:, :3].abs().sum() > 0
    # the last step only ever appears detached, hence no gradient
    assert motion.grad[:, 3].abs().sum() == 0


def test_no_detach_propagates_to_all_steps() -> None:
    loss = MotionTrendConsistencyLoss(detach_target=False)
    motion = torch.randn(1, 4, 2, 8, 8, requires_grad=True)
    mask = torch.ones(1, 4, 8, 8)
    loss(motion, mask=mask)["loss"].backward()
    assert motion.grad[:, 3].abs().sum() > 0


def test_mocast_loss_total_composition() -> None:
    criterion = MoCastLoss(lambda_motion=0.01, motion_cfg={"enabled": True, "mode": "mse_mask"})
    outputs = {"pred": torch.randn(1, 3, 1, 8, 8)}
    target = torch.randn(1, 3, 1, 8, 8)
    mean_motion = torch.randn(1, 4, 2, 8, 8)
    mask = torch.ones(1, 4, 8, 8)
    result = criterion(outputs, target, mean_motion=mean_motion, mask=mask)
    assert torch.allclose(result["loss"],
                          result["precip"] + 0.01 * result["motion"], atol=1e-6)


def test_mocast_loss_without_motion() -> None:
    criterion = MoCastLoss(lambda_motion=0.01, motion_cfg={"enabled": False})
    outputs = {"pred": torch.randn(1, 3, 1, 8, 8)}
    target = torch.randn(1, 3, 1, 8, 8)
    result = criterion(outputs, target)
    assert torch.allclose(result["loss"], result["precip"])


def test_warp_consistency_mode_runs() -> None:
    from mocast.models.advection import Advection

    loss = MotionTrendConsistencyLoss(mode="warp_consistency")
    motion = torch.randn(1, 4, 2, 8, 8)
    frames = torch.rand(1, 5, 1, 16, 16)
    mask = torch.ones(1, 4, 8, 8)
    out = loss(motion, mask=mask, total_motion=motion, advection=Advection(),
               frames_normalized=frames)
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() >= 0


def test_unknown_mode_raises() -> None:
    loss = MotionTrendConsistencyLoss(mode="does-not-exist")
    with pytest.raises(ValueError):
        loss(torch.randn(1, 3, 2, 4, 4), mask=torch.ones(1, 3, 4, 4))

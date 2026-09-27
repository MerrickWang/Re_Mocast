"""UT-04 / FR-MSM-04: gating normalisation."""

from __future__ import annotations

import pytest
import torch

from mocast.models.msm import MSM, MotionAdaptiveGating


@pytest.mark.parametrize("mode", ["softmax", "sigmoid", "uniform"])
def test_ut04_gating_normalisation(mode: str) -> None:
    torch.manual_seed(0)
    gating = MotionAdaptiveGating(motion_channels_per_scale=4, num_experts=3, hidden=8, mode=mode)
    motions = [torch.randn(2, 3, 4, 8, 8) for _ in range(3)]
    weights = gating(motions, size=(8, 8))
    assert weights.shape == (2, 3, 3, 8, 8)
    total = weights.sum(dim=2)
    if mode == "softmax" or mode == "uniform":
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    else:  # sigmoid is a diagnostic alternative - only bound checking
        assert (weights >= 0).all() and (weights <= 1).all()
    assert (weights >= 0).all()


def test_gating_uniform_mode_is_one_over_s() -> None:
    gating = MotionAdaptiveGating(4, num_experts=4, mode="uniform")
    weights = gating([torch.randn(1, 2, 4, 6, 6)], size=(6, 6))
    assert torch.allclose(weights, torch.full_like(weights, 0.25))


def test_gating_handles_different_motion_resolutions() -> None:
    gating = MotionAdaptiveGating(4, num_experts=2, hidden=8, mode="softmax")
    weights = gating([torch.randn(1, 2, 4, 8, 8), torch.randn(1, 2, 4, 4, 4)], size=(8, 8))
    assert weights.shape == (1, 2, 2, 8, 8)
    assert torch.allclose(weights.sum(dim=2), torch.ones_like(weights[:, :, 0]))


def test_msm_output_contract() -> None:
    """Section 5: ``E_s``/``W_gate`` shapes and time alignment with the motion."""
    msm = MSM(cfg={"num_experts": 3, "kernels": [3, 5, 7], "channels": 16, "out_channels": 16,
                   "motion_channels": 12, "expert_hidden": 16, "modulation_hidden": 8,
                   "gate_hidden": 8, "downsample": 2})
    frames = torch.randn(2, 4, 1, 32, 32)                     # T-1 = 4 steps
    motions = [torch.randn(2, 4, 12, 16, 16) for _ in range(3)]
    out = msm(frames, motions)
    assert out["source"].shape == (2, 4, 16, 16, 16)
    assert out["gate"].shape == (2, 4, 3, 16, 16)
    assert torch.allclose(out["gate"].sum(dim=2), torch.ones(2, 4, 16, 16), atol=1e-5)


def test_msm_single_expert_ablation() -> None:
    msm = MSM(cfg={"num_experts": 1, "kernels": [3], "channels": 16, "out_channels": 16,
                   "motion_channels": 12, "multiscale": False, "downsample": 2})
    out = msm(torch.randn(1, 4, 1, 32, 32), [torch.randn(1, 4, 12, 16, 16)])
    assert out["source"].shape == (1, 4, 16, 16, 16)
    assert out["gate"].shape == (1, 4, 1, 16, 16)


def test_expert_modulation_changes_with_motion() -> None:
    """The expert must actually be conditioned on the motion features."""
    msm = MSM(cfg={"num_experts": 1, "kernels": [3], "channels": 8, "out_channels": 8,
                   "motion_channels": 4, "expert_hidden": 8, "modulation_hidden": 8,
                   "downsample": 1})
    frames = torch.randn(1, 2, 1, 16, 16)
    motion_a = [torch.zeros(1, 2, 4, 16, 16)]
    motion_b = [torch.ones(1, 2, 4, 16, 16)]
    out_a = msm(frames, motion_a)["source"]
    out_b = msm(frames, motion_b)["source"]
    assert not torch.allclose(out_a, out_b, atol=1e-3)


def test_expert_modulation_receives_gradients() -> None:
    msm = MSM(cfg={"num_experts": 1, "kernels": [3], "channels": 8, "out_channels": 8,
                   "motion_channels": 4, "expert_hidden": 8, "modulation_hidden": 8,
                   "downsample": 1})
    out = msm(torch.randn(1, 2, 1, 16, 16), [torch.randn(1, 2, 4, 16, 16)])
    out["source"].pow(2).mean().backward()
    grads = [p.grad for expert in msm.experts for head in expert.modulation
             for p in head.parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(float(g.abs().sum()) > 0 for g in grads)


def test_adaLN_zero_variant_is_available() -> None:
    """The zero-initialised modulation (Peebles & Xie 2023) must be selectable."""
    msm = MSM(cfg={"num_experts": 1, "kernels": [3], "channels": 8, "out_channels": 8,
                   "motion_channels": 4, "modulation_init_zero": True, "downsample": 1})
    head = msm.experts[0].modulation[0]
    assert float(head.net[-1].weight.detach().abs().sum()) == 0.0

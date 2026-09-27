"""Tensor contract of section 5 + L1 structural acceptance."""

from __future__ import annotations

from typing import Dict

import pytest
import torch

from mocast.models import MoCast
from .conftest import small_model_cfg


@pytest.fixture()
def contract(device: torch.device) -> Dict[str, torch.Tensor]:
    cfg = small_model_cfg()
    model = MoCast(cfg).to(device).eval()
    x = torch.randn(2, 5, 1, 32, 32, device=device)
    with torch.no_grad():
        out = model(x)
    return out


def test_input_output_shapes(contract: Dict[str, torch.Tensor]) -> None:
    assert contract["pred"].shape == (2, 4, 1, 32, 32)        # [B,P,1,H,W]
    assert contract["motion"].shape == (2, 4, 2, 32, 32)      # [B,P,2,H,W]
    assert contract["source"].shape == (2, 4, 1, 32, 32)      # [B,P,1,H,W]


def test_latent_and_motion_shapes(contract: Dict[str, torch.Tensor]) -> None:
    # encoder downsample = 2 -> latent 16x16, T-1 = 4 steps
    assert contract["latents"].shape == (2, 5, 16, 16, 16)    # [B,T,d,h_d,w_d]
    for key in ("mm", "mf", "ma"):
        assert contract[key].shape == (2, 4, 2, 16, 16), key   # [B,T-1,2,h_d,w_d]


def test_motion_additivity(contract: Dict[str, torch.Tensor]) -> None:
    """Section 5: ``M_a = M_m + M_f``."""
    assert torch.allclose(contract["ma"], contract["mm"] + contract["mf"], atol=1e-6)


def test_gate_softmax_sum_to_one(contract: Dict[str, torch.Tensor]) -> None:
    gate = contract["gate"]
    assert gate.shape[:2] == (2, 4)
    assert torch.allclose(gate.sum(dim=2), torch.ones_like(gate[:, :, 0]), atol=1e-5)
    assert (gate >= 0).all()


def test_channel_order_is_dx_dy(contract: Dict[str, torch.Tensor]) -> None:
    """Motion channels are ``(dx, dy)`` -> a pure +3 pixel x-shift must be recovered."""
    from mocast.models.advection import warp

    blob = torch.zeros(1, 1, 32, 32)
    blob[0, 0, 16, 16] = 1.0
    flow = torch.zeros(1, 2, 32, 32)
    flow[:, 0] = 3.0                       # channel 0 = dx
    warped = warp(blob, flow)
    assert warped[0, 0, 16, 19] > warped[0, 0, 16, 13]


def test_stateless_forward_is_reproducible(device: torch.device) -> None:
    cfg = small_model_cfg()
    torch.manual_seed(0)
    model = MoCast(cfg).to(device).eval()
    x = torch.randn(1, 5, 1, 32, 32, device=device)
    with torch.no_grad():
        a = model(x)["pred"]
        b = model(x)["pred"]
    assert torch.allclose(a, b, atol=1e-6)


def test_overlapping_patches_variant(device: torch.device) -> None:
    """FR-PMM-01: the patch overlap mode must be configurable."""
    cfg = small_model_cfg()
    cfg["pmm"] = {**cfg["pmm"], "patch_size": 4, "patch_stride": 2}
    model = MoCast(cfg).to(device).eval()
    x = torch.randn(1, 5, 1, 32, 32, device=device)
    with torch.no_grad():
        out = model(x)
    assert out["pred"].shape == (1, 4, 1, 32, 32)
    assert torch.isfinite(out["cue"]).all()
    assert out["cue"].shape[:3] == (1, 4, 2 * 3 * 3)


def test_model_rejects_wrong_input_length(device: torch.device) -> None:
    model = MoCast(small_model_cfg()).to(device).eval()
    with pytest.raises(ValueError):
        model(torch.randn(1, 4, 1, 32, 32, device=device))


def test_teacher_forcing_path(device: torch.device) -> None:
    cfg = small_model_cfg()
    cfg["reconstruction"] = {"teacher_forcing": True}
    model = MoCast(cfg).to(device)
    model.train()
    x = torch.randn(1, 5, 1, 32, 32, device=device)
    y = torch.rand(1, 4, 1, 32, 32, device=device)
    out = model(x, y=torch.cat([x, y], dim=1))
    assert out["pred"].shape == (1, 4, 1, 32, 32)
    out["pred"].mean().backward()


def test_autoregressive_prediction_mode(device: torch.device) -> None:
    cfg = small_model_cfg()
    cfg["prediction"] = {"mode": "autoregressive", "spatial_blocks": 1}
    model = MoCast(cfg).to(device).eval()
    x = torch.randn(1, 5, 1, 32, 32, device=device)
    with torch.no_grad():
        out = model(x)
    assert out["pred"].shape == (1, 4, 1, 32, 32)


def test_mocast_plus_shapes_and_sampling(device: torch.device) -> None:
    from mocast.models import MoCastPlus

    backbone = small_model_cfg()
    model = MoCastPlus(
        backbone_cfg=backbone,
        diffusion_cfg={"num_steps": 20, "schedule": "linear", "residual_scale": 1.0},
        unet_cfg={"base_channels": 8, "channel_mults": [1, 2], "num_blocks": 1, "cond_dim": 16,
                  "temporal_attention": True, "conv_gru": True, "num_heads": 2},
    ).to(device).eval()
    x = torch.randn(1, 5, 1, 32, 32, device=device)
    y = torch.rand(1, 4, 1, 32, 32, device=device)
    with torch.no_grad():
        base, samples = model.sample(x, num_samples=2, steps=2)
    assert base.shape == (1, 4, 1, 32, 32)
    assert samples.shape == (1, 2, 4, 1, 32, 32)
    assert torch.isfinite(samples).all()

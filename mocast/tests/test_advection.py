"""UT-05 / FR-ADV-01: warp direction, unit conversion and gradient flow."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mocast.models.advection import Advection, base_grid, warp


def _gaussian_blob(size: int = 64, cx: float = 32.0, cy: float = 32.0, sigma: float = 4.0) -> torch.Tensor:
    y, x = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32),
                       indexing="ij")
    blob = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))).astype(np.float32)
    return torch.from_numpy(blob)[None, None]


def _shift(blob: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Analytic translation: content moves by ``(dx, dy)`` pixels."""
    size = blob.shape[-1]
    y, x = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32),
                       indexing="ij")
    cx, cy = 32.0 + dx, 32.0 + dy
    shifted = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * 4.0**2))).astype(np.float32)
    return torch.from_numpy(shifted)[None, None]


@pytest.mark.parametrize("dx,dy", [(1.0, 0.0), (2.0, 0.0), (0.0, 3.0), (-4.0, 0.0), (2.0, -2.0)])
def test_ut05_translation_recovery(dx: float, dy: float) -> None:
    """A known 1-4 pixel translation must be reproduced by the advection operator."""
    blob = _gaussian_blob()
    flow = torch.tensor([[[[dx]], [[dy]]]]).expand(1, 2, 64, 64).contiguous()
    warped = warp(blob, flow, unit="pixel", semantics="displacement", padding_mode="border",
                  align_corners=True)
    expected = _shift(blob, dx, dy)
    interior = (slice(None), slice(None), slice(16, 48), slice(16, 48))
    error = (warped[interior] - expected[interior]).abs().max().item()
    assert error < 2e-3, f"shift ({dx},{dy}) error {error}"


def test_warp_sign_convention_is_documented_and_consistent() -> None:
    """``displacement`` and ``backward_flow`` differ by the sign of the sampling offset."""
    blob = _gaussian_blob()
    flow = torch.zeros(1, 2, 64, 64)
    flow[:, 0] = 3.0
    forward = warp(blob, flow, semantics="displacement")
    backward = warp(blob, flow, semantics="backward_flow")
    assert torch.allclose(forward, _shift(blob, 3.0, 0.0), atol=2e-3)
    assert torch.allclose(backward, _shift(blob, -3.0, 0.0), atol=2e-3)


def test_normalized_and_pixel_units_agree() -> None:
    blob = _gaussian_blob()
    pixels = torch.zeros(1, 2, 64, 64)
    pixels[:, 0] = 2.0
    normalized = pixels * (2.0 / 63.0)  # align_corners=True mapping
    a = warp(blob, pixels, unit="pixel", align_corners=True)
    b = warp(blob, normalized, unit="normalized", align_corners=True)
    assert torch.allclose(a, b, atol=1e-6)


def test_identity_flow_is_identity() -> None:
    blob = _gaussian_blob()
    identity = torch.zeros(1, 2, 64, 64)
    assert torch.allclose(warp(blob, identity), blob, atol=1e-6)


def test_align_corners_false_mapping() -> None:
    """align_corners=False uses the ``2*dx/W`` pixel-to-grid scale (recorded in the config)."""
    blob = _gaussian_blob()
    flow = torch.zeros(1, 2, 64, 64)
    flow[:, 1] = 2.0
    warped = warp(blob, flow, unit="pixel", align_corners=False)
    expected = _shift(blob, 0.0, 2.0)
    interior = (slice(None), slice(None), slice(16, 48), slice(16, 48))
    assert (warped[interior] - expected[interior]).abs().max().item() < 3e-3
    # and the normalized-unit conversion is consistent with align_corners=False
    normalized = flow * (2.0 / 64.0)
    assert torch.allclose(warped, warp(blob, normalized, unit="normalized",
                                       align_corners=False), atol=1e-6)


def test_advection_module_from_config() -> None:
    module = Advection.from_config({"unit": "pixel", "semantics": "displacement",
                                    "align_corners": True, "padding_mode": "border"})
    blob = _gaussian_blob()
    out = module(blob, torch.full((1, 2, 64, 64), 1.0))
    assert out.shape == blob.shape
    assert torch.allclose(out, _shift(blob, 1.0, 1.0), atol=2e-3)


def test_warp_gradient_flows_to_motion() -> None:
    blob = _gaussian_blob()
    # grid_sample's local gradient vanishes exactly at integer sampling positions,
    # hence the small sub-pixel offset
    flow = torch.full((1, 2, 64, 64), 0.3, requires_grad=True)
    out = warp(blob, flow)
    out.sum().backward()
    assert flow.grad is not None and torch.isfinite(flow.grad).all()
    assert flow.grad.abs().sum() > 0


def test_warp_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        warp(torch.randn(1, 1, 8, 8), torch.randn(1, 2, 4, 4))


def test_base_grid_matches_grid_sample_convention() -> None:
    grid = base_grid(4, 4, align_corners=True)
    assert grid.shape == (1, 4, 4, 2)
    assert torch.allclose(grid[0, 0, 0], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(grid[0, -1, -1], torch.tensor([1.0, 1.0]))


def test_bfloat16_zero_motion_does_not_blur_over_twenty_steps():
    image = torch.zeros(1, 1, 128, 128)
    image[:, :, 40:55, 40:55] = 1
    flow = torch.zeros(1, 2, 128, 128, dtype=torch.bfloat16, requires_grad=True)
    current = image
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for _ in range(20):
            current = warp(current, flow)
    assert current.dtype == torch.float32
    assert torch.allclose(current, image, atol=2e-4)
    current.sum().backward()
    assert flow.grad is not None and torch.isfinite(flow.grad).all()

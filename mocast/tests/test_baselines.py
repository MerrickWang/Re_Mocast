"""Classical baselines (P1): persistence and block-matching optical flow."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mocast.models import BlockMatchingFlow, OpticalFlowBaseline, PersistenceBaseline


def _blob(size: int = 64, cx: float = 32.0, cy: float = 32.0, sigma: float = 5.0) -> torch.Tensor:
    y, x = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32),
                       indexing="ij")
    blob = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))).astype(np.float32)
    return torch.from_numpy(blob)[None, None]


def test_persistence_repeats_the_last_frame() -> None:
    baseline = PersistenceBaseline(output_len=20)
    x = torch.rand(2, 5, 1, 16, 16)
    out = baseline(x)
    assert out["pred"].shape == (2, 20, 1, 16, 16)
    for t in range(20):
        assert torch.allclose(out["pred"][:, t], x[:, -1])


@pytest.mark.parametrize("dx,dy", [(2.0, 0.0), (0.0, 3.0), (-2.0, 2.0)])
def test_block_matching_recovers_uniform_translation(dx: float, dy: float) -> None:
    previous = _blob()
    following = _blob(cx=32.0 + dx, cy=32.0 + dy)
    flow = BlockMatchingFlow(search_radius=4, block_size=8, smooth_kernel=1)(previous, following)
    assert flow.shape == (1, 2, 64, 64)
    # interior blocks must agree with the true displacement
    interior_x = flow[0, 0, 16:48, 16:48].mean().item()
    interior_y = flow[0, 1, 16:48, 16:48].mean().item()
    assert abs(interior_x - dx) <= 1.0
    assert abs(interior_y - dy) <= 1.0


def test_optical_flow_baseline_shapes_and_finiteness() -> None:
    baseline = OpticalFlowBaseline(output_len=5, search_radius=3, block_size=8, smooth_kernel=3)
    x = torch.rand(1, 5, 1, 32, 32)
    out = baseline(x)
    assert out["pred"].shape == (1, 5, 1, 32, 32)
    assert out["motion"].shape == (1, 5, 2, 32, 32)
    assert torch.isfinite(out["pred"]).all()


def test_optical_flow_baseline_follows_a_moving_blob() -> None:
    """Advecting a blob with the block-matching flow beats persistence (L2 sanity)."""
    frames = [torch.rand(1, 1, 64, 64) * 0.0 for _ in range(4)]
    for i in range(4):
        frames[i] = _blob(cx=24.0 + 2.0 * i, cy=32.0)
    x = torch.cat([f[:, None] for f in frames], dim=1)          # [1,4,1,64,64]
    future = torch.cat([_blob(cx=32.0 + 2.0 * i, cy=32.0)[:, None, None] for i in range(3)],
                       dim=1)                                    # [1,3,1,64,64]
    baseline = OpticalFlowBaseline(output_len=3, search_radius=4, block_size=8, smooth_kernel=1)
    pred = baseline(x)["pred"]
    persistence = x[:, -1:].repeat(1, 3, 1, 1, 1)
    flow_mse = torch.mean((pred - future) ** 2).item()
    persist_mse = torch.mean((persistence - future) ** 2).item()
    assert flow_mse < persist_mse

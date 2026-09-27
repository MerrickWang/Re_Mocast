"""UT-01 / UT-02 / FR-PMM-05..07: wavelet transform and fluctuation block."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mocast.models.pmm.wavelet import (
    IDWT2D,
    WaveletFluctuationBlock,
    dwt2,
    filters_2d,
    idwt2,
    wavelet_filters,
)

BASES = ["haar", "db2"]


@pytest.mark.parametrize("basis", BASES)
def test_ut01_perfect_reconstruction(basis: str) -> None:
    """UT-01: DWT/IDWT reconstruction error must be below the numerical tolerance."""
    torch.manual_seed(0)
    filters = filters_2d(basis)
    for size in [(32, 32), (16, 24), (64, 64), (18, 30)]:
        x = torch.randn(2, 3, *size)
        subbands = dwt2(x, filters)
        assert all(tuple(s.shape[-2:]) == (size[0] // 2, size[1] // 2) for s in subbands)
        reconstructed = idwt2(*subbands, filters=filters, out_size=size)
        error = (reconstructed - x).abs().max().item()
        assert error < 1e-5, f"{basis} {size}: reconstruction error {error}"


@pytest.mark.parametrize("basis", BASES)
def test_ut02_constant_field_has_zero_high_frequency(basis: str) -> None:
    """UT-02: constant fields must have (near) zero LH/HL/HH responses."""
    filters = filters_2d(basis)
    constant = torch.full((1, 2, 32, 32), 3.5)
    _, h_h, h_v, h_d = dwt2(constant, filters)
    for band, name in ((h_h, "H_h"), (h_v, "H_v"), (h_d, "H_d")):
        assert band.abs().max().item() < 1e-5, f"{basis}: {name} not zero on a constant field"


def test_wavelet_module_wrappers_match_functional() -> None:
    x = torch.randn(1, 2, 16, 16)
    from mocast.models.pmm.wavelet import DWT2D

    dwt = DWT2D("haar")
    idwt = IDWT2D("haar")
    subbands = dwt(x)
    assert torch.allclose(subbands[0], dwt2(x, dwt.filters)[0])
    assert torch.allclose(idwt(*subbands, out_size=(16, 16)), x, atol=1e-5)


def test_orthonormal_energy_preservation() -> None:
    """Orthonormal filter banks preserve energy (sanity check of the coefficients)."""
    for basis in BASES:
        low, high = wavelet_filters(basis)
        assert torch.allclose(low @ low, torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(high @ high, torch.tensor(1.0), atol=1e-6)
        assert abs(float(low @ high)) < 1e-6
        assert abs(float(low.sum()) - 2 ** 0.5) < 1e-6


def test_fluctuation_block_shapes_and_gradients() -> None:
    block = WaveletFluctuationBlock(dim=8, hidden=8, basis="haar", ll_mode="zero")
    h = torch.randn(2, 5, 8, 16, 16, requires_grad=True)
    x = torch.randn(2, 5, 1, 32, 32, requires_grad=True)
    out = block(h, x)
    assert out["motion"].shape == (2, 4, 2, 16, 16)
    out["motion"].sum().backward()
    assert torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("ll_mode", ["zero", "keep", "learnable"])
def test_ll_modes_run(ll_mode: str) -> None:
    block = WaveletFluctuationBlock(dim=4, hidden=4, ll_mode=ll_mode)
    out = block(torch.randn(1, 3, 4, 8, 8), torch.randn(1, 3, 1, 16, 16))
    assert out["motion"].shape == (1, 2, 2, 8, 8)
    assert torch.isfinite(out["motion"]).all()


def test_wavelet_null_input_gives_zero_fluctuation() -> None:
    """A constant latent + constant frames -> dimensionless fluctuation response."""
    block = WaveletFluctuationBlock(dim=4, hidden=4, ll_mode="zero")
    h = torch.zeros(1, 3, 4, 8, 8)
    x = torch.zeros(1, 3, 1, 16, 16)
    out = block(h, x)
    motion = out["motion"]
    assert np.isfinite(motion.detach().numpy()).all()

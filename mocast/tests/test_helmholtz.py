"""UT-03 / FR-PMM-03 / FR-PMM-04: fixed differential operators and Helmholtz mean motion."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mocast.models.pmm.helmholtz import (
    HelmholtzMeanMotion,
    curl_2d,
    spatial_divergence,
    spatial_gradient,
)


def _linear_potential(h: int, w: int, a: float, b: float) -> np.ndarray:
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32),
                       indexing="ij")
    return a * x + b * y


def _rotational_potential(h: int, w: int, omega: float) -> np.ndarray:
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32),
                       indexing="ij")
    return 0.5 * omega * (x**2 + y**2)


@pytest.mark.parametrize("kernel", ["central", "sobel"])
def test_ut03_gradient_of_linear_potential_is_translation(kernel: str) -> None:
    """A linear potential produces a constant translational velocity field."""
    a, b, h, w = 0.75, -0.5, 32, 32
    phi = torch.from_numpy(_linear_potential(h, w, a, b))[None, None]
    grad = spatial_gradient(phi, kernel=kernel)[0]
    interior = grad[:, 1:-1, 1:-1]
    assert torch.allclose(interior[0], torch.full_like(interior[0], a), atol=1e-5)
    assert torch.allclose(interior[1], torch.full_like(interior[1], b), atol=1e-5)


@pytest.mark.parametrize("kernel", ["central", "sobel"])
def test_ut03_curl_of_rotational_potential(kernel: str) -> None:
    """A quadratic potential produces the expected rotational motion."""
    omega, h, w = 0.03, 32, 32
    psi = torch.from_numpy(_rotational_potential(h, w, omega))[None, None]
    field = curl_2d(psi, kernel=kernel)[0]
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32),
                       indexing="ij")
    expected_x = omega * y
    expected_y = -omega * x
    interior = field[:, 1:-1, 1:-1]
    assert torch.allclose(interior[0], torch.from_numpy(expected_x)[1:-1, 1:-1], atol=1e-4)
    assert torch.allclose(interior[1], torch.from_numpy(expected_y)[1:-1, 1:-1], atol=1e-4)


@pytest.mark.parametrize("kernel", ["central", "sobel"])
def test_helmholtz_structural_identities(kernel: str) -> None:
    """curl-free and divergence-free components really are curl/divergence free."""
    torch.manual_seed(0)
    phi = torch.randn(1, 1, 48, 48)
    psi = torch.randn(1, 1, 48, 48)
    grad_field = spatial_gradient(phi, kernel)   # curl-free candidate
    curl_field = curl_2d(psi, kernel)            # divergence-free candidate

    # div(curl(psi)) == 0
    div_of_curl = spatial_divergence(curl_field, kernel)[0, 0, 1:-1, 1:-1]
    assert div_of_curl.abs().max().item() < 1e-3

    # curl(grad(phi)) == d(vy)/dx - d(vx)/dy == 0
    dvy_dx = spatial_gradient(grad_field[:, 1:2], kernel)[:, 0:1]
    dvx_dy = spatial_gradient(grad_field[:, 0:1], kernel)[:, 1:2]
    curl_of_grad = (dvy_dx - dvx_dy)[0, 0, 1:-1, 1:-1]
    assert curl_of_grad.abs().max().item() < 1e-3


def test_helmholtz_mean_motion_matches_analytic_combination() -> None:
    """``M_mean = grad(phi) + curl(psi)`` with the potentials produced by the nets."""
    module = HelmholtzMeanMotion(in_channels=4, hidden=8, depth=1, motion_scale=1.0)

    class FixedPotential(nn.Module):
        def __init__(self, kind: str) -> None:
            super().__init__()
            self.kind = kind

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            if self.kind == "phi":
                return (x[:, :1] * 0.0) + 0.25
            value = torch.linspace(-1, 1, w)[None, None, None, :].expand(b, 1, h, w)
            return value.clone()

    module.to_phi = FixedPotential("phi")   # type: ignore[assignment]
    module.to_psi = FixedPotential("psi")   # type: ignore[assignment]
    cue = torch.randn(1, 3, 4, 8, 8)
    out = module(cue)
    phi, psi = out["phi"].detach(), out["psi"].detach()
    expected = spatial_gradient(phi[0, 0][None, None], module.kernel) + \
        curl_2d(psi[0, 0][None, None], module.kernel)
    assert out["mean"][0, 0].shape == expected[0].shape
    assert torch.allclose(out["mean"][0, 0], expected[0], atol=1e-5)


def test_mean_motion_gradients_flow() -> None:
    module = HelmholtzMeanMotion(in_channels=6, hidden=8, depth=1)
    cue = torch.randn(1, 2, 6, 8, 8, requires_grad=True)
    out = module(cue)["mean"]
    assert out.shape == (1, 2, 2, 8, 8)
    out.sum().backward()
    assert cue.grad is not None and torch.isfinite(cue.grad).all()
    assert cue.grad.abs().sum() > 0

"""UT-07: every learnable sub-module receives finite, non-zero gradients."""

from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from mocast.models import MoCast
from .conftest import small_model_cfg


def _module_grad_stats(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for name, module in model.named_modules():
        total, nonzero, bad = 0.0, 0.0, 0
        for param in module.parameters(recurse=False):
            if param.grad is None:
                continue
            grad = param.grad.detach()
            total += grad.abs().sum().item()
            nonzero += (grad != 0).sum().item()
            if not torch.isfinite(grad).all():
                bad += 1
        if total or bad:
            stats[name or "<root>"] = {"abs_sum": total, "nonzero": nonzero, "nonfinite": bad}
    return stats


def test_ut07_all_submodules_receive_gradients(device: torch.device) -> None:
    cfg = small_model_cfg()
    model = MoCast(cfg).to(device)
    model.train()
    x = torch.randn(2, int(cfg["input_len"]), 1, 32, 32, device=device)
    y = torch.rand(2, int(cfg["output_len"]), 1, 32, 32, device=device)
    out = model(x, y=y)
    loss = torch.nn.functional.mse_loss(out["pred"], y) + 1e-3 * out["mm"].abs().mean()
    loss.backward()

    stats = _module_grad_stats(model)
    missing: List[str] = []
    zero: List[str] = []
    for name, module in model.named_modules():
        params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        if not params:
            continue
        entry = stats.get(name or "<root>")
        if entry is None or any(p.grad is None for p in params):
            missing.append(name)
            continue
        if entry["abs_sum"] <= 0.0:
            zero.append(name)
    assert not missing, f"modules without gradients: {missing}"
    assert not zero, f"modules with all-zero gradients: {zero}"
    for name, entry in stats.items():
        assert entry["nonfinite"] == 0, f"non-finite gradients in {name}"


def test_ut07_no_nan_or_inf_in_activations(device: torch.device) -> None:
    cfg = small_model_cfg()
    model = MoCast(cfg).to(device)
    model.eval()
    x = torch.randn(2, 5, 1, 32, 32, device=device)
    with torch.no_grad():
        out = model(x)
    for key in ("pred", "motion", "source", "mm", "mf", "ma", "gate", "cue"):
        value = out[key]
        assert torch.isfinite(value).all(), f"non-finite values in '{key}'"


def test_ut07_finite_difference_gradient_check(device: torch.device) -> None:
    """Numerical sanity check of one learnable parameter (small tolerance)."""
    cfg = small_model_cfg()
    model = MoCast(cfg).to(device)
    model.eval()
    x = torch.randn(1, 5, 1, 32, 32, device=device, dtype=torch.float64)
    model = model.double()
    param = model.motion_embed.proj.weight
    with torch.no_grad():
        param.add_(0.01 * torch.randn_like(param))

    def loss_fn() -> torch.Tensor:
        return model(x)["pred"].pow(2).mean()

    with torch.no_grad():
        base = loss_fn()
        eps = 1e-5
        param[0, 0, 0, 0] += eps
        plus = loss_fn()
        param[0, 0, 0, 0] -= 2 * eps
        minus = loss_fn()
        param[0, 0, 0, 0] += eps
    numeric = (plus - minus) / (2 * eps)

    x_grad = x.clone().requires_grad_(True)
    loss = model(x_grad)["pred"].pow(2).mean()
    loss.backward()
    analytic = param.grad[0, 0, 0, 0]
    assert torch.isfinite(analytic)
    assert torch.allclose(numeric, analytic, rtol=1e-3, atol=1e-5), (numeric, analytic)
    assert torch.isfinite(base)

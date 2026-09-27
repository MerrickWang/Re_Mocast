"""UT-06 / risk R4: temporal alignment of motion and source-sink supervision."""

from __future__ import annotations

import pytest
import torch

from mocast.models import MoCast
from .conftest import small_model_cfg


def _frame_dependence(model: MoCast, input_len: int, output: torch.Tensor,
                      size: int = 32) -> torch.Tensor:
    x = torch.randn(1, input_len, 1, size, size, requires_grad=True)
    out = model(x)
    target = out["mm"] if output is None else output
    target.sum().backward()
    return (x.grad.abs().sum(dim=(0, 2, 3, 4)) > 0).float()


def test_ut06_motion_step_pairs_adjacent_frames(device: torch.device) -> None:
    """Motion ``t -> t+1`` may only depend on frames ``t`` and ``t+1``."""
    cfg = small_model_cfg()
    model = MoCast(cfg).to(device)
    model.eval()
    input_len = int(cfg["input_len"])
    x = torch.randn(1, input_len, 1, 32, 32, device=device)
    for step in range(input_len - 1):
        x = x.detach().requires_grad_(True)
        outputs = model(x)
        motion = outputs["mm"][:, step]
        grad = torch.autograd.grad(motion.sum(), x, retain_graph=False)[0]
        used = (grad.abs().sum(dim=(0, 2, 3, 4)) > 0).nonzero().flatten().tolist()
        assert set(used) <= {step, step + 1}, (
            f"motion step {step} depends on frames {used}, expected {{{step}, {step + 1}}}")


def test_ut06_source_sink_uses_the_later_frame(device: torch.device) -> None:
    """``E_s`` at step ``t`` is driven by frame ``t+1`` (spec: input X[:,1:T])."""
    cfg = small_model_cfg()
    model = MoCast(cfg).to(device)
    model.eval()
    input_len = int(cfg["input_len"])
    captured = {}

    def hook(module: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        captured["x"] = inputs[0].detach()

    handle = model.msm.register_forward_hook(hook)

    x = torch.randn(1, input_len, 1, 32, 32, device=device)
    with torch.no_grad():
        model(x)
    handle.remove()
    assert "x" in captured
    expected = x[:, 1:]
    assert torch.allclose(captured["x"], expected)


def test_ut06_prediction_starts_from_last_input_frame(device: torch.device) -> None:
    """Eq. (12): ``Y_hat^T = X^T``; the first prediction equals advection of X^T."""
    cfg = small_model_cfg()
    model = MoCast(cfg).to(device)
    model.eval()
    x = torch.randn(1, int(cfg["input_len"]), 1, 32, 32, device=device)
    with torch.no_grad():
        out = model(x)
    manual = model.advection(x[:, -1], out["motion"][:, 0]) + out["source"][:, 0]
    assert torch.allclose(out["pred"][:, 0], manual, atol=1e-6)


def test_source_sink_zero_ablation(device: torch.device) -> None:
    cfg = small_model_cfg()
    cfg["ablation"] = {"use_source_sink": False}
    model = MoCast(cfg).to(device)
    model.eval()
    x = torch.randn(1, 5, 1, 32, 32, device=device)
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out["source"], torch.zeros_like(out["source"]))


def test_ablation_flags_do_not_break_shapes(device: torch.device) -> None:
    variants = [
        {"helmholtz": False},
        {"use_fluctuation": False},
        {"use_mean": False},
        {"decompose": False},
        {"msm_gating": False},
        {"msm_multiscale": False},
    ]
    x = torch.randn(1, 5, 1, 32, 32, device=device)
    for flags in variants:
        cfg = small_model_cfg()
        cfg["ablation"] = flags
        if not flags.get("msm_multiscale", True):
            cfg["msm"] = {**cfg["msm"], "num_experts": 1, "kernels": [3], "dilations": [1]}
        model = MoCast(cfg).to(device).eval()
        with torch.no_grad():
            out = model(x)
        assert out["pred"].shape == (1, cfg["output_len"], 1, 32, 32), flags
        assert torch.isfinite(out["pred"]).all(), flags

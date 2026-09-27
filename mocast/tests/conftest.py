"""Shared pytest fixtures for the MoCast test-suite."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def small_model_cfg(**overrides: Any) -> Dict[str, Any]:
    """A tiny but complete MoCast configuration (fast on CPU)."""
    cfg: Dict[str, Any] = {
        "name": "mocast",
        "input_len": 5,
        "output_len": 4,
        "target_size": [32, 32],
        "latent_channels": 16,
        "encoder": {"in_channels": 1, "base_channels": 8, "channels": [16, 16], "downsample": 2},
        "pmm": {
            "patch_size": 2,
            "window": 3,
            "heads": 2,
            "attn_dim": 8,
            "attn_scale": "sqrt",
            "potential": {"hidden": 16, "depth": 1, "kernel": "central", "padding": "replicate"},
            "wavelet": {"hidden": 8, "basis": "haar", "ll_mode": "zero", "enhance": "sigmoid"},
            "pyramid": {"mode": "separate", "kernels": [3, 5], "dilations": [1, 1], "channels": 12},
        },
        "msm": {"num_experts": 2, "kernels": [3, 5], "dilations": [1, 1], "channels": 16,
                "out_channels": 16, "expert_hidden": 16, "modulation_hidden": 8, "gate_hidden": 8},
        "temporal": {"embed_dim": 16, "num_blocks": 1},
        "prediction": {"mode": "oneshot", "spatial_blocks": 1},
        "advection": {"unit": "pixel", "semantics": "displacement", "mode": "bilinear",
                      "padding_mode": "border", "align_corners": True},
        "reconstruction": {"teacher_forcing": False},
        "ablation": {},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


@pytest.fixture(scope="session")
def device() -> torch.device:
    return DEVICE


@pytest.fixture()
def tiny_cfg() -> Dict[str, Any]:
    return small_model_cfg()


@pytest.fixture()
def tiny_model(tiny_cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    from mocast.models import MoCast

    model = MoCast(tiny_cfg).to(device)
    model.eval()
    return model


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "slow: long running tests (UT-08 overfit / dry runs)")

"""Config-building logic of the experiment runners (no training involved)."""

from __future__ import annotations

import os

import pytest

from mocast.tools.ablation import BASELINE_VARIANTS, _variant_config
from mocast.tools.sweep import _nested, sample_config
from mocast.utils.config import Config, load_config

CONFIG_ROOT = os.path.join(os.path.dirname(__file__), "..", "configs")


@pytest.fixture()
def l2_config() -> Config:
    return load_config(os.path.join(CONFIG_ROOT, "experiments", "synthetic_l2.yaml"))


@pytest.mark.parametrize("variant,flag", [
    ("A1", "helmholtz"), ("A2", "use_fluctuation"), ("A3", "decompose"),
    ("A5", "msm_gating"), ("A6", "use_source_sink"),
])
def test_ablation_overlay_switches_the_flag(l2_config: Config, variant: str, flag: str) -> None:
    cfg = _variant_config(l2_config, variant,
                          os.path.join(CONFIG_ROOT, "ablation"))
    assert cfg.model.ablation[flag] is False
    assert cfg.run.name == variant
    # geometry of the experiment config must survive the overlay
    assert list(cfg.model.target_size) == list(l2_config.model.target_size)
    assert cfg.model.input_len == l2_config.model.input_len
    assert cfg.model.output_len == l2_config.model.output_len
    assert cfg.dataset.height == l2_config.dataset.height


def test_ablation_a4_uses_a_single_expert(l2_config: Config) -> None:
    cfg = _variant_config(l2_config, "A4", os.path.join(CONFIG_ROOT, "ablation"))
    assert cfg.model.msm.num_experts == 1
    assert cfg.model.ablation.msm_multiscale is False


def test_baseline_variants_replace_the_model(l2_config: Config) -> None:
    cfg = _variant_config(l2_config, "B1", os.path.join(CONFIG_ROOT, "ablation"))
    assert cfg.model.name == "optical_flow"
    assert cfg.model.search_radius == BASELINE_VARIANTS["B1"]["model"]["search_radius"]
    assert cfg.model.target_size == l2_config.model.target_size


def test_variant_restores_full_model_defaults(l2_config: Config) -> None:
    """A0 must not silently inherit the *default* 128x128 geometry of model/mocast.yaml."""
    cfg = _variant_config(l2_config, "A0", os.path.join(CONFIG_ROOT, "ablation"))
    assert list(cfg.model.target_size) == [64, 64]
    assert cfg.model.ablation.use_mean is True


def test_unknown_variant_raises(l2_config: Config) -> None:
    with pytest.raises(FileNotFoundError):
        _variant_config(l2_config, "Z9", os.path.join(CONFIG_ROOT, "ablation"))


def test_sweep_sampling_is_nested_and_in_space() -> None:
    space = {"model.msm.num_experts": [1, 2, 3], "train.optimizer.lr": [1e-4, 2e-4],
             "model.pmm.wavelet.basis": ["haar", "db2"]}
    import numpy as np

    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(20):
        flat = sample_config(rng, space)
        nested = _nested(flat).to_dict()
        assert set(nested) == {"model", "train"}
        assert nested["model"]["msm"]["num_experts"] in (1, 2, 3)
        assert nested["model"]["pmm"]["wavelet"]["basis"] in ("haar", "db2")
        assert nested["train"]["optimizer"]["lr"] in (1e-4, 2e-4)
        seen.add(nested["model"]["msm"]["num_experts"])
    assert seen == {1, 2, 3}


def test_grid_sampling_is_deterministic() -> None:
    space = {"a.b": [1, 2], "a.c": ["x", "y"]}
    import numpy as np

    first = _nested(sample_config(np.random.default_rng(0), space, "grid")).to_dict()
    second = _nested(sample_config(np.random.default_rng(1), space, "grid")).to_dict()
    assert first == second


def test_search_space_file_covers_uncertain_parameters() -> None:
    space = Config.load(os.path.join(CONFIG_ROOT, "search", "space.yaml"))
    for key in ("model.pmm.patch_size", "model.pmm.attn_dim", "model.pmm.wavelet.basis",
                "model.msm.num_experts", "model.temporal.embed_dim", "loss.lambda_motion",
                "train.optimizer.lr"):
        assert key in space, f"search space is missing '{key}'"

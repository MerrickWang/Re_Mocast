"""Configuration system: defaults merging, CLI overrides, round-trip IO."""

from __future__ import annotations

import os

import pytest

from mocast.utils.config import Config, deep_update, load_config, parse_cli_overrides

CONFIG_ROOT = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_attribute_and_dotted_access() -> None:
    cfg = Config({"a": {"b": {"c": 3}}})
    assert cfg.a.b.c == 3
    assert cfg.get_path("a.b.c") == 3
    assert cfg.get_path("a.b.missing", "fallback") == "fallback"
    cfg.set_path("a.b.c", 5)
    assert cfg.a.b.c == 5


def test_deep_update_merges_nested_dicts() -> None:
    base = Config({"a": {"b": 1, "c": 2}})
    deep_update(base, {"a": {"c": 3}})
    assert base.a.b == 1 and base.a.c == 3


def test_parse_cli_overrides() -> None:
    overrides = parse_cli_overrides(["model.msm.num_experts=2", "train.amp=false",
                                     "loss.lambda_motion=0.005", "eval.thresholds=[1,2]"])
    assert overrides["model"]["msm"]["num_experts"] == 2
    assert overrides["train"]["amp"] is False
    assert overrides["loss"]["lambda_motion"] == pytest.approx(0.005)
    assert overrides["eval"]["thresholds"] == [1, 2]


def test_load_config_with_overrides() -> None:
    path = os.path.join(CONFIG_ROOT, "model", "mocast.yaml")
    cfg = load_config(path, overrides=["model.msm.num_experts=4"])
    assert cfg.model.msm.num_experts == 4
    assert cfg.model.encoder.downsample == 2


def test_defaults_key_merges_base_configs() -> None:
    path = os.path.join(CONFIG_ROOT, "ablation", "A4_msm_single_scale.yaml")
    cfg = load_config(path)
    assert cfg.model.name == "mocast"                 # from ../model/mocast.yaml
    assert cfg.model.msm.num_experts == 1             # from the ablation file
    assert cfg.run.name == "A4_msm_single_scale"
    assert cfg.model.ablation.msm_multiscale is False


def test_experiment_config_chain() -> None:
    path = os.path.join(CONFIG_ROOT, "train", "dry_run.yaml")
    cfg = load_config(path)
    assert cfg.train.epochs == 3
    assert cfg.train.optimizer.name == "adamw"        # inherited from default.yaml
    assert cfg.dataset.name == "synthetic"


def test_save_and_reload(tmp_path: object) -> None:
    cfg = load_config(os.path.join(CONFIG_ROOT, "model", "mocast.yaml"))
    out = os.path.join(str(tmp_path), "cfg.yaml")
    cfg.save(out)
    again = Config.load(out)
    assert again.to_dict() == cfg.to_dict()


def test_ablation_configs_are_consistent() -> None:
    """Every ablation file must load and only switch flags / documented knobs."""
    names = ["A0_full", "A1_no_helmholtz", "A2_no_wavelet", "A3_no_reynolds",
             "A4_msm_single_scale", "A5_uniform_gate", "A6_no_source_sink", "A7_impl_details"]
    for name in names:
        cfg = load_config(os.path.join(CONFIG_ROOT, "ablation", f"{name}.yaml"))
        ablation = cfg.model.ablation
        assert "helmholtz" in ablation or name.startswith("A0")
        assert cfg.run.name == name


@pytest.mark.parametrize("experiment,dataset_name,size", [
    ("sevir_mocast", "sevir", 128),
    ("meteonet_mocast", "meteonet", 128),
    ("shanghai_mocast", "shanghai", 128),
    ("synthetic_dryrun", "synthetic", 64),
])
def test_experiment_configs_are_internally_consistent(experiment: str, dataset_name: str,
                                                      size: int) -> None:
    """dataset / model / train sections must agree on names, sizes and thresholds."""
    cfg = load_config(os.path.join(CONFIG_ROOT, "experiments", f"{experiment}.yaml"))
    assert cfg.dataset.name == dataset_name
    assert cfg.model.name in ("mocast", "mocast_plus")
    assert cfg.train.epochs > 0
    # 128x128 everywhere except explicit dry runs (FR-DATA-03 + paper statement)
    target = cfg.model.get("target_size")
    if target is not None and dataset_name != "synthetic":
        assert list(target) == [128, 128]
    assert cfg.dataset.input_len == cfg.model.input_len
    assert cfg.dataset.target_len == cfg.model.output_len
    assert cfg.dataset.thresholds, "metric thresholds must be defined"
    assert cfg.dataset.motion_mask_threshold == min(cfg.dataset.thresholds)
    resize = cfg.dataset.get("resize")
    if resize is not None:
        assert list(resize) == [128, 128]


def test_dry_run_matches_the_small_geometry() -> None:
    cfg = load_config(os.path.join(CONFIG_ROOT, "experiments", "synthetic_dryrun.yaml"))
    assert cfg.dataset.height == cfg.model.target_size[0]
    assert cfg.dataset.width == cfg.model.target_size[1]
    assert cfg.model.encoder.downsample == 4

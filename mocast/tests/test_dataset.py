"""Dataset layer: windowing, temporal continuity, filtering, audit (section 3.1)."""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pytest
import torch

from mocast.datasets import build_dataset, build_dataloader, resolve_dataset_config
from mocast.datasets.base import Normalizer, collate_fn, crop_frame, resize_frame
from mocast.datasets.synthetic import generate_sequence
from mocast.utils.config import Config


def _synthetic_cfg(tmp_path: Any, **overrides: Any) -> Config:
    cfg = Config({
        "name": "synthetic",
        "n_sequences": 6,
        "frames_per_sequence": 25,
        "height": 32,
        "width": 32,
        "stride": 5,
        "input_len": 5,
        "target_len": 20,
        "thresholds": [0.2, 0.5],
        "filter": {"enabled": True, "min_intensity": 0.2, "min_rain_pixels": 4,
                   "min_rain_frames": 1},
        "split_config": {"mode": "ratio", "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                         "seed": 2026},
        "cache_dir": os.path.join(str(tmp_path), "cache"),
        "cache_index": False,
    })
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def test_fr_data_03_frame_contract(tmp_path: Any) -> None:
    """Every sample leaves the dataset as ``[C,128,128]``-shaped tensors."""
    cfg = _synthetic_cfg(tmp_path, resize=[16, 16])
    dataset = build_dataset(cfg, "train")
    item = dataset[0]
    assert item["input"].shape == (5, 1, 16, 16)
    assert item["target"].shape == (20, 1, 16, 16)
    assert item["input"].dtype == torch.float32
    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["input"].shape == (2, 5, 1, 16, 16)


def test_fr_data_02_temporal_continuity(tmp_path: Any) -> None:
    """Windows must be contiguous in time inside their own store."""
    dataset = build_dataset(_synthetic_cfg(tmp_path), "train")
    assert len(dataset) > 0
    for window in dataset.windows:
        store = dataset.stores[window.store]
        assert window.start >= 0
        assert window.start + window.length <= len(store)
        assert window.length == 5 + 20


def test_windows_do_not_cross_sequences(tmp_path: Any) -> None:
    dataset = build_dataset(_synthetic_cfg(tmp_path), "train")
    by_store: Dict[int, list] = {}
    for window in dataset.windows:
        by_store.setdefault(window.store, []).append(window.start)
    assert by_store, "no windows produced"
    for store, starts in by_store.items():
        assert max(starts) + 25 <= len(dataset.stores[store])


def test_physical_targets_are_denormalized(tmp_path: Any) -> None:
    dataset = build_dataset(_synthetic_cfg(tmp_path), "train")
    item = dataset[0]
    restored = dataset.normalizer.denormalize(item["target"].numpy())
    assert np.allclose(restored, item["target_phys"].numpy(), atol=1e-5)


def _store_names(dataset: Any) -> set:
    return {dataset.stores[w.store].name for w in dataset.windows}


def test_split_is_deterministic_and_disjoint(tmp_path: Any) -> None:
    cfg_a = _synthetic_cfg(tmp_path)
    cfg_b = _synthetic_cfg(tmp_path)
    train_a = build_dataset(cfg_a, "train")
    train_b = build_dataset(cfg_b, "train")
    assert [(w.store, w.start) for w in train_a.windows] == \
           [(w.store, w.start) for w in train_b.windows]
    # store *names* are stable across splits (indices are local to each split)
    train = _store_names(train_a)
    val = _store_names(build_dataset(cfg_b, "val"))
    test = _store_names(build_dataset(cfg_b, "test"))
    assert train and val and test
    assert not (train & val) and not (train & test) and not (val & test)


def test_fr_data_05_audit_records_filtering(tmp_path: Any) -> None:
    cfg = _synthetic_cfg(tmp_path, filter={"enabled": True, "min_intensity": 0.95,
                                          "min_rain_pixels": 64, "min_rain_frames": 25})
    dataset = build_dataset(cfg, "train")
    audit = dataset.audit
    assert audit["n_windows_total"] >= audit["n_windows_kept"]
    assert audit["n_windows_dropped_empty"] >= 0
    assert "filter" in audit and audit["filter"]["min_intensity"] == 0.95
    assert len(dataset) == audit["n_windows_kept"]


def test_dataloader_batches(tmp_path: Any) -> None:
    cfg = _synthetic_cfg(tmp_path)
    cfg["train"] = {"batch_size": 2, "num_workers": 0}
    loader = build_dataloader(cfg, "train")
    batch = next(iter(loader))
    assert batch["input"].shape[0] == 2
    assert batch["target"].shape[1] == 20


def test_resolve_dataset_config_defaults() -> None:
    cfg, info = resolve_dataset_config({"name": "sevir"})
    assert cfg["thresholds"] == [16, 74, 133, 160, 181, 219]
    assert cfg["normalization"]["vmax"] == 255.0
    assert cfg["motion_mask_threshold"] == 16
    assert cfg["input_len"] == 5 and cfg["target_len"] == 20
    assert info["frame_interval_minutes"] == 5


def test_crop_and_resize_helpers() -> None:
    frame = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    assert crop_frame(frame, {"mode": "center", "size": [4, 4]}).shape == (4, 4)
    assert resize_frame(frame, [4, 4]).shape == (4, 4)
    block = resize_frame(frame, [4, 4], mode="area")
    assert np.isclose(block[0, 0], frame[:2, :2].mean())
    bilinear = resize_frame(frame, [5, 5], mode="bilinear")
    assert bilinear.shape == (5, 5)


def test_synthetic_generator_matches_known_velocity() -> None:
    """The synthetic dataset exposes a ground-truth velocity field (mechanism tests)."""
    frames, velocity, meta = generate_sequence(n_frames=6, height=64, width=64,
                                               velocity_range=(1.0, 1.0),
                                               vorticity_range=(0.0, 0.0),
                                               wave_range=(0.0, 0.0),
                                               velocity_drift=0.0,
                                               source_strength=0.0, seed=0)
    assert frames.shape == (6, 64, 64)
    assert velocity.shape == (5, 2, 64, 64)
    assert np.allclose(velocity[0], 1.0)
    assert abs(meta["uniform_velocity"][0] - 1.0) < 1e-6
    # the centre of mass must move to the right
    def com(frame: np.ndarray) -> float:
        total = frame.sum()
        xs = (frame * np.arange(frame.shape[1])[None, :]).sum() / max(total, 1e-6)
        return float(xs)

    assert com(frames[3]) > com(frames[0])

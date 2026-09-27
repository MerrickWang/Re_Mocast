"""Shanghai timestamped-PNG adapter and physical-value conversion."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from mocast.datasets import build_dataset
from mocast.datasets.shanghai import build_stores
from mocast.datasets.stores import PngSequenceStore


def _write_frame(root: Path, timestamp: datetime, value: int, suffix: str = "") -> str:
    day = root / timestamp.strftime("%Y%m%d")
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"Z_RADR_I_Z9210_{timestamp:%Y%m%d%H%M%S}{suffix}_O_DOR_SA_CAP.bin.png"
    Image.fromarray(np.full((8, 8), value, dtype=np.uint8), mode="L").save(path)
    return str(path)


def test_png_store_converts_pixels_to_dbz(tmp_path: Path) -> None:
    path = _write_frame(tmp_path, datetime(2018, 7, 1), 255)
    store = PngSequenceStore([path], scale=70.0 / 255.0)
    assert store.shape == (1, 8, 8)
    assert np.allclose(store.read(0), 70.0)


def test_png_frames_are_split_at_time_gaps(tmp_path: Path) -> None:
    start = datetime(2018, 7, 1)
    paths: List[str] = [
        _write_frame(tmp_path, start + timedelta(minutes=6 * index), 255)
        for index in range(25)
    ]
    paths.extend([
        _write_frame(tmp_path, start + timedelta(minutes=6 * 25 + 20 + 6 * index), 128)
        for index in range(5)
    ])
    audit = {}
    stores, _ = build_stores({
        "files": paths,
        "continuity_seconds": [240, 480],
        "pixel_to_dbz": {"scale": 70.0 / 255.0, "offset": 0.0},
        "split_config": {"mode": "all"},
    }, "train", audit)
    assert [len(store) for store in stores] == [25, 5]
    assert audit["n_discontinuities"] == 1


def test_png_dataset_yields_physical_and_normalized_frames(tmp_path: Path) -> None:
    start = datetime(2018, 7, 1)
    paths = [_write_frame(tmp_path, start + timedelta(minutes=6 * index), 255)
             for index in range(25)]
    dataset = build_dataset({
        "name": "shanghai",
        "files": paths,
        "input_len": 5,
        "target_len": 20,
        "stride": 12,
        "resize": [4, 4],
        "resize_mode": "bilinear",
        "crop": {"mode": "none"},
        "sanitize": {"fill": 0.0, "clip": [0.0, 70.0]},
        "normalization": {"mode": "minmax", "vmin": 0.0, "vmax": 70.0},
        "filter": {"enabled": False},
        "cache_index": False,
        "pixel_to_dbz": {"scale": 70.0 / 255.0, "offset": 0.0},
        "split_config": {"mode": "all"},
    }, "train")
    item = dataset[0]
    assert item["input"].shape == (5, 1, 4, 4)
    assert item["target"].shape == (20, 1, 4, 4)
    assert np.allclose(item["input"].numpy(), 1.0)
    assert np.allclose(item["target_phys"].numpy(), 70.0)


def test_png_split_is_disjoint_by_day(tmp_path: Path) -> None:
    paths = [_write_frame(tmp_path, datetime(2018, 7, day), day)
             for day in range(1, 11)]
    cfg = {
        "files": paths,
        "split_config": {
            "mode": "ratio",
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "seed": 2026,
        },
    }
    split_days = {}
    for split in ("train", "val", "test"):
        stores, _ = build_stores(cfg, split)
        split_days[split] = {store.meta["day"] for store in stores}
    assert len(split_days["train"]) == 8
    assert len(split_days["val"]) == 1
    assert len(split_days["test"]) == 1
    assert not (split_days["train"] & split_days["val"])
    assert not (split_days["train"] & split_days["test"])
    assert not (split_days["val"] & split_days["test"])

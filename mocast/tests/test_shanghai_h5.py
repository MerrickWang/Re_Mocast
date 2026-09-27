import pickle

import h5py
import numpy as np
import pytest

from mocast.datasets import build_dataset
from mocast.datasets.shanghai import build_stores
from mocast.datasets.base import _index_signature


def test_grouped_h5_split_and_frames(tmp_path):
    path = tmp_path / "radar.h5"
    with h5py.File(path, "w") as handle:
        for group_name, count in (("train", 10), ("test", 3)):
            group = handle.create_group(group_name)
            group["all_len"] = count
            for index in range(count):
                group[str(index)] = np.broadcast_to(
                    np.arange(25, dtype=np.uint8)[:, None, None], (25, 8, 8))
    cfg = dict(name="shanghai", h5_path=str(path), h5_encoding="pixel_0_255",
               h5_val_fraction=0.2, cache_index=False, resize=[4, 4],
               resize_mode="bilinear", filter={"enabled": False})
    datasets = {split: build_dataset(cfg, split) for split in ("train", "val", "test")}
    assert [len(datasets[s]) for s in datasets] == [8, 2, 3]
    train_keys = {s.name for s in datasets["train"].stores}
    val_keys = {s.name for s in datasets["val"].stores}
    assert not train_keys & val_keys
    assert all(s.name.startswith("test/") for s in datasets["test"].stores)
    repeated, _ = build_stores(cfg, "train")
    assert [s.name for s in repeated] == [s.name for s in datasets["train"].stores]
    store = pickle.loads(pickle.dumps(repeated[0]))
    assert store.read(3).shape == (8, 8)
    assert np.allclose(store.read(3), 3 * 70 / 255)
    frames = datasets["train"].read_frames(datasets["train"].windows[0])
    assert frames.shape == (25, 4, 4)
    assert np.allclose(frames[:, 0, 0], np.arange(25) * 70 / 255)
    other, _ = build_stores({**cfg, "h5_encoding": "dbz"}, "train")
    assert np.allclose(other[0].read(3), 3)
    assert _index_signature(repeated, cfg) != _index_signature(other, cfg)
    with pytest.raises(ValueError, match="h5_encoding"):
        build_stores({**cfg, "h5_encoding": "unknown"}, "train")

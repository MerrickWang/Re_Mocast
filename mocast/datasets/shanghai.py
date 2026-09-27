"""Shanghai dataset adapter (China, 2015-2018, radar echo, 6 min, 0.01 deg).

The local archive is a hierarchy of timestamped, single-frame grayscale PNGs.
Frames are grouped by day, split into continuous acquisition runs, and only
then assigned to train/validation/test so adjacent frames cannot leak between
splits.  The earlier stacked NPY/NPZ layout remains supported.
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .split import assign_split
from .stores import FrameStore, MultiFrameStore, NpyStore, NpzStore, PngSequenceStore, H5SequenceStore

__all__ = ["build_stores", "context"]

NAME = "shanghai"

#: metric thresholds in dBZ, section 8.1
THRESHOLDS = [20, 30, 35, 40]


def context() -> Dict[str, Any]:
    return {
        "name": NAME,
        "region": "Shanghai (Pudong)",
        "years": "2015-2018",
        "variable": "radar echo",
        "unit": "dBZ",
        "frame_interval_minutes": 6,
        "spatial_resolution": "0.01 deg",
        "thresholds": THRESHOLDS,
        "default_normalization": {"mode": "minmax", "vmin": 0.0, "vmax": 70.0},
        "default_crop": {"mode": "none"},
        "default_resize": [128, 128],
    }


def _find_files(raw_root: str, patterns: Optional[Sequence[str]]) -> List[str]:
    patterns = list(patterns or ["**/*.png", "**/*.npy", "**/*.npz", "**/*.h5"])
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(raw_root, pattern), recursive=True))
    return sorted({os.path.abspath(f) for f in files})


def _parse_timestamp(path: str, pattern: str) -> datetime:
    match = re.search(pattern, os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot extract a 14-digit timestamp from Shanghai frame '{path}'")
    token = match.group(1) if match.groups() else match.group(0)
    return datetime.strptime(token, "%Y%m%d%H%M%S")


def _png_runs(files: Sequence[str], cfg: Dict[str, Any]) -> Tuple[Dict[str, List[List[str]]], int]:
    """Group PNGs by day and split each day at duplicates or acquisition gaps."""
    timestamp_pattern = str(cfg.get("timestamp_regex", r"(\d{14})"))
    interval = cfg.get("continuity_seconds", [240, 480])
    min_gap, max_gap = int(interval[0]), int(interval[1])
    by_day: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
    for path in files:
        timestamp = _parse_timestamp(path, timestamp_pattern)
        # Use the archive directory as the split unit.  Falling back to the
        # timestamp date also supports a flat directory layout.
        parent = os.path.basename(os.path.dirname(path))
        day = parent if re.fullmatch(r"\d{8}", parent) else timestamp.strftime("%Y%m%d")
        by_day[day].append((timestamp, path))

    runs_by_day: Dict[str, List[List[str]]] = {}
    n_discontinuities = 0
    for day, entries in sorted(by_day.items()):
        entries.sort(key=lambda item: (item[0], item[1]))
        runs: List[List[str]] = []
        current: List[str] = []
        previous: Optional[datetime] = None
        for timestamp, path in entries:
            delta = (timestamp - previous).total_seconds() if previous is not None else None
            if current and (delta is None or delta < min_gap or delta > max_gap):
                runs.append(current)
                current = []
                n_discontinuities += 1
            current.append(path)
            previous = timestamp
        if current:
            runs.append(current)
        runs_by_day[day] = runs
    return runs_by_day, n_discontinuities


def _build_h5_stores(cfg: Dict[str, Any], split: str,
                     audit: Optional[Dict[str, Any]]) -> Tuple[List[FrameStore], Dict[str, Any]]:
    import h5py

    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    path = os.path.abspath(str(cfg["h5_path"]))
    encoding = cfg.get("h5_encoding")
    if encoding not in ("pixel_0_255", "dbz"):
        raise ValueError("Set h5_encoding explicitly to pixel_0_255 or dbz")
    scale = 70.0 / 255.0 if encoding == "pixel_0_255" else 1.0
    fraction = float(cfg.get("h5_val_fraction", 0.1))
    if not 0 < fraction < 1:
        raise ValueError("h5_val_fraction must be between 0 and 1")
    with h5py.File(path, "r") as handle:
        group_name = "test" if split == "test" else "train"
        if group_name not in handle or not isinstance(handle[group_name], h5py.Group):
            raise ValueError(f"Missing H5 group: {group_name}")
        group = handle[group_name]
        keys = sorted(k for k in group if k != "all_len")
        for key in keys:
            obj = group[key]
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3 or obj.dtype.kind not in "uif":
                raise ValueError(f"Expected numeric [T,H,W] dataset: {group_name}/{key}")
        if "all_len" in group and int(group["all_len"][()]) != len(keys):
            raise ValueError(f"all_len does not match sequence count in {group_name}")
        if split != "test":
            if len(keys) < 2:
                raise ValueError("At least two train sequences are needed for validation")
            selected = assign_split(keys, {
                "mode": "ratio", "seed": int(cfg.get("h5_split_seed", 2026)),
                "ratios": {"train": 1 - fraction, "val": fraction},
            }, split)
            if not selected or len(selected) == len(keys):
                raise ValueError("Validation fraction produces an empty train/val partition")
        else:
            selected = set(keys)
        stores = [H5SequenceStore(path, f"{group_name}/{key}", group[key].shape, scale,
                                 meta={"split": split, "source_group": group_name,
                                       "encoding": encoding}) for key in keys if key in selected]
    info = context()
    info.update(h5_encoding=encoding, h5_split_seed=int(cfg.get("h5_split_seed", 2026)),
                h5_val_fraction=fraction, source_group=group_name,
                n_sequences_selected=len(stores))
    if audit is not None:
        audit.update(info)
    return stores, info


def build_stores(cfg: Dict[str, Any], split: str,
                 audit: Optional[Dict[str, Any]] = None) -> Tuple[List[FrameStore], Dict[str, Any]]:
    if cfg.get("h5_path"):
        return _build_h5_stores(cfg, split, audit)
    info = context()
    raw_root = cfg.get("raw_root")
    files: List[str] = [str(f) for f in cfg.get("files", [])]
    if not files:
        if not raw_root or not os.path.exists(str(raw_root)):
            raise FileNotFoundError(
                f"Shanghai raw_root '{raw_root}' not found. Provide the radar archive "
                "(Chen et al. 2020) or use `dataset.name=synthetic`."
            )
        files = _find_files(str(raw_root), cfg.get("file_patterns"))
    if not files:
        raise FileNotFoundError("No Shanghai frame files found")

    suffixes = {os.path.splitext(path)[1].lower() for path in files}
    if ".png" in suffixes and len(suffixes) != 1:
        raise ValueError(f"Mixed Shanghai PNG/array archives are not supported: {sorted(suffixes)}")

    if suffixes == {".png"}:
        runs_by_day, n_discontinuities = _png_runs(files, cfg)
        # Split whole days, not individual frames, to prevent neighbouring
        # frames from leaking across train/validation/test.
        split_days = assign_split(runs_by_day, cfg.get("split_config"), split)
        scale = float(cfg.get("pixel_to_dbz", {}).get("scale", 70.0 / 255.0))
        offset = float(cfg.get("pixel_to_dbz", {}).get("offset", 0.0))
        stores: List[FrameStore] = []
        for day in sorted(split_days):
            for run_index, paths in enumerate(runs_by_day[day]):
                first = _parse_timestamp(paths[0], str(cfg.get("timestamp_regex", r"(\d{14})")))
                last = _parse_timestamp(paths[-1], str(cfg.get("timestamp_regex", r"(\d{14})")))
                name = f"{day}_run{run_index:03d}_{first:%H%M%S}_{last:%H%M%S}"
                stores.append(PngSequenceStore(
                    paths, name=name, scale=scale, offset=offset,
                    meta={"split": split, "day": day,
                          "start_time": first.isoformat(), "end_time": last.isoformat()},
                ))
        if audit is not None:
            audit.update({"dataset": NAME, "split": split, "n_files": len(files),
                          "n_days": len(runs_by_day), "n_days_selected": len(split_days),
                          "n_sequences": sum(len(runs) for runs in runs_by_day.values()),
                          "n_sequences_selected": len(stores),
                          "n_discontinuities": n_discontinuities,
                          "pixel_to_dbz_scale": scale, "pixel_to_dbz_offset": offset})
        return stores, info

    npz_key = cfg.get("npz_key")
    split_keys = assign_split([os.path.basename(f) for f in files], cfg.get("split_config"), split)
    selected = [f for f in files if os.path.basename(f) in split_keys]

    stores: List[FrameStore] = []
    for path in selected:
        name = os.path.basename(path)
        if path.endswith(".npy"):
            store: FrameStore = NpyStore(path, name=name, meta={"split": split})
        elif path.endswith(".npz"):
            store = NpzStore(path, key=npz_key, name=name, meta={"split": split})
        else:
            store = MultiFrameStore([path], name=name, npz_key=npz_key, meta={"split": split})
        stores.append(store)

    if audit is not None:
        audit.update({"dataset": NAME, "split": split, "n_files": len(files),
                      "n_files_selected": len(selected)})
    return stores, info

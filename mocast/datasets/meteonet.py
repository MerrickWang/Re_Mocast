"""MeteoNet dataset adapter (France, 2016-2018, radar reflectivity, 5 min, 0.01 deg).

The public MeteoNet release stores one archive per timestep.  The adapter accepts

* a directory tree of ``.npz``/``.npy`` single-timestep files (``layout: files``),
* stacked ``.npy`` sequences (``layout: sequences``),
* or an explicit ``files`` list.

Frames are grouped into contiguous stores (one per file / per day) so that
windows never cross gaps; the region crop and the exact split remain
configurable (open question #1).
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .split import assign_split
from .stores import ArrayStore, FrameStore, MultiFrameStore, NpyStore, NpzStore

__all__ = ["build_stores", "context"]

NAME = "meteonet"

#: metric thresholds in dBZ, section 8.1
THRESHOLDS = [12, 18, 24, 32]


def context() -> Dict[str, Any]:
    return {
        "name": NAME,
        "region": "France (NW / SE)",
        "years": "2016-2018",
        "variable": "radar reflectivity",
        "unit": "dBZ",
        "frame_interval_minutes": 5,
        "spatial_resolution": "0.01 deg",
        "thresholds": THRESHOLDS,
        "default_normalization": {"mode": "minmax", "vmin": -10.0, "vmax": 70.0},
        "default_crop": {"mode": "center", "size": [256, 256]},
        "default_resize": [128, 128],
    }


def _find_files(raw_root: str, patterns: Optional[Sequence[str]]) -> List[str]:
    patterns = list(patterns or ["**/*.npz", "**/*.npy"])
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(raw_root, pattern), recursive=True))
    return sorted({os.path.abspath(f) for f in files})


def build_stores(cfg: Dict[str, Any], split: str,
                 audit: Optional[Dict[str, Any]] = None) -> Tuple[List[FrameStore], Dict[str, Any]]:
    info = context()
    raw_root = cfg.get("raw_root")
    layout = str(cfg.get("layout", "files")).lower()
    files: List[str] = [str(f) for f in cfg.get("files", [])]
    if not files:
        if not raw_root or not os.path.exists(str(raw_root)):
            raise FileNotFoundError(
                f"MeteoNet raw_root '{raw_root}' not found. Provide the downloaded "
                "MeteoNet radar archives or use `dataset.name=synthetic`."
            )
        files = _find_files(str(raw_root), cfg.get("file_patterns"))
    if not files:
        raise FileNotFoundError("No MeteoNet frame files found")

    npz_key = cfg.get("npz_key")
    split_keys = assign_split([os.path.basename(f) for f in files], cfg.get("split_config"), split)
    selected = [f for f in files if os.path.basename(f) in split_keys]

    stores: List[FrameStore] = []
    for path in selected:
        frames_per_file = int(cfg.get("frames_per_file", 0))
        if layout == "sequences":
            store: FrameStore = (
                NpyStore(path, name=os.path.basename(path), meta={"split": split})
                if path.endswith(".npy")
                else NpzStore(path, key=npz_key, name=os.path.basename(path), meta={"split": split})
            )
        elif layout == "day":
            store = MultiFrameStore([path], name=os.path.basename(path), npz_key=npz_key,
                                    meta={"split": split})
        else:  # one frame per file
            store = MultiFrameStore([path], name=os.path.basename(path), npz_key=npz_key,
                                    meta={"split": split})
        if frames_per_file and len(store) > frames_per_file:
            store = MultiFrameStore([path], name=os.path.basename(path), npz_key=npz_key,
                                    meta={"split": split})
        stores.append(store)

    if audit is not None:
        audit.update({"dataset": NAME, "split": split, "n_files": len(files),
                      "n_files_selected": len(selected), "layout": layout})
    return stores, info

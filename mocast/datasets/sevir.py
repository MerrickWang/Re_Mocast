"""SEVIR dataset adapter (USA, 2017-2019, VIL, 5 min, ~1 km).

Official event files (``SEVIR_VIL_*.h5``) store
``vil`` with shape ``[N, 49, 384, 384]`` (49 frames at 5 min = 4 h), ``id`` and
``time_utc``.  Every event is exposed as an independent store so that windows
never cross event boundaries, and the split is performed over events.

Preprocessing follows DiffCast (paper: "All datasets are preprocessed following
(Yu et al. 2024), including cropping, normalization, and resizing to 128x128").
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .split import assign_split
from .stores import ArrayStore, FrameStore, H5Store, NpyStore, MultiFrameStore

__all__ = ["build_stores", "context"]

NAME = "sevir"

#: metric thresholds in VIL units (kg/m^2), section 8.1
THRESHOLDS = [16, 74, 133, 160, 181, 219]


def context() -> Dict[str, Any]:
    return {
        "name": NAME,
        "region": "USA (nation-wide)",
        "years": "2017-2019",
        "variable": "VIL (vertically integrated liquid)",
        "unit": "kg/m^2 (0-255)",
        "frame_interval_minutes": 5,
        "spatial_resolution": "~1 km",
        "thresholds": THRESHOLDS,
        "default_normalization": {"mode": "minmax", "vmin": 0.0, "vmax": 255.0},
        "default_crop": {"mode": "center", "size": [384, 384]},
        "default_resize": [128, 128],
        "event_frames": 49,
    }


def _find_event_files(raw_root: str, patterns: Optional[Sequence[str]] = None) -> List[str]:
    patterns = list(patterns or ["SEVIR_VIL*.h5", "**/SEVIR_VIL*.h5", "*.h5", "*.hdf5"])
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(raw_root, pattern), recursive=True))
    return sorted({os.path.abspath(f) for f in files})


def _event_ids(path: str, key: str, n_events: int) -> List[str]:
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            if "id" in handle:
                raw = handle["id"][:n_events]
                ids = []
                for value in raw:
                    ids.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
                return ids
    except Exception:  # pragma: no cover - best effort metadata
        pass
    return [f"{os.path.basename(path)}#{i:05d}" for i in range(n_events)]


def _n_events(path: str, key: str, frames_per_event: int) -> int:
    import h5py

    with h5py.File(path, "r") as handle:
        data = handle[key]
        total = int(data.shape[0]) if data.ndim >= 3 else 1
        if data.ndim == 3:  # [T,H,W] single sequence
            total = 1
        if data.ndim == 4:
            total = int(data.shape[0]) * int(data.shape[1]) // frames_per_event
        return max(total, 1)


def build_stores(cfg: Dict[str, Any], split: str,
                 audit: Optional[Dict[str, Any]] = None) -> Tuple[List[FrameStore], Dict[str, Any]]:
    """Build SEVIR stores restricted to ``split``."""
    info = context()
    raw_root = cfg.get("raw_root")
    if not raw_root or not os.path.exists(str(raw_root)):
        raise FileNotFoundError(
            f"SEVIR raw_root '{raw_root}' not found. Download the official SEVIR VIL "
            "event files or use the synthetic dataset for dry runs "
            "(`dataset.name=synthetic`)."
        )
    h5_key = str(cfg.get("h5_key", "vil"))
    frames_per_event = int(cfg.get("frames_per_event", info["event_frames"]))
    files = list(cfg.get("files", [])) or _find_event_files(str(raw_root), cfg.get("file_patterns"))
    if not files:
        raise FileNotFoundError(f"No SEVIR VIL files found under '{raw_root}'")

    # ---- enumerate events ------------------------------------------------
    events: List[Dict[str, Any]] = []
    for path in files:
        n_events = _n_events(path, h5_key, frames_per_event)
        ids = _event_ids(path, h5_key, n_events)
        for i in range(n_events):
            events.append({"path": path, "index": i, "event_id": ids[i]})

    split_keys = assign_split([e["event_id"] for e in events], cfg.get("split_config"), split)
    selected = [e for e in events if e["event_id"] in split_keys]

    stores: List[FrameStore] = []
    for event in selected:
        offset = event["index"] * frames_per_event
        stores.append(
            H5Store(
                event["path"],
                key=h5_key,
                name=f"{os.path.basename(event['path'])}:{event['index']}",
                frame_offset=offset,
                frame_count=frames_per_event,
                meta={"event_id": event["event_id"], "split": split,
                      "frames_per_group": frames_per_event},
            )
        )
    if audit is not None:
        audit.update({"dataset": NAME, "split": split, "n_files": len(files),
                      "n_events_total": len(events), "n_events_selected": len(selected),
                      "event_ids": [e["event_id"] for e in selected][:1000]})
    return stores, info

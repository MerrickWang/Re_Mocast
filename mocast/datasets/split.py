"""Deterministic split assignment shared by the dataset adapters.

The official train/val/test partitions are not part of the public material
(risk R6 / open question #1), therefore the split is always configurable and is
written to a manifest next to the data statistics (FR-DATA-04).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from ..utils.misc import ensure_dir

__all__ = ["assign_split", "save_split_manifest", "load_split_manifest"]


def _bucket(key: str, seed: int, ratios: Sequence[float]) -> int:
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(0xFFFFFFFF)
    cum = 0.0
    for i, ratio in enumerate(ratios):
        cum += float(ratio)
        if value <= cum:
            return i
    return len(ratios) - 1


def assign_split(keys: Iterable[str], split_cfg: Optional[Dict[str, Any]], split: str) -> Set[str]:
    """Return the keys belonging to ``split``.

    Supported modes:

    ``ratio``   (default) deterministic shuffle of the sorted keys with ``seed``,
                then a cumulative ratio cut - the same call for every split of the
                same key set yields a consistent partition.
    ``hash``    stable per-key hash bucketing (used when a streaming/partial
                enumeration is required).
    ``explicit````assignments`` maps every key to a split name.
    ``all``     every key belongs to every split (debug / dry runs).
    """
    keys = sorted({str(k) for k in keys})
    split_cfg = dict(split_cfg or {})
    mode = str(split_cfg.get("mode", "ratio")).lower()
    ratios_map: Dict[str, float] = dict(split_cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1}))
    order = list(ratios_map.keys())
    ratios = [float(ratios_map[k]) for k in order]
    total = sum(ratios)
    ratios = [r / total for r in ratios]
    seed = int(split_cfg.get("seed", 2026))

    if mode == "all":
        return set(keys)
    if mode == "explicit":
        assignments = {str(k): str(v) for k, v in dict(split_cfg.get("assignments", {})).items()}
        return {k for k in keys if assignments.get(k) == split}
    if mode == "hash":
        idx = order.index(split) if split in order else None
        if idx is None:
            return set()
        return {k for k in keys if _bucket(k, seed, ratios) == idx}
    if mode == "ratio":
        rng = np.random.default_rng(seed)
        shuffled = list(keys)
        rng.shuffle(shuffled)  # type: ignore[arg-type]
        n = len(shuffled)
        counts = [int(round(r * n)) for r in ratios]
        counts[-1] = n - sum(counts[:-1])
        start = 0
        for name, count in zip(order, counts):
            chunk = set(shuffled[start:start + count])
            start += count
            if name == split:
                return chunk
        return set()
    raise ValueError(f"Unknown split mode '{mode}'")


def save_split_manifest(path: str, payload: Dict[str, Any]) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return path


def load_split_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

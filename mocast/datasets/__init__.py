"""Unified dataset entry points (FR-DATA-01 / FR-DATA-04).

``build_dataset(cfg, split)`` returns a :class:`PrecipSequenceDataset` for any of
the three adapted datasets or for the synthetic generator::

    ds_cfg = cfg.dataset
    train_ds = build_dataset(cfg, "train")
    loader = build_dataloader(cfg, "train", shuffle=True)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..utils.config import Config
from . import meteonet, sevir, shanghai, synthetic
from .base import Normalizer, PrecipSequenceDataset, collate_fn

__all__ = [
    "ADAPTERS",
    "build_dataset",
    "build_dataloader",
    "resolve_dataset_config",
    "PrecipSequenceDataset",
    "Normalizer",
    "collate_fn",
]

ADAPTERS = {
    sevir.NAME: sevir,
    meteonet.NAME: meteonet,
    shanghai.NAME: shanghai,
    synthetic.NAME: synthetic,
}


def resolve_dataset_config(dataset_cfg: Dict[str, Any]) -> Tuple[Config, Dict[str, Any]]:
    """Merge the user config with the adapter defaults (thresholds, crop, norm)."""
    cfg = Config(dataset_cfg)
    name = str(cfg.get("name", "synthetic")).lower()
    if name not in ADAPTERS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(ADAPTERS)}")
    info = ADAPTERS[name].context()
    cfg.setdefault("thresholds", info["thresholds"])
    cfg.setdefault("normalization", info["default_normalization"])
    cfg.setdefault("crop", info["default_crop"])
    if info.get("default_resize") is not None:
        cfg.setdefault("resize", info["default_resize"])
    cfg.setdefault("input_len", 5)
    cfg.setdefault("target_len", 20)
    cfg.setdefault("stride", None)
    if cfg.get("stride") is None:
        cfg["stride"] = int(cfg["input_len"]) + int(cfg["target_len"]) // 2
    cfg.setdefault("motion_mask_threshold", min(cfg["thresholds"]) if cfg["thresholds"] else 0.0)
    cfg.setdefault("cache_index", True)
    cfg.setdefault("cache_dir", "outputs/cache")
    cfg.setdefault("return_physical", True)
    cfg.setdefault("sanitize", {"fill": 0.0, "clip": None})
    cfg.setdefault("filter", {"enabled": True, "min_intensity": cfg["motion_mask_threshold"],
                              "min_rain_pixels": 8, "min_rain_frames": 1,
                              "max_empty_ratio": 1.0})
    split_cfg = dict(cfg.get("split_config", {}) or {})
    split_cfg.setdefault("seed", int(cfg.get("seed", 2026)))
    cfg["split_config"] = split_cfg
    return cfg, info


def build_dataset(cfg: Any, split: Optional[str] = None) -> PrecipSequenceDataset:
    """Build the dataset for ``split`` from the ``dataset`` section of ``cfg``."""
    dataset_cfg = cfg.get("dataset", cfg) if isinstance(cfg, dict) else cfg
    dataset_cfg = Config(dataset_cfg)
    split = str(split or dataset_cfg.get("split", "train"))
    resolved, info = resolve_dataset_config(dataset_cfg)
    resolved["split"] = split
    name = str(resolved["name"]).lower()
    adapter = ADAPTERS[name]
    audit: Dict[str, Any] = {}
    stores, context = adapter.build_stores(resolved, split, audit)
    merged = {**info, **audit, **{k: v for k, v in context.items() if k != "name"}}
    dataset = PrecipSequenceDataset(stores, resolved, split=split, name=name, context=merged)
    dataset.info = merged  # type: ignore[attr-defined]
    return dataset


def build_dataloader(cfg: Any, split: str = "train", shuffle: Optional[bool] = None,
                     batch_size: Optional[int] = None, num_workers: Optional[int] = None,
                     drop_last: Optional[bool] = None, dataset: Optional[PrecipSequenceDataset] = None) -> Any:
    """Wrap :func:`build_dataset` in a ``torch.utils.data.DataLoader``."""
    import torch
    from torch.utils.data import DataLoader

    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    dataset = dataset or build_dataset(cfg, split)
    batch_size = int(batch_size or train_cfg.get("batch_size", 6))
    num_workers = int(train_cfg.get("num_workers", 0) if num_workers is None else num_workers)
    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train" and len(dataset) > batch_size
    generator = torch.Generator()
    generator.manual_seed(int(train_cfg.get("seed", 2026)))

    def _worker_init(worker_id: int) -> None:  # pragma: no cover - worker side
        import random

        import numpy as np

        seed = (torch.initial_seed() + worker_id) % 2**32
        np.random.seed(seed)
        random.seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle),
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=bool(drop_last),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        persistent_workers=bool(num_workers > 0),
        worker_init_fn=_worker_init,
        generator=generator,
    )

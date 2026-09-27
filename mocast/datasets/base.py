"""Dataset agnostic windowing, preprocessing, filtering and audit logic.

Implements the functional requirements of section 3.1:

* FR-DATA-01 unified ``Dataset``/``Loader`` layer over the three adapters,
* FR-DATA-02 sliding windows of ``T_in=5`` inputs and ``T_out=20`` targets,
* FR-DATA-03 every frame leaves the dataset as ``[C,128,128]`` while preserving
  the physical meaning of the metric thresholds (round-trip checked by tests),
* FR-DATA-04 split / crop / normalisation are fully configuration driven and a
  split manifest + statistics file is emitted for every run,
* FR-DATA-05 missing values, outliers and empty-precipitation samples are
  handled explicitly and the number of filtered windows is recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.misc import ensure_dir, save_json
from .stores import FrameStore

__all__ = [
    "Window",
    "Normalizer",
    "PrecipSequenceDataset",
    "build_index",
    "crop_frame",
    "resize_frame",
    "sanitize_frame",
    "preprocess_frame",
    "collate_fn",
]


# --------------------------------------------------------------------------- #
#  geometry / value preprocessing
# --------------------------------------------------------------------------- #
def crop_frame(frame: np.ndarray, crop: Optional[Dict[str, Any]]) -> np.ndarray:
    """Crop a ``[H,W]`` frame.

    ``crop`` accepts either ``{"mode": "center", "size": [h, w]}`` or an explicit
    ``{"top": int, "left": int, "height": int, "width": int}`` box.
    """
    if not crop:
        return frame
    h, w = frame.shape[-2:]
    mode = str(crop.get("mode", "center")).lower()
    if mode == "none":
        return frame
    if mode == "center":
        ch, cw = crop.get("size", [h, w])
        ch, cw = min(int(ch), h), min(int(cw), w)
        top = (h - ch) // 2
        left = (w - cw) // 2
    else:
        top, left = int(crop.get("top", 0)), int(crop.get("left", 0))
        ch, cw = int(crop.get("height", h - top)), int(crop.get("width", w - left))
    return frame[..., top:top + ch, left:left + cw]


def _resize_bilinear(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = frame.shape[-2:]
    H, W = int(size[0]), int(size[1])
    ys = (np.arange(H, dtype=np.float32) + 0.5) * (h / H) - 0.5
    xs = (np.arange(W, dtype=np.float32) + 0.5) * (w / W) - 0.5
    y0 = np.clip(np.floor(ys), 0, h - 1).astype(np.int64)
    x0 = np.clip(np.floor(xs), 0, w - 1).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(ys - y0, 0.0, 1.0).astype(np.float32)[:, None]
    wx = np.clip(xs - x0, 0.0, 1.0).astype(np.float32)[None, :]
    a = frame[..., y0[:, None], x0[None, :]]
    b = frame[..., y0[:, None], x1[None, :]]
    c = frame[..., y1[:, None], x0[None, :]]
    d = frame[..., y1[:, None], x1[None, :]]
    top_row = a * (1.0 - wx) + b * wx
    bottom_row = c * (1.0 - wx) + d * wx
    return (top_row * (1.0 - wy) + bottom_row * wy).astype(np.float32)


def resize_frame(frame: np.ndarray, size: Optional[Sequence[int]], mode: str = "area") -> np.ndarray:
    """Resize a ``[..., H, W]`` array.

    ``mode="area"`` (default) performs exact block averaging whenever the
    down-scaling factor is integral (e.g. the SEVIR 384 -> 128 case) and falls
    back to bilinear interpolation otherwise.  Filtering the frames before the
    resize keeps the physical thresholds meaningful (FR-DATA-03).
    """
    if size is None:
        return frame
    H, W = int(size[0]), int(size[1])
    h, w = frame.shape[-2:]
    if (h, w) == (H, W):
        return frame
    if mode == "area" and h % H == 0 and w % W == 0:
        fh, fw = h // H, w // W
        reshaped = frame.reshape(*frame.shape[:-2], H, fh, W, fw)
        return reshaped.mean(axis=(-3, -1)).astype(np.float32)
    if mode == "nearest":
        ys = np.clip((np.arange(H) + 0.5) * (h / H), 0, h - 1).astype(np.int64)
        xs = np.clip((np.arange(W) + 0.5) * (w / W), 0, w - 1).astype(np.int64)
        return frame[..., ys[:, None], xs[None, :]].astype(np.float32)
    if mode == "bilinear":
        return _resize_bilinear(frame.astype(np.float32), (H, W))
    raise ValueError(f"Unknown resize mode '{mode}'")


def sanitize_frame(frame: np.ndarray, fill: float = 0.0,
                   clip: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, int]:
    """Replace non-finite values / outliers; returns ``(frame, n_repaired)``."""
    frame = np.asarray(frame, dtype=np.float32)
    invalid = ~np.isfinite(frame)
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        invalid = invalid | (frame < lo) | (frame > hi)
    n_repaired = int(invalid.sum())
    if n_repaired:
        frame = np.where(invalid, np.float32(fill), frame)
    if clip is not None:
        frame = np.clip(frame, float(clip[0]), float(clip[1]))
    return frame, n_repaired


def preprocess_frame(frame: np.ndarray, crop: Optional[Dict[str, Any]], resize: Optional[Sequence[int]],
                     resize_mode: str, fill: float, clip: Optional[Sequence[float]]) -> Tuple[np.ndarray, int]:
    frame, repaired = sanitize_frame(frame, fill=fill, clip=clip)
    frame = crop_frame(frame, crop)
    frame = resize_frame(frame, resize, mode=resize_mode)
    return np.ascontiguousarray(frame, dtype=np.float32), repaired


# --------------------------------------------------------------------------- #
#  normalisation
# --------------------------------------------------------------------------- #
class Normalizer:
    """Configurable value scaling; keeps the physical mapping invertible."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(cfg or {})
        self.mode = str(cfg.get("mode", "minmax")).lower()
        self.vmin = float(cfg.get("vmin", 0.0))
        self.vmax = float(cfg.get("vmax", 1.0))
        self.mean = float(cfg.get("mean", 0.0))
        self.std = float(cfg.get("std", 1.0)) or 1.0
        self.percentile = cfg.get("percentile", None)
        if self.mode == "minmax" and self.vmax == self.vmin:
            raise ValueError("Normalizer: vmax must differ from vmin")

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        return self._scale(frame, forward=True)

    def denormalize(self, frame: np.ndarray) -> np.ndarray:
        return self._scale(frame, forward=False)

    def _scale(self, frame: Any, forward: bool) -> Any:
        """Works for numpy arrays *and* torch tensors (keeps the device)."""
        if hasattr(frame, "detach"):  # torch tensor
            import torch

            x = frame
            if self.mode == "none":
                return x
            if self.mode == "minmax":
                delta = self.vmax - self.vmin
                return (x - self.vmin) / delta if forward else x * delta + self.vmin
            if self.mode in ("zscore", "meanstd", "standard"):
                return (x - self.mean) / self.std if forward else x * self.std + self.mean
            raise ValueError(f"Unknown normalisation mode '{self.mode}'")
        frame = np.asarray(frame, dtype=np.float32)
        if self.mode == "none":
            return frame
        if self.mode == "minmax":
            delta = self.vmax - self.vmin
            return (frame - self.vmin) / delta if forward else frame * delta + self.vmin
        if self.mode in ("zscore", "meanstd", "standard"):
            return (frame - self.mean) / self.std if forward else frame * self.std + self.mean
        raise ValueError(f"Unknown normalisation mode '{self.mode}'")

    # convenience for metric code that works in physical units
    def to_normalized_threshold(self, threshold: float) -> float:
        return float(self.normalize(np.asarray([threshold], dtype=np.float32))[0])

    def to_physical(self, value: float) -> float:
        return float(self.denormalize(np.asarray([value], dtype=np.float32))[0])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "vmin": self.vmin, "vmax": self.vmax,
            "mean": self.mean, "std": self.std,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Normalizer({self.state_dict()})"


# --------------------------------------------------------------------------- #
#  window index
# --------------------------------------------------------------------------- #
@dataclass
class Window:
    """A ``T_in + T_out`` slice inside a single store."""

    store: int
    start: int
    length: int
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.store}:{self.start}:{self.length}"


def _scan_stats(store: FrameStore, cfg: Dict[str, Any], preprocess_kwargs: Dict[str, Any],
                threshold: float, max_frames: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
    """Count rainy pixels per frame (used for filtering, FR-DATA-05)."""
    n = len(store) if max_frames is None else min(len(store), int(max_frames))
    counts = np.zeros(n, dtype=np.int64)
    repaired_total = 0
    nan_frames = 0
    for i in range(n):
        frame, repaired = preprocess_frame(store.read(i), **preprocess_kwargs)
        repaired_total += repaired
        if repaired:
            nan_frames += 1
        counts[i] = int((frame > threshold).sum())
    return counts, repaired_total, nan_frames


def build_index(stores: Sequence[FrameStore], cfg: Dict[str, Any],
                normalizer: Normalizer) -> Tuple[List[Window], Dict[str, Any]]:
    """Build the list of valid windows plus a full audit record."""
    data_cfg = cfg
    input_len = int(data_cfg.get("input_len", 5))
    target_len = int(data_cfg.get("target_len", 20))
    length = input_len + target_len
    stride = int(data_cfg.get("stride", length // 2)) or length
    filt = dict(data_cfg.get("filter", {}) or {})
    preprocess_kwargs = dict(
        crop=data_cfg.get("crop"),
        resize=data_cfg.get("resize"),
        resize_mode=str(data_cfg.get("resize_mode", "area")),
        fill=float(data_cfg.get("sanitize", {}).get("fill", 0.0)),
        clip=data_cfg.get("sanitize", {}).get("clip"),
    )
    threshold = float(filt.get("min_intensity", data_cfg.get("mask_threshold", 0.0)))
    min_rain_pixels = int(filt.get("min_rain_pixels", 0))
    min_rain_frames = int(filt.get("min_rain_frames", 1 if filt.get("enabled", True) else 0))
    max_empty_ratio = float(filt.get("max_empty_ratio", 1.0))
    enabled = bool(filt.get("enabled", True))
    scan_limit = filt.get("max_scan_frames", None)

    windows: List[Window] = []
    audit: Dict[str, Any] = {
        "stores": [],
        "n_windows_total": 0,
        "n_windows_kept": 0,
        "n_windows_dropped_empty": 0,
        "n_frames_repaired": 0,
        "n_frames_with_repairs": 0,
        "filter": {"enabled": enabled, "min_intensity": threshold,
                   "min_rain_pixels": min_rain_pixels, "min_rain_frames": min_rain_frames,
                   "max_empty_ratio": max_empty_ratio},
        "window": {"input_len": input_len, "target_len": target_len, "stride": stride,
                   "frame_length": length},
    }
    for si, store in enumerate(stores):
        n_frames = len(store)
        store_audit: Dict[str, Any] = {
            "index": si, "name": store.name, "kind": store.kind,
            "frames": int(n_frames), "meta": {k: v for k, v in store.meta.items()
                                              if isinstance(v, (int, float, str, bool))},
        }
        if n_frames < length:
            store_audit.update(kept=0, dropped="too_short")
            audit["stores"].append(store_audit)
            continue
        if enabled and (min_rain_pixels > 0 or min_rain_frames > 0 or max_empty_ratio < 1.0):
            counts, repaired, nan_frames = _scan_stats(store, data_cfg, preprocess_kwargs,
                                                       threshold, scan_limit)
            audit["n_frames_repaired"] += repaired
            audit["n_frames_with_repairs"] += nan_frames
            store_audit["nan_frames"] = nan_frames
            store_audit["mean_rain_pixels"] = float(counts.mean()) if len(counts) else 0.0
        else:
            counts = None
        kept = 0
        starts = list(range(0, n_frames - length + 1, stride))
        for start in starts:
            audit["n_windows_total"] += 1
            if counts is not None:
                window_counts = counts[start:start + length]
                rainy_frames = int((window_counts >= max(min_rain_pixels, 1)).sum())
                empty_ratio = float((window_counts == 0).mean())
                if rainy_frames < min_rain_frames or empty_ratio > max_empty_ratio:
                    audit["n_windows_dropped_empty"] += 1
                    continue
            windows.append(Window(store=si, start=start, length=length,
                                  meta={"store_name": store.name}))
            kept += 1
        store_audit["kept"] = int(kept)
        audit["stores"].append(store_audit)
    audit["n_windows_kept"] = len(windows)
    return windows, audit


def _index_signature(stores: Sequence[FrameStore], cfg: Dict[str, Any]) -> str:
    payload = {
        "name": cfg.get("name"),
        "split": cfg.get("split"),
        "input_len": cfg.get("input_len"),
        "target_len": cfg.get("target_len"),
        "stride": cfg.get("stride"),
        "crop": cfg.get("crop"),
        "resize": cfg.get("resize"),
        "filter": cfg.get("filter"),
        "sanitize": cfg.get("sanitize"),
        "resize_mode": cfg.get("resize_mode"),
        "stores": [[s.name, len(s), s.meta] for s in stores],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
#  dataset
# --------------------------------------------------------------------------- #
class PrecipSequenceDataset:
    """Unified precipitation nowcasting dataset (``[T_in,1,H,W]`` -> ``[T_out,1,H,W]``)."""

    def __init__(self, stores: Sequence[FrameStore], cfg: Dict[str, Any],
                 split: str = "train", name: str = "dataset",
                 context: Optional[Dict[str, Any]] = None) -> None:
        self.stores = list(stores)
        self.cfg = cfg
        self.split = split
        self.name = name
        self.context: Dict[str, Any] = dict(context or {})
        self.input_len = int(cfg.get("input_len", 5))
        self.target_len = int(cfg.get("target_len", 20))
        self.normalizer = Normalizer(cfg.get("normalization"))
        self.return_physical = bool(cfg.get("return_physical", True))
        self.crop = cfg.get("crop")
        self.resize = cfg.get("resize")
        self.resize_mode = str(cfg.get("resize_mode", "area"))
        self.fill = float(cfg.get("sanitize", {}).get("fill", 0.0))
        self.clip = cfg.get("sanitize", {}).get("clip")
        self.cache_index = bool(cfg.get("cache_index", True))
        self.cache_dir = cfg.get("cache_dir", "outputs/cache")

        self.windows, self.audit = self._load_or_build_index()

    # ------------------------------------------------------------------ index
    @property
    def _index_path(self) -> str:
        return os.path.join(str(self.cache_dir), f"index_{self.name}_{self.split}.json")

    def _load_or_build_index(self) -> Tuple[List[Window], Dict[str, Any]]:
        signature = _index_signature(self.stores, self.cfg)
        path = self._index_path
        if self.cache_index and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload.get("signature") == signature:
                    windows = [Window(w["store"], w["start"], w["length"], w.get("meta", {}))
                               for w in payload["windows"]]
                    return windows, payload["audit"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        windows, audit = build_index(self.stores, self.cfg, self.normalizer)
        audit["signature"] = signature
        audit["name"] = self.name
        audit["split"] = self.split
        if self.cache_index:
            ensure_dir(str(self.cache_dir))
            save_json(path, {"signature": signature,
                             "windows": [{"store": w.store, "start": w.start,
                                          "length": w.length, "meta": w.meta} for w in windows],
                             "audit": audit})
        return windows, audit

    # ------------------------------------------------------------------ api
    def __len__(self) -> int:
        return len(self.windows)

    def frame_range(self, window: Window) -> Tuple[int, int]:
        return window.start, window.start + self.input_len

    def read_frames(self, window: Window) -> np.ndarray:
        store = self.stores[window.store]
        frames = store.read_many(window.start, window.length)
        processed = []
        for i in range(window.length):
            frame, _ = preprocess_frame(frames[i], self.crop, self.resize,
                                        self.resize_mode, self.fill, self.clip)
            processed.append(frame)
        return np.stack(processed, axis=0).astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        window = self.windows[index]
        frames = self.read_frames(window)
        normalized = self.normalizer.normalize(frames)[:, None]  # [T,1,H,W]
        t_in = self.input_len
        item: Dict[str, Any] = {
            "input": torch.from_numpy(np.ascontiguousarray(normalized[:t_in])),
            "target": torch.from_numpy(np.ascontiguousarray(normalized[t_in:])),
            # full normalised window: used for teacher forcing / scheduled sampling
            "full": torch.from_numpy(np.ascontiguousarray(normalized)),
            "index": index,
            "store": window.store,
            "start": int(window.start),
            "frame_path": self.stores[window.store].name,
        }
        if self.return_physical:
            item["target_phys"] = torch.from_numpy(np.ascontiguousarray(frames[t_in:][:, None]))
            item["input_phys"] = torch.from_numpy(np.ascontiguousarray(frames[:t_in][:, None]))
            item["full_phys"] = torch.from_numpy(np.ascontiguousarray(frames[:, None]))
        return item

    # -------------------------------------------------------------- helpers
    @property
    def thresholds(self) -> List[float]:
        return [float(t) for t in self.cfg.get("thresholds", [])]

    @property
    def motion_mask_threshold(self) -> float:
        value = self.cfg.get("motion_mask_threshold", None)
        if value is not None:
            return float(value)
        thresholds = self.thresholds
        return float(min(thresholds)) if thresholds else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "split": self.split,
            "n_windows": len(self),
            "input_len": self.input_len,
            "target_len": self.target_len,
            "thresholds": self.thresholds,
            "normalizer": self.normalizer.state_dict(),
            "frame_interval_minutes": self.context.get("frame_interval_minutes"),
            "spatial_resolution": self.context.get("spatial_resolution"),
            "unit": self.context.get("unit"),
            "audit": self.audit,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensor fields, keep metadata as lists."""
    import torch

    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], (int, float, str)):
            out[key] = torch.as_tensor(values) if not isinstance(values[0], str) else values
        else:
            out[key] = values
    return out

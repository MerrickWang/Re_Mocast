"""Synthetic precipitation dataset.

Purpose (sections 8.2 / 9): provide data with a *known* velocity field so that
mechanism tests and level-1 acceptance runs are possible without downloading the
three real datasets.  Blobs are advected by a prescribed (uniform + rotational)
velocity field and modulated by explicit source / sink terms, i.e. exactly the
continuity-equation decomposition the model has to learn.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .split import assign_split
from .stores import ArrayStore, FrameStore

__all__ = ["build_stores", "context", "SyntheticStore", "generate_sequence"]

NAME = "synthetic"

THRESHOLDS = [0.2, 0.4, 0.6, 0.8]


def context() -> Dict[str, Any]:
    return {
        "name": NAME,
        "region": "synthetic",
        "variable": "advected gaussian blobs with source/sink",
        "unit": "normalised reflectivity (0-1)",
        "frame_interval_minutes": 5,
        "spatial_resolution": "synthetic",
        "thresholds": THRESHOLDS,
        "default_normalization": {"mode": "minmax", "vmin": 0.0, "vmax": 1.0},
        "default_crop": {"mode": "none"},
        "default_resize": None,
    }


class SyntheticStore(ArrayStore):
    """Array store that additionally exposes the ground-truth motion field."""

    def __init__(self, array: np.ndarray, name: str, motion: np.ndarray,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(array, name=name, meta=meta)
        self.motion = np.asarray(motion, dtype=np.float32)  # [T-1, 2, H, W] pixel units


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x0 = np.clip(np.floor(xs), 0, w - 1).astype(np.int64)
    y0 = np.clip(np.floor(ys), 0, h - 1).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = np.clip(xs - x0, 0.0, 1.0)
    wy = np.clip(ys - y0, 0.0, 1.0)
    top = img[y0, x0] * (1 - wx) + img[y0, x1] * wx
    bottom = img[y1, x0] * (1 - wx) + img[y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def _velocity_field(height: int, width: int, u0: float, v0: float, vorticity: float,
                    wave: float, rng: np.random.Generator) -> np.ndarray:
    """Return ``[2,H,W]`` (dx, dy) motion in *pixels per frame*."""
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32),
                         np.arange(width, dtype=np.float32), indexing="ij")
    cy, cx = height / 2.0, width / 2.0
    dx = np.full((height, width), u0, dtype=np.float32) - vorticity * (yy - cy)
    dy = np.full((height, width), v0, dtype=np.float32) + vorticity * (xx - cx)
    if wave > 0:
        k = 2.0 * np.pi / max(height, width)
        phase = float(rng.uniform(0, 2 * np.pi))
        dx = dx + wave * np.sin(k * xx * 2 + phase)
        dy = dy + wave * np.cos(k * yy * 2 + phase)
    return np.stack([dx, dy], axis=0).astype(np.float32)


def generate_sequence(
    n_frames: int = 25,
    height: int = 128,
    width: int = 128,
    n_blobs: int = 12,
    sigma_range: Tuple[float, float] = (3.0, 9.0),
    amplitude_range: Tuple[float, float] = (0.15, 1.0),
    velocity_range: Tuple[float, float] = (-2.0, 2.0),
    vorticity_range: Tuple[float, float] = (-0.01, 0.01),
    wave_range: Tuple[float, float] = (0.0, 0.6),
    source_strength: float = 0.08,
    velocity_drift: float = 0.35,
    velocity_period: float = 12.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate ``[T,H,W]`` frames in ``[0,1]`` plus the GT motion ``[T-1,2,H,W]``.

    The sequence contains the three ingredients the model has to deal with:

    * a spatially varying but smooth *mean* motion (uniform flow + rotation),
    * a slowly varying motion *trend* (``velocity_drift`` / ``velocity_period``),
      so that a perfectly persistent flow is sub-optimal,
    * local *source / sink* regions (growth and dissipation of precipitation).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32),
                         np.arange(width, dtype=np.float32), indexing="ij")

    field = np.zeros((height, width), dtype=np.float32)
    for _ in range(int(n_blobs)):
        cy = rng.uniform(0, height)
        cx = rng.uniform(0, width)
        sigma = rng.uniform(*sigma_range)
        amp = rng.uniform(*amplitude_range)
        field += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))

    u0 = float(rng.uniform(*velocity_range))
    v0 = float(rng.uniform(*velocity_range))
    vorticity = float(rng.uniform(*vorticity_range))
    wave = float(rng.uniform(*wave_range))
    base_velocity = _velocity_field(height, width, u0, v0, vorticity, wave, rng)

    # smooth source/sink modulation (growth and dissipation of precipitation)
    source = np.zeros((height, width), dtype=np.float32)
    for _ in range(3):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        sigma = rng.uniform(6.0, 20.0)
        sign = float(rng.choice([-1.0, 1.0]))
        source += sign * source_strength * np.exp(
            -(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
    source = np.clip(source, -0.12, 0.12)

    frames = np.empty((n_frames, height, width), dtype=np.float32)
    motion = np.empty((max(n_frames - 1, 1), 2, height, width), dtype=np.float32)
    frames[0] = np.clip(field, 0.0, 1.2)
    phase = float(rng.uniform(0, 2 * np.pi))
    for t in range(1, n_frames):
        gain = 1.0 + velocity_drift * np.sin(2 * np.pi * (t - 1) / max(velocity_period, 1.0)
                                             + phase)
        velocity = base_velocity * np.float32(gain)
        motion[t - 1] = velocity
        ys = yy - velocity[1]
        xs = xx - velocity[0]
        advected = _bilinear(frames[t - 1], xs, ys)
        src = _bilinear(source, xs, ys)
        frames[t] = np.clip(advected * (1.0 + src), 0.0, 1.2)
    frames = frames * 0.85  # keep most content inside [0,1]

    meta = {
        "seed": int(seed),
        "uniform_velocity": [u0, v0],
        "vorticity": vorticity,
        "wave": wave,
        "source_strength": source_strength,
        "velocity_drift": velocity_drift,
        "velocity_period": velocity_period,
    }
    return frames, motion, meta


def build_stores(cfg: Dict[str, Any], split: str,
                 audit: Optional[Dict[str, Any]] = None) -> Tuple[List[FrameStore], Dict[str, Any]]:
    info = context()
    n_sequences = int(cfg.get("n_sequences", 24))
    frames_per_sequence = int(cfg.get("frames_per_sequence",
                                      int(cfg.get("input_len", 5)) + int(cfg.get("target_len", 20))))
    height = int(cfg.get("height", 128))
    width = int(cfg.get("width", 128))
    gen_seed = int(cfg.get("gen_seed", 1234))
    gen_kwargs = {
        "n_blobs": int(cfg.get("n_blobs", 12)),
        "source_strength": float(cfg.get("source_strength", 0.08)),
        "velocity_drift": float(cfg.get("velocity_drift", 0.35)),
        "velocity_period": float(cfg.get("velocity_period", 12.0)),
    }
    cache_path = cfg.get("cache_sequences")

    keys = [str(i) for i in range(n_sequences)]
    split_cfg = dict(cfg.get("split_config", {}) or {})
    selected = assign_split(keys, split_cfg, split)

    stores: List[FrameStore] = []
    for i in range(n_sequences):
        if str(i) not in selected:
            continue
        frames, motion, meta = generate_sequence(
            n_frames=frames_per_sequence, height=height, width=width, seed=gen_seed + i,
            **gen_kwargs,
        )
        meta.update({"split": split, "sequence": i})
        if cache_path:
            folder = os.path.join(str(cache_path), split)
            os.makedirs(folder, exist_ok=True)
            np.save(os.path.join(folder, f"seq_{i:05d}_x.npy"), frames.astype(np.float32))
            np.save(os.path.join(folder, f"seq_{i:05d}_u.npy"), motion.astype(np.float32))
        stores.append(SyntheticStore(frames, name=f"synthetic:{i}", motion=motion, meta=meta))

    if audit is not None:
        audit.update({"dataset": NAME, "split": split, "n_sequences": len(stores),
                      "frames_per_sequence": frames_per_sequence,
                      "height": height, "width": width, "gen_seed": gen_seed})
    return stores, info

"""Verification scores: CSI, HSS and the tolerance-aware CSI-P4 / CSI-P16.

Definition used everywhere on the platform (section 8.1): frames are binarised at
the dataset specific thresholds *in physical units*, CSI-P4 / CSI-P16 apply a 4x4
/ 16x16 max-pooling before the comparison (exactly equivalent to pooling the
binary masks, since max-pooling commutes with thresholding).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "csi_from_counts",
    "hss_from_counts",
    "csi",
    "hss",
    "pooled_csi",
    "csi_hss_scores",
    "ScoreAccumulator",
    "DEFAULT_EPS",
]

DEFAULT_EPS = 1e-9


def csi_from_counts(hits: float, misses: float, false_alarms: float,
                    eps: float = DEFAULT_EPS) -> float:
    denom = hits + misses + false_alarms
    return float(hits / (denom + eps)) if denom > 0 else 1.0 if hits > 0 else 0.0


def hss_from_counts(hits: float, misses: float, false_alarms: float, correct_negatives: float,
                    eps: float = DEFAULT_EPS) -> float:
    numerator = 2.0 * (hits * correct_negatives - misses * false_alarms)
    denominator = ((hits + misses) * (misses + correct_negatives)
                   + (hits + false_alarms) * (false_alarms + correct_negatives))
    if denominator <= 0:
        return 0.0
    return float(numerator / (denominator + eps))


def _as_bool(array: Any, threshold: float) -> np.ndarray:
    array = np.asarray(array)
    return array > threshold


def csi(pred: Any, target: Any, threshold: float) -> float:
    """CSI for a single threshold on two equally shaped fields."""
    p = _as_bool(pred, threshold)
    t = _as_bool(target, threshold)
    hits = float(np.logical_and(p, t).sum())
    misses = float(np.logical_and(~p, t).sum())
    false_alarms = float(np.logical_and(p, ~t).sum())
    return csi_from_counts(hits, misses, false_alarms)


def hss(pred: Any, target: Any, threshold: float) -> float:
    p = _as_bool(pred, threshold)
    t = _as_bool(target, threshold)
    hits = float(np.logical_and(p, t).sum())
    misses = float(np.logical_and(~p, t).sum())
    false_alarms = float(np.logical_and(p, ~t).sum())
    correct_negatives = float(np.logical_and(~p, ~t).sum())
    return hss_from_counts(hits, misses, false_alarms, correct_negatives)


def _max_pool_np(array: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return array
    h, w = array.shape[-2:]
    ph, pw = (-h) % size, (-w) % size
    if ph or pw:
        array = np.pad(array, [(0, 0)] * (array.ndim - 2) + [(0, ph), (0, pw)],
                       mode="constant", constant_values=-np.inf)
    h, w = array.shape[-2:]
    reshaped = array.reshape(*array.shape[:-2], h // size, size, w // size, size)
    return reshaped.max(axis=(-3, -1))


def pooled_csi(pred: Any, target: Any, threshold: float, pool_size: int) -> float:
    """CSI computed after ``pool_size`` max-pooling (CSI-P4 / CSI-P16)."""
    p = _max_pool_np(np.asarray(pred, dtype=np.float64), pool_size)
    t = _max_pool_np(np.asarray(target, dtype=np.float64), pool_size)
    return csi(p, t, threshold)


def csi_hss_scores(pred: Any, target: Any, thresholds: Sequence[float],
                   pool_sizes: Sequence[int] = (4, 16)) -> Dict[str, Any]:
    """All spatial scores for one sample / batch (mean over thresholds)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    per_threshold: Dict[str, Dict[str, float]] = {}
    for thr in thresholds:
        entry = {
            "csi": csi(pred, target, float(thr)),
            "hss": hss(pred, target, float(thr)),
        }
        for pool in pool_sizes:
            entry[f"csi_p{pool}"] = pooled_csi(pred, target, float(thr), int(pool))
        per_threshold[f"{float(thr):g}"] = entry
    keys = ["csi", "hss"] + [f"csi_p{int(p)}" for p in pool_sizes]
    summary = {
        key: float(np.mean([entry[key] for entry in per_threshold.values()]))
        for key in keys
    }
    summary["per_threshold"] = per_threshold
    return summary


class ScoreAccumulator:
    """Streaming CSI / HSS accumulator with optional per-lead-time reporting."""

    def __init__(self, thresholds: Sequence[float], pool_sizes: Sequence[int] = (4, 16),
                 per_lead_time: bool = False, max_lead_time: int = 0) -> None:
        self.thresholds: List[float] = [float(t) for t in thresholds]
        self.pool_sizes: List[int] = [int(p) for p in pool_sizes]
        self.per_lead_time = bool(per_lead_time)
        self.max_lead_time = int(max_lead_time)
        self.reset()

    # ------------------------------------------------------------------ state
    @property
    def _n_slots(self) -> int:
        return self.max_lead_time + 1 if self.per_lead_time else 1

    def reset(self) -> None:
        n_thr = len(self.thresholds)
        n_slots = self._n_slots
        # counts[threshold, slot, (hits, misses, false_alarms, correct_negatives)]
        self.counts = np.zeros((n_thr, n_slots, 4), dtype=np.float64)
        self.pool_counts = {
            pool: np.zeros((n_thr, n_slots, 4), dtype=np.float64) for pool in self.pool_sizes
        }
        self.n_frames = 0

    # ----------------------------------------------------------------- update
    def update(self, pred: Any, target: Any, lead_offset: int = 0) -> None:
        """``pred`` / ``target`` are ``[..., P, 1, H, W]`` in **physical** units."""
        pred = np.asarray(pred, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        if pred.ndim == 4:  # [P,1,H,W] -> add batch dim
            pred, target = pred[None], target[None]
        steps = pred.shape[-4]
        for t in range(steps):
            p_t = pred[..., t, 0, :, :]
            g_t = target[..., t, 0, :, :]
            slot = min(t + lead_offset, self.max_lead_time) if self.per_lead_time else 0
            for ti, thr in enumerate(self.thresholds):
                self.counts[ti, slot] += _counts(p_t, g_t, thr)
                for pool in self.pool_sizes:
                    self.pool_counts[pool][ti, slot] += _counts(
                        _max_pool_np(p_t, pool), _max_pool_np(g_t, pool), thr)
            self.n_frames += int(pred.shape[0])

    # ---------------------------------------------------------------- compute
    def compute(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        csi_per_thr, hss_per_thr = [], []
        pool_means: Dict[int, List[float]] = {p: [] for p in self.pool_sizes}
        per_threshold: Dict[str, Dict[str, float]] = {}
        per_leadtime: Dict[str, Dict[str, Any]] = {}
        for ti, thr in enumerate(self.thresholds):
            hits, misses, fa, cn = self.counts[ti].sum(axis=0)
            thr_csi = csi_from_counts(hits, misses, fa)
            thr_hss = hss_from_counts(hits, misses, fa, cn)
            csi_per_thr.append(thr_csi)
            hss_per_thr.append(thr_hss)
            entry = _count_metrics(self.counts[ti].sum(axis=0))
            for pool in self.pool_sizes:
                counts = self.pool_counts[pool][ti].sum(axis=0)
                value = csi_from_counts(counts[0], counts[1], counts[2])
                entry[f"csi_p{pool}"] = value
                pool_means[pool].append(value)
            per_threshold[f"{thr:g}"] = entry
        result["csi"] = float(np.mean(csi_per_thr)) if csi_per_thr else 0.0
        result["hss"] = float(np.mean(hss_per_thr)) if hss_per_thr else 0.0
        for pool in self.pool_sizes:
            result[f"csi_p{pool}"] = float(np.mean(pool_means[pool])) if pool_means[pool] else 0.0
        result["per_threshold"] = per_threshold
        result["n_frames"] = int(self.n_frames)
        if self.per_lead_time:
            for slot in range(self._n_slots):
                if self.counts[:, slot].sum() <= 0:
                    continue
                details = {f"{thr:g}": _count_metrics(self.counts[ti, slot])
                           for ti, thr in enumerate(self.thresholds)}
                # Match the paper and the overall score: calculate each
                # threshold independently, then take their arithmetic mean.
                lead_entry: Dict[str, Any] = {key: float(np.mean([v[key] for v in details.values()]))
                                            for key in ("csi", "hss")}
                lead_entry["per_threshold"] = details
                per_leadtime[str(slot + 1)] = lead_entry
            result["per_leadtime"] = per_leadtime
        return result


def _count_metrics(counts: np.ndarray) -> Dict[str, float]:
    hits, misses, fa, cn = [float(v) for v in counts]
    total = max(hits + misses + fa + cn, 1.0)
    return {
        "csi": csi_from_counts(hits, misses, fa),
        "hss": hss_from_counts(hits, misses, fa, cn),
        "hits": hits, "misses": misses, "false_alarms": fa,
        "correct_negatives": cn,
        "precision": hits / (hits + fa) if hits + fa else 0.0,
        "recall": hits / (hits + misses) if hits + misses else 0.0,
        "pred_area": (hits + fa) / total,
        "true_area": (hits + misses) / total,
    }


def _counts(pred: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    predb = pred > threshold
    targetb = target > threshold
    hits = float(np.logical_and(predb, targetb).sum())
    misses = float(np.logical_and(~predb, targetb).sum())
    false_alarms = float(np.logical_and(predb, ~targetb).sum())
    correct_negatives = float(np.logical_and(~predb, ~targetb).sum())
    return np.asarray([hits, misses, false_alarms, correct_negatives], dtype=np.float64)

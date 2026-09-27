"""Read-only provenance, encoding and preprocessing audit of grouped Shanghai H5."""
from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter

import h5py
import numpy as np

from ..datasets import resolve_dataset_config
from ..datasets.shanghai import build_stores
from ..datasets.base import preprocess_frame
from ..utils.config import load_config
from ..utils.misc import save_json


def _attrs(obj):
    return {str(k): str(v)[:2000] for k, v in obj.attrs.items()}


def audit(cfg, samples_per_split=32, frames_per_sequence=5, seed=2026):
    if samples_per_split < 0 or frames_per_sequence < 1:
        raise ValueError("samples_per_split >= 0; frames_per_sequence >= 1")
    ds_cfg, _ = resolve_dataset_config(cfg.get("dataset", cfg))
    path = os.path.abspath(ds_cfg["h5_path"])
    splits = {s: build_stores(ds_cfg, s)[0] for s in ("train", "val", "test")}
    keysets = {s: {store.key for store in stores} for s, stores in splits.items()}
    rng = np.random.default_rng(seed)
    report = {
        "path": path, "file_bytes": os.path.getsize(path), "sampling_seed": seed,
        "frames_per_sequence_requested": frames_per_sequence,
        "configured_encoding": ds_cfg["h5_encoding"],
        "preprocessing": {key: ds_cfg.get(key) for key in (
            "crop", "resize", "resize_mode", "sanitize", "normalization", "thresholds", "filter")},
        "split_sizes": {s: len(stores) for s, stores in splits.items()},
        "key_overlap": {f"{a}/{b}": sorted(keysets[a] & keysets[b])
                        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
        "splits": {}, "duplicate_sequences": [], "cross_split_nonempty_frames": [],
        "limitations": [
            "Numeric ranges alone cannot prove physical units or gray-to-dBZ mapping.",
            "H5 creator/crop/timestamps must be confirmed from source metadata or conversion code.",
            "No duplicate in a sample does not prove no overlap; exact hashes miss transformed/near duplicates.",
            "Frame overlap evidence is content-based, not proof of the same storm event.",
        ],
    }
    hashes, frame_hashes = {}, {}
    with h5py.File(path, "r") as handle:
        report["root_attributes"] = _attrs(handle)
        report["groups"] = {}
        for name, obj in handle.items():
            if not isinstance(obj, h5py.Group):
                continue
            sequences = [obj[k] for k in obj if isinstance(obj[k], h5py.Dataset) and obj[k].ndim == 3]
            report["groups"][name] = {
                "attributes": _attrs(obj), "sequence_count": len(sequences),
                "shapes": dict(Counter(str(s.shape) for s in sequences)),
                "dtypes": dict(Counter(str(s.dtype) for s in sequences)),
                "all_len": int(obj["all_len"][()]) if "all_len" in obj else None,
            }
        for split, stores in splits.items():
            n = len(stores) if samples_per_split == 0 else min(samples_per_split, len(stores))
            indices = sorted(rng.choice(len(stores), n, replace=False).tolist())
            raw_hist = np.zeros(256, dtype=np.int64)
            raw_min, raw_max, nonfinite, pixels = None, None, 0, 0
            candidates = {}
            for encoding in ("pixel_0_255", "dbz"):
                candidates[encoding] = dict(raw_outside_0_70=0, raw_pixels=0, resized_pixels=0,
                                            repaired_pixels=0, before_threshold_counts=np.zeros(len(ds_cfg["thresholds"])),
                                            after_threshold_counts=np.zeros(len(ds_cfg["thresholds"])),
                                            raw_peak_sum=0.0, resized_peak_sum=0.0, frames=0)
            sampled_keys, attrs = [], {}
            for index in indices:
                store = stores[index]
                key = store.key
                raw = handle[key][()]
                sampled_keys.append(key)
                if handle[key].attrs:
                    attrs[key] = _attrs(handle[key])
                digest = hashlib.sha256(str((raw.shape, raw.dtype)).encode() + raw.tobytes()).hexdigest()
                if digest in hashes:
                    report["duplicate_sequences"].append({"first": hashes[digest], "second": [split, key]})
                else:
                    hashes[digest] = [split, key]
                frame_indices = np.unique(np.linspace(0, len(raw) - 1, min(frames_per_sequence, len(raw)), dtype=int))
                for t in frame_indices:
                    frame = raw[t]
                    finite = frame[np.isfinite(frame)]
                    pixels += frame.size
                    nonfinite += frame.size - finite.size
                    if finite.size:
                        lo, hi = float(finite.min()), float(finite.max())
                        raw_min = lo if raw_min is None else min(lo, raw_min)
                        raw_max = hi if raw_max is None else max(hi, raw_max)
                    if frame.dtype == np.uint8:
                        raw_hist += np.bincount(frame.ravel(), minlength=256)
                    if np.any(np.isfinite(frame) & (frame != 0)):
                        fh = hashlib.sha256(str((frame.shape, frame.dtype)).encode() + frame.tobytes()).hexdigest()
                        if fh in frame_hashes and frame_hashes[fh][0] != split:
                            report["cross_split_nonempty_frames"].append(
                                {"first": frame_hashes[fh], "second": [split, key, int(t)]})
                        else:
                            frame_hashes.setdefault(fh, [split, key, int(t)])
                    for encoding, stats in candidates.items():
                        physical = frame.astype(np.float32) * (70 / 255 if encoding == "pixel_0_255" else 1)
                        stats["raw_outside_0_70"] += int(((physical < 0) | (physical > 70) | ~np.isfinite(physical)).sum())
                        processed, repaired = preprocess_frame(
                            physical, ds_cfg.get("crop"), ds_cfg.get("resize"),
                            ds_cfg.get("resize_mode", "bilinear"),
                            ds_cfg.get("sanitize", {}).get("fill", 0),
                            ds_cfg.get("sanitize", {}).get("clip"))
                        stats["raw_pixels"] += physical.size
                        stats["resized_pixels"] += processed.size
                        stats["repaired_pixels"] += repaired
                        stats["frames"] += 1
                        stats["raw_peak_sum"] += float(np.nanmax(physical))
                        stats["resized_peak_sum"] += float(np.nanmax(processed))
                        for j, threshold in enumerate(ds_cfg["thresholds"]):
                            stats["before_threshold_counts"][j] += int((physical > threshold).sum())
                            stats["after_threshold_counts"][j] += int((processed > threshold).sum())
            for stats in candidates.values():
                stats["outside_0_70_fraction"] = stats.pop("raw_outside_0_70") / max(stats["raw_pixels"], 1)
                stats["before_threshold_fraction"] = (stats.pop("before_threshold_counts") / max(stats["raw_pixels"], 1)).tolist()
                stats["after_threshold_fraction"] = (stats.pop("after_threshold_counts") / max(stats["resized_pixels"], 1)).tolist()
                stats["mean_raw_peak"] = stats.pop("raw_peak_sum") / max(stats["frames"], 1)
                stats["mean_resized_peak"] = stats.pop("resized_peak_sum") / max(stats["frames"], 1)
            report["splits"][split] = {
                "sampled_sequences": n, "all_sequences_checked": n == len(stores),
                "sampled_keys": sampled_keys, "sequence_attributes": attrs,
                "raw_range": [raw_min, raw_max], "nonfinite": nonfinite, "pixels": pixels,
                "uint8_histogram": raw_hist.tolist(),
                "candidate_encodings": candidates,
            }
            print(f"[audit] {split}: {n}/{len(stores)} sequences; raw range {raw_min}..{raw_max}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--out", default="outputs/diagnostics/h5_audit.json")
    parser.add_argument("--samples-per-split", type=int, default=32, help="0 = all sequences")
    parser.add_argument("--frames-per-sequence", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)
    result = audit(load_config(args.config, overrides=args.set), args.samples_per_split,
                   args.frames_per_sequence, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_json(args.out, result)
    print("Exact duplicate sampled sequences:", len(result["duplicate_sequences"]))
    print("Cross-split nonempty sampled frames:", len(result["cross_split_nonempty_frames"]))
    print("Encoding is NOT automatically confirmed. Report:", args.out)


if __name__ == "__main__":
    main()

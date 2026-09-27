"""Ablation / baseline runner (section 8.3).

Trains and evaluates a set of variants on the same data split and writes a
comparison table (JSON + CSV + Markdown)::

    python -m mocast.tools.ablation -c mocast/configs/experiments/synthetic_l2.yaml \\
        --variants A0 A1 A2 A3 A4 A5 A6 B0 B1 --epochs 60 --out outputs/ablation_synthetic

Variant names: ``A0``-``A7`` from ``mocast/configs/ablation`` plus the classical
baselines ``B0`` (persistence) and ``B1`` (block-matching optical flow).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from ..engine import Trainer
from ..models import build_model
from ..utils.config import Config, load_config
from ..utils.misc import ensure_dir, fix_seed, save_json
from .common import build_datasets, build_loaders, dump_dataset_summaries
from .eval import evaluate_checkpoint

__all__ = ["main", "run_variant"]

BASELINE_VARIANTS = {
    "B0": {"model": {"name": "persistence", "output_len": 20}},
    "B1": {"model": {"name": "optical_flow", "output_len": 20, "search_radius": 4,
                     "block_size": 8, "smooth_kernel": 3}},
}

#: reference numbers from the paper (SEVIR / MeteoNet / Shanghai, Table 1) for the
#: report; kept here so that the tool can print the delta once a real dataset is used.
PAPER_REFERENCE = {
    "sevir": {"csi": 0.3114, "csi_p4": 0.3279, "csi_p16": 0.3474, "hss": 0.3896,
              "lpips": 0.3430, "ssim": 0.6776},
    "meteonet": {"csi": 0.3489, "csi_p4": 0.3875, "csi_p16": 0.4857, "hss": 0.4738,
                 "lpips": 0.2917, "ssim": 0.8146},
    "shanghai": {"csi": 0.4156, "csi_p4": 0.4810, "csi_p16": 0.5997, "hss": 0.5528,
                 "lpips": 0.1835, "ssim": 0.7812},
}


def _variant_config(base: Config, variant: str, ablation_root: str) -> Config:
    cfg = base.clone()
    if variant in BASELINE_VARIANTS:
        cfg["model"] = Config({**cfg.get("model", {}).to_dict(), **BASELINE_VARIANTS[variant]["model"]})
        cfg.setdefault("run", {})["name"] = f"{variant}_baseline"
        return cfg
    path = os.path.join(ablation_root, f"{variant}_*.yaml")
    import glob

    matches = sorted(glob.glob(path))
    if not matches:
        # accept a direct name such as A0_full
        matches = sorted(glob.glob(os.path.join(ablation_root, f"{variant}*.yaml")))
    if not matches:
        raise FileNotFoundError(f"No ablation config found for variant '{variant}'")
    # NOTE: the raw file is loaded (its own `defaults` are ignored) so that the
    # variant only overrides what it explicitly declares - the data geometry of the
    # experiment config (target_size / input_len / output_len) is preserved.
    overlay = Config.load(matches[0])
    overlay.pop("defaults", None)
    merged = cfg.merge(overlay)
    for key in ("target_size", "input_len", "output_len"):
        merged.setdefault("model", Config())
        if key not in overlay.get("model", {}) and key in cfg.get("model", {}):
            merged["model"][key] = cfg["model"][key]
    merged["run"] = Config({**cfg.get("run", {}).to_dict(), **overlay.get("run", {}).to_dict(),
                            "name": variant})
    return merged


def run_variant(cfg: Config, variant: str, output_root: str, device: torch.device,
                epochs: Optional[int] = None, seed: Optional[int] = None,
                trainable: bool = True) -> Dict[str, Any]:
    run_dir = ensure_dir(os.path.join(output_root, variant))
    cfg = cfg.clone()
    cfg["run"] = Config({**cfg.get("run", {}).to_dict(), "name": variant, "dir": run_dir})
    if seed is not None:
        cfg["run"]["seed"] = int(seed)
        fix_seed(int(seed), deterministic=bool(cfg["run"].get("deterministic", True)))
    if epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(epochs)
    cfg.save(os.path.join(run_dir, "config.yaml"))

    train_ds = build_datasets(cfg, ["train"])["train"]
    val_ds = build_datasets(cfg, ["val"])["val"]
    test_ds = build_datasets(cfg, ["test"])["test"]
    dump_dataset_summaries({"train": train_ds, "val": val_ds, "test": test_ds}, run_dir)

    model = build_model(cfg)
    summary: Dict[str, Any] = {"variant": variant, "run_dir": run_dir,
                               "params": sum(p.numel() for p in model.parameters()),
                               "train_windows": len(train_ds), "val_windows": len(val_ds),
                               "test_windows": len(test_ds)}
    t0 = time.time()
    if trainable and any(p.requires_grad for p in model.parameters()):
        loaders = build_loaders(cfg, {"train": train_ds, "val": val_ds})
        trainer = Trainer(cfg, model, loaders["train"], loaders["val"],
                          normalizer=train_ds.normalizer, output_dir=run_dir, device=device)
        trainer.fit(max_epochs=cfg.get("train", {}).get("epochs", None))
        best = os.path.join(run_dir, "checkpoints", "best.pt")
        if os.path.exists(best):
            trainer.load(best, load_optimizer=False)
        model = trainer.model
    summary["train_seconds"] = time.time() - t0

    metrics = evaluate_checkpoint(cfg, model=model, dataset=test_ds, split="test", device=device)
    save_json(os.path.join(run_dir, "metrics_test.json"), metrics)
    summary["metrics"] = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    print(f"[ablation] {variant:6s} csi={metrics.get('csi', float('nan')):.4f} "
          f"csi_p4={metrics.get('csi_p4', float('nan')):.4f} "
          f"hss={metrics.get('hss', float('nan')):.4f} "
          f"mse={metrics.get('mse', float('nan')):.4f}")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MoCast ablation / baseline matrix")
    parser.add_argument("-c", "--config", nargs="+", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--variants", nargs="+",
                        default=["A0", "A1", "A2", "A3", "A4", "A5", "A6", "B0", "B1"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="outputs/ablation")
    parser.add_argument("--ablation-root", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                             "configs", "ablation"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    base = load_config(args.config, overrides=args.set)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    output_root = ensure_dir(args.out)

    results: List[Dict[str, Any]] = []
    for variant in args.variants:
        cfg = _variant_config(base, variant, args.ablation_root)
        try:
            results.append(run_variant(cfg, variant, output_root, device, args.epochs, args.seed))
        except Exception as exc:  # keep the matrix running, record the failure
            print(f"[ablation] {variant} failed: {exc}")
            results.append({"variant": variant, "error": str(exc)})

    keys = ["csi", "hss", "csi_p4", "csi_p16", "mse", "mae", "ssim", "lpips", "params",
            "train_seconds"]
    csv_path = os.path.join(output_root, "ablation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant"] + keys)
        for entry in results:
            metrics = entry.get("metrics", {})
            writer.writerow([entry["variant"]] + [metrics.get(k, entry.get(k, "")) for k in keys])
    save_json(os.path.join(output_root, "ablation_results.json"), results)
    _write_markdown(results, os.path.join(output_root, "ablation_results.md"))
    print(f"[ablation] wrote {csv_path}")
    return 0


def _write_markdown(results: List[Dict[str, Any]], path: str) -> str:
    header = ("| variant | CSI ↑ | HSS ↑ | CSI-P4 ↑ | CSI-P16 ↑ | MSE ↓ | params | train (s) |\n"
              "| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    rows = []
    for entry in results:
        metrics = entry.get("metrics", {})
        if entry.get("error"):
            rows.append(f"| {entry['variant']} | failed: {entry['error'][:40]} | | | | | | |")
            continue
        rows.append("| {v} | {csi:.4f} | {hss:.4f} | {p4:.4f} | {p16:.4f} | {mse:.4f} | "
                    "{params} | {secs:.1f} |".format(
                        v=entry["variant"],
                        csi=float(metrics.get("csi", float("nan"))),
                        hss=float(metrics.get("hss", float("nan"))),
                        p4=float(metrics.get("csi_p4", float("nan"))),
                        p16=float(metrics.get("csi_p16", float("nan"))),
                        mse=float(metrics.get("mse", float("nan"))),
                        params=entry.get("params", ""),
                        secs=float(entry.get("train_seconds", 0.0))))
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header + "\n".join(rows) + "\n")
    return path


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

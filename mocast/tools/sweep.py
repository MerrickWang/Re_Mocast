"""Random / grid hyper-parameter search over the uncertain parameters.

The paper reports AutoML with NNI but the search space is not public (risk R1,
R3, R7).  This runner reads ``configs/search/space.yaml``, samples configurations,
reuses :func:`mocast.tools.ablation.run_variant` and writes a ranked result table::

    python -m mocast.tools.sweep -c mocast/configs/experiments/synthetic_l2.yaml \\
        --space mocast/configs/search/space.yaml --trials 8 --epochs 30 \\
        --monitor csi --out outputs/sweep_synthetic
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..utils.config import Config, load_config
from ..utils.misc import ensure_dir, save_json
from .ablation import run_variant

__all__ = ["main", "sample_config"]


def _flatten(space: Dict[str, Any]) -> Dict[str, List[Any]]:
    return {str(k): list(v) if isinstance(v, (list, tuple)) else [v] for k, v in space.items()}


def sample_config(rng: np.random.Generator, space: Dict[str, List[Any]],
                  mode: str = "random") -> Dict[str, Any]:
    """Draw one configuration from ``space`` (dotted keys -> nested dict)."""
    flat = _flatten(space)
    if mode == "grid":
        # deterministic round-robin over the first key, random elsewhere
        keys = sorted(flat)
        out: Dict[str, Any] = {}
        for i, key in enumerate(keys):
            out[key] = flat[key][i % len(flat[key])]
        return out
    out = {key: values[int(rng.integers(len(values)))] for key, values in flat.items()}
    return out


def _nested(flat: Dict[str, Any]) -> Config:
    cfg = Config()
    for key, value in flat.items():
        cfg.set_path(key, value)
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Random/grid search over the MoCast search space")
    parser.add_argument("-c", "--config", nargs="+", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--space", type=str, required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--monitor", type=str, default="csi")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "grid"])
    parser.add_argument("--out", type=str, default="outputs/sweep")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    base = load_config(args.config, overrides=args.set)
    space = Config.load(args.space).to_dict()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out_root = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)

    results: List[Dict[str, Any]] = []
    for trial in range(args.trials):
        sampled = sample_config(rng, space, args.mode)
        cfg = base.merge(_nested(sampled))
        name = f"trial{trial:03d}"
        print(f"[sweep] {name}: " + json.dumps(sampled, default=str))
        try:
            summary = run_variant(cfg, name, out_root, device, epochs=args.epochs, seed=args.seed)
            summary["sampled"] = sampled
            results.append(summary)
        except Exception as exc:  # pragma: no cover - report and continue
            print(f"[sweep] {name} failed: {exc}")
            results.append({"variant": name, "error": str(exc), "sampled": sampled})

    def rank(entry: Dict[str, Any]) -> float:
        value = entry.get("metrics", {}).get(args.monitor, None)
        return -float(value) if value is not None else float("inf")

    ranked = sorted(results, key=rank)
    save_json(os.path.join(out_root, "sweep_results.json"), ranked)
    with open(os.path.join(out_root, "sweep_results.csv"), "w", newline="", encoding="utf-8") as h:
        writer = csv.writer(h)
        writer.writerow(["trial", args.monitor, "mse", "params", "sampled"])
        for entry in ranked:
            metrics = entry.get("metrics", {})
            writer.writerow([entry["variant"], metrics.get(args.monitor, ""), metrics.get("mse", ""),
                             entry.get("params", ""), json.dumps(entry.get("sampled", {}))])
    print(f"[sweep] best: {ranked[0]['variant']} -> "
          f"{ranked[0].get('metrics', {}).get(args.monitor)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Shared helpers for the command line tools."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..datasets import PrecipSequenceDataset, build_dataloader, build_dataset
from ..models import build_model
from ..utils.config import Config, load_config
from ..utils.misc import ensure_dir, fix_seed, save_json

__all__ = ["base_parser", "prepare_run", "build_datasets", "build_model_from_config", "timestamp",
           "code_version", "save_run_info"]


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--config", nargs="+", required=True,
                        help="one or more YAML config files (later files override earlier ones)")
    parser.add_argument("--set", nargs="*", default=[],
                        help="config overrides, e.g. train.epochs=5 model.pmm.window=5")
    parser.add_argument("--seed", type=int, default=None, help="override run.seed")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / cuda:1")
    parser.add_argument("--output", type=str, default=None, help="override the run directory")
    parser.add_argument("--tag", type=str, default=None, help="suffix added to the run directory")
    return parser


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def code_version(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort code version record (git commit + dirty flag + package version)."""
    import platform
    import subprocess

    from .. import __version__

    info: Dict[str, Any] = {
        "mocast_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
        "git_commit": None,
        "git_dirty": None,
    }
    try:  # pragma: no cover - depends on the environment
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    root = repo_root or os.getcwd()
    try:  # pragma: no cover - depends on the environment
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=10)
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                text=True, timeout=10)
        if status.returncode == 0:
            info["git_dirty"] = bool(status.stdout.strip())
    except Exception:
        pass
    return info


def save_run_info(run_dir: str, cfg: Config, argv: Optional[List[str]] = None) -> str:
    payload = {
        "timestamp": timestamp(),
        "seed": cfg.get("run", {}).get("seed"),
        "name": cfg.get("run", {}).get("name"),
        "argv": list(argv or []),
        "config_files": cfg.get("_defaults", []) + [cfg.get("_loaded_from")],
        "code": code_version(),
    }
    path = os.path.join(run_dir, "run_info.json")
    save_json(path, payload)
    return path


def prepare_run(cfg: Config, args: Any, default_name: str = "run") -> Tuple[Config, str, torch.device]:
    """Resolve seeds, device and create the run directory (with a config dump)."""
    seed = int(args.seed if args.seed is not None else cfg.get("run", {}).get("seed", 2026))
    cfg.setdefault("run", {})
    cfg["run"]["seed"] = seed
    cfg["run"].setdefault("output_root", "outputs")
    cfg["run"].setdefault("name", default_name)
    cfg["run"].setdefault("deterministic", True)
    fix_seed(seed, deterministic=bool(cfg["run"].get("deterministic", True)))

    name = str(cfg["run"]["name"])
    tag = f"-{args.tag}" if getattr(args, "tag", None) else ""
    run_dir = args.output or os.path.join(str(cfg["run"]["output_root"]), f"{name}{tag}-seed{seed}")
    run_dir = ensure_dir(run_dir)
    cfg["run"]["dir"] = run_dir
    cfg.save(os.path.join(run_dir, "config.yaml"))
    if getattr(args, "device", None):
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_run_info(run_dir, cfg, getattr(args, "_argv", None) or sys.argv)
    return cfg, run_dir, device


def build_datasets(cfg: Config, splits: List[str]) -> Dict[str, PrecipSequenceDataset]:
    datasets: Dict[str, PrecipSequenceDataset] = {}
    for split in splits:
        datasets[split] = build_dataset(cfg, split)
    return datasets


def dump_dataset_summaries(datasets: Dict[str, PrecipSequenceDataset], run_dir: str,
                           splits: Optional[List[str]] = None) -> str:
    payload: Dict[str, Any] = {"splits": {}, "generated_at": timestamp()}
    for split, dataset in datasets.items():
        if splits and split not in splits:
            continue
        payload["splits"][split] = dataset.summary()
    path = os.path.join(run_dir, "dataset_manifest.json")
    save_json(path, payload)
    return path


def build_model_from_config(cfg: Config) -> torch.nn.Module:
    return build_model(cfg)


def build_loaders(cfg: Config, datasets: Dict[str, PrecipSequenceDataset],
                  splits: Optional[List[str]] = None) -> Dict[str, Any]:
    loaders = {}
    for split, dataset in datasets.items():
        if splits and split not in splits:
            continue
        loaders[split] = build_dataloader(cfg, split, dataset=dataset)
    return loaders

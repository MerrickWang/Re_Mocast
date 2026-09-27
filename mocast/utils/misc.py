"""Small helpers shared by tools, engine and tests."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Optional

import numpy as np

__all__ = [
    "AverageMeter",
    "Timer",
    "count_parameters",
    "ensure_dir",
    "fix_seed",
    "save_json",
    "seed_worker",
    "worker_init_fn",
    "get_device",
]

_REPRO_SEEDS = [2026, 2027, 2028]  # section 7: 随机种子 2026/2027/2028


def fix_seed(seed: int = 2026, deterministic: bool = True) -> int:
    """Seed python / numpy / torch (and cuda) for reproducible runs."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard requirement in practice
        pass
    return seed


def seed_worker(worker_id: int) -> None:  # pragma: no cover - depends on torch RNG
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


worker_init_fn = seed_worker


class AverageMeter:
    """Track running mean of a scalar (loss / metric logging)."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __float__(self) -> float:
        return self.avg

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.name}={self.avg:.6f} (n={self.count})"


class Timer:
    """Context manager returning elapsed seconds."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0


def count_parameters(module: Any, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def count_parameters_by_module(model: Any, prefix: str = "") -> dict:
    """Parameter count per top level sub-module (used by the efficiency report)."""
    out = {}
    for name, child in model.named_children():
        out[f"{prefix}{name}"] = count_parameters(child)
    return out


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload: Any, indent: int = 2) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, default=_default)
    return path


def _default(obj: Any) -> Any:  # json fallback for numpy / torch scalars
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def get_device(prefer: Optional[str] = None) -> "Any":
    """Return cuda if available (the paper trains on a single A100)."""
    import torch

    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def human_readable(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}" if unit else f"{n:.0f}"
        n /= 1000.0
    return f"{n:.2f}P"

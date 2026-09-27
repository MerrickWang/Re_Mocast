"""Efficiency benchmark (paper Table 5: training time, GPU memory, CSI trade-off).

Reports parameters, FLOPs, latency and peak GPU memory for a single forward pass::

    python -m mocast.tools.benchmark -c <cfg> --batch 1 --size 128
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from ..utils.config import load_config
from ..utils.misc import count_parameters, count_parameters_by_module, ensure_dir, save_json
from .common import base_parser, build_model_from_config, prepare_run

__all__ = ["main", "benchmark_model"]


def benchmark_model(model: torch.nn.Module, device: torch.device, batch: int = 1,
                    size: int = 128, input_len: int = 5, warmup: int = 3,
                    iters: int = 10, backward: bool = False) -> Dict[str, Any]:
    model = model.to(device).eval()
    x = torch.randn(batch, input_len, 1, size, size, device=device)
    result: Dict[str, Any] = {
        "params": count_parameters(model),
        "params_by_module": count_parameters_by_module(model),
        "batch_size": batch, "size": size, "device": str(device),
    }
    # ---- FLOPs (torch built-in counter, falls back to None) ---------------
    flops = None
    try:  # pragma: no cover - depends on torch version
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with counter:
            model(x)
        flops = counter.get_total_flops()
    except Exception:
        flops = None
    result["flops"] = flops

    # ---- latency ----------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / max(iters, 1)
    result["latency_ms"] = elapsed * 1000.0
    result["throughput_samples_per_s"] = batch / max(elapsed, 1e-9)
    if device.type == "cuda":
        result["peak_memory_mb_forward"] = torch.cuda.max_memory_allocated(device) / 1024**2
    if backward:
        model.train()
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        out = model(x)
        loss = out["pred"].mean()
        loss.backward()
        if device.type == "cuda":
            result["peak_memory_mb_train_step"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = base_parser("Benchmark MoCast (params / FLOPs / latency / memory)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--size", type=int, default=None,
                        help="spatial size (defaults to model.target_size)")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    cfg, run_dir, device = prepare_run(cfg, args, default_name="benchmark")
    target_size = cfg.get("model", {}).get("target_size", None)
    size = int(args.size if args.size is not None else
               (target_size[0] if target_size else 128))
    model = build_model_from_config(cfg)
    stats = benchmark_model(model, device, args.batch, size,
                            input_len=int(cfg.get("model", {}).get("input_len", 5)),
                            iters=args.iters, backward=args.backward)
    stats["torch"] = torch.__version__
    stats["python"] = platform.python_version()
    stats["platform"] = platform.platform()
    path = save_json(os.path.join(run_dir, "benchmark.json"), stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "params_by_module"},
                     indent=2, default=str))
    print(f"[mocast] wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

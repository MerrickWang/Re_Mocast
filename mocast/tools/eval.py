"""Evaluation entry point: full split metrics on physical frames (section 8.1).

Example::

    python -m mocast.tools.eval -c mocast/configs/experiments/sevir_mocast.yaml \\
        --checkpoint outputs/sevir_mocast-seed2026/checkpoints/best.pt --split test
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import torch

from ..datasets import build_dataloader, build_dataset
from ..engine import load_checkpoint
from ..metrics import Evaluator
from ..utils.config import load_config
from ..utils.misc import save_json
from .common import base_parser, build_model_from_config, prepare_run

__all__ = ["main", "evaluate_checkpoint"]


@torch.no_grad()
def evaluate_checkpoint(cfg: Any, checkpoint: Optional[str] = None, split: str = "test",
                        device: Optional[torch.device] = None,
                        max_batches: Optional[int] = None, samples: int = 1,
                        model: Optional[torch.nn.Module] = None,
                        dataset: Optional[Any] = None) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset if dataset is not None else build_dataset(cfg, split)
    loader = build_dataloader(cfg, split, shuffle=False, dataset=dataset)
    if model is None:
        model = build_model_from_config(cfg)
    model = model.to(device)
    if checkpoint:
        load_checkpoint(checkpoint, model, map_location=str(device))
    model.eval()

    eval_cfg = cfg.get("eval", {}) or {}
    thresholds = eval_cfg.get("thresholds") or dataset.thresholds
    normalizer = dataset.normalizer
    data_range = (normalizer.vmax - normalizer.vmin) if normalizer.mode == "minmax" else 1.0
    evaluator = Evaluator(thresholds,
                          pool_sizes=tuple(eval_cfg.get("pool_sizes", (4, 16))),
                          perceptual=bool(eval_cfg.get("perceptual", True)),
                          data_range=data_range,
                          device=str(device),
                          per_lead_time=bool(eval_cfg.get("per_lead_time", True)),
                          max_lead_time=int(cfg.get("model", {}).get("output_len", 20)),
                          enable_lpips=bool(eval_cfg.get("lpips", True)))
    limit = max_batches or eval_cfg.get("max_batches")
    for i, batch in enumerate(loader):
        if limit and i >= int(limit):
            break
        x = batch["input"].to(device)
        target = batch["target_phys"].to(device)
        outputs = model(x)
        pred = outputs.get("pred", outputs.get("base_pred"))
        if samples > 1 and hasattr(model, "sample"):
            base, sampled = model.sample(x, num_samples=samples)  # type: ignore[attr-defined]
            for s in range(sampled.shape[1]):
                evaluator.update(normalizer.denormalize(sampled[:, s].detach()), target)
            continue
        evaluator.update(normalizer.denormalize(pred), target)
    metrics = evaluator.compute()
    metrics["split"] = split
    metrics["samples_per_input"] = int(samples)
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = base_parser("Evaluate MoCast / MoCast+")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--samples", type=int, default=1, help="probabilistic samples (MoCast+)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    cfg, run_dir, device = prepare_run(cfg, args, default_name="eval")
    metrics = evaluate_checkpoint(cfg, args.checkpoint, args.split, device,
                                 args.max_batches, args.samples)
    out_path = os.path.join(run_dir, f"metrics_{args.split}.json")
    save_json(out_path, metrics)
    print(f"[mocast] metrics -> {out_path}")
    for key in ("csi", "hss", "csi_p4", "csi_p16", "ssim", "lpips", "mse"):
        if key in metrics:
            print(f"  {key:8s}: {metrics[key]:.6f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

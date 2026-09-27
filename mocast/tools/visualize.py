"""Visualisation tools: frame comparisons, motion fields, expert gates, data samples.

Examples::

    python -m mocast.tools.visualize -c <cfg> --checkpoint <ckpt> --kind prediction
    python -m mocast.tools.visualize -c <cfg> --kind motion --motion source
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..datasets import build_dataset
from ..engine import load_checkpoint
from ..utils.config import load_config
from ..utils.misc import ensure_dir
from .common import base_parser, build_model_from_config, prepare_run

__all__ = ["main", "save_frame_grid", "save_motion_figure"]


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().float().cpu().numpy()
    return np.asarray(tensor)


def save_frame_grid(rows: List[Dict[str, Any]], path: str, titles: Optional[List[str]] = None,
                    cmap: str = "turbo", value_range: Optional[tuple] = None) -> str:
    """Save a ``(row, lead time)`` grid of frames."""
    plt = _import_matplotlib()
    n_rows = len(rows)
    n_cols = max(len(row["frames"]) for row in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.2 * n_rows), squeeze=False)
    arrays = np.concatenate([np.asarray(row["frames"]).reshape(-1) for row in rows])
    vmin, vmax = value_range or (float(np.nanmin(arrays)), float(np.nanmax(arrays)))
    for r, row in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            if c < len(row["frames"]):
                ax.imshow(np.asarray(row["frames"][c]).squeeze(), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(str(row.get("name", "")))
            if r == 0 and titles:
                ax.set_title(str(titles[c]) if c < len(titles) else "")
    fig.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_motion_figure(motion: Any, speed: Optional[Any] = None, path: str = "motion.png",
                       title: str = "", stride: int = 4, background: Any = None) -> str:
    """Speed background + arrows, mirroring Fig. 4 of the paper."""
    plt = _import_matplotlib()
    motion = _to_numpy(motion)
    if motion.ndim == 4:
        motion = motion[0]
    u, v = motion[0], motion[1]
    if speed is None:
        speed = np.sqrt(u**2 + v**2)
    speed = _to_numpy(speed).squeeze()
    fig, ax = plt.subplots(figsize=(5, 5))
    background_array = _to_numpy(background).squeeze() if background is not None else speed
    im = ax.imshow(background_array, cmap="turbo")
    y, x = np.mgrid[0:u.shape[0]:stride, 0:u.shape[1]:stride]
    ax.quiver(x, y, u[::stride, ::stride], v[::stride, ::stride], color="white",
              angles="xy", scale_units="xy", scale=1.0, width=0.003)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


@torch.no_grad()
def main(argv: Optional[List[str]] = None) -> int:
    parser = base_parser("Visualise MoCast predictions / motion / gates")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--kind", type=str, default="prediction",
                        choices=["prediction", "motion", "gate", "data"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    cfg, run_dir, device = prepare_run(cfg, args, default_name="visualize")
    out_dir = ensure_dir(args.out or os.path.join(run_dir, "visuals"))
    dataset = build_dataset(cfg, args.split)
    item = dataset[args.sample]
    x = item["input"][None].to(device)
    target_phys = item["target_phys"][None]

    if args.kind == "data":
        frames = [item["input_phys"][i, 0].numpy() for i in range(5)]
        path = save_frame_grid([{"name": "input", "frames": frames}],
                               os.path.join(out_dir, "data_input.png"))
        print(f"[mocast] wrote {path}")
        return 0

    model = build_model_from_config(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()
    outputs = model(x)
    pred = outputs.get("pred", outputs.get("base_pred"))
    normalizer = dataset.normalizer
    pred_phys = normalizer.denormalize(pred[0].squeeze(1).cpu().numpy())
    gt_phys = target_phys[0].squeeze(1).numpy()
    last_input = item["input_phys"][-1, 0].numpy()

    if args.kind == "prediction":
        leads = [0, 4, 9, 14, 19]
        rows = [
            {"name": "input(last)", "frames": [last_input] * len(leads)},
            {"name": "pred", "frames": [pred_phys[i] for i in leads]},
            {"name": "truth", "frames": [gt_phys[i] for i in leads]},
        ]
        path = save_frame_grid(rows, os.path.join(out_dir, "prediction.png"),
                               titles=[f"+{(i + 1) * float(dataset.cfg.get('frame_interval_minutes', dataset.context.get('frame_interval_minutes', 1))):g}min" for i in leads])
    elif args.kind == "motion":
        steps = [0, len(outputs["ma"][0]) - 1]
        for i, step in enumerate(steps):
            mean = outputs["mm"][0, step].cpu().numpy()
            total = outputs["ma"][0, step].cpu().numpy()
            save_motion_figure(mean, path=os.path.join(out_dir, f"motion_mean_{i}.png"),
                               title=f"mean motion t={step}")
            save_motion_figure(total, path=os.path.join(out_dir, f"motion_total_{i}.png"),
                               title=f"total motion t={step}")
        path = os.path.join(out_dir, "motion_mean_0.png")
    else:  # gate
        gate = outputs["gate"][0].mean(dim=0).cpu().numpy()
        plt = _import_matplotlib()
        fig, axes = plt.subplots(1, gate.shape[0], figsize=(4 * gate.shape[0], 4))
        for i in range(gate.shape[0]):
            axes[i].imshow(gate[i], cmap="viridis")
            axes[i].set_title(f"expert {i} (scale {i})")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        fig.tight_layout()
        path = os.path.join(out_dir, "gate.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
    print(f"[mocast] wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

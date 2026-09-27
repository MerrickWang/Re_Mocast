"""Validation diagnostics: persistence, motion/source contributions, lead-time skill.

No prepare_run(): diagnostic runs never overwrite training config/checkpoints.
Branch removal is a post-hoc probe, not a retrained ablation.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from ..datasets import build_dataset, build_dataloader
from ..engine import load_checkpoint
from ..metrics.csi import ScoreAccumulator
from ..models.advection import warp, _flow_to_grid
from ..utils.config import load_config
from ..utils.misc import save_json, fix_seed
from .common import build_model_from_config


def operator_probes():
    """CPU probes separate numerical warping artifacts from learned predictions."""
    image = torch.zeros(1, 1, 128, 128)
    image[:, :, 40:55, 40:55] = 1
    zero = torch.zeros(1, 2, 128, 128, dtype=torch.bfloat16)
    legacy_grid = _flow_to_grid(zero, 128, 128).float()
    legacy, fixed = image, image
    for _ in range(20):
        legacy = torch.nn.functional.grid_sample(legacy, legacy_grid, align_corners=True,
                                                  padding_mode="border")
        fixed = warp(fixed, zero)
    impulse = torch.zeros_like(image)
    impulse[:, :, 50, 50] = 1
    half_pixel = torch.zeros(1, 2, 128, 128)
    half_pixel[:, 0] = 0.5
    repeated = impulse
    for _ in range(20):
        repeated = warp(repeated, half_pixel)
    single = warp(impulse, half_pixel * 20)
    right = torch.zeros_like(half_pixel)
    right[:, 0] = 1
    shifted = warp(impulse, right)
    return {
        "legacy_bf16_zero_flow_20step_mse": float((legacy - image).square().mean()),
        "fixed_zero_flow_20step_mse": float((fixed - image).square().mean()),
        "half_pixel_20step_peak": float(repeated.max()),
        "equivalent_single_10pixel_peak": float(single.max()),
        "positive_dx_moves_right": bool(shifted[0, 0, 50, 51] > 0.99),
        "interpretation": "Subpixel repeated bilinear interpolation smooths peaks even in FP32; constant-flow single warp is only a probe, not a replacement for variable-flow recurrence.",
    }


@torch.inference_mode()
def diagnose(cfg, checkpoint, device, max_batches=None, interval=None, amp=False):
    fix_seed(int(cfg.get("run", {}).get("seed", 2026)))
    ds = build_dataset(cfg, "val")
    if not len(ds):
        raise ValueError("Validation dataset is empty")
    loader = build_dataloader(cfg, "val", dataset=ds, shuffle=False, num_workers=0,
                              drop_last=False)
    model = build_model_from_config(cfg).to(device).eval()
    load_checkpoint(checkpoint, model, map_location=str(device))
    if not hasattr(model, "advection"):
        raise ValueError("This diagnostic requires a deterministic MoCast model")
    names = ("model", "persistence", "motion_only", "source_only")
    meters = {name: ScoreAccumulator(ds.thresholds, (), per_lead_time=True,
                                    max_lead_time=ds.target_len) for name in names}
    squared = {name: np.zeros(ds.target_len) for name in names}
    branch_sums = np.zeros((ds.target_len, 5))
    samples = pixels = 0
    amp_dtype = torch.bfloat16 if str(cfg.get("train", {}).get("amp_dtype", "float16")) in (
        "bfloat16", "bf16") else torch.float16
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["input"].to(device)
        truth = batch["target_phys"].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            output = model(x)
        motion, source = output["motion"].float(), output["source"].float()
        current, only_motion, only_source = x[:, -1].float(), x[:, -1].float(), x[:, -1].float()
        motion_frames, source_frames = [], []
        for t in range(ds.target_len):
            advected = model.advection(current, motion[:, t])
            delta = advected - current
            values = (
                motion[:, t].square().sum(dim=1).sqrt(),
                source[:, t], source[:, t].abs(), delta.abs(), advected,
            )
            for j, value in enumerate(values):
                branch_sums[t, j] += float(value.sum().cpu())
            current = advected + source[:, t]
            only_motion = model.advection(only_motion, motion[:, t])
            only_source = only_source + source[:, t]
            motion_frames.append(only_motion)
            source_frames.append(only_source)
        predictions = {
            "model": output["pred"].float(),
            "persistence": x[:, -1:].expand(-1, ds.target_len, -1, -1, -1),
            "motion_only": torch.stack(motion_frames, dim=1),
            "source_only": torch.stack(source_frames, dim=1),
        }
        target = truth.cpu().numpy()
        for name, pred in predictions.items():
            physical = ds.normalizer.denormalize(pred).cpu().numpy()
            if not np.isfinite(physical).all():
                raise ValueError(f"Non-finite prediction in {name}")
            meters[name].update(physical, target)
            squared[name] += ((physical.astype(np.float64) - target) ** 2).sum(axis=(0, 2, 3, 4))
        samples += len(x)
        pixels += int(truth[:, 0].numel())
        if (i + 1) % 10 == 0:
            print(f"[diagnose] {i + 1}/{len(loader)} batches", flush=True)
    if not samples:
        raise ValueError("No batches evaluated")
    scores = {name: meter.compute() for name, meter in meters.items()}
    for name in names:
        for t in range(ds.target_len):
            scores[name]["per_leadtime"][str(t + 1)]["mse"] = float(squared[name][t] / pixels)
    minutes = float(interval if interval is not None else ds.cfg.get(
        "frame_interval_minutes", ds.context.get("frame_interval_minutes", 1)))
    model_leads, base_leads = scores["model"]["per_leadtime"], scores["persistence"]["per_leadtime"]
    worse = [t for t in range(1, ds.target_len + 1)
             if model_leads[str(t)]["csi"] < base_leads[str(t)]["csi"]]
    return {
        "operator_probes": operator_probes(),
        "checkpoint": os.path.abspath(checkpoint), "split": "val", "n_samples": samples,
        "partial": samples < len(ds), "amp": amp, "amp_dtype": str(amp_dtype) if amp else None,
        "frame_interval_minutes": minutes, "interval_source": "CLI" if interval else "config/context (verify provenance)",
        "thresholds": ds.thresholds, "scores": scores,
        "first_lead_below_persistence": worse[0] if worse else None,
        "all_leads_below_persistence": worse,
        "branch_stat_columns": ["motion_magnitude_configured_units", "source_mean_normalized",
                                "source_abs_mean_normalized", "advection_abs_change_normalized",
                                "advected_mean_normalized"],
        "branch_stats": (branch_sums / pixels).tolist(),
        "advection": dict(cfg.get("model", {}).get("advection", {})),
        "resolved_dataset": ds.cfg.to_dict(),
        "note": "motion_only/source_only reuse learned fields without retraining; not paper ablation results",
    }


def write_report(report, out):
    os.makedirs(out, exist_ok=True)
    save_json(os.path.join(out, "forecast_diagnostics.json"), report)
    rows = []
    for name, result in report["scores"].items():
        for lead, scores in result["per_leadtime"].items():
            for threshold, detail in scores["per_threshold"].items():
                rows.append({"variant": name, "lead": int(lead),
                             "minutes": int(lead) * report["frame_interval_minutes"],
                             "threshold": threshold, "csi_macro": scores["csi"],
                             "mse": scores["mse"], **detail})
    with open(os.path.join(out, "lead_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, result in report["scores"].items():
        steps = sorted(map(int, result["per_leadtime"]))
        x = [s * report["frame_interval_minutes"] for s in steps]
        for ax, metric in zip(axes, ("csi", "mse")):
            ax.plot(x, [result["per_leadtime"][str(s)][metric] for s in steps], label=name)
            ax.set(xlabel="Lead time (min; configured interval)", ylabel=metric.upper())
            ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "lead_skill.png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/diagnostics/forecast")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--interval-minutes", type=float)
    parser.add_argument("--amp", action="store_true", help="match saved training AMP dtype; default FP32")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)
    report = diagnose(load_config(args.config, overrides=args.set), args.checkpoint,
                      torch.device(args.device), args.max_batches, args.interval_minutes, args.amp)
    write_report(report, args.out)
    for name, result in report["scores"].items():
        print(f"{name:12s} CSI={result['csi']:.6f}")
    print("First lead below persistence:", report["first_lead_below_persistence"])
    print("Report:", args.out)


if __name__ == "__main__":
    main()

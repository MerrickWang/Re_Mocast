"""UT-08: small-sample overfitting + end-to-end training pipeline smoke test."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np
import pytest
import torch

from mocast.datasets import build_dataset, build_dataloader
from mocast.engine import Trainer
from mocast.models import MoCast
from mocast.utils.config import Config
from .conftest import small_model_cfg


def _dry_run_config(tmp_path: str, epochs: int = 300, n_sequences: int = 10,
                    size: int = 32) -> Config:
    cfg = Config({
        "dataset": {
            "name": "synthetic",
            "n_sequences": n_sequences,
            "frames_per_sequence": 25,
            "height": size,
            "width": size,
            "stride": 5,
            "input_len": 5,
            "target_len": 20,
            "resize": None,
            "crop": {"mode": "none"},
            "normalization": {"mode": "minmax", "vmin": 0.0, "vmax": 1.0},
            "thresholds": [0.2, 0.4],
            "motion_mask_threshold": 0.2,
            "filter": {"enabled": True, "min_intensity": 0.2, "min_rain_pixels": 4,
                       "min_rain_frames": 1},
            "split_config": {"mode": "ratio", "ratios": {"train": 0.8, "val": 0.2}, "seed": 2026},
            "cache_dir": os.path.join(tmp_path, "cache"),
            "cache_index": False,
            "return_physical": True,
        },
        "model": {**small_model_cfg(), "target_size": [size, size], "output_len": 20},
        "train": {
            "epochs": epochs, "batch_size": 4, "num_workers": 0, "amp": False,
            "log_every": 0, "val_every": 1, "grad_clip": 1.0,
            "optimizer": {"name": "adamw", "lr": 3e-3, "weight_decay": 0.0},
            "scheduler": {"name": "cosine", "min_lr": 1e-5},
            "early_stop": {"patience": 1000, "monitor": "val_csi", "mode": "max"},
            "device": "cpu",
        },
        "loss": {"lambda_motion": 0.01,
                 "precip": {"name": "mse"},
                 "motion": {"enabled": True, "mode": "mse_mask", "reduce": "any"}},
        "eval": {"thresholds": [0.2, 0.4], "perceptual": False, "lpips": False,
                 "max_batches": 2, "pool_sizes": [4, 16]},
        "run": {"name": "ut08", "output_root": tmp_path, "seed": 2026, "deterministic": True},
    })
    return cfg


@pytest.mark.slow
def test_ut08_small_sample_overfitting(tmp_path: Any, device: torch.device) -> None:
    """8-16 sequences must be learnable: the training loss has to drop sharply."""
    cfg = _dry_run_config(str(tmp_path))
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    assert 6 <= len(train_ds) <= 16, f"expected 8-16 training sequences, got {len(train_ds)}"

    train_loader = build_dataloader(cfg, "train", dataset=train_ds)
    val_loader = build_dataloader(cfg, "val", dataset=val_ds, shuffle=False)
    model = MoCast(cfg["model"])
    trainer = Trainer(cfg, model, train_loader, val_loader,
                      normalizer=train_ds.normalizer,
                      output_dir=os.path.join(str(tmp_path), "run"), device=device)
    trainer.epochs = int(cfg["train"]["epochs"])
    history = []
    for epoch in range(int(cfg["train"]["epochs"])):
        stats = trainer.train_epoch(epoch)
        history.append(stats["loss"])
        if stats["loss"] < 2e-3:
            break
    assert len(history) >= 2
    initial = float(np.mean(history[:2]))
    final = float(np.mean(history[-2:]))
    assert final < 0.5 * initial, f"loss barely moved: {initial:.6f} -> {final:.6f}"
    assert np.isfinite(final)

    # the artefacts required by the acceptance criteria must exist
    trainer.save_checkpoint(os.path.join(str(tmp_path), "run", "checkpoints", "last.pt"), 0, final)
    assert os.path.exists(os.path.join(str(tmp_path), "run", "checkpoints", "last.pt"))


@pytest.mark.slow
def test_ut08_overfit_beats_persistence(tmp_path: Any, device: torch.device) -> None:
    """A short fit on synthetic data must be better than the persistence baseline."""
    cfg = _dry_run_config(str(tmp_path), epochs=250, n_sequences=12)
    train_ds = build_dataset(cfg, "train")
    loader = build_dataloader(cfg, "train", dataset=train_ds)
    model = MoCast(cfg["model"])
    trainer = Trainer(cfg, model, loader, None, normalizer=train_ds.normalizer,
                      output_dir=os.path.join(str(tmp_path), "run2"), device=device)
    for epoch in range(int(cfg["train"]["epochs"])):
        stats = trainer.train_epoch(epoch)
        if stats["loss"] < 2e-3:
            break
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        x = batch["input"].to(device)
        target = batch["target"].to(device)
        pred = model(x)["pred"]
        model_mse = torch.mean((pred - target) ** 2).item()
        persistence = torch.mean((x[:, -1:] - target) ** 2).item()
    assert model_mse < persistence, f"model {model_mse:.5f} vs persistence {persistence:.5f}"


@pytest.mark.slow
def test_ut08_mocast_plus_training_step(tmp_path: Any, device: torch.device) -> None:
    """The diffusion stage must train through the same engine (FR-DIFF-01)."""
    from mocast.models import MoCastPlus

    cfg = _dry_run_config(str(tmp_path), epochs=2, n_sequences=10)
    cfg["model"] = {"name": "mocast_plus", "backbone": cfg["model"].to_dict(),
                    "diffusion": {"num_steps": 20, "schedule": "linear"},
                    "unet": {"base_channels": 8, "channel_mults": [1, 2], "num_blocks": 1,
                             "cond_dim": 16, "temporal_attention": True, "conv_gru": True,
                             "num_heads": 2}}
    train_ds = build_dataset(cfg, "train")
    loader = build_dataloader(cfg, "train", dataset=train_ds)
    model = MoCastPlus(cfg["model"]["backbone"], cfg["model"]["diffusion"], cfg["model"]["unet"])
    trainer = Trainer(cfg, model, loader, None, normalizer=train_ds.normalizer,
                      output_dir=os.path.join(str(tmp_path), "run3"), device=device)
    stats = trainer.train_epoch(0)
    assert np.isfinite(stats["loss"]) and stats["loss"] > 0
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        base, samples = model.sample(batch["input"].to(device), num_samples=2, steps=2)
    assert base.shape == samples.shape[:1] + samples.shape[2:]
    assert torch.isfinite(samples).all()

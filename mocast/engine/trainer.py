"""Training engine: optimisation loop, validation, checkpointing, logging."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..losses import MoCastLoss, build_motion_mask
from ..metrics import Evaluator
from ..utils.config import Config
from ..utils.misc import AverageMeter, ensure_dir, save_json

__all__ = ["Trainer", "build_optimizer", "build_scheduler", "load_checkpoint"]


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-2))
    betas = tuple(cfg.get("betas", (0.9, 0.999)))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(cfg.get("momentum", 0.9)),
                               weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any],
                    steps_per_epoch: int = 0) -> Optional[Any]:
    name = str(cfg.get("name", "cosine")).lower()
    epochs = int(cfg.get("epochs", 200))
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    if name in ("none", "constant", ""):
        return None
    if name.startswith("cosine"):
        eta_min = float(cfg.get("min_lr", 1e-6))
        base = [lambda e, epochs=epochs, eta_min=eta_min: 1.0]
        if warmup_epochs > 0:
            def warmup(epoch: int) -> float:
                return float(epoch + 1) / float(max(warmup_epochs, 1)) if epoch < warmup_epochs else 1.0
        else:
            def warmup(epoch: int) -> float:
                return 1.0
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: warmup(e) * (
                eta_min / max(float(optimizer.param_groups[0]["initial_lr"]), 1e-12)
                + (1 - eta_min / max(float(optimizer.param_groups[0]["initial_lr"]), 1e-12))
                * (1 + math.cos(math.pi * min(e / max(epochs - 1, 1), 1.0))) / 2
            )
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.get("step_size", 50)),
                                               gamma=float(cfg.get("gamma", 0.1)))
    raise ValueError(f"Unknown scheduler '{name}'")


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[Any] = None,
                    map_location: str = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


class Trainer:
    """Deterministic training loop with AMP, gradient clipping and early stopping."""

    def __init__(self, cfg: Any, model: nn.Module, train_loader: Any = None,
                 val_loader: Any = None, normalizer: Any = None,
                 output_dir: str = "outputs/run", device: Optional[torch.device] = None,
                 criterion: Optional[nn.Module] = None) -> None:
        self.cfg = Config(cfg)
        train_cfg = Config(self.cfg.get("train", {}) or {})
        self.train_cfg = train_cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalizer = normalizer
        self.output_dir = ensure_dir(output_dir)
        self.epochs = int(train_cfg.get("epochs", 200))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.amp = bool(train_cfg.get("amp", True)) and self.device.type == "cuda"
        amp_dtype = str(train_cfg.get("amp_dtype", "float16")).lower()
        if amp_dtype in ("bf16", "bfloat16"):
            self.amp_dtype = torch.bfloat16
        elif amp_dtype in ("fp16", "float16", "half"):
            self.amp_dtype = torch.float16
        else:
            raise ValueError(f"Unknown amp_dtype '{amp_dtype}' (use float16 or bfloat16)")
        self.teacher_forcing = bool(self.cfg.get("model", {}).get("reconstruction", {}).get(
            "teacher_forcing", False))
        self.log_every = int(train_cfg.get("log_every", 50))
        self.val_every = int(train_cfg.get("val_every", 1))
        self.max_val_batches = train_cfg.get("max_val_batches", None)
        self.motion_cfg = Config(self.cfg.get("loss", {}).get("motion", {}) or {})
        self.lambda_motion = float(self.cfg.get("loss", {}).get("lambda_motion", 0.01))
        self.eval_cfg = Config(self.cfg.get("eval", {}) or {})
        self.latent_size = self._infer_latent_size()
        # Eq. 13: theta is a *dataset provided* significance threshold - fall back to
        # dataset.motion_mask_threshold instead of silently using a wrong default.
        if "threshold" not in self.motion_cfg:
            dataset_threshold = self.cfg.get("dataset", {}).get("motion_mask_threshold", None)
            if dataset_threshold is not None:
                self.motion_cfg["threshold"] = float(dataset_threshold)
        if self.latent_size is not None:
            # Eq. 13 downsamples the precipitation mask onto the latent motion grid;
            # the size is injected here so that the criterion is built correctly.
            self.motion_cfg.setdefault("latent_size", list(self.latent_size))

        self.optimizer = build_optimizer(self.model, Config(train_cfg.get("optimizer", {}) or {}))
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        sched_cfg = Config(train_cfg.get("scheduler", {}) or {})
        sched_cfg.setdefault("epochs", self.epochs)
        self.scheduler = build_scheduler(self.optimizer, sched_cfg,
                                          steps_per_epoch=len(train_loader) if train_loader else 0)
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = MoCastLoss(
                lambda_motion=self.lambda_motion,
                motion_cfg=self.motion_cfg,
                precip_cfg=self.cfg.get("loss", {}).get("precip", {}),
            )
        early = Config(train_cfg.get("early_stop", {}) or {})
        self.patience = int(early.get("patience", 20))
        self.monitor = str(early.get("monitor", "val_csi"))
        self.monitor_mode = str(early.get("mode", "max"))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.history: List[Dict[str, Any]] = []
        self.best_score = -np.inf if self.monitor_mode == "max" else np.inf
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.global_step = 0
        self.start_epoch = 0
        self._writer = None
        try:  # optional tensorboard
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(os.path.join(self.output_dir, "tensorboard"))
        except Exception:  # pragma: no cover - optional dependency
            self._writer = None

    # ------------------------------------------------------------------ utils
    def _infer_latent_size(self) -> Optional[Tuple[int, int]]:
        model = getattr(self.model, "backbone", self.model)  # MoCast+ wraps MoCast
        downsample = getattr(model, "downsample", None)
        target = getattr(model, "target_size", None)
        if downsample and target:
            return (int(target[0]) // int(downsample), int(target[1]) // int(downsample))
        return None

    def _move(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value.to(self.device, non_blocking=True)
            else:
                out[key] = value
        return out

    def _loss_kwargs(self, batch: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        backbone = outputs.get("backbone", outputs)
        mean_motion = backbone.get("mm", None)
        if mean_motion is not None:
            kwargs["mean_motion"] = mean_motion
            kwargs["total_motion"] = backbone.get("ma", None)
        frames_phys = batch.get("full_phys", None)
        input_len = int(self.cfg.get("model", {}).get("input_len", 5))
        if frames_phys is not None:
            kwargs["frames"] = frames_phys[:, :input_len]
        kwargs["advection"] = getattr(self.model, "advection", None) or getattr(
            getattr(self.model, "backbone", None), "advection", None)
        kwargs["frames_normalized"] = batch.get("input", None)
        return kwargs

    def _target(self, batch: Dict[str, Any]) -> torch.Tensor:
        return batch["target"]

    # ------------------------------------------------------------------ train
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meters = {k: AverageMeter(k) for k in ("loss", "precip", "motion")}
        t0 = time.time()
        for i, batch in enumerate(self.train_loader):
            batch = self._move(batch)
            target = self._target(batch)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):
                outputs = self.model(batch["input"], y=batch["full"] if "full" in batch else None)
                losses = self.criterion(outputs, target, **self._loss_kwargs(batch, outputs))
                if "diffusion_loss" in outputs:
                    losses["loss"] = losses["loss"] + outputs["diffusion_loss"]
                loss = losses["loss"]
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1
            meters["loss"].update(float(loss.detach()))
            meters["precip"].update(float(losses["precip"].detach()))
            if "motion" in losses:
                meters["motion"].update(float(losses["motion"].detach()))
            if self.log_every and (i + 1) % self.log_every == 0:
                print(f"    step {i + 1}/{len(self.train_loader)} "
                      f"loss={meters['loss'].avg:.6f} "
                      f"precip={meters['precip'].avg:.6f}")
        stats = {k: m.avg for k, m in meters.items()}
        stats["epoch_time"] = time.time() - t0
        stats["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return stats

    # --------------------------------------------------------------- validate
    @torch.no_grad()
    def validate(self, epoch: int = 0) -> Dict[str, Any]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        thresholds = self.eval_cfg.get("thresholds", None)
        if not thresholds:
            dataset = getattr(self.val_loader, "dataset", None)
            thresholds = getattr(dataset, "thresholds", []) if dataset is not None else []
        data_range = None
        if self.normalizer is not None and getattr(self.normalizer, "mode", "none") == "minmax":
            data_range = float(self.normalizer.vmax - self.normalizer.vmin)
        evaluator = Evaluator(
            thresholds,
            pool_sizes=tuple(self.eval_cfg.get("pool_sizes", (4, 16))),
            perceptual=bool(self.eval_cfg.get("perceptual", False)),
            data_range=data_range or 1.0,
            device=str(self.device),
            enable_lpips=bool(self.eval_cfg.get("lpips", False)),
        )
        loss_meter = AverageMeter("val_loss")
        n_batches = len(self.val_loader)
        if self.max_val_batches:
            n_batches = min(n_batches, int(self.max_val_batches))
        for i, batch in enumerate(self.val_loader):
            if i >= n_batches:
                break
            batch = self._move(batch)
            target = self._target(batch)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):
                outputs = self.model(batch["input"])
                losses = self.criterion(outputs, target, **self._loss_kwargs(batch, outputs))
            loss_meter.update(float(losses["loss"].detach()))
            pred = outputs["pred"] if "pred" in outputs else outputs["base_pred"]
            pred = self._denormalize(pred)
            gt = batch.get("target_phys", target)
            evaluator.update(pred, gt)
        metrics = evaluator.compute()
        metrics["val_loss"] = loss_meter.avg
        return metrics

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return tensor
        if getattr(self.normalizer, "mode", "none") == "minmax":
            return tensor * (self.normalizer.vmax - self.normalizer.vmin) + self.normalizer.vmin
        if getattr(self.normalizer, "mode", "none") in ("zscore", "meanstd", "standard"):
            return tensor * self.normalizer.std + self.normalizer.mean
        return tensor

    # -------------------------------------------------------------------- fit
    def fit(self, max_epochs: Optional[int] = None) -> Dict[str, Any]:
        epochs = int(max_epochs or self.epochs)
        self.cfg.save(os.path.join(self.output_dir, "config.yaml"))
        log_path = os.path.join(self.output_dir, "metrics.jsonl")
        for epoch in range(self.start_epoch, epochs):
            train_stats = self.train_epoch(epoch)
            record: Dict[str, Any] = {"epoch": epoch, **train_stats}
            if self.val_loader is not None and (epoch + 1) % max(self.val_every, 1) == 0:
                val_metrics = self.validate(epoch)
                record.update({f"val_{k}": v for k, v in val_metrics.items() if
                               not isinstance(v, dict)})
                record["val_metrics"] = val_metrics
                score = self._monitor_value(record)
                if self._is_improvement(score):
                    self.best_score = score
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(os.path.join(self.output_dir, "checkpoints", "best.pt"),
                                         epoch, score)
                else:
                    self.epochs_without_improvement += 1
            if self.scheduler is not None:
                self.scheduler.step()
            self.history.append(record)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
            self._log_epoch(record)
            self.save_checkpoint(os.path.join(self.output_dir, "checkpoints", "last.pt"),
                                 epoch, self._monitor_value(record))
            if self.epochs_without_improvement >= self.patience:
                print(f"  early stopping at epoch {epoch} "
                      f"(best epoch {self.best_epoch}, score {self.best_score:.6f})")
                break
        summary = {
            "best_epoch": self.best_epoch,
            "best_score": float(self.best_score) if np.isfinite(self.best_score) else None,
            "monitor": self.monitor,
            "epochs_run": len(self.history),
            "output_dir": self.output_dir,
        }
        save_json(os.path.join(self.output_dir, "summary.json"), summary)
        if self._writer is not None:  # pragma: no cover - optional dependency
            self._writer.flush()
            self._writer.close()
        return summary

    # -------------------------------------------------------------- plumbing
    def _monitor_value(self, record: Dict[str, Any]) -> float:
        key = self.monitor
        value = record.get(key, record.get(f"val_{key}", None))
        if value is None:
            value = record.get("val_loss", record.get("loss", np.nan))
        return float(value) if value is not None and np.isfinite(float(value)) else (
            -np.inf if self.monitor_mode == "max" else np.inf)

    def _is_improvement(self, score: float) -> bool:
        if not np.isfinite(score):
            return False
        if self.monitor_mode == "max":
            return score > self.best_score + 1e-8
        return score < self.best_score - 1e-8

    def _log_epoch(self, record: Dict[str, Any]) -> None:
        parts = [f"epoch {record['epoch']}"]
        for key in ("loss", "precip", "motion", "lr", "epoch_time"):
            if key in record:
                parts.append(f"{key}={record[key]:.6g}")
        for key in ("val_csi", "val_hss", "val_csi_p4", "val_csi_p16", "val_ssim", "val_loss"):
            if key in record and np.isfinite(record[key]):
                parts.append(f"{key}={record[key]:.6f}")
        print("  " + "  ".join(parts))
        if self._writer is not None:  # pragma: no cover
            for key, value in record.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    self._writer.add_scalar(key, value, record["epoch"])

    def save_checkpoint(self, path: str, epoch: int, score: float) -> str:
        ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "score": float(score) if np.isfinite(score) else None,
            "config": self.cfg.to_dict(),
            "monitor": self.monitor,
        }, path)
        return path

    def load(self, path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        payload = load_checkpoint(path, self.model, self.optimizer if load_optimizer else None,
                                  map_location=str(self.device))
        self.start_epoch = int(payload.get("epoch", -1)) + 1
        self.best_score = float(payload.get("score", self.best_score) or self.best_score)
        return payload

"""Reconstruction loss (FR-LOSS-01) and motion trend-consistency loss (FR-LOSS-02)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PrecipMSELoss", "MotionTrendConsistencyLoss", "MoCastLoss", "build_motion_mask"]


class PrecipMSELoss(nn.Module):
    """Pixel-wise MSE between predicted and ground-truth frames (``L_precip``)."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
        return F.mse_loss(pred, target, reduction=self.reduction)


def build_motion_mask(frames: torch.Tensor, threshold: float,
                      smooth_window: int = 1, latent_size: Optional[Sequence[int]] = None,
                      reduce: str = "average") -> torch.Tensor:
    """Binary mask ``W`` highlighting precipitation-effective regions.

    Implements ``W^t = Binary[max(avg(X^t), avg(X^{t+1})) > theta]`` (Eq. 13) and
    the subsequent average-pooling downsampling onto the latent motion grid.

    Args:
        frames: ``[B,T,1,H,W]`` precipitation frames in **physical units**.
        threshold: dataset provided significance threshold ``theta``.
        smooth_window: ``avg`` window (1 disables the local averaging).
        latent_size: target ``(h_d, w_d)``; ``None`` keeps the pixel grid.
        reduce: average coverage (paper), or legacy ``any`` / ``majority``.
    """
    if frames.ndim != 5:
        raise ValueError(f"expected [B,T,1,H,W], got {tuple(frames.shape)}")
    b, t, c, h, w = frames.shape
    if t < 2:
        raise ValueError("motion mask needs at least two frames")
    frames = frames.float()
    if smooth_window and int(smooth_window) > 1:
        k = int(smooth_window)
        if k % 2 == 0:
            raise ValueError("smooth_window must be odd")
        frames = F.avg_pool2d(frames.reshape(b * t, c, h, w), kernel_size=k, stride=1,
                              padding=k // 2).reshape(b, t, c, h, w)
    pair = torch.maximum(frames[:, :-1], frames[:, 1:])
    mask = (pair > float(threshold)).to(frames.dtype)
    if latent_size is not None:
        lh, lw = int(latent_size[0]), int(latent_size[1])
        flat = mask.reshape(b * (t - 1), c, h, w)
        pooled = F.adaptive_avg_pool2d(flat, (lh, lw))
        if str(reduce).lower() == "average":
            mask = pooled
        elif str(reduce).lower() in ("majority", "mean"):
            mask = (pooled > 0.5).to(frames.dtype)
        elif str(reduce).lower() == "any":
            mask = (pooled > 0).to(frames.dtype)
        else:
            raise ValueError(f"Unknown mask reduction: {reduce}")
        mask = mask.reshape(b, t - 1, c, lh, lw)
    return mask


class MotionTrendConsistencyLoss(nn.Module):
    """``L_motion`` of Eq. (13): temporal coherence of the learned mean motion.

    Only precipitation-effective regions contribute, and the magnitude is
    normalised by the number of active cells (per sample) so that the weight
    ``lambda`` stays comparable across batches with different rain coverage.

    ``mode``:
        ``mse_mask`` (paper default) - masked temporal smoothness of ``M_mean``.
        ``warp_consistency`` - self-supervised alternative (risk R8): the total
            motion must advect ``X^t`` onto ``X^{t+1}`` on the mask.
        ``both`` - sum of the two terms.
    """

    def __init__(self, threshold: float = 16.0, mode: str = "mse_mask", smooth_window: int = 1,
                 latent_size: Optional[Sequence[int]] = None, reduce: str = "average",
                 detach_target: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.mode = str(mode).lower()
        self.smooth_window = int(smooth_window)
        self.latent_size = tuple(latent_size) if latent_size else None
        self.reduce = str(reduce)
        self.detach_target = bool(detach_target)
        self.eps = float(eps)

    # ------------------------------------------------------------------ api
    def forward(self, mean_motion: torch.Tensor, frames: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                total_motion: Optional[torch.Tensor] = None,
                advection: Optional[nn.Module] = None,
                frames_normalized: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """``mean_motion``: ``[B,T-1,2,h,w]``.

        Returns a dict with ``loss`` and the diagnostic terms so that every
        variant can be reported separately (risk R8).
        """
        if mean_motion.ndim != 5 or mean_motion.shape[1] < 2:
            raise ValueError("motion loss needs [B,T-1,2,h,w] with T-1 >= 2")
        # Squared differences and spatial sums must not overflow in AMP.
        mean_motion = mean_motion.float()
        if mask is None:
            if frames is None:
                raise ValueError("motion loss requires either `frames` or `mask`")
            mask = build_motion_mask(frames, self.threshold, self.smooth_window,
                                     self.latent_size, self.reduce)
        else:
            mask = mask.to(mean_motion.dtype)
        if mask.ndim == 5:
            mask = mask[:, :, 0]                                    # [B,T-1,h,w]
        mask = mask.to(mean_motion.dtype)

        # ---- trend consistency of the mean component (Eq. 13) ------------
        target = mean_motion[:, 1:]
        source = mean_motion[:, :-1]
        if self.detach_target:
            target = target.detach()
        diff = (target - source) ** 2                                # [B,T-2,2,h,w]
        weight = mask[:, :-1]                                        # t = 1..T-2 pairs
        weighted = (diff * weight[:, :, None]).sum(dim=(2, 3, 4))    # [B,T-2]
        # normalise by the number of active cells (per sample) so that lambda stays
        # comparable across batches with different rain coverage
        active = weight.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        # Eq.13 sums the squared vector norm over all times and cells, then
        # divides once by mask mass. No extra channel/time averaging.
        smoothness = weighted.sum(dim=1) / active
        out = {"smoothness": smoothness.mean()}

        warp_term = torch.zeros((), dtype=mean_motion.dtype, device=mean_motion.device)
        if self.mode in ("warp_consistency", "warp", "both"):
            if total_motion is None or advection is None or frames_normalized is None:
                raise ValueError("warp_consistency needs total_motion, advection and frames")
            b, steps = total_motion.shape[0], total_motion.shape[1]
            if frames_normalized.shape[1] < steps + 1:
                raise ValueError("warp_consistency needs T >= steps + 1 frames")
            current = frames_normalized[:, :steps].reshape(
                b * steps, 1, *frames_normalized.shape[-2:])
            nxt = frames_normalized[:, 1:steps + 1].reshape(
                b * steps, 1, *frames_normalized.shape[-2:])
            motion = total_motion.reshape(b * steps, 2, *total_motion.shape[-2:])
            if tuple(motion.shape[-2:]) != tuple(current.shape[-2:]):
                motion = F.interpolate(motion, size=current.shape[-2:], mode="bilinear",
                                       align_corners=False)
            warped = advection(current, motion)
            residual = (warped - nxt) ** 2
            pair_weight = mask[:, :steps].reshape(-1, *mask.shape[-2:])[:, None]
            if tuple(pair_weight.shape[-2:]) != tuple(residual.shape[-2:]):
                pair_weight = F.interpolate(pair_weight, size=residual.shape[-2:], mode="nearest")
            warp_term = ((residual * pair_weight).sum()
                         / (pair_weight.sum() * residual.shape[1] + self.eps))
            out["warp_consistency"] = warp_term

        if self.mode in ("mse_mask", "smoothness", "mask"):
            loss = out["smoothness"]
        elif self.mode in ("warp_consistency", "warp"):
            loss = warp_term
        elif self.mode == "both":
            loss = out["smoothness"] + warp_term
        else:
            raise ValueError(f"Unknown motion loss mode '{self.mode}'")
        out["loss"] = loss
        return out


class MoCastLoss(nn.Module):
    """``L_final = L_precip + lambda * L_motion`` (Eq. 14)."""

    def __init__(self, lambda_motion: float = 0.01, motion_cfg: Optional[Dict[str, Any]] = None,
                 precip_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.lambda_motion = float(lambda_motion)
        precip_cfg = dict(precip_cfg or {})
        self.precip = PrecipMSELoss(reduction=str(precip_cfg.get("reduction", "mean")))
        motion_cfg = dict(motion_cfg or {})
        self.motion_enabled = bool(motion_cfg.get("enabled", True))
        self.motion = MotionTrendConsistencyLoss(
            threshold=float(motion_cfg.get("threshold", 16.0)),
            mode=str(motion_cfg.get("mode", "mse_mask")),
            smooth_window=int(motion_cfg.get("smooth_window", 1)),
            latent_size=motion_cfg.get("latent_size"),
            reduce=str(motion_cfg.get("reduce", "average")),
            detach_target=bool(motion_cfg.get("detach_target", False)),
        )

    def forward(self, outputs: Dict[str, Any], target: torch.Tensor,
                mean_motion: Optional[torch.Tensor] = None,
                frames: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                total_motion: Optional[torch.Tensor] = None,
                advection: Optional[nn.Module] = None,
                frames_normalized: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if isinstance(outputs, dict):
            # MoCast+ returns the deterministic forecast as ``base_pred``
            pred = outputs.get("pred", None)
            if pred is None:
                pred = outputs.get("base_pred", None)
            if pred is None:
                raise KeyError("model outputs contain neither 'pred' nor 'base_pred'")
        else:
            pred = outputs
        precip = self.precip(pred, target)
        total = precip
        result: Dict[str, torch.Tensor] = {"precip": precip, "loss": precip}
        if self.motion_enabled and mean_motion is not None:
            motion_out = self.motion(mean_motion, frames=frames, mask=mask,
                                     total_motion=total_motion, advection=advection,
                                     frames_normalized=frames_normalized)
            total = total + self.lambda_motion * motion_out["loss"]
            result.update({f"motion_{k}": v for k, v in motion_out.items() if k != "loss"})
            result["motion"] = motion_out["loss"]
        result["loss"] = total
        return result

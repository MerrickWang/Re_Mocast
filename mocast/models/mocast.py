"""MoCast: the deterministic model (section 6 of the specification, Eq. 12).

Forward contract (section 5 "张量接口规范")::

    X            [B,T,1,H,W]      normalised historical frames (T = 5)
    H_m          [B,T,d,h_d,w_d]  shared encoder representation (time preserved)
    M_a          [B,T-1,2,h_d,w_d] total motion over the input interval
    E_s          [B,T-1,C_s,h_s,w_s] gated source-sink embedding
    W_gate       [B,T-1,S,h_s,w_s] softmax weights over experts
    M_hat        [B,P,2,H,W]      future motion (pixel units, target grid)
    S_hat        [B,P,1,H,W]      future source-sink
    Y_hat        [B,P,1,H,W]      predicted frames

    Y_hat^{t+1} = Adv(Y_hat^t, M_hat^{t->t+1}) + S_hat^{t+1},   Y_hat^T = X^T
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .advection import Advection
from .encoder import SpatialEncoder
from .msm import MSM
from .pmm import PMM
from .temporal import TemporalEmbedding, TemporalPredictor
from ..utils.config import Config

__all__ = ["MoCast"]


class MoCast(nn.Module):
    """Physics-guided turbulent motion model for precipitation nowcasting."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg = Config(cfg or {})
        self.cfg = cfg
        enc = Config(cfg.get("encoder", {}) or {})
        self.input_len = int(cfg.get("input_len", 5))
        self.output_len = int(cfg.get("output_len", 20))
        target_size = cfg.get("target_size", None)
        self.target_size: Optional[Sequence[int]] = tuple(target_size) if target_size else None
        downsample = int(enc.get("downsample", 2))
        self.downsample = downsample
        latent_size = None
        if self.target_size is not None:
            latent_size = (int(self.target_size[0]) // downsample, int(self.target_size[1]) // downsample)

        # ---------------------------------------------------------- encoder
        self.encoder = SpatialEncoder(
            in_channels=int(enc.get("in_channels", 1)),
            base_channels=int(enc.get("base_channels", 32)),
            channels=enc.get("channels", (64, 64)),
            downsample=downsample,
            norm=str(enc.get("norm", "group")),
            act=str(enc.get("act", "gelu")),
        )
        dim = int(cfg.get("latent_channels", self.encoder.out_channels))
        self.latent_channels = dim

        # -------------------------------------------------------------- PMM
        ab = Config(cfg.get("ablation", {}) or {})
        pmm_cfg = Config(cfg.get("pmm", {}) or {})
        pmm_cfg["use_mean"] = bool(ab.get("use_mean", True))
        pmm_cfg["use_fluctuation"] = bool(ab.get("use_fluctuation", True))
        pmm_cfg["decompose"] = bool(ab.get("decompose", True))
        potential_cfg = Config(pmm_cfg.get("potential", {}) or {})
        potential_cfg["use_helmholtz"] = bool(ab.get("helmholtz", True))
        pmm_cfg["potential"] = potential_cfg
        self.pmm = PMM(dim=dim, cfg=pmm_cfg, target_size=self.target_size, latent_size=latent_size)

        # -------------------------------------------------------------- MSM
        msm_cfg = Config(cfg.get("msm", {}) or {})
        msm_cfg["multiscale"] = bool(ab.get("msm_multiscale", msm_cfg.get("multiscale", True)))
        msm_cfg["gate_mode"] = (
            str(msm_cfg.get("gate_mode", "softmax"))
            if bool(ab.get("msm_gating", True)) else "uniform"
        )
        msm_cfg["motion_channels"] = self.pmm.motion_channels
        msm_cfg["downsample"] = downsample
        self.msm = MSM(cfg=msm_cfg, in_channels=int(enc.get("in_channels", 1)),
                       downsample=downsample)
        self.source_channels = int(msm_cfg.get("out_channels", msm_cfg.get("channels", dim)))

        # ------------------------------------------------------- prediction
        temp_cfg = Config(cfg.get("temporal", {}) or {})
        pred_cfg = Config(cfg.get("prediction", {}) or {})
        embed_dim = int(temp_cfg.get("embed_dim", dim))
        steps = self.input_len - 1
        self.motion_embed = TemporalEmbedding(2, embed_dim, int(temp_cfg.get("num_blocks", 2)),
                                              int(temp_cfg.get("kernel_size", 3)))
        self.source_embed = TemporalEmbedding(self.source_channels, embed_dim,
                                              int(temp_cfg.get("num_blocks", 2)),
                                              int(temp_cfg.get("kernel_size", 3)))
        common = dict(latent_size=latent_size, target_size=self.target_size,
                      spatial_blocks=int(pred_cfg.get("spatial_blocks", 2)),
                      mode=str(pred_cfg.get("mode", "oneshot")))
        self.motion_predictor = TemporalPredictor(embed_dim, steps, self.output_len, 2, **common)
        self.source_predictor = TemporalPredictor(embed_dim, steps, self.output_len, 1, **common)

        # ---------------------------------------------------------- advection
        self.advection = Advection.from_config(Config(cfg.get("advection", {}) or {}))
        self.teacher_forcing = bool((cfg.get("reconstruction", {}) or {}).get("teacher_forcing", False))
        self.motion_clip = cfg.get("prediction", {}).get("motion_clip", None)
        self.use_source_sink = bool(ab.get("use_source_sink", True))

    # ------------------------------------------------------------------ api
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None,
                return_aux: bool = True) -> Dict[str, Any]:
        """``x``: ``[B,T,1,H,W]`` normalised input frames.

        ``y`` (optional, training only) supplies the ground-truth frames for
        teacher forcing / scheduled sampling.
        """
        if x.ndim != 5:
            raise ValueError(f"MoCast expects [B,T,1,H,W], got {tuple(x.shape)}")
        b, t = x.shape[0], x.shape[1]
        if t != self.input_len:
            raise ValueError(f"expected {self.input_len} input frames, got {t}")
        if self.target_size is not None and tuple(x.shape[-2:]) != tuple(self.target_size):
            raise ValueError(
                f"input spatial size {tuple(x.shape[-2:])} does not match model.target_size "
                f"{tuple(self.target_size)}; set dataset.resize (FR-DATA-03: 128x128 by default) "
                "or model.target_size accordingly")

        # 1) shared spatial encoder ------------------------------------
        h = self.encoder(x)                                     # [B,T,d,h_d,w_d]

        # 2) PMM: mean / fluctuation / total motion --------------------
        pmm_out = self.pmm(h, x)
        motion_features: List[torch.Tensor] = pmm_out["motion_features"]

        # 3) MSM: motion-guided source-sink (frames X[:,1:T] per spec) --
        msm_out = self.msm(x[:, 1:], motion_features)           # [B,T-1,C_s,h_s,w_s]
        if tuple(msm_out["source"].shape[-2:]) != tuple(pmm_out["total"].shape[-2:]):
            raise ValueError(
                "source-sink grid "
                f"{tuple(msm_out['source'].shape[-2:])} does not match the motion grid "
                f"{tuple(pmm_out['total'].shape[-2:])}; check msm.stem_stride vs "
                "encoder.downsample (the specification forbids implicit broadcasting)")

        # 4) temporal embedding + prediction ---------------------------
        emb_motion = self.motion_embed(pmm_out["total"])
        emb_source = self.source_embed(msm_out["source"])
        motion_hat = self.motion_predictor(emb_motion)           # [B,P,2,H,W]
        source_hat = self.source_predictor(emb_source)           # [B,P,1,H,W]
        if not self.use_source_sink:                             # ablation A6
            source_hat = torch.zeros_like(source_hat)
        if self.motion_clip is not None:
            clip = float(self.motion_clip)
            motion_hat = torch.clamp(motion_hat, -clip, clip)

        # 5) differentiable advection + source-sink residual -----------
        predictions = self.reconstruct(x[:, -1], motion_hat, source_hat,
                                       targets=y[:, self.input_len:] if y is not None else None)

        out: Dict[str, Any] = {
            "pred": predictions,
            "motion": motion_hat,
            "source": source_hat,
            "mm": pmm_out["mean"],
            "mf": pmm_out["fluctuation"],
            "ma": pmm_out["total"],
            "motion_features": motion_features,
            "gate": msm_out["gate"],
            "latents": h,
        }
        if return_aux:
            for key in ("cue", "phi", "psi", "curl_free", "divergence_free", "attention",
                        "fluctuation"):
                if key in pmm_out:
                    out[key] = pmm_out[key]
            out["sin"] = None
        return out

    def reconstruct(self, first_frame: torch.Tensor, motion: torch.Tensor,
                    source: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Unroll Eq. (12) over the prediction horizon."""
        steps = motion.shape[1]
        current = first_frame
        outputs: List[torch.Tensor] = []
        for i in range(steps):
            prev = current
            if self.training and self.teacher_forcing and targets is not None and i > 0:
                prev = targets[:, i - 1]
            current = self.advection(prev, motion[:, i]) + source[:, i]
            outputs.append(current)
        return torch.stack(outputs, dim=1)

    # -------------------------------------------------------------- helpers
    def motion_for_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """Motion used by the trend-consistency loss (mean component, Eq. 13)."""
        return outputs["mm"]

    @property
    def motion_unit(self) -> str:
        return self.advection.unit

    def describe(self) -> Dict[str, Any]:
        from ..utils.misc import count_parameters

        return {
            "params": count_parameters(self),
            "latent_channels": self.latent_channels,
            "downsample": self.downsample,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "advection": {"unit": self.advection.unit, "semantics": self.advection.semantics,
                          "mode": self.advection.mode,
                          "padding_mode": self.advection.padding_mode,
                          "align_corners": self.advection.align_corners},
        }

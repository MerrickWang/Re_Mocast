"""Physics-guided Motion Modeling module (PMM).

Pipeline (Fig. 1 of the paper)::

    H_m ---------------> local patch attention -> motion cue D~
     |                                                  |
     |                                    Helmholtz (phi, A) -> M_mean
     +--> DWT -> directional enhancement -> IDWT -> f_f -> M_fluctuation
                                        |
                              M_total = M_mean + M_fluctuation
                                        |
                              multi-scale motion features for MSM

Ablation switches (section 8.3) are first-class arguments so that A1-A3 and A7
run through exactly the same code path as the full model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .attention import LocalPatchAttention
from .helmholtz import GradientHead, HelmholtzMeanMotion
from .motion_pyramid import MotionPyramid
from .wavelet import WaveletFluctuationBlock

__all__ = ["PMM", "LocalPatchAttention", "HelmholtzMeanMotion", "WaveletFluctuationBlock",
           "MotionPyramid", "GradientHead"]


class PMM(nn.Module):
    """Learn turbulent motions from mean and fluctuating components."""

    def __init__(self, dim: int, cfg: Optional[Dict[str, Any]] = None,
                 target_size: Optional[Sequence[int]] = None,
                 latent_size: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        cfg = dict(cfg or {})
        self.cfg = cfg
        self.dim = int(dim)
        self.use_mean = bool(cfg.get("use_mean", True))
        self.use_fluctuation = bool(cfg.get("use_fluctuation", True))
        self.decompose = bool(cfg.get("decompose", True))
        self.use_attention = bool(cfg.get("use_attention", True))

        # ---- Helmholtz / mean motion ------------------------------------
        pot = dict(cfg.get("potential", {}) or {})
        use_helmholtz = bool(pot.get("use_helmholtz", True))
        self.patch_size = int(cfg.get("patch_size", 4))
        self.window = int(cfg.get("window", 3))
        cue_channels = 2 * self.window * self.window
        motion_scale = cfg.get("motion_scale", "auto")
        if isinstance(motion_scale, str) and motion_scale == "auto":
            if target_size is not None and latent_size is not None:
                motion_scale = float(int(target_size[0]) // int(latent_size[0]))
            else:
                motion_scale = 2.0
        self.motion_scale = float(motion_scale)

        if self.use_attention:
            self.attention: Optional[LocalPatchAttention] = LocalPatchAttention(
                dim=self.dim,
                patch_size=self.patch_size,
                patch_stride=cfg.get("patch_stride", None),
                window=self.window,
                heads=int(cfg.get("heads", 4)),
                attn_dim=int(cfg.get("attn_dim", 64)),
                scale=str(cfg.get("attn_scale", "sqrt")),
                dropout=float(cfg.get("attn_dropout", 0.0)),
            )
        else:
            self.attention = None
            cue_channels = self.dim

        self.mean_block = HelmholtzMeanMotion(
            cue_channels,
            hidden=int(pot.get("hidden", 64)),
            depth=int(pot.get("depth", 2)),
            kernel=str(pot.get("kernel", "central")),
            padding=str(pot.get("padding", "replicate")),
            motion_scale=self.motion_scale,
            use_helmholtz=use_helmholtz,
        )
        self.direct_head: Optional[GradientHead] = None
        if not self.decompose:  # ablation A3: no Reynolds decomposition at all
            self.direct_head = GradientHead(cue_channels, hidden=int(pot.get("hidden", 64)),
                                            out_scale=self.motion_scale)

        # ---- wavelet fluctuation ----------------------------------------
        wav = dict(cfg.get("wavelet", {}) or {})
        self.fluctuation_block = WaveletFluctuationBlock(
            dim=self.dim,
            hidden=int(wav.get("hidden", 64)),
            basis=str(wav.get("basis", "haar")),
            ll_mode=str(wav.get("ll_mode", "zero")),
            enhance=str(wav.get("enhance", "sigmoid")),
            grad_kernel=str(wav.get("grad_kernel", "central")),
            padding=str(pot.get("padding", "replicate")),
            motion_scale=self.motion_scale,
        )

        # ---- multi-scale motion features for the MSM ---------------------
        pyr = dict(cfg.get("pyramid", {}) or {})
        self.pyramid = MotionPyramid(
            in_channels=2,
            channels=int(pyr.get("channels", 64)),
            kernels=pyr.get("kernels", (3, 5, 7)),
            dilations=pyr.get("dilations", (1, 1, 1)),
            mode=str(pyr.get("mode", "separate")),
            num_motion=3,
            norm=str(pyr.get("norm", "group")),
        )

    # ------------------------------------------------------------------ api
    @property
    def motion_channels(self) -> int:
        return self.pyramid.out_channels

    @property
    def num_scales(self) -> int:
        return self.pyramid.num_scales

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> Dict[str, Any]:
        """``h``: ``[B,T,d,h_d,w_d]`` encoder features, ``x``: ``[B,T,1,H,W]`` frames."""
        out: Dict[str, Any] = {}
        if self.attention is not None:
            attn_out = self.attention(h)
            cue = attn_out["cue"]
            out["attention"] = attn_out["attention"]
        else:  # degenerate case used by diagnostics only
            cue = h[:, 1:].transpose(1, 2)  # type: ignore[assignment]
            out["attention"] = None

        if self.decompose:
            mean_out = self.mean_block(cue)
            m_mean = mean_out["mean"]
            if self.use_fluctuation:
                fluct_out = self.fluctuation_block(h, x)
                m_fluct = fluct_out["motion"]
            else:
                fluct_out = {}
                m_fluct = torch.zeros_like(m_mean)  # ablation A2
            if not self.use_mean:  # ablation A1 (mean branch removed)
                m_mean_used = torch.zeros_like(m_fluct)
            else:
                m_mean_used = m_mean
            m_total = m_mean_used + m_fluct
            out.update({k: v for k, v in mean_out.items() if k != "mean"})
            out["fluctuation"] = fluct_out.get("fluctuation")
        else:
            m_total = self.direct_head(cue) if self.direct_head is not None else torch.zeros(
                1, dtype=cue.dtype, device=cue.device)
            m_mean = m_total
            m_fluct = torch.zeros_like(m_total)

        out.update({"cue": cue, "mean": m_mean, "fluctuation": m_fluct, "total": m_total})
        out["motion_features"] = self.pyramid([m_mean, m_fluct, m_total])
        return out

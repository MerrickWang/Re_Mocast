"""Motion-guided Source-sink Modeling module (MSM).

``E = sum_s W_gate^s * Resize(E'_s)``  (Eq. 11), with ``S=3`` experts by default
(paper hyper-parameter study: S=3 is optimal; ablation A4 keeps the code path but
uses a single expert).

Temporal alignment (risk R4): the module consumes ``X[:, 1:T]``, i.e. the frame at
the *end* of each motion interval, so motion step ``t -> t+1`` is paired with the
frame ``t+1`` - verified by ``UT-06``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import resize_tensor
from .experts import MotionGuidedExpert, SourceSinkStem
from .gating import MotionAdaptiveGating

__all__ = ["MSM", "SourceSinkStem", "MotionGuidedExpert", "MotionAdaptiveGating"]


class MSM(nn.Module):
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, in_channels: int = 1,
                 downsample: int = 2) -> None:
        super().__init__()
        cfg = dict(cfg or {})
        self.cfg = cfg
        num_experts = int(cfg.get("num_experts", 3))
        self.num_experts = num_experts
        self.use_multiscale = bool(cfg.get("multiscale", True))
        self.gate_mode = str(cfg.get("gate_mode", "softmax"))
        kernels = list(cfg.get("kernels", (3, 5, 7)))[:max(num_experts, 1)] if self.use_multiscale \
            else list(cfg.get("kernels", (3, 5, 7)))[:1]
        dilations = list(cfg.get("dilations", (1, 1, 1)))[:len(kernels)]
        self.channels = int(cfg.get("channels", 64))
        norm = str(cfg.get("norm", "group"))
        act = str(cfg.get("act", "gelu"))
        self.kernels = kernels
        self.dilations = dilations
        self.num_scales = len(kernels)

        stem_stride = cfg.get("stem_stride", "auto")
        if isinstance(stem_stride, str) and stem_stride.lower() == "auto":
            stem_stride = int(downsample)   # keep source-sink aligned with the motion grid
        self.stem = SourceSinkStem(
            in_channels=in_channels, channels=self.channels, kernels=kernels,
            dilations=dilations, stride=int(stem_stride),
            norm=norm, act=act, num_blocks=int(cfg.get("stem_blocks", 1)),
        )
        self.experts = nn.ModuleList([
            MotionGuidedExpert(
                channels=self.channels,
                motion_channels=int(cfg.get("motion_channels", 64)),
                num_motion=int(cfg.get("num_motion", 3)),
                hidden=int(cfg.get("expert_hidden", self.channels)),
                norm=norm, act=act,
                modulation_hidden=int(cfg.get("modulation_hidden", 32)),
                cfg_init_zero=bool(cfg.get("modulation_init_zero", False)),
            )
            for _ in kernels
        ])
        self.gating = MotionAdaptiveGating(
            motion_channels_per_scale=int(cfg.get("motion_channels", 64)),
            num_experts=self.num_scales,
            hidden=int(cfg.get("gate_hidden", 32)),
            mode=self.gate_mode, act=act,
        )
        self.out = nn.Conv2d(self.channels, int(cfg.get("out_channels", self.channels)), 1)

    # ------------------------------------------------------------------ api
    def forward(self, x: torch.Tensor, motion_features: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """``x``: ``[B,T,1,H,W]`` (use ``X[:,1:T]``); ``motion_features``: ``S`` tensors."""
        motions = list(motion_features)[: self.num_scales]
        if not motions:
            raise ValueError("MSM needs at least one motion feature tensor")
        if len(motions) < self.num_scales:  # ablation A4 may pass fewer scales
            motions = motions + [motions[-1]] * (self.num_scales - len(motions))
        b, t = x.shape[0], x.shape[1]
        flat_x = x.reshape(b * t, *x.shape[2:])
        flat_motions = [m.reshape(b * t, *m.shape[2:]) for m in motions]
        candidates = self.stem(flat_x)
        size = candidates[0].shape[-2:]
        expert_outputs = []
        for candidate, motion, expert in zip(candidates, flat_motions, self.experts):
            aligned = self._align_motion(motion, candidate.shape)
            if not self.use_multiscale:  # ablation A4: single-scale expert
                aligned = [aligned[0]] * expert.num_motion
            expert_outputs.append(expert(candidate, aligned))
        gate = self.gating(flat_motions, size)                # [B*T,S,h,w]
        fused = None
        for s, out in enumerate(expert_outputs):
            weight = gate[:, s: s + 1]
            resized = resize_tensor(out, (int(size[-2]), int(size[-1])), mode="bilinear",
                                    align_corners=False)
            contribution = weight * resized
            fused = contribution if fused is None else fused + contribution
        assert fused is not None
        source = self.out(fused)
        gate = gate.reshape(b, t, self.num_scales, *gate.shape[-2:])
        candidates = [c.reshape(b, t, *c.shape[1:]) for c in candidates]
        return {"source": source.reshape(b, t, *source.shape[1:]), "gate": gate,
                "candidates": candidates}

    @staticmethod
    def _align_motion(motion: torch.Tensor, shape: Sequence[int]) -> List[torch.Tensor]:
        """Split the ``3*Cm`` channel motion tensor into the three motion groups."""
        channels = motion.shape[2] // 3
        groups = [motion[:, :, i * channels:(i + 1) * channels] for i in range(3)]
        out = []
        for group in groups:
            if tuple(group.shape[-2:]) != (int(shape[-2]), int(shape[-1])):
                group = F.interpolate(group, size=(int(shape[-2]), int(shape[-1])),
                                      mode="bilinear", align_corners=False)
            out.append(group)
        return out

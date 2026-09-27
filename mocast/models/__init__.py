"""Model zoo: MoCast, MoCast+ and the advection / persistence baselines."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..utils.config import Config
from ..utils.registry import MODELS
from .advection import Advection
from .baselines import BlockMatchingFlow, OpticalFlowBaseline, PersistenceBaseline
from .encoder import SpatialEncoder
from .mocast import MoCast
from .mocast_plus import MoCastPlus

__all__ = [
    "MoCast",
    "MoCastPlus",
    "SpatialEncoder",
    "Advection",
    "PersistenceBaseline",
    "AdvectionBaseline",
    "OpticalFlowBaseline",
    "BlockMatchingFlow",
    "build_model",
]


@MODELS(name="mocast")
def _build_mocast(cfg: Dict[str, Any]) -> nn.Module:
    return MoCast(cfg)


@MODELS(name="mocast_plus")
def _build_mocast_plus(cfg: Dict[str, Any]) -> nn.Module:
    cfg = Config(cfg)
    return MoCastPlus(
        backbone_cfg=cfg.get("backbone", cfg.get("mocast", {})),
        diffusion_cfg=cfg.get("diffusion", {}),
        unet_cfg=cfg.get("unet", {}),
    )


class AdvectionBaseline(nn.Module):
    """Advection-only baseline: persistent motion obtained from the last interval."""

    def __init__(self, output_len: int = 20, unit: str = "pixel",
                 semantics: str = "displacement", align_corners: bool = True) -> None:
        super().__init__()
        self.output_len = int(output_len)
        self.advection = Advection(unit=unit, semantics=semantics, align_corners=align_corners)

    def forward(self, x: torch.Tensor, flow: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        if flow is None:
            raise ValueError("AdvectionBaseline requires an external motion field")
        current = x[:, -1]
        outputs = []
        for i in range(self.output_len):
            current = self.advection(current, flow[:, min(i, flow.shape[1] - 1)])
            outputs.append(current)
        return {"pred": torch.stack(outputs, dim=1), "motion": flow}


@MODELS(name="persistence")
def _build_persistence(cfg: Dict[str, Any]) -> nn.Module:
    return PersistenceBaseline(output_len=int(cfg.get("output_len", 20)))


@MODELS(name="optical_flow")
def _build_optical_flow(cfg: Dict[str, Any]) -> nn.Module:
    return OpticalFlowBaseline(
        output_len=int(cfg.get("output_len", 20)),
        search_radius=int(cfg.get("search_radius", 4)),
        block_size=int(cfg.get("block_size", 8)),
        smooth_kernel=int(cfg.get("smooth_kernel", 3)),
    )


@MODELS(name="advection")
def _build_advection(cfg: Dict[str, Any]) -> nn.Module:
    return AdvectionBaseline(output_len=int(cfg.get("output_len", 20)),
                             unit=str(cfg.get("unit", "pixel")),
                             semantics=str(cfg.get("semantics", "displacement")),
                             align_corners=bool(cfg.get("align_corners", True)))


def build_model(cfg: Any) -> nn.Module:
    """Instantiate a model from the ``model`` section of the configuration."""
    model_cfg = Config(cfg.get("model", cfg) if isinstance(cfg, dict) else cfg)
    name = str(model_cfg.get("name", "mocast")).lower()
    builder = MODELS.get(name)
    return builder(model_cfg)

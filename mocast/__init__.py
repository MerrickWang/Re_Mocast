"""MoCast: Learning Turbulent Motions Under Physical Guidance for Precipitation Nowcasting.

Reference implementation of the AAAI-2026 paper by Wu et al. (DOI: 10.1609/aaai.v40i19.38628).

The package is organised following the specification in
``MoCast代码复现需求文档`` (section 11. 建议代码结构)::

    mocast/
      configs/     # dataset / model / train / ablation YAML
      datasets/    # unified adapters (sevir, meteonet, shanghai, synthetic)
      models/      # encoder, pmm, msm, temporal, advection, mocast, mocast_plus
      losses/      # reconstruction + motion trend-consistency
      metrics/     # CSI / HSS / pooled CSI / LPIPS / SSIM
      tests/       # unit, shape, synthetic-motion and overfit tests
      tools/       # train / eval / visualize / benchmark / build_manifest
      outputs/     # run artefacts (configs, logs, checkpoints, metrics)
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]

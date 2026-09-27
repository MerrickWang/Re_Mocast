"""Ad-hoc smoke check: build MoCast and run a forward/backward pass.

Usage::

    python -m mocast.tools.smoke

Kept dependency free (no pytest) so it can be run in any environment.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

import torch


def _default_cfg() -> Dict:
    return {
        "input_len": 5,
        "output_len": 20,
        "target_size": [128, 128],
        "latent_channels": 64,
        "encoder": {"in_channels": 1, "base_channels": 32, "channels": [64, 64], "downsample": 2},
        "pmm": {
            "patch_size": 4, "window": 3, "heads": 4, "attn_dim": 64,
            "potential": {"hidden": 64, "depth": 2, "kernel": "central", "padding": "replicate"},
            "wavelet": {"basis": "haar", "ll_mode": "zero", "enhance": "sigmoid", "hidden": 32},
            "pyramid": {"mode": "separate", "kernels": [3, 5, 7], "dilations": [1, 1, 1],
                        "channels": 48},
        },
        "msm": {"num_experts": 3, "kernels": [3, 5, 7], "dilations": [1, 1, 1], "channels": 48,
                "gate_hidden": 32, "modulation_hidden": 24},
        "temporal": {"embed_dim": 64, "num_blocks": 2},
        "prediction": {"mode": "oneshot", "spatial_blocks": 2},
        "advection": {"unit": "pixel", "semantics": "displacement", "mode": "bilinear",
                      "padding_mode": "border", "align_corners": True},
        "ablation": {},
    }


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(description="MoCast forward/backward smoke test")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--backward", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    from mocast.models import MoCast

    device = torch.device(args.device)
    model = MoCast(_default_cfg()).to(device)
    x = torch.randn(args.batch, 5, 1, args.size, args.size, device=device)
    y = torch.randn(args.batch, 20, 1, args.size, args.size, device=device)

    out = model(x, y=y)
    print(f"device            : {device}")
    print(f"parameters        : {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    for key, value in out.items():
        if torch.is_tensor(value):
            print(f"{key:18s}: {tuple(value.shape)}")
    loss = torch.nn.functional.mse_loss(out["pred"], y)
    if args.backward:
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        finite = all(torch.isfinite(g).all() for g in grads)
        print(f"loss              : {loss.item():.6f}")
        print(f"grads finite      : {finite} ({len(grads)} tensors)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

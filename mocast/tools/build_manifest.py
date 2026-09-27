"""Build split manifests + data audit for a dataset (FR-DATA-04 / FR-DATA-05).

Usage::

    python -m mocast.tools.build_manifest -c mocast/configs/dataset/sevir.yaml \\
        --out data/manifests/sevir.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from ..datasets import build_dataset
from ..utils.config import load_config
from ..utils.misc import ensure_dir, save_json
from .common import base_parser, timestamp

__all__ = ["main"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the dataset split manifest + audit")
    parser.add_argument("-c", "--config", nargs="+", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--max-windows", type=int, default=2000, help="windows stored per split")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    payload: Dict[str, Any] = {
        "generated_at": timestamp(),
        "config": {k: v for k, v in cfg.to_dict().items() if not k.startswith("_")},
        "splits": {},
    }
    for split in args.splits:
        dataset = build_dataset(cfg, split)
        payload["splits"][split] = {
            "summary": dataset.summary(),
            "windows": [
                {"store": w.store, "start": int(w.start), "length": int(w.length),
                 "store_name": dataset.stores[w.store].name}
                for w in dataset.windows[: args.max_windows]
            ],
            "n_windows": len(dataset),
        }
        print(f"[mocast] {split:5s}: {len(dataset):6d} windows, "
              f"audit kept={dataset.audit.get('n_windows_kept')} "
              f"dropped={dataset.audit.get('n_windows_dropped_empty')}")
    out = args.out or os.path.join(str(cfg.get("dataset", {}).get("cache_dir", "outputs/cache")),
                                   f"manifest_{cfg.get('dataset', {}).get('name', 'dataset')}.json")
    ensure_dir(os.path.dirname(os.path.abspath(out)) or ".")
    save_json(out, payload)
    print(f"[mocast] wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Training entry point (P0-P7 of the implementation plan).

Example::

    python -m mocast.tools.train -c mocast/configs/experiments/sevir_mocast.yaml
    python -m mocast.tools.train -c mocast/configs/train/dry_run.yaml \\
        mocast/configs/model/mocast_small.yaml --set train.epochs=2
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from ..engine import Trainer
from ..utils.config import Config, load_config
from ..utils.misc import count_parameters, save_json
from .common import (base_parser, build_datasets, build_loaders, build_model_from_config,
                     dump_dataset_summaries, prepare_run)

__all__ = ["main"]


def main(argv: Optional[List[str]] = None) -> int:
    args = base_parser("Train MoCast / MoCast+").parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    cfg, run_dir, device = prepare_run(cfg, args, default_name="mocast")

    print(f"[mocast] run dir : {run_dir}")
    print(f"[mocast] device  : {device}")

    # ---------------------------------------------------------------- data
    train_ds = build_datasets(cfg, ["train"])["train"]
    val_ds = build_datasets(cfg, ["val"])["val"]
    dump_dataset_summaries({"train": train_ds, "val": val_ds}, run_dir)
    print(f"[mocast] dataset : {train_ds.name}  train windows={len(train_ds)}  "
          f"val windows={len(val_ds)}")

    loaders = build_loaders(cfg, {"train": train_ds, "val": val_ds})
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Empty split: check dataset.raw_root / filter settings")

    # --------------------------------------------------------------- model
    model = build_model_from_config(cfg)
    print(f"[mocast] model   : {type(model).__name__}  "
          f"params={count_parameters(model) / 1e6:.2f} M")
    if hasattr(model, "describe"):
        save_json(os.path.join(run_dir, "model_description.json"), model.describe())  # type: ignore[attr-defined]

    # --------------------------------------------------------------- train
    trainer = Trainer(cfg, model, loaders["train"], loaders["val"],
                      normalizer=train_ds.normalizer, output_dir=run_dir, device=device)
    resume = cfg.get("run", {}).get("resume", None)
    if resume:
        trainer.load(str(resume))
        print(f"[mocast] resumed from {resume} (epoch {trainer.start_epoch})")
    summary = trainer.fit(max_epochs=cfg.get("train", {}).get("epochs", None))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

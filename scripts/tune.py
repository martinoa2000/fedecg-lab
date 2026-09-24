#!/usr/bin/env python
"""Score a training setting on the validation fold only (baseline tuning).

The place to try a wider model, augmentation, another loss or an ensemble
before committing to it: this script trains exactly as
`scripts/train_centralized.py` does, but never loads a test prediction, so
settings can be compared as often as needed without leaking the test fold.
Once a setting wins here, put it in a config and train it for real.

Writes, named after the config (or `--name`):

    results/tables/tuning.csv              one row per setting, replaced on rerun
    results/tables/tuning_<name>_history.csv   per-epoch validation curve (first member)

Usage:
    python scripts/tune.py --config tune_wide.yaml
    python scripts/tune.py --config path/to/generated.yaml --name my_try
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fedecg.config import load_config
from fedecg.data.constants import SUPERCLASSES
from fedecg.data.pipeline import prepare_data
from fedecg.models.resnet1d import build_model, count_parameters
from fedecg.paths import CACHE_DIR, PTBXL_DIR, TABLES_DIR, ensure_dir
from fedecg.training.augment import AUGMENTATIONS
from fedecg.training.centralized import ensemble_seeds, predict_ensemble, train_centralized
from fedecg.training.loop import resolve_device
from fedecg.training.metrics import per_class_auroc

TUNING_COLUMNS = (
    "run",
    "setting",
    "base_channels",
    "n_parameters",
    "crop_samples",
    "augment",
    "loss",
    "seeds",
    "val_macro_auroc",
    *(f"val_auroc_{c}" for c in SUPERCLASSES),
    "member_val_macro_auroc",
    "best_epoch",
    "seconds",
)


def describe_augment(augment: dict | None) -> str:
    """`amplitude 0.1, noise 0.05`, or `none`."""
    active = [
        f"{k} {augment[k]}" for k in AUGMENTATIONS if augment and float(augment.get(k, 0)) > 0
    ]
    return ", ".join(active) or "none"


def record_tuning(row: dict, path: Path) -> pd.DataFrame:
    """Insert or replace `row` (matched on `run`) in the tuning table."""
    table = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=list(TUNING_COLUMNS))
    table = table[table["run"] != row["run"]]
    new = pd.DataFrame([row])
    table = new if table.empty else pd.concat([table, new], ignore_index=True)
    table = table[[c for c in TUNING_COLUMNS if c in table.columns]]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(path, index=False)
    return table


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Config to score on validation")
    parser.add_argument("--name", default=None, help="Row name (default: the config's stem)")
    parser.add_argument("--root", type=Path, default=PTBXL_DIR, help="PTB-XL directory")
    parser.add_argument(
        "--cache", type=Path, default=CACHE_DIR / "ptbxl_100hz.npz", help="Waveform cache"
    )
    parser.add_argument("--no-cache", action="store_true", help="Always parse WFDB records")
    parser.add_argument("--tables", type=Path, default=TABLES_DIR, help="Output for CSVs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    config = load_config(args.config)
    name = args.name or Path(args.config).stem
    train_cfg = config["training"]
    started = time.perf_counter()

    data = prepare_data(config, root=args.root, cache_path=None if args.no_cache else args.cache)
    device = resolve_device(train_cfg.get("device", "auto"))
    n_parameters = count_parameters(build_model(config["model"]))
    seeds = ensemble_seeds(config)
    print(f"tuning {name}: {n_parameters:,} parameters, seeds {seeds}, on {device}", flush=True)

    members = train_centralized(config, data, device)
    prob, true = predict_ensemble(
        [m.model for m in members], data.x_val, data.y_val, device, train_cfg
    )
    per_class = per_class_auroc(true, prob)
    member_scores = [m.result.best_score for m in members]
    score = float(np.nanmean(per_class))
    print(
        f"validation macro AUROC {score:.4f} (members: {', '.join(f'{s:.4f}' for s in member_scores)})"
    )

    tables = ensure_dir(args.tables)
    pd.DataFrame(members[0].result.history).round(4).to_csv(
        tables / f"tuning_{name}_history.csv", index=False
    )
    row = {
        "run": name,
        "setting": config.get("experiment", {}).get("setting", name),
        "base_channels": config["model"].get("base_channels"),
        "n_parameters": n_parameters,
        "crop_samples": train_cfg.get("crop_samples"),
        "augment": describe_augment(train_cfg.get("augment")),
        "loss": train_cfg.get("loss", "bce"),
        "seeds": len(seeds),
        "val_macro_auroc": score,
        **{f"val_auroc_{c}": float(v) for c, v in zip(SUPERCLASSES, per_class, strict=True)},
        "member_val_macro_auroc": float(np.mean(member_scores)),
        "best_epoch": float(np.mean([m.result.best_epoch for m in members])),
        "seconds": round(time.perf_counter() - started, 1),
    }
    record_tuning(row, tables / "tuning.csv")
    print(f"Row written to {tables / 'tuning.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

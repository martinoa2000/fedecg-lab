#!/usr/bin/env python
"""Train the centralized baseline (roadmap phase 3).

Pipeline:

1. Load PTB-XL labels and waveforms (through the `.npz` cache after the first run).
2. Split with the official folds; optionally subsample the training set.
3. Band-pass every split; fit the per-lead standardizer on train only.
4. Train the 1D ResNet with early stopping on validation macro AUROC.
5. Tune one F1 threshold per class on validation, then evaluate test once.

Writes, named after the config (`centralized` by default):

    results/tables/<name>_test_metrics.csv   per-class AUROC / F1 / threshold on test
    results/tables/<name>_history.csv        per-epoch losses and val AUROC
    checkpoints/<name>.pt                    best weights, standardizer, thresholds

and logs the same to MLflow when `tracking.enabled` is true.

Usage:
    python scripts/train_centralized.py                      # full run
    python scripts/train_centralized.py --config smoke.yaml  # 2-minute check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from fedecg.config import load_config
from fedecg.data.preprocess import preprocess_splits
from fedecg.data.ptbxl import load_dataset, split_by_folds
from fedecg.models.resnet1d import build_model, count_parameters
from fedecg.paths import CACHE_DIR, CHECKPOINT_DIR, PTBXL_DIR, TABLES_DIR, ensure_dir
from fedecg.seed import new_generator, set_seed
from fedecg.training.loop import fit, make_loader, predict, resolve_device
from fedecg.training.metrics import best_f1_thresholds, metrics_table
from fedecg.training.tracking import start_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="centralized.yaml", help="Experiment config")
    parser.add_argument("--root", type=Path, default=PTBXL_DIR, help="PTB-XL directory")
    parser.add_argument(
        "--cache", type=Path, default=CACHE_DIR / "ptbxl_100hz.npz", help="Waveform cache"
    )
    parser.add_argument("--no-cache", action="store_true", help="Always parse WFDB records")
    parser.add_argument("--tables", type=Path, default=TABLES_DIR, help="Output for CSVs")
    parser.add_argument(
        "--checkpoints", type=Path, default=CHECKPOINT_DIR, help="Output for weights"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    config = load_config(args.config)
    name = Path(args.config).stem
    data_cfg, train_cfg = config["data"], config["training"]
    fs = data_cfg["sampling_rate_hz"]
    seed = int(config["seed"])
    set_seed(seed)

    meta, arrays = load_dataset(
        args.root,
        sampling_rate=fs,
        cache_path=None if args.no_cache else args.cache,
        progress=True,
    )
    split = split_by_folds(
        meta,
        train_folds=data_cfg["train_folds"],
        val_fold=data_cfg["val_fold"],
        test_fold=data_cfg["test_fold"],
    )
    train_ids = split.train.to_numpy()
    if data_cfg.get("subsample") is not None and data_cfg["subsample"] < len(train_ids):
        rng = new_generator(seed)
        train_ids = np.sort(rng.choice(train_ids, size=data_cfg["subsample"], replace=False))
    train, val, test = (arrays.select(ids) for ids in (train_ids, split.val, split.test))
    print(f"{len(train)} train, {len(val)} val, {len(test)} test records")

    x_train, others, standardizer = preprocess_splits(
        train.signals, {"val": val.signals, "test": test.signals}, data_cfg["preprocess"], fs=fs
    )

    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg.get("num_workers", 0))
    train_loader = make_loader(
        x_train, train.labels, batch_size=batch_size, shuffle=True, seed=seed, num_workers=workers
    )
    val_loader = make_loader(others["val"], val.labels, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(others["test"], test.labels, batch_size=batch_size, shuffle=False)

    device = resolve_device(train_cfg.get("device", "auto"))
    model = build_model(config["model"]).to(device)
    print(f"resnet1d with {count_parameters(model):,} parameters on {device}")

    with start_run(config.get("tracking", {}), run_name=name) as tracker:
        tracker.log_params(config)
        tracker.log_params({"n_train": len(train), "n_parameters": count_parameters(model)})

        result = fit(
            model,
            train_loader,
            val_loader,
            train_cfg,
            device,
            on_epoch=lambda row: tracker.log_metrics(
                {k: v for k, v in row.items() if k != "epoch"}, step=int(row["epoch"])
            ),
            progress=True,
        )
        print(f"best epoch {result.best_epoch}: val macro AUROC {result.best_score:.4f}")

        val_prob, val_true = predict(model, val_loader, device)
        thresholds = best_f1_thresholds(val_true, val_prob)
        test_prob, test_true = predict(model, test_loader, device)
        table = pd.DataFrame(metrics_table(test_true, test_prob, thresholds)).T
        table.index.name = "superclass"
        print(table.round(4).to_string())

        tables = ensure_dir(args.tables)
        metrics_path = tables / f"{name}_test_metrics.csv"
        history_path = tables / f"{name}_history.csv"
        table.round(4).to_csv(metrics_path)
        pd.DataFrame(result.history).round(4).to_csv(history_path, index=False)

        checkpoint_path = ensure_dir(args.checkpoints) / f"{name}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "standardizer": standardizer.to_dict() if standardizer else None,
                "thresholds": thresholds.tolist(),
                "best_epoch": result.best_epoch,
            },
            checkpoint_path,
        )

        tracker.log_metrics(
            {
                "best_epoch": result.best_epoch,
                "test_macro_auroc": table.loc["macro", "auroc"],
                "test_macro_f1": table.loc["macro", "f1"],
            }
        )
        for path in (metrics_path, history_path):
            tracker.log_artifact(path)

    print(f"Tables written to {tables}\nCheckpoint written to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

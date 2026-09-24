#!/usr/bin/env python
"""Train the centralized baseline (roadmap phase 3).

Pipeline:

1. Load PTB-XL labels and waveforms (through the `.npz` cache after the first run).
2. Split with the official folds; optionally subsample the training set.
3. Band-pass every split; fit the per-lead standardizer on train only.
4. Train the 1D ResNet with early stopping on validation macro AUROC.
5. Tune one F1 threshold per class on validation, then evaluate test once.

With `privacy.enabled`, step 4 runs DP-SGD (roadmap phase 6) and the epsilon
spent is reported and recorded.

Writes, named after the config (`centralized` by default):

    results/tables/<name>_test_metrics.csv   per-class AUROC / F1 / threshold on test
    results/tables/<name>_history.csv        per-epoch losses and val AUROC
    results/tables/experiments.csv           one summary row, replaced on rerun
    checkpoints/<name>.pt                    best weights, standardizer, thresholds

and logs the same to MLflow when `tracking.enabled` is true.

Usage:
    python scripts/train_centralized.py                      # full run
    python scripts/train_centralized.py --config smoke.yaml  # 2-minute check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from fedecg.config import load_config
from fedecg.data.pipeline import prepare_data
from fedecg.models.resnet1d import build_model, count_parameters
from fedecg.paths import CACHE_DIR, CHECKPOINT_DIR, PTBXL_DIR, TABLES_DIR, ensure_dir
from fedecg.privacy.dp_sgd import private_training_from_config
from fedecg.seed import set_seed
from fedecg.training.evaluate import final_evaluation
from fedecg.training.loop import fit, make_loader, resolve_device
from fedecg.training.results import experiment_row, record_experiment
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
    parser.add_argument(
        "--experiments",
        type=Path,
        default=None,
        help="Summary table to update (default: <tables>/experiments.csv)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    config = load_config(args.config)
    name = Path(args.config).stem
    train_cfg = config["training"]
    seed = int(config["seed"])
    set_seed(seed)
    started = time.perf_counter()

    data = prepare_data(config, root=args.root, cache_path=None if args.no_cache else args.cache)
    print(f"{len(data.x_train)} train, {len(data.x_val)} val, {len(data.x_test)} test records")

    batch_size = int(train_cfg["batch_size"])
    train_loader = make_loader(
        data.x_train,
        data.y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=int(train_cfg.get("num_workers", 0)),
        crop_samples=train_cfg.get("crop_samples"),
    )
    val_loader = make_loader(data.x_val, data.y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(data.x_test, data.y_test, batch_size=batch_size, shuffle=False)

    device = resolve_device(train_cfg.get("device", "auto"))
    model = build_model(config["model"]).to(device)
    print(f"resnet1d with {count_parameters(model):,} parameters on {device}")

    # DP-SGD (phase 6) when the config enables it; the budget covers every
    # planned epoch, whether or not early stopping ends training sooner.
    private = private_training_from_config(
        config, n_records=len(data.x_train), epochs=int(train_cfg["epochs"])
    )
    if private is not None:
        print(
            f"DP-SGD: target epsilon {private.target_epsilon} at delta {private.delta}, "
            f"noise multiplier {private.noise_multiplier:.3f}, clip {private.max_grad_norm}"
        )

    with start_run(config.get("tracking", {}), run_name=name) as tracker:
        tracker.log_params(config)
        tracker.log_params({"n_train": len(data.x_train), "n_parameters": count_parameters(model)})

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
            wrap=private.wrap if private is not None else None,
        )
        print(f"best epoch {result.best_epoch}: val macro AUROC {result.best_score:.4f}")
        epsilon = None
        if private is not None:
            private.unwrap()
            epsilon = private.epsilon()
            print(f"epsilon spent: {epsilon:.3f} at delta {private.delta}")

        table, thresholds = final_evaluation(model, val_loader, test_loader, device, train_cfg)
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
                "standardizer": data.standardizer.to_dict() if data.standardizer else None,
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
                **({"epsilon": epsilon} if epsilon is not None else {}),
            }
        )
        for path in (metrics_path, history_path):
            tracker.log_artifact(path)

    record_experiment(
        experiment_row(
            config,
            name,
            table,
            algorithm="centralized",
            partition="none",
            n_clients=1,
            best_step=result.best_epoch,
            steps_run=len(result.history),
            communication_mb=0.0,
            epsilon=None if epsilon is None else round(epsilon, 3),
            seconds=round(time.perf_counter() - started, 1),
        ),
        args.experiments or tables / "experiments.csv",
    )

    print(f"Tables written to {tables}\nCheckpoint written to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

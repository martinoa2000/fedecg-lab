#!/usr/bin/env python
"""Train across simulated hospitals with FedAvg or FedProx (roadmap phases 4 and 5).

Pipeline:

1. Load and preprocess PTB-XL exactly as the centralized baseline does.
2. Split the training records across hospitals (`federated.partition`).
3. Run Flower's FedAvg or FedProx for up to `federated.rounds` rounds, with
   every client training `federated.local_epochs` passes per round.
4. Keep the global model from the round with the best validation macro AUROC.
5. Tune per-class thresholds on validation, then evaluate test once.

With `privacy.enabled`, every hospital trains with DP-SGD (roadmap phase 6)
and the largest epsilon spent by any hospital is recorded.

Validation and test never leave the server and are never split: every model
in every phase is selected and scored on the same two folds.

Writes, named after the config:

    results/tables/<name>_test_metrics.csv   per-class AUROC / F1 / threshold on test
    results/tables/<name>_history.csv        per-round losses, val AUROC, traffic
    results/tables/<name>_clients.csv        records and label mix per hospital
    results/tables/experiments.csv           one summary row, replaced on rerun
    checkpoints/<name>.pt                    best global weights and thresholds

Usage:
    python scripts/train_federated.py --config fedavg_iid_10.yaml
    python scripts/train_federated.py --config fedprox_site.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from flwr.common import ndarrays_to_parameters

from fedecg.config import load_config
from fedecg.data.partition import partition_from_config, partition_summary
from fedecg.data.pipeline import prepare_data
from fedecg.federated.client import ECGClient, get_weights
from fedecg.federated.simulation import (
    InProcessClientProxy,
    centralized_evaluate_fn,
    run_federated,
)
from fedecg.federated.strategy import build_strategy
from fedecg.models.resnet1d import build_model, count_parameters
from fedecg.paths import CACHE_DIR, CHECKPOINT_DIR, PTBXL_DIR, TABLES_DIR, ensure_dir
from fedecg.privacy.dp_sgd import private_training_from_config
from fedecg.seed import new_generator, set_seed
from fedecg.training.evaluate import final_evaluation
from fedecg.training.loop import (
    bce_from_probs,
    lr_factor,
    make_loader,
    predict_from_config,
    resolve_device,
)
from fedecg.training.metrics import macro_auroc
from fedecg.training.results import experiment_row, record_experiment
from fedecg.training.tracking import start_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="Federated experiment config")
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
    train_cfg, fed_cfg = config["training"], config["federated"]
    seed = int(config["seed"])
    set_seed(seed)
    started = time.perf_counter()

    data = prepare_data(config, root=args.root, cache_path=None if args.no_cache else args.cache)
    partition = partition_from_config(fed_cfg, data.meta_train, data.y_train, new_generator(seed))
    clients_table = partition_summary(partition, data.y_train)
    print(f"{len(data.x_train)} train records across {len(partition)} hospitals:")
    print(clients_table.to_string())

    device = resolve_device(train_cfg.get("device", "auto"))
    model = build_model(config["model"]).to(device)
    print(f"resnet1d with {count_parameters(model):,} parameters on {device}")

    batch_size = int(train_cfg["batch_size"])
    val_loader = make_loader(data.x_val, data.y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(data.x_test, data.y_test, batch_size=batch_size, shuffle=False)

    def evaluate(m: torch.nn.Module) -> tuple[float, float]:
        prob, true = predict_from_config(m, val_loader, device, train_cfg)
        return bce_from_probs(true, prob), macro_auroc(true, prob)

    rounds = int(fed_cfg.get("rounds", train_cfg["epochs"]))
    local_epochs = int(fed_cfg.get("local_epochs", 1))

    # DP-SGD inside every hospital (phase 6): each one calibrates its own noise
    # so that all its rounds together spend at most the target epsilon.
    privates = [
        private_training_from_config(config, n_records=len(idx), epochs=rounds * local_epochs)
        for idx in partition.clients
    ]
    if privates[0] is not None:
        print(
            f"DP-SGD per hospital: target epsilon {privates[0].target_epsilon}, noise multipliers "
            + ", ".join(f"{p.noise_multiplier:.2f}" for p in privates)
        )

    # Each client's seed is derived from the experiment seed and its index, so
    # adding a hospital never changes the batch order of the others.
    clients = [
        InProcessClientProxy(
            str(i),
            ECGClient(
                data.x_train[idx],
                data.y_train[idx],
                model,
                device,
                train_cfg,
                seed=seed * 1000 + i,
                private=privates[i],
            ).to_client(),
        )
        for i, idx in enumerate(partition.clients)
    ]

    base_lr = float(train_cfg["learning_rate"])
    schedule = train_cfg.get("lr_schedule", "constant")
    warmup = float(train_cfg.get("warmup_fraction", 0.0))

    def fit_config(server_round: int) -> dict:
        # Midpoint of the round, so round 1 does not start at a learning rate of 0.
        progress = (server_round - 0.5) / rounds
        return {
            "learning_rate": base_lr * lr_factor(progress, schedule, warmup),
            "local_epochs": local_epochs,
        }

    strategy = build_strategy(
        fed_cfg,
        n_clients=len(clients),
        initial_parameters=ndarrays_to_parameters(get_weights(model)),
        evaluate_fn=centralized_evaluate_fn(model, evaluate),
        on_fit_config_fn=fit_config,
    )

    with start_run(config.get("tracking", {}), run_name=name) as tracker:
        tracker.log_params(config)
        tracker.log_params(
            {
                "n_train": len(data.x_train),
                "n_clients": len(clients),
                "n_parameters": count_parameters(model),
            }
        )
        result = run_federated(
            strategy,
            clients,
            model,
            rounds=rounds,
            patience=int(
                fed_cfg.get("early_stopping_patience", train_cfg["early_stopping_patience"])
            ),
            on_round=lambda row: tracker.log_metrics(
                {k: v for k, v in row.items() if k != "round"}, step=int(row["round"])
            ),
            progress=True,
        )
        print(f"best round {result.best_round}: val macro AUROC {result.best_score:.4f}")
        # The guarantee holds per hospital; report the weakest one.
        epsilon = max(p.epsilon() for p in privates) if privates[0] is not None else None
        if epsilon is not None:
            print(f"epsilon spent (largest over hospitals): {epsilon:.3f}")

        table, thresholds = final_evaluation(model, val_loader, test_loader, device, train_cfg)
        print(table.round(4).to_string())

        tables = ensure_dir(args.tables)
        paths = {
            "metrics": tables / f"{name}_test_metrics.csv",
            "history": tables / f"{name}_history.csv",
            "clients": tables / f"{name}_clients.csv",
        }
        table.round(4).to_csv(paths["metrics"])
        pd.DataFrame(result.history).round(6).to_csv(paths["history"], index=False)
        clients_table.to_csv(paths["clients"])

        checkpoint_path = ensure_dir(args.checkpoints) / f"{name}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "standardizer": data.standardizer.to_dict() if data.standardizer else None,
                "thresholds": thresholds.tolist(),
                "best_round": result.best_round,
                "client_names": partition.names,
            },
            checkpoint_path,
        )

        tracker.log_metrics(
            {
                "best_round": result.best_round,
                "test_macro_auroc": table.loc["macro", "auroc"],
                "test_macro_f1": table.loc["macro", "f1"],
                "communication_mb": result.communication_mb,
                **({"epsilon": epsilon} if epsilon is not None else {}),
            }
        )
        for path in paths.values():
            tracker.log_artifact(path)

    record_experiment(
        experiment_row(
            config,
            name,
            table,
            algorithm=fed_cfg.get("strategy", "fedavg"),
            partition=partition.scheme,
            n_clients=len(clients),
            best_step=result.best_round,
            steps_run=len(result.history),
            communication_mb=round(result.communication_mb, 1),
            epsilon=None if epsilon is None else round(epsilon, 3),
            seconds=round(time.perf_counter() - started, 1),
        ),
        args.experiments or tables / "experiments.csv",
    )
    print(f"Tables written to {tables}\nCheckpoint written to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

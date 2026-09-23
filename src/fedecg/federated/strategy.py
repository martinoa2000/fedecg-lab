"""FedAvg and FedProx, configured from a config's `federated` section.

Both are Flower's own strategy classes; this module only wires them up.

**FedAvg** (McMahan et al., 2017) averages the clients' weights, each weighted
by its number of training records.

**FedProx** (Li et al., 2020) aggregates the same way, but asks each client to
minimize `loss + mu/2 * ||w - w_global||^2`. The proximal term limits how far a
hospital with an unusual label mix can pull its local model away from the
global one within a round. Flower's `FedProx` only sends `proximal_mu` to the
clients; the term itself is applied in `ECGClient.fit`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from flwr.common import Metrics, Parameters
from flwr.server.strategy import FedAvg, FedProx, Strategy

STRATEGIES: tuple[str, ...] = ("fedavg", "fedprox")

EvaluateFn = Callable[[int, list, dict], tuple[float, dict[str, Any]] | None]
FitConfigFn = Callable[[int], dict[str, Any]]


def weighted_mean(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Record-weighted mean of every numeric client metric (e.g. train loss)."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    keys = set.intersection(*(set(m) for _, m in metrics))
    return {k: sum(n * float(m[k]) for n, m in metrics) / total for k in sorted(keys)}


def build_strategy(
    federated_config: Mapping[str, Any],
    *,
    n_clients: int,
    initial_parameters: Parameters,
    evaluate_fn: EvaluateFn,
    on_fit_config_fn: FitConfigFn,
) -> Strategy:
    """Instantiate the strategy named by `federated_config["strategy"]`.

    Args:
        federated_config: Uses `strategy`, `fraction_fit` and, for FedProx,
            `proximal_mu`.
        n_clients: Number of simulated hospitals.
        initial_parameters: The global model at round 0.
        evaluate_fn: Server-side evaluation on the validation split.
        on_fit_config_fn: Per-round client config (learning rate, epochs).
    """
    name = federated_config.get("strategy", "fedavg")
    kwargs: dict[str, Any] = {
        "fraction_fit": float(federated_config.get("fraction_fit", 1.0)),
        # Evaluation happens once, on the server, against the validation split
        # that model selection uses in every phase; clients do not evaluate.
        "fraction_evaluate": 0.0,
        "min_fit_clients": min(2, n_clients),
        "min_evaluate_clients": 0,
        "min_available_clients": n_clients,
        "evaluate_fn": evaluate_fn,
        "on_fit_config_fn": on_fit_config_fn,
        "initial_parameters": initial_parameters,
        "fit_metrics_aggregation_fn": weighted_mean,
    }
    if name == "fedavg":
        return FedAvg(**kwargs)
    if name == "fedprox":
        return FedProx(**kwargs, proximal_mu=float(federated_config["proximal_mu"]))
    raise ValueError(f"Unknown strategy {name!r}; use one of {STRATEGIES}")

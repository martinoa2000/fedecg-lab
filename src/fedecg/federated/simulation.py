"""Run a federated experiment with every hospital in this process.

Flower's usual simulation engine starts one Ray worker per client. Here the
clients run one after another in the main process instead, driven round by
round through the Flower `Strategy` interface:

    configure_fit -> each client's fit -> aggregate_fit -> evaluate

The strategy code (client sampling, weighted averaging, FedProx's `mu`) is
Flower's own; only the transport is replaced by direct calls. The design
choices behind this:

- **Memory.** The preprocessed training set is ~700 MB. Ray workers would
  each hold a copy; sequential clients share one, which fits a 16 GB laptop.
- **Determinism.** No thread or actor scheduling, so a seed fixes the run.
- **Testability.** The whole loop runs in CI on a tiny model without Ray.

What is lost is exactly what the README's Limitations section already
disclaims: network latency, stragglers and dropped clients.

The server keeps the global model that scored best on the validation split
and stops after `patience` rounds without improvement: the same model
selection rule as the centralized baseline, applied per round, not per epoch.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from flwr.client import Client
from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    ReconnectIns,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from torch import nn

from fedecg.federated.client import set_weights
from fedecg.training.loop import EarlyStopping


class InProcessClientProxy(ClientProxy):
    """A `ClientProxy` that calls a Flower `Client` directly, with no transport."""

    def __init__(self, cid: str, client: Client):
        super().__init__(cid)
        self.client = client

    def get_properties(  # noqa: D102
        self, ins: GetPropertiesIns, timeout: float | None, group_id: int | None
    ) -> GetPropertiesRes:
        return self.client.get_properties(ins)

    def get_parameters(  # noqa: D102
        self, ins: GetParametersIns, timeout: float | None, group_id: int | None
    ) -> GetParametersRes:
        return self.client.get_parameters(ins)

    def fit(self, ins: FitIns, timeout: float | None, group_id: int | None) -> FitRes:  # noqa: D102
        return self.client.fit(ins)

    def evaluate(  # noqa: D102
        self, ins: EvaluateIns, timeout: float | None, group_id: int | None
    ) -> EvaluateRes:
        return self.client.evaluate(ins)

    def reconnect(  # noqa: D102
        self, ins: ReconnectIns, timeout: float | None, group_id: int | None
    ) -> DisconnectRes:
        return DisconnectRes(reason="in-process client")


@dataclass
class FederatedResult:
    """Outcome of `run_federated`. The model is left holding the best round."""

    history: list[dict[str, float]] = field(default_factory=list)
    """One dict per round: round, lr, clients, train_loss, val_loss,
    val_macro_auroc, communication_mb, seconds."""
    best_round: int = 0
    best_score: float = float("-inf")
    stopped_early: bool = False
    communication_mb: float = 0.0
    """Total weights sent to and from clients over all rounds run."""


RoundCallback = Callable[[dict[str, float]], None]


def parameters_mb(parameters: Any) -> float:
    """Serialized size of a Flower `Parameters` object in megabytes."""
    return sum(len(t) for t in parameters.tensors) / 1e6


def run_federated(
    strategy: Strategy,
    clients: Sequence[InProcessClientProxy],
    model: nn.Module,
    *,
    rounds: int,
    patience: int,
    on_round: RoundCallback | None = None,
    progress: bool = False,
) -> FederatedResult:
    """Train for up to `rounds` rounds with early stopping on validation AUROC.

    Args:
        strategy: A Flower strategy with `initial_parameters` and an
            `evaluate_fn` returning `(val_loss, {"val_macro_auroc": ...})`.
        clients: All simulated hospitals.
        model: The global model, used to hold the best weights at the end.
        rounds: Maximum number of rounds.
        patience: Rounds without a better validation score before stopping.
        on_round: Called with each round's metrics dict.
        progress: Print one line per round.
    """
    manager = SimpleClientManager()
    for proxy in clients:
        manager.register(proxy)

    parameters = strategy.initialize_parameters(manager)
    if parameters is None:
        raise ValueError("The strategy needs initial_parameters")
    stopper = EarlyStopping(patience)
    result = FederatedResult()

    for server_round in range(1, rounds + 1):
        start = time.perf_counter()
        instructions = strategy.configure_fit(server_round, parameters, manager)
        # Fixed client order: summation order in the average stays the same.
        instructions = sorted(instructions, key=lambda pair: int(pair[0].cid))
        results = [(proxy, proxy.fit(ins, None, server_round)) for proxy, ins in instructions]
        aggregated, fit_metrics = strategy.aggregate_fit(server_round, results, [])
        if aggregated is None:
            raise RuntimeError(f"Round {server_round}: aggregation returned no parameters")

        # Down: the global model to each selected client. Up: each client's update.
        sent = sum(parameters_mb(ins.parameters) for _, ins in instructions)
        received = sum(parameters_mb(res.parameters) for _, res in results)
        result.communication_mb += sent + received
        parameters = aggregated

        evaluated = strategy.evaluate(server_round, parameters)
        if evaluated is None:
            raise ValueError("The strategy needs an evaluate_fn")
        val_loss, val_metrics = evaluated
        score = float(val_metrics["val_macro_auroc"])

        row = {
            "round": server_round,
            "lr": float(instructions[0][1].config["learning_rate"]),
            "clients": len(results),
            "train_loss": float(fit_metrics.get("train_loss", float("nan"))),
            "val_loss": float(val_loss),
            "val_macro_auroc": score,
            "communication_mb": result.communication_mb,
            "seconds": time.perf_counter() - start,
        }
        result.history.append(row)
        if on_round is not None:
            on_round(row)
        if progress:
            print(
                f"round {server_round:3d}  lr {row['lr']:.2e}  clients {len(results)}  "
                f"train_loss {row['train_loss']:.4f}  val_loss {val_loss:.4f}  "
                f"val_macro_auroc {score:.4f}  ({row['seconds']:.1f}s)"
            )

        set_weights(model, parameters_to_ndarrays(parameters))
        if stopper.update(server_round, score, model):
            result.stopped_early = server_round < rounds
            break

    result.best_round, result.best_score = stopper.best_step, stopper.best_score
    stopper.restore(model)
    return result


def centralized_evaluate_fn(
    model: nn.Module,
    evaluate: Callable[[nn.Module], tuple[float, float]],
) -> Callable[[int, list, dict], tuple[float, dict[str, Any]]]:
    """Wrap `evaluate(model) -> (val_loss, val_macro_auroc)` as a Flower `evaluate_fn`.

    Flower calls it with the aggregated weights; they are loaded into `model`
    before scoring.
    """

    @torch.no_grad()
    def evaluate_fn(server_round: int, weights: list, config: dict) -> tuple[float, dict]:
        set_weights(model, weights)
        loss, auroc = evaluate(model)
        return loss, {"val_macro_auroc": auroc}

    return evaluate_fn

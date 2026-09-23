"""Tests for the Flower client, strategies and in-process simulation (tiny, CPU-only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg, FedProx

from fedecg.federated.client import ECGClient, get_weights, set_weights
from fedecg.federated.simulation import (
    InProcessClientProxy,
    centralized_evaluate_fn,
    parameters_mb,
    run_federated,
)
from fedecg.federated.strategy import build_strategy, weighted_mean
from fedecg.models.resnet1d import ResNet1d
from fedecg.seed import new_generator, set_seed
from fedecg.training.loop import make_loader, predict
from fedecg.training.metrics import macro_auroc

CPU = torch.device("cpu")
TRAIN_CFG = {"batch_size": 16, "weight_decay": 0.0}


def separable_data(n: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Class k is present iff lead k carries a large offset."""
    rng = new_generator(seed)
    labels = rng.integers(0, 2, size=(n, 5)).astype(np.float32)
    signals = rng.standard_normal((n, 12, 100)).astype(np.float32) * 0.1
    signals[:, :5, :] += 2.0 * labels[:, :, None]
    return signals, labels


def tiny_model() -> ResNet1d:
    return ResNet1d(base_channels=8, blocks_per_stage=(1,), norm_groups=4, dropout=0.0)


def simulate(strategy_name: str = "fedavg", rounds: int = 6, n_clients: int = 3, seed: int = 0):
    """FedAvg/FedProx on a separable problem split across `n_clients`."""
    set_seed(seed)
    x, y = separable_data(96, seed=seed)
    model = tiny_model()
    val_loader = make_loader(x, y, batch_size=96, shuffle=False)

    def evaluate(m):
        prob, true = predict(m, val_loader, CPU)
        return 0.0, macro_auroc(true, prob)

    shares = np.array_split(np.arange(len(x)), n_clients)
    clients = [
        InProcessClientProxy(
            str(i),
            ECGClient(x[idx], y[idx], model, CPU, TRAIN_CFG, seed=i).to_client(),
        )
        for i, idx in enumerate(shares)
    ]
    strategy = build_strategy(
        {"strategy": strategy_name, "proximal_mu": 0.1},
        n_clients=n_clients,
        initial_parameters=ndarrays_to_parameters(get_weights(model)),
        evaluate_fn=centralized_evaluate_fn(model, evaluate),
        on_fit_config_fn=lambda rnd: {"learning_rate": 0.01, "local_epochs": 3},
    )
    result = run_federated(strategy, clients, model, rounds=rounds, patience=rounds)
    return result, model


class TestWeights:
    def test_round_trip(self):
        a, b = tiny_model(), tiny_model()
        set_weights(b, get_weights(a))
        for pa, pb in zip(a.state_dict().values(), b.state_dict().values(), strict=True):
            torch.testing.assert_close(pa, pb)

    def test_rejects_the_wrong_number_of_arrays(self):
        with pytest.raises(ValueError, match="Expected"):
            set_weights(tiny_model(), get_weights(tiny_model())[:-1])


class TestStrategy:
    def test_builds_flower_strategies(self):
        kwargs = {
            "n_clients": 4,
            "initial_parameters": ndarrays_to_parameters([np.zeros(1)]),
            "evaluate_fn": lambda *a: None,
            "on_fit_config_fn": lambda rnd: {},
        }
        assert type(build_strategy({"strategy": "fedavg"}, **kwargs)) is FedAvg
        prox = build_strategy({"strategy": "fedprox", "proximal_mu": 0.5}, **kwargs)
        assert isinstance(prox, FedProx) and prox.proximal_mu == 0.5
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_strategy({"strategy": "scaffold"}, **kwargs)

    def test_weighted_mean_weights_by_records(self):
        assert weighted_mean([(1, {"loss": 1.0}), (3, {"loss": 3.0})]) == {"loss": 2.5}


class TestSimulation:
    def test_fedavg_learns_a_separable_problem(self):
        result, _ = simulate("fedavg")
        assert result.best_score > 0.95
        assert [row["round"] for row in result.history] == list(range(1, 7))
        assert all(row["clients"] == 3 for row in result.history)

    def test_counts_traffic_both_ways_every_round(self):
        result, model = simulate("fedavg", rounds=2)
        # Serialized size, i.e. what would actually cross the network.
        model_mb = parameters_mb(ndarrays_to_parameters(get_weights(model)))
        # 3 clients x (download + upload) x 2 rounds.
        assert result.communication_mb == pytest.approx(12 * model_mb)

    def test_is_deterministic_given_the_seed(self):
        a, _ = simulate("fedprox", rounds=3)
        b, _ = simulate("fedprox", rounds=3)
        assert [r["val_macro_auroc"] for r in a.history] == [
            r["val_macro_auroc"] for r in b.history
        ]
        assert [r["train_loss"] for r in a.history] == [r["train_loss"] for r in b.history]

    def test_fedprox_sends_mu_and_changes_training(self):
        avg, _ = simulate("fedavg", rounds=3)
        prox, _ = simulate("fedprox", rounds=3)
        assert [r["train_loss"] for r in avg.history] != [r["train_loss"] for r in prox.history]

    def test_leaves_the_model_holding_the_best_round(self):
        result, model = simulate("fedavg", rounds=4)
        x, y = separable_data(96)
        prob, true = predict(model, make_loader(x, y, batch_size=96, shuffle=False), CPU)
        assert macro_auroc(true, prob) == pytest.approx(result.best_score, abs=1e-6)

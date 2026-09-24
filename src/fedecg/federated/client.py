"""A Flower client: one simulated hospital training on its own records.

Each round the client receives the global weights, trains for
`local_epochs` passes over its own data with the shared training loop, and
sends back the new weights and its record count (FedAvg weights clients by it).
Nothing else leaves the client: no records, no gradients per record.

The optimizer is created fresh every round, as in the reference FedAvg and
FedProx implementations, so no optimizer state carries over from one round's
global model to the next. The learning rate for the round comes from the
server in the fit config, which is how the cosine schedule spans rounds.

With DP (phase 6), each hospital owns a `PrivateTraining`: its model is
wrapped for DP-SGD for the round and unwrapped afterwards, while its privacy
accountant keeps counting across rounds. Only the hospital's noisy, clipped
updates reach the server.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from flwr.client import NumPyClient
from torch import nn

from fedecg.privacy.dp_sgd import PrivateTraining
from fedecg.training.loop import make_loader, train_one_epoch

NDArrays = list[np.ndarray]


def get_weights(model: nn.Module) -> NDArrays:
    """Model state as a list of NumPy arrays, in `state_dict` order."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: NDArrays) -> None:
    """Load a list produced by `get_weights` back into `model`."""
    state = model.state_dict()
    if len(weights) != len(state):
        raise ValueError(f"Expected {len(state)} arrays, got {len(weights)}")
    model.load_state_dict(
        {k: torch.from_numpy(np.asarray(w)) for k, w in zip(state, weights, strict=True)}
    )


class ECGClient(NumPyClient):
    """Trains the shared model on one hospital's records.

    Clients run one after another in the same process, so they share a single
    `model` object: each `fit` overwrites it with the global weights first.
    Every client keeps its own seeded loader, so its batch order and crops
    differ from round to round but are fixed by the seed.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        model: nn.Module,
        device: torch.device,
        training_config: Mapping[str, Any],
        *,
        seed: int,
        private: PrivateTraining | None = None,
    ):
        self.model = model
        self.private = private
        self.device = device
        self.training_config = training_config
        self.n_records = len(signals)
        self.loader = make_loader(
            signals,
            labels,
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            seed=seed,
            crop_samples=training_config.get("crop_samples"),
        )

    def get_parameters(self, config: dict[str, Any]) -> NDArrays:  # noqa: D102
        return get_weights(self.model)

    def fit(self, parameters: NDArrays, config: dict[str, Any]) -> tuple[NDArrays, int, dict]:
        """Train locally from the global `parameters`.

        Reads `learning_rate`, `local_epochs` and (FedProx) `proximal_mu` from
        the server's `config`.
        """
        set_weights(self.model, parameters)
        mu = float(config.get("proximal_mu", 0.0))
        anchor = [p.detach().clone() for p in self.model.parameters()] if mu > 0 else None
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
        )
        loss_fn = nn.BCEWithLogitsLoss()
        model, loader = self.model, self.loader
        if self.private is not None:
            model, optimizer, loader = self.private.wrap(model, optimizer, loader)
        losses = [
            train_one_epoch(
                model,
                loader,
                optimizer,
                loss_fn,
                self.device,
                proximal_mu=mu,
                proximal_anchor=anchor,
            )
            for _ in range(int(config.get("local_epochs", 1)))
        ]
        metrics = {"train_loss": float(losses[-1])}
        if self.private is not None:
            self.private.unwrap()
            metrics["epsilon"] = self.private.epsilon()
        return get_weights(self.model), self.n_records, metrics

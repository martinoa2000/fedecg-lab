"""Epoch-level training with early stopping on validation macro AUROC.

The loop is written against plain `DataLoader`s and an `nn.Module` so that the
federated phase can reuse `train_one_epoch` and `predict` inside each Flower
client, and the DP phase can hand in an Opacus-wrapped model and optimizer.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedecg.training.metrics import macro_auroc


def resolve_device(name: str = "auto") -> torch.device:
    """Map `auto | cpu | cuda | mps` to a `torch.device`.

    `auto` prefers CUDA, then Apple's MPS, then CPU.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unknown device {name!r}; use auto, cpu, cuda or mps")
    return torch.device(name)


def make_loader(
    signals: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap in-memory arrays in a `DataLoader`.

    Shuffling draws from its own seeded `torch.Generator`, so the batch order
    is reproducible and independent of any other use of torch's global RNG.
    """
    dataset = TensorDataset(torch.from_numpy(signals), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """One pass over `loader`. Returns the mean loss per record."""
    model.train()
    total, n_seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
        n_seen += len(x)
    return total / max(n_seen, 1)


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Sigmoid probabilities and true labels for every record in `loader`."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> float:
    """Mean loss per record without updating the model."""
    model.eval()
    total, n_seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += loss_fn(model(x), y).item() * len(x)
        n_seen += len(x)
    return total / max(n_seen, 1)


@dataclass
class FitResult:
    """Outcome of `fit`. The model passed in is left holding the best weights."""

    history: list[dict[str, float]] = field(default_factory=list)
    """One dict per epoch: epoch, train_loss, val_loss, val_macro_auroc, seconds."""
    best_epoch: int = 0
    best_score: float = -math.inf
    stopped_early: bool = False


EpochCallback = Callable[[dict[str, float]], None]


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: Mapping[str, Any],
    device: torch.device,
    *,
    on_epoch: EpochCallback | None = None,
    progress: bool = False,
) -> FitResult:
    """Train with AdamW and stop when validation macro AUROC stops improving.

    Args:
        model: Already on `device`.
        train_loader: Shuffled training batches.
        val_loader: Validation batches, used for model selection only.
        training_config: The config's `training` section (`epochs`,
            `learning_rate`, `weight_decay`, `early_stopping_patience`).
        device: Where `model` lives.
        on_epoch: Called with each epoch's metrics dict, e.g. to log to MLflow.
        progress: Print one line per epoch.

    Returns:
        A `FitResult`. On return, `model` holds the weights of the best epoch,
        not the last one.
    """
    metric = training_config.get("early_stopping_metric", "val_macro_auroc")
    if metric != "val_macro_auroc":
        raise ValueError(f"Unsupported early_stopping_metric {metric!r}")
    epochs = int(training_config["epochs"])
    patience = int(training_config.get("early_stopping_patience", epochs))

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )

    result = FitResult()
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        val_prob, val_true = predict(model, val_loader, device)
        score = macro_auroc(val_true, val_prob)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auroc": score,
            "seconds": time.perf_counter() - start,
        }
        result.history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        if progress:
            print(
                f"epoch {epoch:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
                f"val_macro_auroc {score:.4f}  ({row['seconds']:.1f}s)"
            )

        # NaN (e.g. a validation set without positives) never counts as better.
        if score > result.best_score:
            result.best_score, result.best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                result.stopped_early = epoch < epochs
                break

    model.load_state_dict(best_state)
    return result

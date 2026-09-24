"""Train the centralized model, or an ensemble of them, from a config.

Shared by `scripts/train_centralized.py` (which then scores test once) and
`scripts/tune.py` (which only ever looks at validation), so a setting tuned
on validation is trained exactly the same way when it is finally tested.

The `training` config section this reads, beyond what `fit` reads:

- `augment`: see `fedecg.training.augment` (off by default);
- `loss`: `bce` (default), `weighted_bce` or `focal`, see `make_loss`;
- `ensemble_seeds`: a list of seeds; one model is trained per seed and their
  probabilities are averaged. Without it, one model with the config's `seed`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from fedecg.data.pipeline import PreparedData
from fedecg.models.resnet1d import build_model
from fedecg.seed import set_seed
from fedecg.training.loop import (
    EpochCallback,
    FitResult,
    WrapFn,
    fit,
    make_loader,
    make_loss,
    predict_from_config,
)


@dataclass
class TrainedModel:
    """One member of a (possibly single-member) ensemble."""

    seed: int
    model: nn.Module
    result: FitResult


def ensemble_seeds(config: Mapping[str, Any]) -> list[int]:
    """The seeds to train: `training.ensemble_seeds`, else just `seed`."""
    seeds = config["training"].get("ensemble_seeds")
    return [int(s) for s in seeds] if seeds else [int(config["seed"])]


def train_centralized(
    config: Mapping[str, Any],
    data: PreparedData,
    device: torch.device,
    *,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
    progress: bool = True,
    wrap: WrapFn | None = None,
) -> list[TrainedModel]:
    """Train one model per ensemble seed on `data.x_train`.

    Args:
        config: A resolved experiment config.
        data: Output of `prepare_data`.
        device: Where to train.
        on_epoch: Called as `on_epoch(member_index, row)` after every epoch.
        progress: Print one line per epoch (the training server parses these).
        wrap: DP-SGD hook (see `fit`); only for a single-model run, since each
            member of an ensemble would spend its own privacy budget.
    """
    train_cfg = config["training"]
    seeds = ensemble_seeds(config)
    if wrap is not None and len(seeds) > 1:
        raise ValueError("DP-SGD with an ensemble would spend the privacy budget once per member")
    loss_fn = make_loss(train_cfg, data.y_train)
    batch_size = int(train_cfg["batch_size"])
    val_loader = make_loader(data.x_val, data.y_val, batch_size=batch_size, shuffle=False)

    members = []
    for index, seed in enumerate(seeds):
        if len(seeds) > 1 and progress:
            print(f"ensemble member {index + 1} of {len(seeds)} (seed {seed})", flush=True)
        set_seed(seed)
        train_loader = make_loader(
            data.x_train,
            data.y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=int(train_cfg.get("num_workers", 0)),
            crop_samples=train_cfg.get("crop_samples"),
            augment=train_cfg.get("augment"),
        )
        model = build_model(config["model"]).to(device)
        callback: EpochCallback | None = (
            (lambda row, i=index: on_epoch(i, row)) if on_epoch is not None else None
        )
        result = fit(
            model,
            train_loader,
            val_loader,
            train_cfg,
            device,
            on_epoch=callback,
            progress=progress,
            wrap=wrap,
            loss_fn=loss_fn,
        )
        members.append(TrainedModel(seed=seed, model=model, result=result))
    return members


def predict_ensemble(
    models: Sequence[nn.Module],
    signals: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    training_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean sigmoid probability over `models`, and the true labels."""
    loader = make_loader(signals, labels, batch_size=256, shuffle=False)
    probs = [predict_from_config(m, loader, device, training_config)[0] for m in models]
    return np.mean(probs, axis=0), labels

"""DP-SGD with Opacus, calibrated to a target (epsilon, delta) budget.

DP-SGD changes two things in each optimizer step. Every record's gradient is
clipped to an L2 norm of at most `max_grad_norm`, so no single ECG can move
the model much, and Gaussian noise scaled to that bound is added to the summed
gradient. Batches are drawn by Poisson sampling (each record joins a batch
independently with probability `batch_size / n_records`), which is what the
privacy accountant assumes.

**Budget.** The noise multiplier is chosen once, before training, so that the
whole planned run (every epoch, or every round times local epochs) spends at
most `target_epsilon` at the given `delta`. The accountant then tracks what was
actually spent; `epsilon()` reports it. Keeping the checkpoint with the best
validation AUROC is post-processing: the validation fold is not training data,
so model selection costs no extra privacy.

**What is protected.** Record-level privacy for the data owner that trains:
the whole training set in the centralized setting, one hospital's records in
the federated one. In the federated case each hospital runs its own
`PrivateTraining`, with its own accountant that persists across rounds.

**Delta.** `1e-5` by default, below `1 / n_records` for every data owner here
(the smallest hospital in the DP experiments holds ~3,400 records).

**Not accounted for.** Hyperparameters (learning rate, clipping bound) were
chosen on the validation fold without privacy accounting, as is common in DP
papers. A deployment would tune them on public data or pay for the search.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class PrivateTraining:
    """One data owner's DP-SGD state: noise level and privacy accountant.

    Args:
        privacy_config: The config's `privacy` section (`target_epsilon`,
            `delta`, `max_grad_norm`, `accountant`).
        n_records: Size of the data owner's training set.
        batch_size: Expected batch size; sets the Poisson sampling rate.
        epochs: Total passes over the data the budget must cover (for a
            hospital: rounds x local epochs).
    """

    def __init__(
        self,
        privacy_config: Mapping[str, Any],
        *,
        n_records: int,
        batch_size: int,
        epochs: int,
    ):
        if n_records < batch_size:
            raise ValueError(f"Need at least batch_size={batch_size} records, got {n_records}")
        self.target_epsilon = float(privacy_config["target_epsilon"])
        self.delta = float(privacy_config.get("delta", 1e-5))
        self.max_grad_norm = float(privacy_config.get("max_grad_norm", 1.0))
        self.accountant = str(privacy_config.get("accountant", "prv"))
        self.sample_rate = batch_size / n_records
        self.noise_multiplier = get_noise_multiplier(
            target_epsilon=self.target_epsilon,
            target_delta=self.delta,
            sample_rate=self.sample_rate,
            epochs=epochs,
            accountant=self.accountant,
        )
        self.engine = PrivacyEngine(accountant=self.accountant)
        self._wrapped: nn.Module | None = None

    def wrap(
        self, model: nn.Module, optimizer: Optimizer, loader: DataLoader
    ) -> tuple[nn.Module, Optimizer, DataLoader]:
        """Return DP versions of the three, sharing this owner's accountant.

        `model` keeps its parameters: the returned module wraps it, so weights
        trained through it are visible on `model` itself. Call `unwrap` when
        done to remove the per-sample gradient hooks.
        """
        # Opacus refuses a model in eval mode, which is how the server's
        # evaluation leaves the shared model between federated rounds.
        model.train()
        private_model, private_optimizer, private_loader = self.engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            poisson_sampling=True,
        )
        self._wrapped = private_model
        return private_model, private_optimizer, private_loader

    def unwrap(self) -> None:
        """Remove the hooks `wrap` attached to the model."""
        if self._wrapped is not None:
            self._wrapped.to_standard_module()
            self._wrapped = None

    def epsilon(self) -> float:
        """Epsilon spent so far at `delta` (0 before the first step)."""
        if not self.engine.accountant.history:
            return 0.0
        return float(self.engine.get_epsilon(self.delta))


def private_training_from_config(
    config: Mapping[str, Any], *, n_records: int, epochs: int
) -> PrivateTraining | None:
    """A `PrivateTraining` when `privacy.enabled` is true, else None."""
    privacy = config.get("privacy", {})
    if not privacy.get("enabled", False):
        return None
    return PrivateTraining(
        privacy,
        n_records=n_records,
        batch_size=int(config["training"]["batch_size"]),
        epochs=epochs,
    )

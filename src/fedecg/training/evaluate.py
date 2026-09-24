"""The one final evaluation every experiment ends with.

Thresholds are tuned per class on validation, then the test split is scored
exactly once with them. Sharing this function is what guarantees that the
centralized, federated and private models are measured identically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fedecg.training.loop import predict_from_config
from fedecg.training.metrics import best_f1_thresholds, metrics_table


def final_evaluation(
    model: nn.Module | Sequence[nn.Module],
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    training_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Tune thresholds on validation, then score test.

    Args:
        model: One model, or several whose probabilities are averaged (an
            ensemble).
        val_loader: Validation records, for the per-class thresholds.
        test_loader: Test records, scored once.
        device: Where the models live.
        training_config: The config's `training` section (evaluation windowing).

    Returns:
        `(table, thresholds)`: per-class AUROC / F1 / threshold on test with a
        `macro` row (index named `superclass`), and the tuned thresholds.
    """
    models = [model] if isinstance(model, nn.Module) else list(model)

    def scores(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        outputs = [predict_from_config(m, loader, device, training_config) for m in models]
        return np.mean([prob for prob, _ in outputs], axis=0), outputs[0][1]

    val_prob, val_true = scores(val_loader)
    thresholds = best_f1_thresholds(val_true, val_prob)
    test_prob, test_true = scores(test_loader)
    table = pd.DataFrame(metrics_table(test_true, test_prob, thresholds)).T
    table.index.name = "superclass"
    return table, thresholds

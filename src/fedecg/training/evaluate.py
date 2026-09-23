"""The one final evaluation every experiment ends with.

Thresholds are tuned per class on validation, then the test split is scored
exactly once with them. Sharing this function is what guarantees that the
centralized, federated and private models are measured identically.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fedecg.training.loop import predict_from_config
from fedecg.training.metrics import best_f1_thresholds, metrics_table


def final_evaluation(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    training_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Tune thresholds on validation, then score test.

    Returns:
        `(table, thresholds)`: per-class AUROC / F1 / threshold on test with a
        `macro` row (index named `superclass`), and the tuned thresholds.
    """
    val_prob, val_true = predict_from_config(model, val_loader, device, training_config)
    thresholds = best_f1_thresholds(val_true, val_prob)
    test_prob, test_true = predict_from_config(model, test_loader, device, training_config)
    table = pd.DataFrame(metrics_table(test_true, test_prob, thresholds)).T
    table.index.name = "superclass"
    return table, thresholds

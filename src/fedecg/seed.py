"""Seed control for reproducible experiments.

Reproducibility matters twice over here. As usual we want a rerun to reproduce
a reported number, but federated experiments add a second source of randomness:
how the dataset is split across simulated hospitals. A different split is a
different experiment, so partitioning uses its own explicitly-passed generator
rather than global state (see `fedecg.data.partition`).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch.

    Args:
        seed: The seed to apply to all three generators.
        deterministic: If True, ask cuDNN for deterministic kernels. This costs
            some speed and is a no-op on CPU, but without it two runs on the
            same GPU can differ in the last decimal places.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Seeding the legacy global state on purpose: scikit-learn, neurokit2 and
    # parts of PyTorch still draw from it, and they cannot be handed a
    # `Generator`. Our own sampling uses `new_generator` instead.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def new_generator(seed: int) -> np.random.Generator:
    """Return an independent NumPy generator.

    Preferred over the global `np.random` state whenever randomness defines the
    experiment itself (client partitions, Dirichlet label skew), because it
    cannot be perturbed by unrelated library calls.
    """
    return np.random.default_rng(seed)

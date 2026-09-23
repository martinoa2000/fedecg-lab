"""From a config to preprocessed train / validation / test arrays.

Every experiment script starts the same way, and the comparison between phases
is only fair if it starts *exactly* the same way: same folds, same subsample,
same filter, same standardizer. This module is that shared start.

On the standardizer in the federated phases: per-lead mean and standard
deviation can be computed exactly from per-hospital sums, sums of squares and
counts (24 numbers per lead), which a real federation can aggregate without
moving any record. The pooled statistics used here are therefore attainable
under federation, and identical to the centralized ones by construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fedecg.data.preprocess import LeadStandardizer, preprocess_splits
from fedecg.data.ptbxl import load_dataset, split_by_folds
from fedecg.paths import PTBXL_DIR
from fedecg.seed import new_generator


@dataclass(frozen=True)
class PreparedData:
    """Model-ready arrays for the three splits, plus the training metadata."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    meta_train: pd.DataFrame
    """Metadata rows of the training records, in the order of `x_train`."""
    standardizer: LeadStandardizer | None


def prepare_data(
    config: Mapping[str, Any],
    *,
    root: str | Path = PTBXL_DIR,
    cache_path: str | Path | None = None,
) -> PreparedData:
    """Load PTB-XL, split by folds, subsample, filter and standardize.

    Args:
        config: A resolved experiment config (uses `seed` and `data`).
        root: PTB-XL directory.
        cache_path: Optional `.npz` waveform cache (see `load_dataset`).
    """
    data_cfg = config["data"]
    fs = data_cfg["sampling_rate_hz"]
    meta, arrays = load_dataset(root, sampling_rate=fs, cache_path=cache_path, progress=True)
    split = split_by_folds(
        meta,
        train_folds=data_cfg["train_folds"],
        val_fold=data_cfg["val_fold"],
        test_fold=data_cfg["test_fold"],
    )
    train_ids = split.train.to_numpy()
    if data_cfg.get("subsample") is not None and data_cfg["subsample"] < len(train_ids):
        rng = new_generator(int(config["seed"]))
        train_ids = np.sort(rng.choice(train_ids, size=data_cfg["subsample"], replace=False))
    train, val, test = (arrays.select(ids) for ids in (train_ids, split.val, split.test))

    x_train, others, standardizer = preprocess_splits(
        train.signals, {"val": val.signals, "test": test.signals}, data_cfg["preprocess"], fs=fs
    )
    return PreparedData(
        x_train=x_train,
        y_train=train.labels,
        x_val=others["val"],
        y_val=val.labels,
        x_test=others["test"],
        y_test=test.labels,
        meta_train=meta.loc[train_ids],
        standardizer=standardizer,
    )

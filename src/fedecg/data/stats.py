"""Descriptive statistics of the label structure.

These tables answer the questions that shape every later design choice: how
imbalanced are the classes (which is why the headline metric is macro AUROC
rather than accuracy), how often do labels co-occur (which is why the output
layer uses independent sigmoids rather than a softmax), and do the official
folds preserve the class balance (which is what makes the validation and test
numbers trustworthy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fedecg.data.constants import SUPERCLASSES
from fedecg.data.ptbxl import Split


def label_distribution(meta: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Count and prevalence of each superclass, overall and per split.

    Args:
        meta: Labeled metadata (see `fedecg.data.ptbxl.attach_labels`).
        split: The train / val / test record IDs.

    Returns:
        One row per superclass; for each of `all`, `train`, `val`, `test`, a
        `<split>_n` count column and a `<split>_pct` prevalence column. Rows do
        not sum to 100% because records can carry several labels.
    """
    parts = {"all": meta.index, "train": split.train, "val": split.val, "test": split.test}
    columns = {}
    for name, ids in parts.items():
        labels = meta.loc[ids, list(SUPERCLASSES)]
        columns[f"{name}_n"] = labels.sum().astype(int)
        columns[f"{name}_pct"] = (100 * labels.mean()).round(2) if len(ids) else np.nan
    table = pd.DataFrame(columns)
    table.index.name = "superclass"
    return table


def cooccurrence(meta: pd.DataFrame) -> pd.DataFrame:
    """Square matrix: entry `(a, b)` counts records labeled with both a and b.

    The diagonal is the per-class count.
    """
    labels = meta[list(SUPERCLASSES)].to_numpy(dtype=np.int64)
    counts = labels.T @ labels
    return pd.DataFrame(counts, index=list(SUPERCLASSES), columns=list(SUPERCLASSES))


def label_cardinality(meta: pd.DataFrame) -> pd.Series:
    """How many records carry exactly 0, 1, 2, ... superclasses."""
    per_record = meta[list(SUPERCLASSES)].sum(axis=1)
    counts = per_record.value_counts().sort_index()
    counts.index.name = "n_labels"
    counts.name = "n_records"
    return counts


def records_by(meta: pd.DataFrame, column: str) -> pd.DataFrame:
    """Record count and per-class prevalence (%) for each value of `column`.

    Used on `site` and `device` to see how much natural heterogeneity the
    metadata offers for the non-IID federated partitions.
    """
    groups = meta.groupby(column, dropna=False)
    table = (100 * groups[list(SUPERCLASSES)].mean()).round(2)
    table.insert(0, "n_records", groups.size())
    return table.sort_values("n_records", ascending=False)


def example_ids(meta: pd.DataFrame, seed_index: int = 0) -> dict[str, int]:
    """Pick one representative `ecg_id` per superclass.

    Prefers records carrying that superclass *alone*, so the example shows the
    class rather than a mixture; falls back to any record containing it.
    `seed_index` selects which of the candidates to take, for browsing.
    """
    labels = meta[list(SUPERCLASSES)].astype(bool)
    single = labels.sum(axis=1) == 1
    chosen = {}
    for name in SUPERCLASSES:
        candidates = meta.index[labels[name] & single]
        if len(candidates) == 0:
            candidates = meta.index[labels[name]]
        if len(candidates):
            chosen[name] = int(candidates[seed_index % len(candidates)])
    return chosen


def label_combinations(meta: pd.DataFrame, top: int = 10) -> pd.Series:
    """The most frequent exact label sets, e.g. `MI+STTC`."""
    labels = meta[list(SUPERCLASSES)].astype(bool)
    names = labels.apply(
        lambda row: "+".join(c for c in SUPERCLASSES if row[c]) or "(none)", axis=1
    )
    counts = names.value_counts().head(top)
    counts.index.name = "labels"
    counts.name = "n_records"
    return counts

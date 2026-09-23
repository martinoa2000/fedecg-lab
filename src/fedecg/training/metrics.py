"""Multi-label evaluation metrics.

**Macro AUROC** is the headline number, as in the PTB-XL benchmark. It is
threshold-free and weights every superclass equally, so the rare `HYP` (12% of
records) counts as much as `NORM` (44%).

**Per-class F1** is reported alongside because a deployed model has to commit
to a threshold. A single 0.5 cut-off is a poor choice for imbalanced classes,
so thresholds are tuned per class on the *validation* split and then applied
unchanged to the test split. Tuning them on test would inflate F1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from fedecg.data.constants import SUPERCLASSES


def per_class_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """AUROC for each column. NaN for a class with only one label value present.

    A tiny subsample (or a small federated client) can easily contain no
    positive `HYP` record; returning NaN keeps that visible instead of raising.
    """
    _check_shapes(y_true, y_prob)
    scores = np.full(y_true.shape[1], np.nan)
    for k in range(y_true.shape[1]):
        if len(np.unique(y_true[:, k])) == 2:
            scores[k] = roc_auc_score(y_true[:, k], y_prob[:, k])
    return scores


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean per-class AUROC over the classes where it is defined."""
    scores = per_class_auroc(y_true, y_prob)
    if np.isnan(scores).all():
        return float("nan")
    return float(np.nanmean(scores))


def best_f1_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-class probability threshold that maximizes F1 on the given data.

    Call this on validation predictions only. Classes without positives fall
    back to 0.5.
    """
    _check_shapes(y_true, y_prob)
    thresholds = np.full(y_true.shape[1], 0.5)
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() == 0:
            continue
        precision, recall, cutoffs = precision_recall_curve(y_true[:, k], y_prob[:, k])
        # The last precision/recall pair has no threshold attached.
        precision, recall = precision[:-1], recall[:-1]
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        thresholds[k] = cutoffs[int(np.argmax(f1))]
    return thresholds


def per_class_f1(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """F1 for each class after binarizing `y_prob` at `thresholds`."""
    _check_shapes(y_true, y_prob)
    y_pred = (y_prob >= np.asarray(thresholds)[None, :]).astype(int)
    # Column by column: given a single column, sklearn would switch to binary
    # mode and return the F1 of both the negative and the positive label.
    return np.array(
        [f1_score(y_true[:, k], y_pred[:, k], zero_division=0) for k in range(y_true.shape[1])]
    )


def metrics_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    class_names: Sequence[str] = SUPERCLASSES,
) -> dict[str, dict[str, float]]:
    """Per-class AUROC, F1 and threshold, plus a `macro` row.

    Returns a nested dict (`{class: {metric: value}}`) that converts directly
    into a DataFrame with `pd.DataFrame(table).T`.
    """
    auroc = per_class_auroc(y_true, y_prob)
    f1 = per_class_f1(y_true, y_prob, thresholds)
    table = {
        name: {"auroc": float(auroc[k]), "f1": float(f1[k]), "threshold": float(thresholds[k])}
        for k, name in enumerate(class_names)
    }
    table["macro"] = {
        "auroc": macro_auroc(y_true, y_prob),
        "f1": float(np.mean(f1)),
        "threshold": float("nan"),
    }
    return table


def _check_shapes(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    if y_true.shape != y_prob.shape or y_true.ndim != 2:
        raise ValueError(
            f"y_true and y_prob must both be (n, classes); got {y_true.shape}, {y_prob.shape}"
        )

"""Tests for multi-label metrics."""

from __future__ import annotations

import numpy as np
import pytest

from fedecg.training.metrics import (
    best_f1_thresholds,
    macro_auroc,
    metrics_table,
    per_class_auroc,
    per_class_f1,
)

Y_TRUE = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0]])


class TestAuroc:
    def test_perfect_ranking_scores_one(self):
        assert macro_auroc(Y_TRUE, Y_TRUE * 0.9 + 0.05) == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero(self):
        assert macro_auroc(Y_TRUE, 1 - Y_TRUE) == pytest.approx(0.0)

    def test_is_the_unweighted_mean_over_classes(self):
        prob = np.column_stack([Y_TRUE[:, 0], 1 - Y_TRUE[:, 1]]).astype(float)
        assert per_class_auroc(Y_TRUE, prob).tolist() == [1.0, 0.0]
        assert macro_auroc(Y_TRUE, prob) == pytest.approx(0.5)

    def test_class_without_positives_is_nan_and_skipped(self):
        y = np.column_stack([Y_TRUE[:, 0], np.zeros(5)])
        scores = per_class_auroc(y, y.astype(float))
        assert scores[0] == 1.0 and np.isnan(scores[1])
        assert macro_auroc(y, y.astype(float)) == 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            macro_auroc(Y_TRUE, Y_TRUE[:, :1])


class TestThresholds:
    def test_picks_a_threshold_that_separates_the_classes(self):
        y = np.array([[0], [0], [1], [1]])
        prob = np.array([[0.1], [0.2], [0.3], [0.4]])
        threshold = best_f1_thresholds(y, prob)
        assert per_class_f1(y, prob, threshold)[0] == pytest.approx(1.0)
        # 0.5 would have predicted nothing positive.
        assert per_class_f1(y, prob, np.array([0.5]))[0] == 0.0

    def test_falls_back_to_half_without_positives(self):
        assert best_f1_thresholds(np.zeros((3, 1)), np.full((3, 1), 0.7)).tolist() == [0.5]


def test_metrics_table_has_one_row_per_class_plus_macro():
    table = metrics_table(Y_TRUE, Y_TRUE.astype(float), np.full(2, 0.5), class_names=("A", "B"))
    assert list(table) == ["A", "B", "macro"]
    assert table["macro"]["auroc"] == pytest.approx(1.0)
    assert table["macro"]["f1"] == pytest.approx(1.0)

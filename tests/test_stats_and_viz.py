"""Tests for label statistics and plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from fedecg.data.constants import NUM_LEADS, SIGNAL_LENGTH, SUPERCLASSES
from fedecg.data.ptbxl import attach_labels, load_diagnostic_map, load_metadata, split_by_folds
from fedecg.data.stats import (
    cooccurrence,
    example_ids,
    label_cardinality,
    label_combinations,
    label_distribution,
    records_by,
)
from fedecg.viz import plot_class_examples, plot_ecg


@pytest.fixture(scope="module")
def labeled(fake_ptbxl):
    return attach_labels(load_metadata(fake_ptbxl), load_diagnostic_map(fake_ptbxl))


class TestLabelDistribution:
    def test_counts_add_up_across_splits(self, labeled):
        table = label_distribution(labeled, split_by_folds(labeled))
        assert list(table.index) == list(SUPERCLASSES)
        np.testing.assert_array_equal(
            table["all_n"], table["train_n"] + table["val_n"] + table["test_n"]
        )

    def test_prevalence_is_a_percentage_of_records(self, labeled):
        table = label_distribution(labeled, split_by_folds(labeled))
        expected = 100 * labeled["NORM"].mean()
        assert table.loc["NORM", "all_pct"] == pytest.approx(expected, abs=0.01)


class TestCooccurrence:
    def test_is_symmetric_with_class_counts_on_the_diagonal(self, labeled):
        matrix = cooccurrence(labeled)
        np.testing.assert_array_equal(matrix.to_numpy(), matrix.to_numpy().T)
        np.testing.assert_array_equal(np.diag(matrix), labeled[list(SUPERCLASSES)].sum())

    def test_counts_known_pairs(self, labeled):
        # Fixture records with both MI and STTC: ids 2 and 20.
        assert cooccurrence(labeled).loc["MI", "STTC"] == 2


class TestCardinality:
    def test_sums_to_the_number_of_records(self, labeled):
        counts = label_cardinality(labeled)
        assert counts.sum() == len(labeled)
        assert 0 not in counts.index  # unlabeled records were dropped

    def test_combinations_name_label_sets(self, labeled):
        combos = label_combinations(labeled)
        assert combos.index[0] == "NORM"
        assert "CD+HYP" in combos.index


class TestRecordsBy:
    def test_counts_every_record_once(self, labeled):
        table = records_by(labeled, "site")
        assert table["n_records"].sum() == len(labeled)
        assert list(table.columns) == ["n_records", *SUPERCLASSES]


class TestExampleIds:
    def test_prefers_single_label_records(self, labeled):
        examples = example_ids(labeled)
        assert set(examples) == set(SUPERCLASSES)
        for name, ecg_id in examples.items():
            row = labeled.loc[ecg_id, list(SUPERCLASSES)]
            assert row[name] == 1
            assert row.sum() == 1


class TestPlots:
    def test_plot_ecg_draws_one_axis_per_lead(self):
        signal = np.zeros((NUM_LEADS, SIGNAL_LENGTH))
        fig = plot_ecg(signal, overlay=signal + 0.1, title="test")
        assert len(fig.axes) == NUM_LEADS
        plt.close(fig)

    def test_plot_ecg_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            plot_ecg(np.zeros((SIGNAL_LENGTH, NUM_LEADS)))

    def test_plot_class_examples(self):
        signals = np.zeros((3, NUM_LEADS, SIGNAL_LENGTH))
        fig = plot_class_examples(signals, ["A", "B", "C"])
        assert len(fig.axes) == 3
        plt.close(fig)


def test_plot_curves_draws_one_line_per_run_plus_reference():
    import pandas as pd

    from fedecg.viz import plot_curves

    curves = {
        "centralized": pd.DataFrame({"epoch": [1, 2, 3], "val_macro_auroc": [0.8, 0.85, 0.9]}),
        "fedavg": pd.DataFrame({"round": [1, 2], "val_macro_auroc": [0.7, 0.8]}),
    }
    fig = plot_curves(curves, reference=0.9)
    assert len(fig.axes[0].lines) == 3

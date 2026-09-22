"""Tests for PTB-XL metadata parsing, labels, splits and waveform loading."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import RECORDS, UNLABELED_ECG_ID, synthetic_signal
from fedecg.data.constants import NUM_CLASSES, NUM_LEADS, SIGNAL_LENGTH, SUPERCLASSES
from fedecg.data.ptbxl import (
    attach_labels,
    label_matrix,
    load_dataset,
    load_diagnostic_map,
    load_metadata,
    load_signals,
    split_by_folds,
    superclasses_of,
)


@pytest.fixture(scope="module")
def diagnostic_map(fake_ptbxl):
    return load_diagnostic_map(fake_ptbxl)


@pytest.fixture(scope="module")
def labeled(fake_ptbxl, diagnostic_map):
    return attach_labels(load_metadata(fake_ptbxl), diagnostic_map)


class TestMetadata:
    def test_indexed_by_ecg_id_with_parsed_codes(self, fake_ptbxl):
        meta = load_metadata(fake_ptbxl)
        assert meta.index.name == "ecg_id"
        assert len(meta) == len(RECORDS)
        assert meta.loc[1, "scp_codes"] == {"NORM": 100.0, "SR": 0.0}

    def test_missing_dataset_points_at_the_download_script(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"download_data\.py"):
            load_metadata(tmp_path)


class TestDiagnosticMap:
    def test_maps_diagnostic_codes_to_superclasses(self, diagnostic_map):
        assert diagnostic_map["IMI"] == "MI"
        assert diagnostic_map["CLBBB"] == "CD"

    def test_excludes_rhythm_and_form_statements(self, diagnostic_map):
        assert "SR" not in diagnostic_map
        assert "PVC" not in diagnostic_map


class TestLabels:
    def test_ignores_likelihood_and_non_diagnostic_codes(self, diagnostic_map):
        # A 15% likelihood still counts, as in the PTB-XL benchmark.
        assert superclasses_of({"IMI": 15.0, "SR": 0.0}, diagnostic_map) == {"MI"}

    def test_two_codes_of_one_class_give_one_label(self, diagnostic_map):
        matrix = label_matrix([{"IMI": 100.0, "ASMI": 100.0}], diagnostic_map)
        assert matrix.tolist() == [[0, 1, 0, 0, 0]]

    def test_multi_label_columns_follow_superclass_order(self, diagnostic_map):
        matrix = label_matrix([{"CLBBB": 100.0, "LVH": 50.0}], diagnostic_map)
        assert matrix.dtype == np.float32
        assert dict(zip(SUPERCLASSES, matrix[0], strict=True)) == {
            "NORM": 0,
            "MI": 0,
            "STTC": 0,
            "CD": 1,
            "HYP": 1,
        }

    def test_unknown_superclass_raises(self):
        with pytest.raises(ValueError, match="Unknown superclass"):
            label_matrix([{"X": 1.0}], {"X": "NEW"})

    def test_empty_input_has_the_right_shape(self, diagnostic_map):
        assert label_matrix([], diagnostic_map).shape == (0, NUM_CLASSES)

    def test_unlabeled_records_are_dropped_by_default(self, labeled):
        assert UNLABELED_ECG_ID not in labeled.index
        assert len(labeled) == len(RECORDS) - 1
        assert (labeled[list(SUPERCLASSES)].sum(axis=1) > 0).all()

    def test_unlabeled_records_can_be_kept(self, fake_ptbxl, diagnostic_map):
        meta = attach_labels(load_metadata(fake_ptbxl), diagnostic_map, drop_unlabeled=False)
        assert UNLABELED_ECG_ID in meta.index


class TestSplit:
    def test_official_split_is_by_fold(self, labeled):
        split = split_by_folds(labeled)
        assert set(labeled.loc[split.val, "strat_fold"]) == {9}
        assert set(labeled.loc[split.test, "strat_fold"]) == {10}
        assert set(labeled.loc[split.train, "strat_fold"]) <= set(range(1, 9))

    def test_partitions_are_disjoint_and_complete(self, labeled):
        split = split_by_folds(labeled)
        parts = [set(split.train), set(split.val), set(split.test)]
        assert sum(len(p) for p in parts) == len(labeled)
        assert set.union(*parts) == set(labeled.index)

    def test_overlapping_folds_are_rejected(self, labeled):
        with pytest.raises(ValueError, match="disjoint"):
            split_by_folds(labeled, train_folds=(1, 2, 9), val_fold=9, test_fold=10)


class TestSignals:
    def test_shape_is_leads_first_and_values_round_trip(self, fake_ptbxl, labeled):
        signals = load_signals(labeled.iloc[:3], fake_ptbxl)
        assert signals.shape == (3, NUM_LEADS, SIGNAL_LENGTH)
        assert signals.dtype == np.float32
        expected = synthetic_signal(int(labeled.index[0])).T
        # 16-bit storage with gain 1000 quantizes to 1 microvolt.
        np.testing.assert_allclose(signals[0], expected, atol=1e-3)

    def test_rejects_unsupported_sampling_rate(self, fake_ptbxl, labeled):
        with pytest.raises(ValueError, match="sampling_rate"):
            load_signals(labeled, fake_ptbxl, sampling_rate=250)


class TestLoadDataset:
    def test_rows_align_with_metadata(self, fake_ptbxl):
        meta, arrays = load_dataset(fake_ptbxl)
        assert len(arrays) == len(meta)
        assert arrays.ecg_ids.tolist() == meta.index.tolist()
        np.testing.assert_array_equal(arrays.labels, meta[list(SUPERCLASSES)].to_numpy())

    def test_cache_round_trip(self, fake_ptbxl, tmp_path):
        cache = tmp_path / "cache" / "ptbxl100.npz"
        _, first = load_dataset(fake_ptbxl, cache_path=cache)
        assert cache.is_file()
        _, second = load_dataset(fake_ptbxl, cache_path=cache)
        np.testing.assert_array_equal(first.signals, second.signals)

    def test_stale_cache_is_rebuilt(self, fake_ptbxl, tmp_path):
        cache = tmp_path / "stale.npz"
        np.savez(cache, signals=np.zeros((1, NUM_LEADS, SIGNAL_LENGTH)), ecg_ids=np.array([999]))
        meta, arrays = load_dataset(fake_ptbxl, cache_path=cache)
        assert len(arrays) == len(meta)
        with np.load(cache) as rebuilt:
            assert rebuilt["ecg_ids"].tolist() == meta.index.tolist()

    def test_select_follows_the_requested_order(self, fake_ptbxl):
        _, arrays = load_dataset(fake_ptbxl)
        subset = arrays.select([3, 1])
        assert subset.ecg_ids.tolist() == [3, 1]
        np.testing.assert_array_equal(subset.signals[1], arrays.select([1]).signals[0])

    def test_select_unknown_id_raises(self, fake_ptbxl):
        _, arrays = load_dataset(fake_ptbxl)
        with pytest.raises(KeyError):
            arrays.select([UNLABELED_ECG_ID])

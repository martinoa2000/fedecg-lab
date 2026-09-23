"""Tests for splitting the training set across simulated hospitals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedecg.data.constants import SUPERCLASSES
from fedecg.data.partition import (
    dirichlet_partition,
    group_partition,
    iid_partition,
    partition_from_config,
    partition_summary,
    primary_label,
)
from fedecg.seed import new_generator


def assert_is_a_partition(clients: list[np.ndarray], n_records: int) -> None:
    """Every record lands in exactly one client."""
    merged = np.concatenate(clients)
    assert len(merged) == n_records
    np.testing.assert_array_equal(np.sort(merged), np.arange(n_records))
    assert all(np.all(np.diff(c) > 0) for c in clients)


def random_labels(n: int, seed: int = 0) -> np.ndarray:
    rng = new_generator(seed)
    labels = (rng.random((n, len(SUPERCLASSES))) < [0.45, 0.25, 0.25, 0.2, 0.1]).astype(float)
    labels[labels.sum(axis=1) == 0, 0] = 1.0
    return labels.astype(np.float32)


class TestIid:
    def test_covers_every_record_in_near_equal_shares(self):
        clients = iid_partition(103, 10, new_generator(0))
        assert_is_a_partition(clients, 103)
        assert {len(c) for c in clients} == {10, 11}

    def test_depends_only_on_the_generator(self):
        a = iid_partition(50, 4, new_generator(7))
        b = iid_partition(50, 4, new_generator(7))
        c = iid_partition(50, 4, new_generator(8))
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
        assert not all(np.array_equal(x, y) for x, y in zip(a, c, strict=True))

    def test_rejects_more_clients_than_records(self):
        with pytest.raises(ValueError):
            iid_partition(3, 4, new_generator(0))


class TestGroup:
    def test_one_client_per_value_with_small_values_pooled(self):
        values = pd.Series(["a"] * 5 + ["b"] * 4 + ["c"] * 1 + ["d"] * 2)
        clients, names = group_partition(values, min_records=3)
        assert names == ["a", "b", "other"]
        assert [len(c) for c in clients] == [5, 4, 3]
        assert_is_a_partition(clients, len(values))

    def test_strips_padding_and_keeps_missing_values(self):
        values = pd.Series(["CS-12   ", "CS-12", None, None])
        _, names = group_partition(values)
        assert sorted(names) == ["CS-12", "missing"]

    def test_needs_at_least_two_clients(self):
        with pytest.raises(ValueError, match="fewer than two"):
            group_partition(pd.Series(["a", "a", "b"]), min_records=3)


class TestDirichlet:
    def test_is_a_partition_with_a_minimum_size(self):
        labels = random_labels(600)
        clients = dirichlet_partition(labels, 5, 0.5, new_generator(0), min_records=40)
        assert_is_a_partition(clients, 600)
        assert min(len(c) for c in clients) >= 40

    def test_small_alpha_skews_labels_more_than_large_alpha(self):
        labels = random_labels(3000)

        def spread(alpha: float) -> float:
            clients = dirichlet_partition(labels, 6, alpha, new_generator(1), min_records=20)
            primary = primary_label(labels)
            # Mean over classes of the std of each client's share of that class.
            shares = np.array([np.bincount(primary[c], minlength=5) / len(c) for c in clients])
            return float(shares.std(axis=0).mean())

        assert spread(0.1) > 2 * spread(100.0)

    def test_primary_label_is_the_rarest_positive_class(self):
        labels = np.array([[1, 0, 0, 0, 1], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0]])
        # Frequencies: NORM 4, MI 2, STTC 1, CD 0, HYP 1.
        np.testing.assert_array_equal(primary_label(labels), [4, 1, 0, 2])

    def test_gives_up_when_the_minimum_cannot_be_met(self):
        labels = random_labels(100)
        with pytest.raises(RuntimeError, match="No Dirichlet"):
            dirichlet_partition(labels, 10, 0.01, new_generator(0), min_records=10, max_attempts=5)


class TestFromConfig:
    @pytest.fixture
    def train(self):
        n = 120
        meta = pd.DataFrame(
            {
                "site": [0.0] * 60 + [1.0] * 50 + [7.0] * 9 + [np.nan],
                "device": ["CS-12   E"] * 30 + ["AT-6 C 5.5"] * 90,
            }
        )
        return meta, random_labels(n)

    @pytest.mark.parametrize(
        ("config", "names"),
        [
            ({"partition": "iid", "n_clients": 3}, ["hospital 1", "hospital 2", "hospital 3"]),
            ({"partition": "site", "min_client_records": 20}, ["site 0", "site 1", "other"]),
            ({"partition": "device"}, ["device AT-6 C 5.5", "device CS-12   E"]),
        ],
    )
    def test_schemes(self, train, config, names):
        meta, labels = train
        partition = partition_from_config(config, meta, labels, new_generator(0))
        assert_is_a_partition(partition.clients, len(labels))
        assert partition.names == names

    def test_dirichlet_from_config(self, train):
        meta, labels = train
        config = {"partition": "dirichlet", "n_clients": 3, "dirichlet_alpha": 1.0}
        partition = partition_from_config(config, meta, labels, new_generator(0))
        assert len(partition) == 3
        assert partition.sizes().sum() == len(labels)

    def test_rejects_unknown_scheme(self, train):
        meta, labels = train
        with pytest.raises(ValueError, match="Unknown partition"):
            partition_from_config({"partition": "zip"}, meta, labels, new_generator(0))

    def test_summary_has_one_row_per_client_with_prevalence(self, train):
        meta, labels = train
        partition = partition_from_config({"partition": "site"}, meta, labels, new_generator(0))
        summary = partition_summary(partition, labels)
        assert list(summary.index) == partition.names
        assert list(summary.columns) == ["n_records", *SUPERCLASSES]
        first = partition.clients[0]
        assert summary.iloc[0]["HYP"] == pytest.approx(labels[first, 4].mean() * 100, abs=0.01)

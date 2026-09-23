"""Tests for the shared experiments table."""

from __future__ import annotations

import pandas as pd
import pytest

from fedecg.data.constants import SUPERCLASSES
from fedecg.training.results import experiment_row, record_experiment


def metrics(macro: float) -> pd.DataFrame:
    rows = {c: {"auroc": 0.9, "f1": 0.7, "threshold": 0.5} for c in SUPERCLASSES}
    rows["macro"] = {"auroc": macro, "f1": 0.7, "threshold": float("nan")}
    return pd.DataFrame(rows).T


def test_row_takes_phase_and_setting_from_the_config():
    config = {"experiment": {"phase": 4, "setting": "FedAvg, IID, 10 hospitals"}}
    row = experiment_row(config, "fedavg_iid_10", metrics(0.91), n_clients=10)
    assert row["phase"] == 4
    assert row["setting"] == "FedAvg, IID, 10 hospitals"
    assert row["macro_auroc"] == 0.91
    assert row["auroc_HYP"] == 0.9
    assert row["n_clients"] == 10


def test_rerunning_a_config_replaces_its_row(tmp_path):
    path = tmp_path / "experiments.csv"
    record_experiment({"phase": 4, "run": "b", "setting": "B", "macro_auroc": 0.8}, path)
    record_experiment({"phase": 3, "run": "a", "setting": "A", "macro_auroc": 0.9}, path)
    table = record_experiment({"phase": 4, "run": "b", "setting": "B", "macro_auroc": 0.85}, path)

    assert list(table["run"]) == ["a", "b"]  # sorted by phase
    assert table.loc[table["run"] == "b", "macro_auroc"].item() == 0.85
    assert list(pd.read_csv(path)["run"]) == ["a", "b"]


def test_rejects_rows_without_required_columns(tmp_path):
    with pytest.raises(ValueError, match="macro_auroc"):
        record_experiment({"phase": 3, "run": "a", "setting": "A"}, tmp_path / "e.csv")

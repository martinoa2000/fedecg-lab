"""Smoke tests for the command-line scripts, run against the fake dataset."""

from __future__ import annotations

import importlib.util
import json

import pandas as pd
import pytest
import torch

from fedecg.paths import PROJECT_ROOT


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExploreData:
    def test_writes_every_table_and_figure(self, fake_ptbxl, tmp_path):
        script = load_script("explore_data")
        tables, figures = tmp_path / "tables", tmp_path / "figures"

        exit_code = script.main(
            ["--root", str(fake_ptbxl), "--tables", str(tables), "--figures", str(figures)]
        )

        assert exit_code == 0
        assert {p.name for p in tables.iterdir()} == {
            "label_distribution.csv",
            "label_cooccurrence.csv",
            "label_cardinality.csv",
            "label_combinations.csv",
            "records_by_site.csv",
            "records_by_device.csv",
        }
        assert {p.name for p in figures.iterdir()} == {
            "label_distribution.png",
            "class_examples.png",
            "preprocessing.png",
        }


class TestExportDashboard:
    def test_writes_the_files_the_app_reads(self, fake_ptbxl, tmp_path):
        script = load_script("export_dashboard")

        exit_code = script.main(
            [
                "--root", str(fake_ptbxl),
                "--out", str(tmp_path),
                "--examples", "2",
                "--tables", str(tmp_path / "none"),
            ]
        )  # fmt: skip

        assert exit_code == 0
        assert {p.name for p in tmp_path.iterdir()} == {
            "dataset.json",
            "signals.json",
            "experiments.json",
        }

    def test_experiments_carry_curves_and_client_mixes(self, fake_ptbxl, tmp_path):
        script = load_script("export_dashboard")
        tables = tmp_path / "tables"
        tables.mkdir()
        pd.DataFrame(
            [
                {"phase": 3, "run": "central", "setting": "C", "macro_auroc": 0.9},
                {"phase": 5, "run": "fed", "setting": "F", "macro_auroc": 0.8},
            ]
        ).to_csv(tables / "experiments.csv", index=False)
        history = {"train_loss": [0.5], "val_loss": [0.4], "val_macro_auroc": [0.8]}
        pd.DataFrame({"epoch": [1], **history}).to_csv(tables / "central_history.csv", index=False)
        pd.DataFrame({"round": [1], **history}).to_csv(tables / "fed_history.csv", index=False)
        pd.DataFrame({"client": ["site 0"], "n_records": [10], "NORM": [50.0]}).to_csv(
            tables / "fed_clients.csv", index=False
        )
        out = tmp_path / "out"

        script.main(["--root", str(fake_ptbxl), "--out", str(out), "--tables", str(tables)])

        experiments = json.loads((out / "experiments.json").read_text())
        assert [r["run"] for r in experiments["rows"]] == ["central", "fed"]
        assert experiments["histories"]["central"][0]["step"] == 1
        assert experiments["histories"]["fed"][0]["step"] == 1
        assert experiments["clients"] == {
            "fed": [{"client": "site 0", "n_records": 10, "NORM": 50.0}]
        }
        assert experiments["published"][0]["macro_auroc"] == 0.93

    def test_signals_are_integer_microvolts_per_lead(self, fake_ptbxl, tmp_path):
        script = load_script("export_dashboard")
        script.main(["--root", str(fake_ptbxl), "--out", str(tmp_path), "--examples", "1"])

        signals = json.loads((tmp_path / "signals.json").read_text())
        record = signals["records"][0]
        assert len(record["raw"]) == len(signals["leads"]) == 12
        assert all(isinstance(v, int) for v in record["raw"][0])
        assert len(record["raw"][0]) == 10 * signals["fs"]

    def test_examples_are_single_label_records_of_their_class(self, fake_ptbxl, tmp_path):
        script = load_script("export_dashboard")
        script.main(["--root", str(fake_ptbxl), "--out", str(tmp_path), "--examples", "3"])

        signals = json.loads((tmp_path / "signals.json").read_text())
        labels = {r["ecg_id"]: r["labels"] for r in signals["records"]}
        for cls, ids in signals["examples"].items():
            assert all(labels[i] == [cls] for i in ids)


class TestTrainCentralized:
    def test_trains_and_writes_metrics_history_and_checkpoint(self, fake_ptbxl, tmp_path):
        script = load_script("train_centralized")
        config = tmp_path / "tiny.yaml"
        config.write_text(
            "extends: smoke.yaml\n"
            "model: {base_channels: 8, blocks_per_stage: [1], norm_groups: 4}\n"
            "training: {epochs: 1, batch_size: 4, device: cpu}\n"
        )
        tables, checkpoints = tmp_path / "tables", tmp_path / "ckpt"

        exit_code = script.main(
            [
                "--config", str(config),
                "--root", str(fake_ptbxl),
                "--no-cache",
                "--tables", str(tables),
                "--checkpoints", str(checkpoints),
            ]
        )  # fmt: skip

        assert exit_code == 0
        assert {p.name for p in tables.iterdir()} == {
            "tiny_test_metrics.csv",
            "tiny_history.csv",
            "experiments.csv",
        }
        state = torch.load(checkpoints / "tiny.pt", weights_only=False)
        assert set(state) >= {"model_state", "standardizer", "thresholds", "config"}


class TestTrainFederated:
    @pytest.mark.parametrize("partition", ["iid", "site"])
    def test_trains_and_writes_metrics_history_clients_and_summary(
        self, fake_ptbxl, tmp_path, partition
    ):
        script = load_script("train_federated")
        config = tmp_path / f"tiny_{partition}.yaml"
        config.write_text(
            "extends: smoke_federated.yaml\n"
            "data: {subsample: null}\n"
            "model: {base_channels: 8, blocks_per_stage: [1], norm_groups: 4}\n"
            "training: {batch_size: 4, device: cpu}\n"
            f"federated: {{partition: {partition}, n_clients: 2, rounds: 2}}\n"
        )
        tables, checkpoints = tmp_path / "tables", tmp_path / "ckpt"

        exit_code = script.main(
            [
                "--config", str(config),
                "--root", str(fake_ptbxl),
                "--no-cache",
                "--tables", str(tables),
                "--checkpoints", str(checkpoints),
            ]
        )  # fmt: skip

        assert exit_code == 0
        name = f"tiny_{partition}"
        assert {p.name for p in tables.iterdir()} == {
            f"{name}_test_metrics.csv",
            f"{name}_history.csv",
            f"{name}_clients.csv",
            "experiments.csv",
        }
        history = pd.read_csv(tables / f"{name}_history.csv")
        assert list(history["round"]) == [1, 2]
        summary = pd.read_csv(tables / "experiments.csv")
        assert summary.loc[0, "run"] == name
        assert summary.loc[0, "partition"] == partition
        state = torch.load(checkpoints / f"{name}.pt", weights_only=False)
        assert set(state) >= {"model_state", "thresholds", "best_round", "client_names"}

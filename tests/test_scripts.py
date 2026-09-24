"""Smoke tests for the command-line scripts, run against the fake dataset."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
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
            "explain.json",
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

    def test_explain_carries_tables_and_example_maps(self, fake_ptbxl, tmp_path):
        script = load_script("export_dashboard")
        tables = tmp_path / "tables"
        tables.mkdir()
        pd.DataFrame(
            [{"superclass": "MI", "method": "grad_cam", "segment": "st", "enrichment": 0.5}]
        ).to_csv(tables / "saliency_segments.csv", index=False)
        masks = {s: np.zeros(1000, dtype=bool) for s in ("qrs", "st", "t", "other")}
        masks["st"][100:120] = True
        np.savez(
            tmp_path / "examples.npz",
            fs=100,
            leads=np.array(["I", "II"]),
            MI_signal=np.full((2, 1000), 0.5, dtype=np.float32),
            MI_ig=np.linspace(-1, 2, 2000, dtype=np.float32).reshape(2, 1000),
            MI_probability=np.float32(0.9),
            **{f"MI_{s}": m for s, m in masks.items()},
        )
        out = tmp_path / "out"

        script.main(
            [
                "--root", str(fake_ptbxl),
                "--out", str(out),
                "--tables", str(tables),
                "--saliency", str(tmp_path / "examples.npz"),
            ]
        )  # fmt: skip

        explain = json.loads((out / "explain.json").read_text())
        assert explain["segments"][0]["enrichment"] == 0.5
        example = explain["examples"][0]
        assert example["superclass"] == "MI"
        assert example["signal"][0][0] == 500  # microvolts
        assert max(max(lead) for lead in example["attribution"]) == 100
        assert min(min(lead) for lead in example["attribution"]) >= 0
        assert example["segments"]["st"] == [[100, 120]]

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


class TestPrivateTraining:
    @pytest.mark.parametrize("script_name", ["train_centralized", "train_federated"])
    def test_records_the_epsilon_spent(self, fake_ptbxl, tmp_path, script_name):
        script = load_script(script_name)
        config = tmp_path / "tiny_dp.yaml"
        base = "smoke_federated.yaml" if script_name == "train_federated" else "smoke.yaml"
        config.write_text(
            f"extends: {base}\n"
            "data: {subsample: null}\n"
            "model: {base_channels: 8, blocks_per_stage: [1], norm_groups: 4}\n"
            "training: {epochs: 2, batch_size: 4, device: cpu}\n"
            "federated: {n_clients: 2, rounds: 2}\n"
            "privacy: {enabled: true, target_epsilon: 5.0}\n"
            "experiment: {phase: 6, setting: tiny}\n"
        )
        tables = tmp_path / "tables"

        exit_code = script.main(
            [
                "--config", str(config),
                "--root", str(fake_ptbxl),
                "--no-cache",
                "--tables", str(tables),
                "--checkpoints", str(tmp_path / "ckpt"),
            ]
        )  # fmt: skip

        assert exit_code == 0
        row = pd.read_csv(tables / "experiments.csv").iloc[0]
        assert row["phase"] == 6
        assert 0 < row["epsilon"] <= 5.0 * 1.05


class TestReproduce:
    def test_every_config_it_runs_exists(self):
        """A renamed config must not silently break the full reproduction."""
        import itertools
        import re

        # Walk the script with a stack of enclosing `for` loops, expanding each
        # --config template over the loops it sits in.
        names, stack = set(), []
        for line in (PROJECT_ROOT / "scripts" / "reproduce.sh").read_text().splitlines():
            if loop := re.search(r"for (\w+) in ([\w ]+); do", line):
                stack.append((loop[1], loop[2].split()))
            elif line.strip() == "done":
                stack.pop()
            elif template := re.search(r'--config "([^"]+)"', line):
                for combo in itertools.product(*(values for _, values in stack)):
                    name = template[1]
                    for (var, _), value in zip(stack, combo, strict=True):
                        name = re.sub(rf"\$\{{?{var}\}}?", value, name)
                    names.add(name)
        assert len(names) == 29  # phase 3: 4, tuning: 8, phases 4-6: 3 + 6 + 8
        missing = sorted(n for n in names if not (PROJECT_ROOT / "configs" / n).is_file())
        assert missing == []

    def test_every_script_it_runs_exists(self):
        import re

        script = (PROJECT_ROOT / "scripts" / "reproduce.sh").read_text()
        for name in set(re.findall(r"run (scripts/\w+\.py)", script)):
            assert (PROJECT_ROOT / name).is_file(), name


class TestTune:
    def test_scores_validation_and_writes_a_tuning_row(self, fake_ptbxl, tmp_path):
        script = load_script("tune")
        config = tmp_path / "tiny_tune.yaml"
        config.write_text(
            "extends: smoke.yaml\n"
            "data: {subsample: null}\n"
            "model: {base_channels: 8, blocks_per_stage: [1], norm_groups: 4}\n"
            "training: {epochs: 1, batch_size: 4, device: cpu, loss: focal, ensemble_seeds: [1, 2],"
            " augment: {amplitude: 0.0, noise: 0.05, wander: 0.0, lead_dropout: 0.0}}\n"
            "experiment: {setting: Tiny try}\n"
        )
        tables = tmp_path / "tables"

        exit_code = script.main(
            [
                "--config",
                str(config),
                "--root",
                str(fake_ptbxl),
                "--no-cache",
                "--tables",
                str(tables),
            ]
        )

        assert exit_code == 0
        assert {p.name for p in tables.iterdir()} == {"tuning.csv", "tuning_tiny_tune_history.csv"}
        row = pd.read_csv(tables / "tuning.csv").iloc[0]
        assert row["setting"] == "Tiny try"
        assert row["seeds"] == 2 and row["loss"] == "focal" and row["augment"] == "noise 0.05"
        # Never the test fold: no experiments.csv, no test metrics.
        assert not (tables / "experiments.csv").exists()

    def test_centralized_training_trains_every_ensemble_member(self, fake_ptbxl, tmp_path):
        script = load_script("train_centralized")
        config = tmp_path / "tiny_ens.yaml"
        config.write_text(
            "extends: smoke.yaml\n"
            "data: {subsample: null}\n"
            "model: {base_channels: 8, blocks_per_stage: [1], norm_groups: 4}\n"
            "training: {epochs: 1, batch_size: 4, device: cpu, ensemble_seeds: [1, 2, 3]}\n"
        )
        script.main(
            [
                "--config", str(config),
                "--root", str(fake_ptbxl),
                "--no-cache",
                "--tables", str(tmp_path / "tables"),
                "--checkpoints", str(tmp_path / "ckpt"),
            ]
        )  # fmt: skip
        state = torch.load(tmp_path / "ckpt" / "tiny_ens.pt", weights_only=False)
        assert len(state["ensemble_states"]) == 3

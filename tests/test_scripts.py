"""Smoke tests for the command-line scripts, run against the fake dataset."""

from __future__ import annotations

import importlib.util

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
        }
        state = torch.load(checkpoints / "tiny.pt", weights_only=False)
        assert set(state) >= {"model_state", "standardizer", "thresholds", "config"}

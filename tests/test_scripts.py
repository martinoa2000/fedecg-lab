"""Smoke tests for the command-line scripts, run against the fake dataset."""

from __future__ import annotations

import importlib.util
import json

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
            ["--root", str(fake_ptbxl), "--out", str(tmp_path), "--examples", "2"]
        )

        assert exit_code == 0
        assert {p.name for p in tmp_path.iterdir()} == {
            "dataset.json",
            "signals.json",
            "experiments.json",
        }

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

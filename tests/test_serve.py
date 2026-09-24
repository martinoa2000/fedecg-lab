"""Tests for the dashboard's local training server (no training, no dataset)."""

from __future__ import annotations

import http.client
import json
import sys
import threading
import time

import pytest
import yaml

from fedecg.config import load_config
from fedecg.paths import PROJECT_ROOT


def load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while being created.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serve = load_script("serve")


def wait_for(trainer, run_id, statuses=("done", "failed", "stopped"), timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = trainer.detail(run_id)
        if run["status"] in statuses:
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} still {run['status']}")


def fake_run(trainer, script: str):
    run = serve.Run(
        serve.uuid.uuid4().hex[:8], "tune", "fake", "fake", [sys.executable, "-c", script]
    )
    return trainer._enqueue(run)


class TestOptions:
    def test_defaults_fill_every_option(self):
        options = serve.validate_options({})
        assert set(options) == set(serve.OPTIONS)
        # The phase 3 recipe in configs/centralized.yaml.
        assert options["base_channels"] == 64 and options["loss"] == "bce"
        assert options["noise"] == 0.05

    @pytest.mark.parametrize(
        "bad",
        [
            {"base_channels": 48},
            {"epochs": 0},
            {"epochs": 1.5},
            {"seeds": True},
            {"noise": 0.9},
            {"loss": "mse"},
            {"script": "rm -rf /"},
            ["not", "an", "object"],
        ],
    )
    def test_rejects_anything_outside_the_whitelist(self, bad):
        with pytest.raises(ValueError):
            serve.validate_options(bad)

    def test_generated_config_resolves_to_the_chosen_setting(self, tmp_path):
        options = serve.validate_options(
            {"base_channels": 64, "loss": "focal", "seeds": 3, "noise": 0.05, "epochs": 20}
        )
        path = tmp_path / "try.yaml"
        path.write_text(yaml.safe_dump(serve.tune_config(options, serve.describe(options))))
        config = load_config(path)
        assert config["model"]["base_channels"] == 64
        assert config["training"]["loss"] == "focal"
        assert config["training"]["ensemble_seeds"] == [42, 43, 44]
        assert config["training"]["augment"]["noise"] == 0.05
        assert config["training"]["epochs"] == 20
        # Inherited from the centralized baseline, untouched.
        assert config["data"]["train_folds"] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert (
            "64 ch" in config["experiment"]["setting"]
            and "3 seeds" in config["experiment"]["setting"]
        )


class TestExperiments:
    def test_lists_experiment_configs_with_their_script(self):
        experiments = {e["name"]: e for e in serve.list_experiments()}
        assert experiments["centralized"]["script"] == "train_centralized"
        assert experiments["fedavg_site"]["script"] == "train_federated"
        assert experiments["fed_dp_eps8"]["phase"] == 6
        for base in ("default", "federated", "dp", "fed_dp", "smoke", "tune_wide"):
            assert base not in experiments

    def test_rejects_unknown_configs(self, tmp_path):
        trainer = serve.Trainer(runs_dir=tmp_path, export=False)
        with pytest.raises(ValueError):
            trainer.submit_experiment("../../etc/passwd")


class TestRuns:
    def test_parses_progress_and_result(self, tmp_path):
        trainer = serve.Trainer(runs_dir=tmp_path, export=False)
        run = fake_run(
            trainer,
            "print('ensemble member 1 of 2 (seed 42)');"
            "print('epoch   1  lr 1e-03  train_loss 0.5  val_loss 0.4  val_macro_auroc 0.8123  (2.0s)');"
            "print('epoch   2  lr 1e-03  train_loss 0.4  val_loss 0.3  val_macro_auroc 0.8456  (2.0s)');"
            "print('validation macro AUROC 0.8500 (members: 0.84, 0.85)')",
        )
        done = wait_for(trainer, run.id)
        assert done["status"] == "done"
        assert [p["step"] for p in done["progress"]] == [1, 2]
        assert done["progress"][1]["val_macro_auroc"] == pytest.approx(0.8456)
        assert done["members"] == 2
        assert done["result"] == pytest.approx(0.85)
        assert "validation macro AUROC" in done["log"][-1]

    def test_runs_one_at_a_time_and_can_be_stopped(self, tmp_path):
        trainer = serve.Trainer(runs_dir=tmp_path, export=False)
        first = fake_run(trainer, "import time; time.sleep(30)")
        second = fake_run(trainer, "print('never')")
        wait_for(trainer, first.id, statuses=("running",))
        assert trainer.detail(second.id)["status"] == "queued"

        assert trainer.stop(second.id).status == "stopped"
        trainer.stop(first.id)
        assert wait_for(trainer, first.id)["status"] == "stopped"

    def test_failed_scripts_are_reported(self, tmp_path):
        trainer = serve.Trainer(runs_dir=tmp_path, export=False)
        run = fake_run(trainer, "raise SystemExit(3)")
        done = wait_for(trainer, run.id)
        assert done["status"] == "failed" and done["returncode"] == 3


class TestHttp:
    @pytest.fixture
    def server(self, tmp_path):
        server = serve.serve(0, serve.Trainer(runs_dir=tmp_path, export=False))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield server
        server.shutdown()

    def request(self, server, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
        conn.request(
            method, path, body=json.dumps(body) if body is not None else None, headers=headers or {}
        )
        response = conn.getresponse()
        return response.status, json.loads(response.read())

    def test_health_and_options(self, server):
        assert self.request(server, "GET", "/api/health") == (200, {"ok": True, "busy": False})
        status, options = self.request(server, "GET", "/api/options")
        assert status == 200 and "base_channels" in options

    def test_refuses_requests_from_other_sites(self, server):
        status, _ = self.request(
            server, "GET", "/api/runs", headers={"Origin": "https://evil.example"}
        )
        assert status == 403
        status, _ = self.request(server, "GET", "/api/runs", headers={"Host": "evil.example"})
        assert status == 403

    def test_bad_options_are_a_client_error(self, server):
        status, body = self.request(server, "POST", "/api/tune", {"options": {"epochs": 1000}})
        assert status == 400 and "epochs" in body["error"]

    def test_unknown_run_is_not_found(self, server):
        assert self.request(server, "GET", "/api/runs/deadbeef")[0] == 404

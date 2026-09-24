#!/usr/bin/env python
"""A local training server for the dashboard's Train page.

Lets the dashboard start training runs on this machine and follow them live.
Two kinds of run:

- **tune**: a setting assembled from whitelisted options (width, crops,
  augmentation, loss, ensemble, epochs, learning rate) is written to a config
  that extends `configs/centralized.yaml` and scored on the validation fold
  by `scripts/tune.py`. Never touches the test fold.
- **experiment**: an existing config from `configs/` is trained for real by
  `scripts/train_centralized.py` or `scripts/train_federated.py`, which score
  the test fold and update `results/tables/experiments.csv`.

Runs execute one at a time (there is one GPU) in a queue. Each run's output is
kept in `runs/<id>.log`, its epoch or round lines parsed into progress, and
after every successful run the dashboard data is exported again.

Security: the server binds to 127.0.0.1 only, rejects requests whose Host or
Origin is not local (so a web page elsewhere cannot drive it), runs only the
scripts named above, and only accepts option values from fixed ranges.

Usage:
    uv run python scripts/serve.py            # then: npm --prefix app run dev
    uv run python scripts/serve.py --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"
SCRIPTS = PROJECT_ROOT / "scripts"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

# ------------------------------------------------------------------ options

OPTIONS: dict[str, dict[str, Any]] = {
    "base_channels": {"type": "choice", "values": [16, 32, 64, 96], "default": 32},
    "crop_samples": {"type": "choice", "values": [None, 250, 500], "default": 250},
    "epochs": {"type": "int", "min": 1, "max": 100, "default": 50},
    "learning_rate": {"type": "choice", "values": [0.0003, 0.001, 0.003], "default": 0.001},
    "loss": {"type": "choice", "values": ["bce", "weighted_bce", "focal"], "default": "bce"},
    "seeds": {"type": "int", "min": 1, "max": 5, "default": 1},
    "amplitude": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.0},
    "noise": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.0},
    "wander": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.0},
    "lead_dropout": {"type": "float", "min": 0.0, "max": 0.5, "default": 0.0},
}
"""Every option the Train page may set, with its allowed values."""


def validate_options(raw: Any) -> dict[str, Any]:
    """Defaults filled in, every value checked against `OPTIONS`.

    Raises:
        ValueError: On an unknown option or a value outside its range.
    """
    if not isinstance(raw, dict):
        raise ValueError("options must be an object")
    unknown = set(raw) - set(OPTIONS)
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    out = {}
    for name, spec in OPTIONS.items():
        value = raw.get(name, spec["default"])
        if spec["type"] == "choice":
            if value not in spec["values"]:
                raise ValueError(f"{name} must be one of {spec['values']}")
        elif spec["type"] == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{name} must be in [{spec['min']}, {spec['max']}]")
        else:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a number")
            if not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{name} must be in [{spec['min']}, {spec['max']}]")
            value = float(value)
        out[name] = value
    return out


def describe(options: dict[str, Any]) -> str:
    """A short human label for a tuning setting, e.g. `64 ch, focal, 3 seeds`."""
    parts = [f"{options['base_channels']} ch"]
    parts.append(f"crop {options['crop_samples']}" if options["crop_samples"] else "full records")
    augment = [k for k in ("amplitude", "noise", "wander", "lead_dropout") if options[k] > 0]
    if augment:
        parts.append("aug " + "+".join(augment))
    if options["loss"] != "bce":
        parts.append(options["loss"].replace("_", " "))
    if options["seeds"] > 1:
        parts.append(f"{options['seeds']} seeds")
    parts.append(f"lr {options['learning_rate']:g}, {options['epochs']} ep")
    return ", ".join(parts)


def tune_config(options: dict[str, Any], label: str) -> dict[str, Any]:
    """A config extending the centralized baseline with `options` applied."""
    augment = {k: options[k] for k in ("amplitude", "noise", "wander", "lead_dropout")}
    training: dict[str, Any] = {
        "epochs": options["epochs"],
        "learning_rate": options["learning_rate"],
        "crop_samples": options["crop_samples"],
        "loss": options["loss"],
        "augment": augment,
    }
    if options["seeds"] > 1:
        training["ensemble_seeds"] = [42 + i for i in range(options["seeds"])]
    return {
        "extends": str(CONFIG_DIR / "centralized.yaml"),
        "experiment": {"setting": label},
        "model": {"base_channels": options["base_channels"]},
        "training": training,
        "tracking": {"enabled": False},
    }


def list_experiments(config_dir: Path = CONFIG_DIR) -> list[dict[str, Any]]:
    """Experiment configs: those declaring their own `experiment` section.

    Base configs (`federated.yaml`, `dp.yaml`, ...) only carry shared
    settings; smoke and tuning configs are not experiments. The phase may be
    inherited, and a config is federated if it or any parent has a
    `federated` section.
    """

    def chain(path: Path) -> list[dict[str, Any]]:
        configs = []
        while path.is_file():
            raw = yaml.safe_load(path.read_text()) or {}
            configs.append(raw)
            if not raw.get("extends"):
                break
            path = config_dir / Path(raw["extends"]).name
        return configs

    out = []
    for path in sorted(config_dir.glob("*.yaml")):
        if path.stem.startswith(("smoke", "tune_")):
            continue
        configs = chain(path)
        own = configs[0].get("experiment")
        if not own:
            continue
        phase = next(
            (c["experiment"]["phase"] for c in configs if (c.get("experiment") or {}).get("phase")),
            None,
        )
        out.append(
            {
                "name": path.stem,
                "phase": phase,
                "setting": own.get("setting", path.stem),
                "script": "train_federated"
                if any("federated" in c for c in configs)
                else "train_centralized",
            }
        )
    return out


# --------------------------------------------------------------------- runs

EPOCH = re.compile(r"^(epoch|round)\s+(\d+)\b.*val_macro_auroc\s+([0-9.]+)")
MEMBER = re.compile(r"^ensemble member (\d+) of (\d+)")
RESULT = re.compile(r"^(?:validation macro AUROC|macro)\s+([0-9.]+)")


@dataclass
class Run:
    """One queued, running or finished training run."""

    id: str
    kind: str
    label: str
    config: str
    command: list[str]
    options: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    returncode: int | None = None
    member: int = 1
    members: int = 1
    progress: list[dict[str, float]] = field(default_factory=list)
    result: float | None = None

    def summary(self) -> dict[str, Any]:
        """JSON-ready view without the command line."""
        data = asdict(self)
        data.pop("command")
        return data


class Trainer:
    """Queue of runs executed one at a time by a background thread."""

    def __init__(self, runs_dir: Path = RUNS_DIR, export: bool = True):
        self.runs_dir = runs_dir
        self.export = export
        self.runs: dict[str, Run] = {}
        self.queue: deque[str] = deque()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.process: subprocess.Popen | None = None
        self.current: str | None = None
        (runs_dir / "configs").mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._work, daemon=True).start()

    # -- submitting

    def submit_tune(self, raw_options: Any) -> Run:
        """Queue a validation-only run of a setting built from options."""
        options = validate_options(raw_options)
        run_id = uuid.uuid4().hex[:8]
        label = describe(options)
        config_path = self.runs_dir / "configs" / f"app_{run_id}.yaml"
        config_path.write_text(yaml.safe_dump(tune_config(options, label), sort_keys=False))
        command = [
            sys.executable, "-u", str(SCRIPTS / "tune.py"),
            "--config", str(config_path), "--name", f"app_{run_id}",
        ]  # fmt: skip
        return self._enqueue(Run(run_id, "tune", label, config_path.stem, command, options))

    def submit_experiment(self, name: Any) -> Run:
        """Queue a real training run of an existing experiment config."""
        experiments = {e["name"]: e for e in list_experiments()}
        if not isinstance(name, str) or name not in experiments:
            raise ValueError("unknown experiment config")
        experiment = experiments[name]
        command = [
            sys.executable, "-u", str(SCRIPTS / f"{experiment['script']}.py"),
            "--config", f"{name}.yaml",
        ]  # fmt: skip
        return self._enqueue(
            Run(uuid.uuid4().hex[:8], "experiment", experiment["setting"], name, command)
        )

    def _enqueue(self, run: Run) -> Run:
        with self.lock:
            self.runs[run.id] = run
            self.queue.append(run.id)
        self.wake.set()
        return run

    def stop(self, run_id: str) -> Run:
        """Cancel a queued run or terminate the running one."""
        with self.lock:
            run = self.runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            if run.status == "queued":
                self.queue.remove(run_id)
                run.status = "stopped"
                run.finished = time.time()
            elif run.status == "running" and self.process is not None:
                run.status = "stopping"
                self.process.terminate()
        return run

    # -- worker

    def _work(self) -> None:
        while True:
            self.wake.wait()
            with self.lock:
                run_id = self.queue.popleft() if self.queue else None
                if run_id is None:
                    self.wake.clear()
                    continue
                run = self.runs[run_id]
                run.status, run.started = "running", time.time()
            self._execute(run)

    def _execute(self, run: Run) -> None:
        log_path = self.runs_dir / f"{run.id}.log"
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}
        with log_path.open("w") as log:
            process = subprocess.Popen(
                run.command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            with self.lock:
                self.process, self.current = process, run.id
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                self._parse(run, line.strip())
            returncode = process.wait()
        with self.lock:
            self.process, self.current = None, None
            run.returncode, run.finished = returncode, time.time()
            run.status = (
                "stopped" if run.status == "stopping" else ("done" if returncode == 0 else "failed")
            )
        if run.status == "done" and self.export:
            subprocess.run(
                [sys.executable, str(SCRIPTS / "export_dashboard.py")],
                cwd=PROJECT_ROOT,
                capture_output=True,
                env=env,
            )

    def _parse(self, run: Run, line: str) -> None:
        if match := MEMBER.match(line):
            run.member, run.members = int(match[1]), int(match[2])
        elif match := EPOCH.match(line):
            run.progress.append(
                {"member": run.member, "step": int(match[2]), "val_macro_auroc": float(match[3])}
            )
        elif match := RESULT.match(line):
            run.result = float(match[1])

    # -- reading

    def list(self) -> list[dict[str, Any]]:
        """Every run, newest first, without logs."""
        with self.lock:
            return [r.summary() for r in sorted(self.runs.values(), key=lambda r: -r.created)]

    def detail(self, run_id: str, lines: int = 80) -> dict[str, Any]:
        """One run with the tail of its log."""
        with self.lock:
            run = self.runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            data = run.summary()
        log_path = self.runs_dir / f"{run_id}.log"
        tail = (
            log_path.read_text(errors="replace").splitlines()[-lines:] if log_path.exists() else []
        )
        # Progress bars redraw with carriage returns; keep only the last state.
        data["log"] = [line.split("\r")[-1] for line in tail]
        return data


# --------------------------------------------------------------------- http


def make_handler(trainer: Trainer) -> type[BaseHTTPRequestHandler]:
    """Request handler bound to `trainer`."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "fedecg-train/1"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass  # keep the terminal for training output

        def _local(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            origin = self.headers.get("Origin")
            origin_host = re.sub(r"^https?://", "", origin or "").rsplit(":", 1)[0]
            return host in LOCAL_HOSTS and (origin is None or origin_host in LOCAL_HOSTS)

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 10_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:
            if not self._local():
                return self._send(HTTPStatus.FORBIDDEN, {"error": "local requests only"})
            path = self.path.split("?")[0]
            if path == "/api/health":
                return self._send(HTTPStatus.OK, {"ok": True, "busy": trainer.current is not None})
            if path == "/api/options":
                return self._send(HTTPStatus.OK, OPTIONS)
            if path == "/api/experiments":
                return self._send(HTTPStatus.OK, list_experiments())
            if path == "/api/runs":
                return self._send(HTTPStatus.OK, trainer.list())
            if match := re.fullmatch(r"/api/runs/([0-9a-f]{8})", path):
                try:
                    return self._send(HTTPStatus.OK, trainer.detail(match[1]))
                except KeyError:
                    return self._send(HTTPStatus.NOT_FOUND, {"error": "no such run"})
            return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._local():
                return self._send(HTTPStatus.FORBIDDEN, {"error": "local requests only"})
            path = self.path.split("?")[0]
            try:
                body = self._body()
                if path == "/api/tune":
                    run = trainer.submit_tune(body.get("options", {}))
                    return self._send(HTTPStatus.CREATED, run.summary())
                if path == "/api/experiments":
                    run = trainer.submit_experiment(body.get("config"))
                    return self._send(HTTPStatus.CREATED, run.summary())
                if match := re.fullmatch(r"/api/runs/([0-9a-f]{8})/stop", path):
                    return self._send(HTTPStatus.OK, trainer.stop(match[1]).summary())
            except (ValueError, json.JSONDecodeError) as error:
                return self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except KeyError:
                return self._send(HTTPStatus.NOT_FOUND, {"error": "no such run"})
            return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    return Handler


def serve(port: int, trainer: Trainer | None = None) -> ThreadingHTTPServer:
    """An HTTP server on 127.0.0.1:`port` (not yet serving)."""
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(trainer or Trainer()))


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8765, help="Port on 127.0.0.1")
    args = parser.parse_args(argv)
    server = serve(args.port)
    print(f"Training server on http://127.0.0.1:{args.port} (Ctrl+C to stop)")
    print("Open the dashboard's Train page: npm --prefix app run dev")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

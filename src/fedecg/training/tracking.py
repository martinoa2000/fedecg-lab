"""Thin MLflow wrapper that can be switched off.

Every run logs its fully resolved config (after `extends` inheritance), so a
number in the MLflow UI can always be traced back to the exact settings that
produced it. Tracking is disabled in tests and by `tracking.enabled: false`;
the rest of the code calls the same methods either way.

Browse local runs with::

    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

Runs go to a SQLite database inside `mlruns/`, with artifacts next to it. The
plain-directory store is deprecated in MLflow 3.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fedecg.paths import MLRUNS_DIR


def flatten(config: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested config into dotted keys, e.g. `training.batch_size`."""
    flat: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{name}."))
        else:
            flat[name] = str(value)
    return flat


class Tracker:
    """Log params, metrics and files to MLflow, or do nothing when disabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log a (possibly nested) mapping of parameters."""
        if self.enabled:
            import mlflow

            mlflow.log_params(flatten(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log numeric metrics, optionally at a step (the epoch)."""
        if self.enabled:
            import mlflow

            mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str | Path) -> None:
        """Attach a local file to the run."""
        if self.enabled:
            import mlflow

            mlflow.log_artifact(str(path))


@contextmanager
def start_run(
    tracking_config: Mapping[str, Any],
    *,
    run_name: str | None = None,
    tracking_dir: str | Path = MLRUNS_DIR,
) -> Iterator[Tracker]:
    """Open an MLflow run for the duration of the block.

    Args:
        tracking_config: The config's `tracking` section (`enabled`,
            `experiment_name`).
        run_name: Human-readable run name shown in the UI.
        tracking_dir: Local directory holding `mlflow.db` and `artifacts/`.
    """
    if not tracking_config.get("enabled", False):
        yield Tracker(enabled=False)
        return

    import mlflow

    tracking_dir = Path(tracking_dir).resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")
    experiment = tracking_config.get("experiment_name", "fedecg")
    if mlflow.get_experiment_by_name(experiment) is None:
        # Pin artifacts under `tracking_dir`; the default is relative to the cwd.
        mlflow.create_experiment(
            experiment, artifact_location=(tracking_dir / "artifacts").as_uri()
        )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        yield Tracker(enabled=True)

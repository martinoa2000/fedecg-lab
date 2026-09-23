"""The shared results table every experiment appends to.

`results/tables/experiments.csv` holds one row per run, keyed by `run` (the
config file's stem). Rerunning a config replaces its row instead of adding a
second one, so the table always reflects the latest run of each experiment.
The dashboard and the notebooks read this table; the per-run CSVs next to it
hold the detail (per-class metrics, per-epoch or per-round history).

Required columns: `phase`, `run`, `setting`, `macro_auroc`. Everything else is
optional and may be empty for phases where it does not apply (for example
`n_clients` for the centralized baseline, `epsilon` before phase 6).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from fedecg.data.constants import SUPERCLASSES
from fedecg.paths import TABLES_DIR

EXPERIMENTS_CSV = TABLES_DIR / "experiments.csv"

COLUMNS: tuple[str, ...] = (
    "phase",
    "run",
    "setting",
    "algorithm",
    "partition",
    "n_clients",
    "macro_auroc",
    "macro_f1",
    *(f"auroc_{c}" for c in SUPERCLASSES),
    "best_step",
    "steps_run",
    "communication_mb",
    "epsilon",
    "seconds",
)
"""Column order of the table. `best_step` is an epoch or a round."""


def experiment_row(
    config: Mapping[str, Any],
    run: str,
    metrics: pd.DataFrame,
    **extra: Any,
) -> dict[str, Any]:
    """Build a row from a config's `experiment` section and a metrics table.

    Args:
        config: The resolved config. Its `experiment` section supplies `phase`
            and `setting` (a human-readable label for plots).
        run: Unique run name, normally the config file's stem.
        metrics: Output of `metrics_table` as a DataFrame (classes as rows).
        **extra: Any other column, e.g. `n_clients=10`.
    """
    experiment = config.get("experiment", {})
    row: dict[str, Any] = {
        "phase": experiment.get("phase"),
        "run": run,
        "setting": experiment.get("setting", run),
        "macro_auroc": float(metrics.loc["macro", "auroc"]),
        "macro_f1": float(metrics.loc["macro", "f1"]),
    }
    for name in SUPERCLASSES:
        row[f"auroc_{name}"] = float(metrics.loc[name, "auroc"])
    row.update(extra)
    return row


def record_experiment(row: Mapping[str, Any], path: str | Path = EXPERIMENTS_CSV) -> pd.DataFrame:
    """Insert or replace `row` (matched on `run`) and rewrite the table.

    Rows are kept sorted by phase, then run, so diffs of the committed CSV stay
    readable. Returns the updated table.
    """
    for key in ("phase", "run", "setting", "macro_auroc"):
        if row.get(key) is None:
            raise ValueError(f"Experiment row is missing {key!r}")
    path = Path(path)
    table = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=list(COLUMNS))
    table = table[table["run"] != row["run"]]
    new = pd.DataFrame([dict(row)])
    table = new if table.empty else pd.concat([table, new], ignore_index=True)
    ordered = [c for c in COLUMNS if c in table.columns]
    table = table[ordered + [c for c in table.columns if c not in COLUMNS]]
    table = table.sort_values(["phase", "run"], kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(path, index=False)
    return table

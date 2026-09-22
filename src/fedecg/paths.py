"""Canonical filesystem locations for the project.

Notebooks run with their own directory as the working directory, scripts run
from the repository root, and pytest runs from somewhere else again. Resolving
paths from the location of this file makes all three agree.
"""

from __future__ import annotations

from pathlib import Path

# src/fedecg/paths.py -> src/fedecg -> src -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
"""Root for all datasets. Git-ignored: nothing in here is ever committed."""

RAW_DATA_DIR = DATA_DIR / "raw"
"""Downloaded archives, exactly as retrieved from the source."""

PTBXL_DIR = DATA_DIR / "ptbxl"
"""Extracted PTB-XL tree (contains ptbxl_database.csv and records100/)."""

CACHE_DIR = DATA_DIR / "cache"
"""Derived arrays (e.g. all waveforms in one `.npz`). Safe to delete."""

CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
"""Local MLflow tracking store. Git-ignored."""


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path

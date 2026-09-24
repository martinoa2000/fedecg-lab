#!/usr/bin/env python
"""Export the data the dashboard in `app/` reads.

Writes into `app/public/data/` (git-ignored, since it contains PTB-XL
waveforms, which this repository does not redistribute):

    dataset.json      label statistics, split sizes, site and device breakdowns
    signals.json      a few example records per superclass, raw and band-passed
    experiments.json  rows of `results/tables/experiments.csv`, each run's
                      training curve and, for federated runs, every hospital's
                      size and label mix; plus the validation-only tuning
                      study from `results/tables/tuning.csv`
    explain.json      saliency tables from `scripts/explain_model.py` and one
                      example record per class with its attribution map

Signals are stored as integer microvolts to keep the file small; the app
divides by 1000 to get millivolts back.

Usage:
    python scripts/export_dashboard.py
    python scripts/export_dashboard.py --examples 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fedecg.config import load_config
from fedecg.data.constants import LEAD_NAMES, SUPERCLASSES
from fedecg.data.preprocess import filter_from_config
from fedecg.data.ptbxl import (
    attach_labels,
    load_diagnostic_map,
    load_metadata,
    load_signals,
    split_by_folds,
)
from fedecg.data.stats import (
    cooccurrence,
    label_cardinality,
    label_combinations,
    label_distribution,
    records_by,
)
from fedecg.explain.saliency import SEGMENTS
from fedecg.paths import PROJECT_ROOT, PTBXL_DIR, RESULTS_DIR, TABLES_DIR, ensure_dir

APP_DATA_DIR = PROJECT_ROOT / "app" / "public" / "data"

PUBLISHED = [
    {
        "model": "resnet1d_wang",
        "macro_auroc": 0.930,
        "source": "Strodthoff et al. (2021), superdiagnostic task, 100 Hz",
    },
]
"""Published PTB-XL results the baseline is compared against."""


def _records(frame: pd.DataFrame, key: str) -> list[dict]:
    """Turn a table indexed by `key` into a list of JSON-ready rows."""
    table = frame.reset_index()
    table = table.rename(columns={table.columns[0]: key})
    return json.loads(table.to_json(orient="records"))


def pick_examples(meta: pd.DataFrame, per_class: int) -> dict[str, list[int]]:
    """Choose single-label records per superclass, human-validated first."""
    labels = meta[list(SUPERCLASSES)].astype(bool)
    single = labels.sum(axis=1) == 1
    validated = meta["validated_by_human"].astype(bool)
    chosen = {}
    for name in SUPERCLASSES:
        pool = meta[labels[name] & single]
        ordered = pd.concat([pool[validated.loc[pool.index]], pool[~validated.loc[pool.index]]])
        chosen[name] = [int(i) for i in ordered.index[:per_class]]
    return chosen


def dataset_summary(meta: pd.DataFrame, raw_meta: pd.DataFrame, config: dict) -> dict:
    """Everything the Dataset view shows, computed from the labeled metadata."""
    data_cfg = config["data"]
    split = split_by_folds(
        meta,
        train_folds=data_cfg["train_folds"],
        val_fold=data_cfg["val_fold"],
        test_fold=data_cfg["test_fold"],
    )
    ages = meta.loc[meta["age"] < 300, "age"]
    return {
        "superclasses": list(SUPERCLASSES),
        "records_total": len(raw_meta),
        "records_labeled": len(meta),
        "patients": int(meta["patient_id"].nunique()),
        "split": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "split_folds": {
            "train": data_cfg["train_folds"],
            "val": data_cfg["val_fold"],
            "test": data_cfg["test_fold"],
        },
        "sampling_rate_hz": data_cfg["sampling_rate_hz"],
        "bandpass_hz": [
            data_cfg["preprocess"]["bandpass_low_hz"],
            data_cfg["preprocess"]["bandpass_high_hz"],
        ],
        "distribution": _records(label_distribution(meta, split), "superclass"),
        "cooccurrence": cooccurrence(meta).astype(int).values.tolist(),
        "cardinality": _records(label_cardinality(meta).to_frame("n_records"), "n_labels"),
        "combinations": _records(label_combinations(meta).to_frame("n_records"), "labels"),
        "by_site": _records(records_by(meta, "site"), "site"),
        "by_device": _records(records_by(meta, "device"), "device"),
        "age": {
            "median": float(ages.median()),
            "histogram": np.histogram(ages, bins=range(0, 95, 5))[0].tolist(),
            "bin_width": 5,
        },
        "sex": {"male": int((meta["sex"] == 0).sum()), "female": int((meta["sex"] == 1).sum())},
    }


def signal_examples(meta: pd.DataFrame, config: dict, root: Path, per_class: int) -> dict:
    """Raw and band-passed waveforms for the example records."""
    data_cfg = config["data"]
    fs = data_cfg["sampling_rate_hz"]
    examples = pick_examples(meta, per_class)
    ids = [i for name in SUPERCLASSES for i in examples[name]]
    raw = load_signals(meta.loc[ids], root, sampling_rate=fs)
    filtered = filter_from_config(raw, data_cfg["preprocess"], fs=fs)

    def microvolts(x: np.ndarray) -> list[list[int]]:
        return np.rint(x * 1000).astype(int).tolist()

    records = []
    for k, ecg_id in enumerate(ids):
        row = meta.loc[ecg_id]
        records.append(
            {
                "ecg_id": ecg_id,
                "labels": [c for c in SUPERCLASSES if row[c]],
                "age": None if row["age"] >= 300 else int(row["age"]),
                "sex": "male" if row["sex"] == 0 else "female",
                "device": str(row["device"]).strip(),
                "site": None if pd.isna(row["site"]) else int(row["site"]),
                "report": str(row["report"]).strip(),
                "scp_codes": row["scp_codes"],
                "validated": bool(row["validated_by_human"]),
                "strat_fold": int(row["strat_fold"]),
                "raw": microvolts(raw[k]),
                "filtered": microvolts(filtered[k]),
            }
        )
    return {"fs": fs, "leads": list(LEAD_NAMES), "examples": examples, "records": records}


def tuning_results(tables: Path) -> dict:
    """Validation-only tuning rows (scripts/tune.py) and their curves."""
    summary = tables / "tuning.csv"
    if not summary.exists():
        return {"rows": [], "histories": {}}
    table = pd.read_csv(summary)
    histories = {}
    for run in table["run"]:
        history = tables / f"tuning_{run}_history.csv"
        if history.exists():
            curve = pd.read_csv(history).rename(columns={"epoch": "step"})
            keep = ["step", "train_loss", "val_loss", "val_macro_auroc"]
            histories[run] = json.loads(curve[keep].round(4).to_json(orient="records"))
    return {"rows": json.loads(table.to_json(orient="records")), "histories": histories}


def experiment_results(tables: Path) -> dict:
    """Summary rows plus, per run, its training curve and client breakdown.

    Curves are keyed by run name. Centralized histories are indexed by epoch
    and federated ones by round; both are exported as a common `step`. The
    validation-only tuning study rides along under `tuning`.
    """
    summary = tables / "experiments.csv"
    tuning = tuning_results(tables)
    if not summary.exists():
        return {
            "rows": [],
            "histories": {},
            "clients": {},
            "published": PUBLISHED,
            "tuning": tuning,
        }
    table = pd.read_csv(summary)
    histories, clients = {}, {}
    for run in table["run"]:
        history = tables / f"{run}_history.csv"
        if history.exists():
            curve = pd.read_csv(history).rename(columns={"epoch": "step", "round": "step"})
            keep = ["step", "train_loss", "val_loss", "val_macro_auroc"]
            histories[run] = json.loads(curve[keep].round(4).to_json(orient="records"))
        breakdown = tables / f"{run}_clients.csv"
        if breakdown.exists():
            clients[run] = json.loads(pd.read_csv(breakdown).to_json(orient="records"))
    return {
        "rows": json.loads(table.to_json(orient="records")),
        "histories": histories,
        "clients": clients,
        "published": PUBLISHED,
        "tuning": tuning,
    }


def _spans(mask: np.ndarray) -> list[list[int]]:
    """`[start, end)` sample ranges where a boolean mask is true."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return [
        [int(a), int(b)]
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    ]


def explain_results(tables: Path, examples_path: Path) -> dict:
    """Saliency tables plus, if present, one example record per class.

    Attribution is exported as |IG| per lead and sample, scaled to integers
    0-100 by the record's 99.5th percentile (clipped above), so leads are
    comparable within a record and the bulk of the map stays visible.
    Signals are integer microvolts, as in signals.json.
    """
    out: dict = {"segments": [], "leads": [], "sanity": [], "examples": []}
    for key, name in (
        ("segments", "saliency_segments"),
        ("leads", "saliency_leads"),
        ("sanity", "saliency_sanity"),
    ):
        path = tables / f"{name}.csv"
        if path.exists():
            out[key] = json.loads(pd.read_csv(path).to_json(orient="records"))
    if not examples_path.exists():
        return out
    with np.load(examples_path) as saved:
        out["fs"] = int(saved["fs"])
        out["lead_names"] = [str(lead) for lead in saved["leads"]]
        for name in SUPERCLASSES:
            if f"{name}_signal" not in saved:
                continue
            ig = np.abs(saved[f"{name}_ig"])
            # A few samples carry most of the attribution; scaling by the
            # 99.5th percentile (and clipping) keeps the rest visible.
            scale = max(float(np.percentile(ig, 99.5)), 1e-12)
            out["examples"].append(
                {
                    "superclass": name,
                    "probability": round(float(saved[f"{name}_probability"]), 3),
                    "signal": np.rint(saved[f"{name}_signal"] * 1000).astype(int).tolist(),
                    "attribution": np.rint(np.clip(ig / scale, 0, 1) * 100).astype(int).tolist(),
                    "segments": {
                        seg: _spans(saved[f"{name}_{seg}"]) for seg in SEGMENTS if seg != "other"
                    },
                }
            )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=PTBXL_DIR, help="PTB-XL directory")
    parser.add_argument("--config", default="default.yaml", help="Config for split/filter")
    parser.add_argument("--out", type=Path, default=APP_DATA_DIR, help="Output directory")
    parser.add_argument("--examples", type=int, default=4, help="Example records per class")
    parser.add_argument(
        "--tables", type=Path, default=TABLES_DIR, help="Where experiment results are read from"
    )
    parser.add_argument(
        "--saliency",
        type=Path,
        default=RESULTS_DIR / "saliency_examples.npz",
        help="Example maps written by scripts/explain_model.py",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    config = load_config(args.config)
    out = ensure_dir(args.out)

    raw_meta = load_metadata(args.root)
    meta = attach_labels(raw_meta, load_diagnostic_map(args.root))

    outputs = {
        "dataset.json": dataset_summary(meta, raw_meta, config),
        "signals.json": signal_examples(meta, config, args.root, args.examples),
    }
    outputs["experiments.json"] = experiment_results(args.tables)
    outputs["explain.json"] = explain_results(args.tables, args.saliency)

    for name, payload in outputs.items():
        (out / name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"{out / name}  {(out / name).stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

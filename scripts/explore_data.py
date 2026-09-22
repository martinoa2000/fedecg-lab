#!/usr/bin/env python
"""Produce the exploration tables and figures for PTB-XL (roadmap phase 2).

Writes into `results/`:

    tables/label_distribution.csv   per-class counts and prevalence, per split
    tables/label_cooccurrence.csv   how often two superclasses share a record
    tables/label_cardinality.csv    records with 1, 2, ... superclasses
    tables/label_combinations.csv   most frequent exact label sets
    tables/records_by_site.csv      records and class mix per recording site
    tables/records_by_device.csv    records and class mix per recording device
    figures/label_distribution.png
    figures/class_examples.png      lead II of one record per superclass
    figures/preprocessing.png       one record before and after band-pass

The tables are small and committed; they are what the notebook narrates.

Usage:
    python scripts/explore_data.py
    python scripts/explore_data.py --root path/to/ptbxl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fedecg.config import load_config
from fedecg.data.constants import SUPERCLASSES
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
    example_ids,
    label_cardinality,
    label_combinations,
    label_distribution,
    records_by,
)
from fedecg.paths import FIGURES_DIR, PTBXL_DIR, TABLES_DIR, ensure_dir
from fedecg.viz import plot_class_examples, plot_ecg


def plot_label_distribution(table) -> plt.Figure:
    """Grouped bar chart of class prevalence in each split."""
    fig, ax = plt.subplots(figsize=(8, 4))
    splits = ["train", "val", "test"]
    width = 0.8 / len(splits)
    for i, name in enumerate(splits):
        positions = [x + (i - 1) * width for x in range(len(table))]
        ax.bar(positions, table[f"{name}_pct"], width=width, label=name)
    ax.set_xticks(range(len(table)), table.index)
    ax.set_ylabel("records with label (%)")
    ax.set_title("PTB-XL superclass prevalence by split")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=PTBXL_DIR, help="PTB-XL directory")
    parser.add_argument("--config", default="default.yaml", help="Config for split/filter")
    parser.add_argument("--tables", type=Path, default=TABLES_DIR, help="Output for CSVs")
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR, help="Output for PNGs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    config = load_config(args.config)
    data_cfg = config["data"]
    fs = data_cfg["sampling_rate_hz"]
    tables, figures = ensure_dir(args.tables), ensure_dir(args.figures)

    meta = attach_labels(load_metadata(args.root), load_diagnostic_map(args.root))
    split = split_by_folds(
        meta,
        train_folds=data_cfg["train_folds"],
        val_fold=data_cfg["val_fold"],
        test_fold=data_cfg["test_fold"],
    )
    print(
        f"{len(meta)} labeled records: {len(split.train)} train, "
        f"{len(split.val)} val, {len(split.test)} test"
    )

    distribution = label_distribution(meta, split)
    distribution.to_csv(tables / "label_distribution.csv")
    cooccurrence(meta).to_csv(tables / "label_cooccurrence.csv", index_label="superclass")
    label_cardinality(meta).to_csv(tables / "label_cardinality.csv")
    label_combinations(meta).to_csv(tables / "label_combinations.csv")
    for column in ("site", "device"):
        records_by(meta, column).to_csv(tables / f"records_by_{column}.csv")
    print(distribution.to_string())

    fig = plot_label_distribution(distribution)
    fig.savefig(figures / "label_distribution.png", dpi=150)
    plt.close(fig)

    examples = example_ids(meta)
    names = [c for c in SUPERCLASSES if c in examples]
    signals = load_signals(meta.loc[[examples[c] for c in names]], args.root, sampling_rate=fs)
    filtered = filter_from_config(signals, data_cfg["preprocess"], fs=fs)

    fig = plot_class_examples(filtered, names, fs=fs)
    fig.savefig(figures / "class_examples.png", dpi=150)
    plt.close(fig)

    fig = plot_ecg(
        signals[0],
        overlay=filtered[0],
        fs=fs,
        title=f"ecg_id {examples[names[0]]} ({names[0]}): raw vs. band-passed",
    )
    fig.savefig(figures / "preprocessing.png", dpi=120)
    plt.close(fig)

    print(f"Tables written to {tables}\nFigures written to {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

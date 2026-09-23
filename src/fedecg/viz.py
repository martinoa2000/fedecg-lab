"""Plotting helpers shared by notebooks and scripts.

Figures are returned rather than shown, so the same function serves a notebook
cell and a script that saves a PNG into `results/figures/` without a display.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fedecg.data.constants import LEAD_NAMES, NUM_LEADS, SAMPLING_RATE_HZ


def plot_ecg(
    signal: np.ndarray,
    *,
    fs: float = SAMPLING_RATE_HZ,
    title: str | None = None,
    overlay: np.ndarray | None = None,
    overlay_label: str = "filtered",
    leads: Sequence[str] = LEAD_NAMES,
) -> Figure:
    """Plot one 12-lead record as a column of strips sharing a time axis.

    Args:
        signal: `(NUM_LEADS, n_samples)` array in millivolts.
        fs: Sampling rate in Hz.
        title: Figure title.
        overlay: Optional second signal of the same shape drawn on top, e.g. the
            filtered version, to show what preprocessing removed.
        overlay_label: Legend label for `overlay`.
        leads: Lead names, one per row.
    """
    if signal.ndim != 2 or signal.shape[0] != NUM_LEADS:
        raise ValueError(f"Expected ({NUM_LEADS}, n_samples), got {signal.shape}")
    if overlay is not None and overlay.shape != signal.shape:
        raise ValueError("overlay must have the same shape as signal")

    time = np.arange(signal.shape[1]) / fs
    fig, axes = plt.subplots(NUM_LEADS, 1, figsize=(12, 14), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(time, signal[i], color="0.35", linewidth=0.8, label="raw")
        if overlay is not None:
            ax.plot(time, overlay[i], color="C3", linewidth=0.8, label=overlay_label)
        ax.set_ylabel(leads[i], rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("time (s)")
    if overlay is not None:
        axes[0].legend(loc="upper right", fontsize=8, ncols=2)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_class_examples(
    signals: np.ndarray,
    class_names: Sequence[str],
    *,
    lead: int = 1,
    fs: float = SAMPLING_RATE_HZ,
    seconds: float = 4.0,
) -> Figure:
    """One lead of one example record per class, stacked for comparison.

    Args:
        signals: `(n_classes, NUM_LEADS, n_samples)`: one record per class.
        class_names: A label for each row of `signals`.
        lead: Which lead to show; lead II (index 1) is the conventional rhythm
            strip.
        fs: Sampling rate in Hz.
        seconds: Length of the excerpt, from the start of the record.
    """
    if len(signals) != len(class_names):
        raise ValueError("Need exactly one signal per class name")
    n = int(seconds * fs)
    time = np.arange(n) / fs
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 1.8 * len(signals)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sig, name in zip(axes, signals, class_names, strict=True):
        ax.plot(time, sig[lead, :n], linewidth=0.9)
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(f"time (s), lead {LEAD_NAMES[lead]}")
    fig.tight_layout()
    return fig


def plot_curves(
    curves: Mapping[str, pd.DataFrame],
    *,
    metric: str = "val_macro_auroc",
    step_label: str = "epoch / round",
    reference: float | None = None,
    reference_label: str = "reference",
) -> Figure:
    """One line per run of `metric` against the epoch or round.

    Args:
        curves: Run label -> a history table with an `epoch` or `round` column,
            as written by the training scripts.
        metric: History column to plot.
        step_label: X-axis label.
        reference: Optional horizontal line, e.g. the centralized test AUROC.
        reference_label: Legend label for `reference`.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, history in curves.items():
        step = "epoch" if "epoch" in history.columns else "round"
        ax.plot(history[step], history[metric], linewidth=1.6, label=label)
    if reference is not None:
        ax.axhline(reference, color="0.4", linewidth=1, label=reference_label)
    ax.set_xlabel(step_label)
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig

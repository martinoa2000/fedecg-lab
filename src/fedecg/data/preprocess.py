"""Signal preprocessing: band-pass filtering and per-lead standardization.

Two steps, applied in this order:

1. **Band-pass filter** (default 0.5-40 Hz). The low cut removes baseline
   wander from breathing and electrode movement; the high cut removes muscle
   noise and mains interference. The filter is zero-phase (applied forwards and
   backwards), so it does not shift the timing of waves relative to each other.
   That matters: an ST-segment elevation that moved in time relative to the QRS
   would no longer look like one.

2. **Per-lead standardization.** Each lead is shifted and scaled by a mean and
   standard deviation computed on the *training* set only, then reused for
   validation and test. Fitting on all data would leak test statistics into the
   model. In the federated phases the same question returns in a sharper form:
   each hospital can only see its own data, so it has to fit its own
   statistics; `LeadStandardizer` is serializable so both options can be
   compared.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal as sp_signal

from fedecg.data.constants import NUM_LEADS, SAMPLING_RATE_HZ

DEFAULT_FILTER_ORDER = 3
"""Butterworth order per pass. Filtering forwards and backwards doubles it."""

_EPS = 1e-6
"""Floor on the standard deviation, so a flat-lined lead is not divided by 0."""


def bandpass(
    signals: np.ndarray,
    *,
    low_hz: float,
    high_hz: float,
    fs: float = SAMPLING_RATE_HZ,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter along the last (time) axis.

    Args:
        signals: Array whose last axis is time, e.g. `(n, leads, samples)`.
        low_hz: Lower cut-off frequency.
        high_hz: Upper cut-off frequency. Must be below Nyquist (`fs / 2`).
        fs: Sampling rate in Hz.
        order: Filter order of each pass.

    Returns:
        Filtered float32 array with the same shape as `signals`.
    """
    nyquist = fs / 2
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(f"Need 0 < low_hz < high_hz < fs/2, got {low_hz}, {high_hz}, fs={fs}")
    # Second-order sections are numerically stable at low normalized cut-offs
    # such as 0.5 Hz / 50 Hz, where the transfer-function form can blow up.
    sos = sp_signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, signals, axis=-1).astype(np.float32)


@dataclass
class LeadStandardizer:
    """Standardize each lead with statistics fitted on a training set.

    Signals are expected as `(n_records, NUM_LEADS, n_samples)`. The statistics
    pool every sample of every record, so each lead gets one mean and one
    standard deviation.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether `fit` (or `from_dict`) has been called."""
        return self.mean is not None and self.std is not None

    def fit(self, signals: np.ndarray) -> LeadStandardizer:
        """Compute per-lead mean and standard deviation from `signals`."""
        _check_layout(signals)
        # float64 accumulation: summing ~20M float32 values per lead loses
        # precision otherwise.
        self.mean = signals.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
        self.std = signals.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
        return self

    def transform(self, signals: np.ndarray) -> np.ndarray:
        """Apply the fitted standardization. Returns a new float32 array."""
        if not self.is_fitted:
            raise RuntimeError("LeadStandardizer must be fitted before transform")
        _check_layout(signals)
        mean = self.mean[None, :, None]
        std = np.maximum(self.std, _EPS)[None, :, None]
        return ((signals - mean) / std).astype(np.float32)

    def fit_transform(self, signals: np.ndarray) -> np.ndarray:
        """Fit on `signals` and return them standardized."""
        return self.fit(signals).transform(signals)

    def to_dict(self) -> dict[str, list[float]]:
        """Plain-Python form, suitable for YAML/JSON or an MLflow param."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted LeadStandardizer")
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> LeadStandardizer:
        """Inverse of `to_dict`."""
        mean = np.asarray(state["mean"], dtype=np.float32)
        std = np.asarray(state["std"], dtype=np.float32)
        if mean.shape != (NUM_LEADS,) or std.shape != (NUM_LEADS,):
            raise ValueError(f"Expected {NUM_LEADS} per-lead values for mean and std")
        return cls(mean=mean, std=std)


def _check_layout(signals: np.ndarray) -> None:
    if signals.ndim != 3 or signals.shape[1] != NUM_LEADS:
        raise ValueError(
            f"Expected signals shaped (n_records, {NUM_LEADS}, n_samples), got {signals.shape}"
        )


def filter_from_config(
    signals: np.ndarray, preprocess_config: Mapping[str, Any], *, fs: float
) -> np.ndarray:
    """Apply the band-pass filter described by a config's `data.preprocess`.

    A missing or `null` cut-off disables filtering.
    """
    low = preprocess_config.get("bandpass_low_hz")
    high = preprocess_config.get("bandpass_high_hz")
    if low is None or high is None:
        return signals.astype(np.float32, copy=False)
    return bandpass(signals, low_hz=low, high_hz=high, fs=fs)


def preprocess_splits(
    train: np.ndarray,
    others: Mapping[str, np.ndarray],
    preprocess_config: Mapping[str, Any],
    *,
    fs: float = SAMPLING_RATE_HZ,
) -> tuple[np.ndarray, dict[str, np.ndarray], LeadStandardizer | None]:
    """Filter every split, then standardize with statistics from `train` only.

    Args:
        train: Training signals; the only split the standardizer is fitted on.
        others: Remaining splits by name (e.g. `{"val": ..., "test": ...}`).
        preprocess_config: The `data.preprocess` section of a config.
        fs: Sampling rate in Hz.

    Returns:
        `(train, others, standardizer)`, where `standardizer` is None when the
        config's `normalize` is `null` / `none`.
    """
    normalize = preprocess_config.get("normalize")
    if normalize not in (None, "none", "per_lead"):
        raise ValueError(f"Unknown normalize mode {normalize!r}; use 'per_lead' or null")

    train = filter_from_config(train, preprocess_config, fs=fs)
    others = {name: filter_from_config(x, preprocess_config, fs=fs) for name, x in others.items()}
    if normalize != "per_lead":
        return train, others, None

    standardizer = LeadStandardizer()
    train = standardizer.fit_transform(train)
    others = {name: standardizer.transform(x) for name, x in others.items()}
    return train, others, standardizer

"""Tests for filtering and per-lead standardization."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from fedecg.data.constants import NUM_LEADS, SAMPLING_RATE_HZ, SIGNAL_LENGTH
from fedecg.data.preprocess import LeadStandardizer, bandpass, preprocess_splits
from fedecg.seed import new_generator

FS = SAMPLING_RATE_HZ
TIME = np.arange(SIGNAL_LENGTH) / FS


def tone(freq_hz: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq_hz * TIME)


def rms(x: np.ndarray) -> float:
    # Ignore the edges, where any IIR filter has a transient.
    core = x[..., 100:-100]
    return float(np.sqrt(np.mean(core**2)))


def random_signals(n: int, seed: int = 0) -> np.ndarray:
    rng = new_generator(seed)
    scale = np.linspace(0.2, 2.0, NUM_LEADS)[None, :, None]
    offset = np.linspace(-1.0, 1.0, NUM_LEADS)[None, :, None]
    return (rng.standard_normal((n, NUM_LEADS, SIGNAL_LENGTH)) * scale + offset).astype(np.float32)


class TestBandpass:
    def test_keeps_the_diagnostic_band(self):
        assert rms(bandpass(tone(10.0), low_hz=0.5, high_hz=40.0)) == pytest.approx(
            rms(tone(10.0)), rel=0.02
        )

    def test_removes_baseline_wander(self):
        wander = tone(0.1, amplitude=2.0)
        assert rms(bandpass(wander, low_hz=0.5, high_hz=40.0)) < 0.05 * rms(wander)

    def test_attenuates_mains_interference(self):
        # 50 Hz is Nyquist at 100 Hz, so test just below it.
        hum = tone(47.0)
        assert rms(bandpass(hum, low_hz=0.5, high_hz=40.0)) < 0.35 * rms(hum)

    def test_removes_dc_offset(self):
        filtered = bandpass(np.full(SIGNAL_LENGTH, 3.0), low_hz=0.5, high_hz=40.0)
        assert abs(filtered[100:-100].mean()) < 1e-2

    def test_is_zero_phase(self):
        """The filtered wave must peak where the raw wave peaks."""
        signal = tone(5.0)
        filtered = bandpass(signal, low_hz=0.5, high_hz=40.0)
        core = slice(200, 400)
        assert np.argmax(filtered[core]) == np.argmax(signal[core])

    def test_preserves_shape_and_filters_each_lead_independently(self):
        batch = np.stack([np.stack([tone(10.0)] * NUM_LEADS)] * 2)
        batch[1, 3] = 0.0
        filtered = bandpass(batch, low_hz=0.5, high_hz=40.0)
        assert filtered.shape == batch.shape
        assert filtered.dtype == np.float32
        np.testing.assert_allclose(filtered[1, 3], 0.0, atol=1e-6)
        np.testing.assert_allclose(filtered[0, 0], filtered[1, 0], atol=1e-6)

    @pytest.mark.parametrize(("low", "high"), [(0.0, 40.0), (40.0, 0.5), (0.5, 50.0)])
    def test_rejects_invalid_cutoffs(self, low, high):
        with pytest.raises(ValueError, match="fs/2"):
            bandpass(tone(10.0), low_hz=low, high_hz=high)


class TestLeadStandardizer:
    def test_training_set_ends_up_zero_mean_unit_std_per_lead(self):
        standardized = LeadStandardizer().fit_transform(random_signals(16))
        np.testing.assert_allclose(standardized.mean(axis=(0, 2)), 0.0, atol=1e-4)
        np.testing.assert_allclose(standardized.std(axis=(0, 2)), 1.0, atol=1e-4)

    def test_uses_training_statistics_on_other_data(self):
        train = random_signals(8, seed=1)
        shifted = train + 5.0
        standardizer = LeadStandardizer().fit(train)
        np.testing.assert_allclose(
            standardizer.transform(shifted),
            standardizer.transform(train) + 5.0 / standardizer.std[None, :, None],
            rtol=1e-4,
        )

    def test_flat_lead_does_not_produce_nan(self):
        signals = random_signals(4)
        signals[:, 5] = 0.0
        standardized = LeadStandardizer().fit_transform(signals)
        assert np.isfinite(standardized).all()

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fitted"):
            LeadStandardizer().transform(random_signals(1))

    def test_rejects_wrong_layout(self):
        with pytest.raises(ValueError, match="n_records"):
            LeadStandardizer().fit(np.zeros((2, SIGNAL_LENGTH, NUM_LEADS), dtype=np.float32))

    def test_serialization_round_trip(self):
        signals = random_signals(4)
        original = LeadStandardizer().fit(signals)
        restored = LeadStandardizer.from_dict(original.to_dict())
        np.testing.assert_array_equal(original.transform(signals), restored.transform(signals))

    def test_from_dict_checks_lead_count(self):
        with pytest.raises(ValueError, match="per-lead"):
            LeadStandardizer.from_dict({"mean": [0.0], "std": [1.0]})


class TestPreprocessSplits:
    CONFIG: ClassVar[dict] = {
        "bandpass_low_hz": 0.5,
        "bandpass_high_hz": 40.0,
        "normalize": "per_lead",
    }

    def test_fits_on_train_only(self):
        train, val = random_signals(8, seed=0), random_signals(4, seed=1) + 3.0
        train_out, others, standardizer = preprocess_splits(train, {"val": val}, self.CONFIG)
        assert standardizer is not None
        np.testing.assert_allclose(train_out.mean(axis=(0, 2)), 0.0, atol=1e-3)
        # The band-pass already removed val's DC offset, but its statistics
        # must not have influenced the fit: refitting on train alone agrees.
        refit = LeadStandardizer().fit(bandpass(train, low_hz=0.5, high_hz=40.0))
        np.testing.assert_allclose(standardizer.std, refit.std, rtol=1e-6)
        assert others["val"].shape == val.shape

    def test_normalization_can_be_disabled(self):
        config = {**self.CONFIG, "normalize": None}
        _, _, standardizer = preprocess_splits(random_signals(2), {}, config)
        assert standardizer is None

    def test_filtering_can_be_disabled(self):
        config = {"bandpass_low_hz": None, "bandpass_high_hz": None, "normalize": None}
        train = random_signals(2)
        out, _, _ = preprocess_splits(train, {}, config)
        np.testing.assert_array_equal(out, train)

    def test_unknown_normalize_mode_raises(self):
        with pytest.raises(ValueError, match="normalize"):
            preprocess_splits(random_signals(1), {}, {"normalize": "global"})

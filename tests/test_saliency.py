"""Tests for attribution, beat segmentation and enrichment (no dataset needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fedecg.explain.saliency import (
    SEGMENTS,
    beat_segments,
    blurred_baseline,
    grad_cam,
    integrated_gradients,
    lead_shares,
    randomization_similarity,
    segment_shares,
    summarize_enrichment,
)
from fedecg.models.resnet1d import ResNet1d

FS = 100


@pytest.fixture(scope="module")
def simulated_lead() -> np.ndarray:
    """Ten seconds of a clean simulated ECG at 100 Hz, in millivolts."""
    import neurokit2 as nk

    return nk.ecg_simulate(duration=10, sampling_rate=FS, heart_rate=70, noise=0.0, random_state=0)


def tiny_model() -> ResNet1d:
    torch.manual_seed(0)
    return ResNet1d(base_channels=8, blocks_per_stage=(1, 1), norm_groups=4, dropout=0.0).eval()


class TestSegments:
    def test_masks_partition_the_record(self, simulated_lead):
        masks = beat_segments(simulated_lead, FS)
        assert masks is not None
        stacked = np.stack([masks[s] for s in SEGMENTS])
        assert (stacked.sum(axis=0) == 1).all()  # every sample in exactly one segment

    def test_segments_have_plausible_durations(self, simulated_lead):
        masks = beat_segments(simulated_lead, FS)
        beats = 70 * 10 / 60
        qrs_ms = masks["qrs"].sum() / beats / FS * 1000
        assert 40 <= qrs_ms <= 160  # a normal QRS lasts roughly 80-120 ms
        assert masks["st"].sum() > 0 and masks["t"].sum() > 0

    def test_flat_line_has_no_beats(self):
        assert beat_segments(np.zeros(1000), FS) is None


class TestShares:
    def test_attribution_confined_to_a_segment_gets_all_of_it(self, simulated_lead):
        masks = beat_segments(simulated_lead, FS)
        attribution = np.zeros((12, len(simulated_lead)))
        attribution[:, masks["st"]] = -1.0  # sign does not matter
        shares = segment_shares(attribution, masks)
        assert shares["st"][0] == pytest.approx(1.0)
        assert shares["st"][1] == pytest.approx(masks["st"].mean())
        assert shares["qrs"][0] == 0.0

    def test_uniform_attribution_has_enrichment_one(self, simulated_lead):
        masks = beat_segments(simulated_lead, FS)
        shares = segment_shares(np.ones(len(simulated_lead)), masks)
        summary = summarize_enrichment([shares, shares])
        for segment in SEGMENTS:
            assert summary[segment]["enrichment"] == pytest.approx(1.0)

    def test_lead_shares_sum_to_one(self):
        attribution = np.zeros((12, 50))
        attribution[3] = 2.0
        attribution[7] = -2.0
        shares = lead_shares(attribution)
        assert shares.sum() == pytest.approx(1.0)
        assert shares[3] == shares[7] == pytest.approx(0.5)


class TestAttribution:
    def test_integrated_gradients_are_complete(self):
        """Attributions sum to logit(x) - logit(blurred x), up to Riemann error."""
        model = tiny_model()
        x = torch.randn(3, 12, 200)
        attr = integrated_gradients(model, x, target=1, steps=128)
        assert attr.shape == x.shape
        with torch.no_grad():
            gap = model(x)[:, 1] - model(blurred_baseline(x, 20.0))[:, 1]
        np.testing.assert_allclose(attr.sum(axis=(1, 2)), gap.numpy(), rtol=0.05, atol=0.02)

    def test_the_model_is_scale_invariant_so_a_zero_baseline_is_useless(self):
        """Why IG uses a blurred baseline: f(a x) == f(x) for a > 0, f(0) differs."""
        model = tiny_model()
        x = torch.randn(2, 12, 200)
        with torch.no_grad():
            torch.testing.assert_close(model(0.1 * x), model(x), atol=1e-3, rtol=1e-3)
            assert not torch.allclose(model(torch.zeros_like(x)), model(x), atol=1e-3)

    def test_grad_cam_matches_input_length_and_is_non_negative(self):
        model = tiny_model()
        cam = grad_cam(model, torch.randn(2, 12, 200), target=0, layer=model.blocks[-1])
        assert cam.shape == (2, 200)
        assert (cam >= 0).all()

    def test_randomization_similarity(self):
        rng = np.random.default_rng(0)
        maps = rng.standard_normal((4, 12, 50))
        np.testing.assert_allclose(randomization_similarity(maps, maps), 1.0)
        unrelated = randomization_similarity(maps, rng.standard_normal((4, 12, 50)))
        assert np.abs(unrelated).max() < 0.2
        with pytest.raises(ValueError, match="shapes"):
            randomization_similarity(maps, maps[:, :6])

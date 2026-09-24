"""Where on the ECG does the model look? Saliency, beat segments, enrichment.

A saliency map on its own is a picture; this module turns it into a number
that can be checked against cardiology. Three steps:

1. **Attribution.** Integrated Gradients (Sundararajan et al., 2017) credits
   every sample of every lead with part of the model's logit for one class,
   relative to a baseline input. Grad-CAM (Selvaraju et al., 2017) gives a
   coarser, lead-free map from the last residual stage.

   **The baseline is a blurred copy of the record, not zeros.** The ResNet's
   first convolution has no bias and is followed by GroupNorm, so the network
   is scale-invariant: `f(a * x) == f(x)` for every `a > 0` (checked on the
   trained model down to `a = 0.01`). The straight path from a zero baseline
   is exactly such a rescaling, so the output does not change along it until
   it jumps at 0, and zero-baseline IG measures nothing. A Gaussian blur of
   0.2 s removes the QRS, ST and T morphology but keeps the slow trend
   (Sturmfels et al., 2020, "Visualizing the Impact of Feature Attribution
   Baselines"), so the attributions answer: what does the *shape* of the
   waveform contribute to this prediction?

2. **Beat segments.** Lead II is delineated with neurokit2 into, per beat, the
   QRS complex (ventricular depolarization), the ST segment (from the J point
   to the start of the T wave) and the T wave (repolarization). All twelve
   leads are recorded simultaneously, so the same timing applies to each.

3. **Enrichment.** The share of absolute attribution that falls in a segment,
   divided by the share of the record's time the segment covers. 1 means the
   model attends to that segment no more than to any other stretch of signal;
   2 means twice as much. Myocardial infarction and ST/T changes are
   diagnosed from the ST segment and T wave, so a model that reasons like a
   cardiologist should show enrichment there for those classes.

Enrichment only means something if the maps reflect what the model learned.
`randomization_similarity` implements the model-randomization check of
Adebayo et al. (2018): maps from the trained model are compared with maps
from the same architecture with random weights. If they are alike, the maps
show the input's structure, not the model's reasoning.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

SEGMENTS: tuple[str, ...] = ("qrs", "st", "t", "other")
"""Beat segments, in the order tables report them. `other` is everything
outside the first three: P wave, PR and TP baseline, and unlabeled stretches."""


def blurred_baseline(signals: torch.Tensor, sigma: float) -> torch.Tensor:
    """Each lead of each record smoothed by a Gaussian of `sigma` samples."""
    from scipy.ndimage import gaussian_filter1d

    blurred = gaussian_filter1d(signals.detach().cpu().numpy(), sigma, axis=-1, mode="nearest")
    return torch.from_numpy(blurred.astype(np.float32)).to(signals.device)


def integrated_gradients(
    model: nn.Module,
    signals: torch.Tensor,
    target: int,
    *,
    steps: int = 32,
    blur_sigma: float = 20.0,
    batch_size: int = 16,
) -> np.ndarray:
    """Integrated Gradients of the `target` logit, from a blurred baseline.

    Args:
        model: In eval mode, on the same device as `signals`.
        signals: `(n, leads, samples)`, standardized as the model saw them.
        target: Class index (column of the model's output).
        steps: Riemann steps along the path from the baseline.
        blur_sigma: Gaussian width of the baseline in samples (20 = 0.2 s at
            100 Hz). See the module docstring for why not zeros.
        batch_size: Records per Captum call (memory is `batch_size * steps`).

    Returns:
        float32 `(n, leads, samples)` attributions. Their sum over leads and
        samples approximates `logit(x) - logit(blur(x))` (completeness).
    """
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    out = []
    for start in range(0, len(signals), batch_size):
        x = signals[start : start + batch_size]
        baseline = blurred_baseline(x, blur_sigma)
        attr = ig.attribute(x, baselines=baseline, target=target, n_steps=steps)
        out.append(attr.detach().float().cpu().numpy())
    return np.concatenate(out)


def grad_cam(model: nn.Module, signals: torch.Tensor, target: int, layer: nn.Module) -> np.ndarray:
    """Grad-CAM of the `target` logit at `layer`, upsampled to input length.

    Returns:
        float32 `(n, samples)`, non-negative (ReLU of the weighted activations).
    """
    from captum.attr import LayerAttribution, LayerGradCam

    cam = LayerGradCam(model, layer).attribute(signals, target=target, relu_attributions=True)
    cam = LayerAttribution.interpolate(cam, (signals.shape[-1],), interpolate_mode="linear")
    return cam.squeeze(1).detach().float().cpu().numpy()


def beat_segments(lead: np.ndarray, fs: float) -> dict[str, np.ndarray] | None:
    """Boolean masks over samples for each segment in `SEGMENTS`.

    Args:
        lead: One lead in millivolts, ideally lead II.
        fs: Sampling rate in Hz.

    Returns:
        `{segment: bool mask of len(lead)}`, or None when delineation finds
        no complete beat (flat or very noisy leads).
    """
    import neurokit2 as nk

    with warnings.catch_warnings():
        # neurokit2 trips pandas' chained-assignment and NumPy warnings on
        # every call; they do not affect the delineation.
        warnings.simplefilter("ignore")
        try:
            clean = nk.ecg_clean(lead, sampling_rate=fs)
            _, peaks = nk.ecg_peaks(clean, sampling_rate=fs)
            r_peaks = peaks["ECG_R_Peaks"]
            if len(r_peaks) < 2:
                return None
            _, waves = nk.ecg_delineate(clean, r_peaks, sampling_rate=fs, method="dwt")
        except (ValueError, IndexError, ZeroDivisionError):
            return None

    def points(key: str) -> np.ndarray:
        return np.asarray(waves.get(key, []), dtype=float)

    bounds = {
        "qrs": (points("ECG_R_Onsets"), points("ECG_R_Offsets")),
        "st": (points("ECG_R_Offsets"), points("ECG_T_Onsets")),
        "t": (points("ECG_T_Onsets"), points("ECG_T_Offsets")),
    }
    masks = {name: np.zeros(len(lead), dtype=bool) for name in SEGMENTS}
    max_len = {"qrs": 0.2, "st": 0.4, "t": 0.5}  # seconds; longer means a mis-delineation
    for name, (starts, ends) in bounds.items():
        for start, end in zip(starts, ends, strict=False):
            if np.isnan(start) or np.isnan(end) or not 0 < end - start <= max_len[name] * fs:
                continue
            masks[name][int(start) : int(end) + 1] = True
    # A sample belongs to the first segment that claims it, in SEGMENTS order.
    masks["st"] &= ~masks["qrs"]
    masks["t"] &= ~(masks["qrs"] | masks["st"])
    masks["other"] = ~(masks["qrs"] | masks["st"] | masks["t"])
    if not (masks["qrs"].any() and masks["st"].any()):
        return None
    return masks


def segment_shares(
    attribution: np.ndarray, masks: Mapping[str, np.ndarray]
) -> dict[str, tuple[float, float]]:
    """Per segment: `(share of |attribution|, share of time)`.

    Args:
        attribution: `(leads, samples)` or `(samples,)`.
        masks: Output of `beat_segments`.
    """
    weight = np.abs(attribution)
    if weight.ndim == 2:
        weight = weight.sum(axis=0)
    total = weight.sum()
    n = len(weight)
    return {
        name: (
            float(weight[masks[name]].sum() / total) if total > 0 else float("nan"),
            float(masks[name].sum() / n),
        )
        for name in SEGMENTS
    }


def lead_shares(attribution: np.ndarray) -> np.ndarray:
    """Share of |attribution| carried by each lead of one `(leads, samples)` map."""
    weight = np.abs(attribution).sum(axis=1)
    return weight / weight.sum() if weight.sum() > 0 else np.full(len(weight), np.nan)


def randomization_similarity(maps_a: np.ndarray, maps_b: np.ndarray) -> np.ndarray:
    """Spearman correlation of |attribution| between paired maps, per record.

    Args:
        maps_a: `(n, ...)` attributions from one model (e.g. the trained one).
        maps_b: Attributions for the same records from another model (e.g.
            the same architecture with random weights), same shape.

    Returns:
        `(n,)` rank correlations. Values near 1 mean the maps do not depend on
        the trained weights; a faithful method gives values near 0.
    """
    if maps_a.shape != maps_b.shape:
        raise ValueError(f"Map shapes differ: {maps_a.shape} vs {maps_b.shape}")
    flat_a = np.abs(maps_a.reshape(len(maps_a), -1))
    flat_b = np.abs(maps_b.reshape(len(maps_b), -1))
    return np.array([spearmanr(a, b).statistic for a, b in zip(flat_a, flat_b, strict=True)])


def summarize_enrichment(
    shares: Sequence[Mapping[str, tuple[float, float]]],
) -> dict[str, dict[str, float]]:
    """Mean attribution share, time share and their ratio over many records.

    The ratio is taken of the means (pooled), not the mean of per-record
    ratios, which a record with a tiny delineated segment would dominate.
    """
    summary = {}
    for name in SEGMENTS:
        attr = np.nanmean([s[name][0] for s in shares])
        time = np.nanmean([s[name][1] for s in shares])
        summary[name] = {
            "attribution_share": float(attr),
            "time_share": float(time),
            "enrichment": float(attr / time) if time > 0 else float("nan"),
        }
    return summary

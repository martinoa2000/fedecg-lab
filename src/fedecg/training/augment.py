"""Training-time augmentation of standardized 12-lead signals.

Random cropping (in `fedecg.training.loop`) shifts each record in time. These
transforms add the other variations real recordings have, each drawn per
record from the dataset's own seeded generator:

- **amplitude**: every lead is scaled by `1 + N(0, amplitude)`, as electrode
  placement and skin impedance change a lead's gain from one recording to the
  next;
- **noise**: Gaussian noise with standard deviation `noise` (in standardized
  units), like muscle artifact the band-pass filter left behind;
- **wander**: a slow sinusoid per lead, amplitude up to `wander` and 0.05 to
  0.5 Hz, like breathing-driven baseline drift;
- **lead_dropout**: each lead is zeroed with probability `lead_dropout` (at
  least one lead always survives), like a detached electrode.

All are off by default (0). They apply to training batches only; validation
and test records are never augmented.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

AUGMENTATIONS: tuple[str, ...] = ("amplitude", "noise", "wander", "lead_dropout")


def augment_record(
    signal: torch.Tensor,
    config: Mapping[str, Any],
    generator: torch.Generator,
    *,
    fs: float = 100.0,
) -> torch.Tensor:
    """Return an augmented copy of one `(leads, samples)` record."""
    unknown = set(config) - set(AUGMENTATIONS)
    if unknown:
        raise ValueError(f"Unknown augmentations {sorted(unknown)}; use {AUGMENTATIONS}")
    x = signal.clone()
    leads, samples = x.shape

    if (s := float(config.get("amplitude", 0.0))) > 0:
        x *= 1 + s * torch.randn(leads, 1, generator=generator)
    if (s := float(config.get("noise", 0.0))) > 0:
        x += s * torch.randn(leads, samples, generator=generator)
    if (s := float(config.get("wander", 0.0))) > 0:
        t = torch.arange(samples) / fs
        freq = 0.05 + 0.45 * torch.rand(leads, 1, generator=generator)
        phase = 2 * math.pi * torch.rand(leads, 1, generator=generator)
        amp = s * torch.rand(leads, 1, generator=generator)
        x += amp * torch.sin(2 * math.pi * freq * t + phase)
    if (p := float(config.get("lead_dropout", 0.0))) > 0:
        keep = torch.rand(leads, generator=generator) >= p
        if not keep.any():
            keep[int(torch.randint(leads, (1,), generator=generator))] = True
        x *= keep[:, None].to(x.dtype)
    return x


def is_enabled(config: Mapping[str, Any] | None) -> bool:
    """Whether any augmentation in `config` is switched on."""
    return bool(config) and any(float(config.get(name, 0.0)) > 0 for name in AUGMENTATIONS)

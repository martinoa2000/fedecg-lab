"""Fixed facts about PTB-XL that the rest of the code depends on.

Centralized here so that a label ordering or a fold assignment is defined once.
Silently reordering `SUPERCLASSES` would invalidate every saved checkpoint and
every per-class metric table in `results/`.
"""

from __future__ import annotations

SUPERCLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")
"""The five diagnostic superclasses, in the column order used by every model.

NORM: normal ECG.
MI:   myocardial infarction.
STTC: ST/T-wave changes.
CD:   conduction disturbance.
HYP:  hypertrophy.

A record may carry several of these at once, which is what makes the task
multi-label rather than multi-class.
"""

NUM_CLASSES: int = len(SUPERCLASSES)

LEAD_NAMES: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)  # fmt: skip
"""Standard 12-lead layout, in the channel order PTB-XL stores them."""

NUM_LEADS: int = len(LEAD_NAMES)

SAMPLING_RATE_HZ: int = 100
"""This project uses the 100 Hz version of PTB-XL to stay laptop-friendly."""

SIGNAL_LENGTH: int = 1000
"""Samples per record: every PTB-XL strip is 10 seconds, so 10 s * 100 Hz."""

# The PTB-XL authors assigned every record to one of ten stratified folds and
# recommend this exact split. Folds 9 and 10 were human-validated, which is why
# they are reserved for validation and test rather than used for training.
# Reusing the split is what makes results comparable with published baselines.
TRAIN_FOLDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLD: int = 9
TEST_FOLD: int = 10

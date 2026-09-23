"""Shared fixtures.

`fake_ptbxl` writes a miniature PTB-XL tree with the same file layout, CSV
columns and WFDB record format as the real dataset, so the loading code can be
tested in CI without downloading 1.8 GB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb

from fedecg.data.constants import LEAD_NAMES, SAMPLING_RATE_HZ, SIGNAL_LENGTH

# A subset of the real scp_statements.csv: diagnostic codes with their class,
# plus rhythm/form codes whose `diagnostic` column is empty.
STATEMENTS = pd.DataFrame(
    {
        "description": [
            "normal ECG",
            "inferior myocardial infarction",
            "anteroseptal myocardial infarction",
            "non-diagnostic T abnormalities",
            "complete left bundle branch block",
            "left ventricular hypertrophy",
            "sinus rhythm",
            "ventricular premature complex",
        ],
        "diagnostic": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
        "form": [np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 1.0],
        "rhythm": [np.nan] * 6 + [1.0, np.nan],
        "diagnostic_class": ["NORM", "MI", "MI", "STTC", "CD", "HYP", np.nan, np.nan],
        "diagnostic_subclass": ["NORM", "IMI", "AMI", "STTC", "CLBBB", "LVH", np.nan, np.nan],
    },
    index=pd.Index(["NORM", "IMI", "ASMI", "NDT", "CLBBB", "LVH", "SR", "PVC"]),
)

# (scp_codes, strat_fold). Two records per fold, covering: single labels,
# multi-label records, a record whose two codes map to the same class, and a
# record with only non-diagnostic codes (it must be dropped).
RECORDS: list[tuple[dict[str, float], int]] = [
    ({"NORM": 100.0, "SR": 0.0}, 1),
    ({"IMI": 80.0, "NDT": 50.0}, 1),
    ({"NORM": 100.0}, 2),
    ({"CLBBB": 100.0, "LVH": 50.0}, 2),
    ({"IMI": 100.0, "ASMI": 100.0}, 3),
    ({"SR": 0.0, "PVC": 100.0}, 3),  # no diagnostic code -> unlabeled
    ({"NORM": 80.0}, 4),
    ({"NDT": 100.0}, 4),
    ({"LVH": 100.0}, 5),
    ({"NORM": 100.0}, 5),
    ({"CLBBB": 100.0}, 6),
    ({"IMI": 15.0}, 6),
    ({"NORM": 100.0}, 7),
    ({"NDT": 100.0, "LVH": 100.0}, 7),
    ({"NORM": 100.0}, 8),
    ({"ASMI": 100.0}, 8),
    ({"NORM": 100.0}, 9),
    ({"IMI": 100.0, "CLBBB": 100.0}, 9),
    ({"NORM": 100.0}, 10),
    ({"NDT": 100.0, "IMI": 100.0}, 10),
]

UNLABELED_ECG_ID = 6


def synthetic_signal(ecg_id: int, n_samples: int = SIGNAL_LENGTH) -> np.ndarray:
    """A deterministic `(n_samples, 12)` millivolt signal, distinct per record."""
    time = np.arange(n_samples) / SAMPLING_RATE_HZ
    leads = np.arange(len(LEAD_NAMES))
    # ~72 bpm fundamental, a per-lead amplitude and offset, and a per-record phase.
    return (
        0.5 * (1 + 0.1 * leads) * np.sin(2 * np.pi * 1.2 * time[:, None] + 0.3 * ecg_id)
        + 0.05 * leads
    )


@pytest.fixture(scope="session")
def fake_ptbxl(tmp_path_factory) -> Path:
    """A miniature PTB-XL tree; returns its root directory."""
    root = tmp_path_factory.mktemp("ptbxl")
    STATEMENTS.to_csv(root / "scp_statements.csv")

    rows = []
    for ecg_id, (codes, fold) in enumerate(RECORDS, start=1):
        subdir = root / "records100" / "00000"
        subdir.mkdir(parents=True, exist_ok=True)
        record_name = f"{ecg_id:05d}_lr"
        wfdb.wrsamp(
            record_name,
            fs=SAMPLING_RATE_HZ,
            units=["mV"] * len(LEAD_NAMES),
            sig_name=list(LEAD_NAMES),
            p_signal=synthetic_signal(ecg_id),
            fmt=["16"] * len(LEAD_NAMES),
            adc_gain=[1000.0] * len(LEAD_NAMES),
            baseline=[0] * len(LEAD_NAMES),
            write_dir=str(subdir),
        )
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": 1000.0 + ecg_id,
                "age": 60.0,
                "sex": ecg_id % 2,
                "site": float(ecg_id % 3),
                "device": "CS-12   E" if ecg_id % 2 else "AT-6 C 5.8",
                "report": f"synthetic report {ecg_id}",
                "scp_codes": str(codes),
                "validated_by_human": ecg_id % 3 != 0,
                "strat_fold": fold,
                "filename_lr": f"records100/00000/{record_name}",
                "filename_hr": f"records500/00000/{ecg_id:05d}_hr",
            }
        )
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    return root

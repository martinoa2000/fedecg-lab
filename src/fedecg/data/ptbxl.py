"""Read PTB-XL metadata, build superclass labels, and load waveforms.

The PTB-XL tree produced by `scripts/download_data.py` looks like::

    data/ptbxl/
        ptbxl_database.csv     one row per record: patient, fold, SCP codes, ...
        scp_statements.csv     the SCP-ECG code table, incl. diagnostic class
        records100/00000/00001_lr.{dat,hea}
        ...

Label construction follows the PTB-XL authors' reference code and the
Strodthoff et al. (2021) benchmark: every SCP code attached to a record that is
a *diagnostic* statement is mapped to its `diagnostic_class`, regardless of the
likelihood the cardiologist gave it. Records that end up with no superclass at
all (they only carry form or rhythm statements) are dropped for this task,
exactly as the benchmark does, so that published numbers stay comparable.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from fedecg.data.constants import (
    NUM_LEADS,
    SAMPLING_RATE_HZ,
    SIGNAL_LENGTH,
    SUPERCLASSES,
    TEST_FOLD,
    TRAIN_FOLDS,
    VAL_FOLD,
)
from fedecg.paths import PTBXL_DIR

DATABASE_CSV = "ptbxl_database.csv"
STATEMENTS_CSV = "scp_statements.csv"

_FILENAME_COLUMN = {100: "filename_lr", 500: "filename_hr"}


def load_metadata(root: str | Path = PTBXL_DIR) -> pd.DataFrame:
    """Load `ptbxl_database.csv`, indexed by `ecg_id`.

    The `scp_codes` column is stored in the CSV as the string form of a Python
    dict (`"{'NORM': 100.0, 'SR': 0.0}"`); it is parsed back into a dict here.
    """
    path = Path(root) / DATABASE_CSV
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download PTB-XL first: python scripts/download_data.py"
        )
    meta = pd.read_csv(path, index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].map(ast.literal_eval)
    return meta


def load_diagnostic_map(root: str | Path = PTBXL_DIR) -> dict[str, str]:
    """Map every diagnostic SCP code to its superclass (e.g. `IMI` -> `MI`).

    Non-diagnostic statements (rhythm and form codes such as `SR` or `PVC`) are
    left out, which is what makes them invisible to the superclass labels.
    """
    path = Path(root) / STATEMENTS_CSV
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Download PTB-XL first.")
    statements = pd.read_csv(path, index_col=0)
    diagnostic = statements[statements["diagnostic"] == 1]
    return diagnostic["diagnostic_class"].dropna().astype(str).to_dict()


def superclasses_of(scp_codes: Mapping[str, float], diagnostic_map: Mapping[str, str]) -> set[str]:
    """Return the set of superclasses implied by one record's SCP codes."""
    return {diagnostic_map[code] for code in scp_codes if code in diagnostic_map}


def label_matrix(
    scp_codes: Iterable[Mapping[str, float]], diagnostic_map: Mapping[str, str]
) -> np.ndarray:
    """Multi-hot label matrix of shape `(n_records, NUM_CLASSES)`.

    Columns follow `SUPERCLASSES`. Superclasses outside that tuple (there are
    none in PTB-XL 1.0.3, but a future release could add one) raise rather than
    being silently dropped.
    """
    column = {name: i for i, name in enumerate(SUPERCLASSES)}
    rows = []
    for codes in scp_codes:
        row = np.zeros(len(SUPERCLASSES), dtype=np.float32)
        for superclass in superclasses_of(codes, diagnostic_map):
            if superclass not in column:
                raise ValueError(f"Unknown superclass {superclass!r}; expected {SUPERCLASSES}")
            row[column[superclass]] = 1.0
        rows.append(row)
    if not rows:
        return np.zeros((0, len(SUPERCLASSES)), dtype=np.float32)
    return np.stack(rows)


def attach_labels(
    meta: pd.DataFrame, diagnostic_map: Mapping[str, str], *, drop_unlabeled: bool = True
) -> pd.DataFrame:
    """Add one 0/1 column per superclass to `meta`.

    Args:
        meta: Output of `load_metadata`.
        diagnostic_map: Output of `load_diagnostic_map`.
        drop_unlabeled: Drop records with no diagnostic superclass, as the
            PTB-XL benchmark does for the superdiagnostic task.

    Returns:
        A copy of `meta` with integer columns `NORM`, `MI`, `STTC`, `CD`, `HYP`.
    """
    labels = label_matrix(meta["scp_codes"], diagnostic_map).astype(np.int8)
    labeled = meta.copy()
    for i, name in enumerate(SUPERCLASSES):
        labeled[name] = labels[:, i]
    if drop_unlabeled:
        labeled = labeled[labels.sum(axis=1) > 0]
    return labeled


@dataclass(frozen=True)
class Split:
    """Record IDs of the train / validation / test partitions."""

    train: pd.Index
    val: pd.Index
    test: pd.Index


def split_by_folds(
    meta: pd.DataFrame,
    *,
    train_folds: Sequence[int] = TRAIN_FOLDS,
    val_fold: int = VAL_FOLD,
    test_fold: int = TEST_FOLD,
) -> Split:
    """Partition records using the official `strat_fold` assignment.

    PTB-XL's folds are stratified *by patient*: all recordings of one patient
    share a fold. Splitting by fold therefore keeps patients from leaking
    between train and test, which a random row-level split would not.
    """
    folds = {*train_folds, val_fold, test_fold}
    if len(folds) != len(train_folds) + 2:
        raise ValueError("train_folds, val_fold and test_fold must be disjoint")
    fold = meta["strat_fold"]
    return Split(
        train=meta.index[fold.isin(train_folds)],
        val=meta.index[fold == val_fold],
        test=meta.index[fold == test_fold],
    )


def load_signals(
    meta: pd.DataFrame,
    root: str | Path = PTBXL_DIR,
    *,
    sampling_rate: int = SAMPLING_RATE_HZ,
    progress: bool = False,
) -> np.ndarray:
    """Read the waveforms for every row of `meta`.

    Returns:
        A float32 array of shape `(n_records, NUM_LEADS, n_samples)` in
        millivolts. Leads come first because that is the `(channels, time)`
        layout `torch.nn.Conv1d` expects.
    """
    if sampling_rate not in _FILENAME_COLUMN:
        raise ValueError(f"sampling_rate must be one of {sorted(_FILENAME_COLUMN)}")
    root = Path(root)
    filenames = meta[_FILENAME_COLUMN[sampling_rate]]
    n_samples = SIGNAL_LENGTH * sampling_rate // SAMPLING_RATE_HZ
    signals = np.empty((len(meta), NUM_LEADS, n_samples), dtype=np.float32)

    iterator: Iterable[tuple[int, str]] = enumerate(filenames)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, total=len(filenames), desc="Reading records", unit="rec")

    for i, filename in iterator:
        signal, fields = wfdb.rdsamp(str(root / filename))
        if signal.shape != (n_samples, NUM_LEADS) or fields["fs"] != sampling_rate:
            raise ValueError(
                f"{filename}: expected {n_samples} samples x {NUM_LEADS} leads at "
                f"{sampling_rate} Hz, got {signal.shape} at {fields['fs']} Hz"
            )
        signals[i] = signal.T
    # A handful of PTB-XL records contain flat-lined or missing leads that wfdb
    # reads as NaN. Zero is the neutral value after per-lead standardization.
    np.nan_to_num(signals, copy=False)
    return signals


@dataclass(frozen=True)
class PTBXLArrays:
    """Waveforms, labels and IDs for the whole labeled dataset, in memory."""

    signals: np.ndarray
    """float32, `(n_records, NUM_LEADS, n_samples)`, millivolts."""
    labels: np.ndarray
    """float32, `(n_records, NUM_CLASSES)`, multi-hot in `SUPERCLASSES` order."""
    ecg_ids: np.ndarray
    """int64, `(n_records,)`, the PTB-XL `ecg_id` of each row."""

    def select(self, ids: Iterable[int]) -> PTBXLArrays:
        """Return the subset for the given `ecg_id`s, in the order given."""
        position = pd.Index(self.ecg_ids).get_indexer(list(ids))
        if (position < 0).any():
            raise KeyError("Some requested ecg_ids are not in this dataset")
        return PTBXLArrays(self.signals[position], self.labels[position], self.ecg_ids[position])

    def __len__(self) -> int:
        return len(self.ecg_ids)


def load_dataset(
    root: str | Path = PTBXL_DIR,
    *,
    sampling_rate: int = SAMPLING_RATE_HZ,
    cache_path: str | Path | None = None,
    progress: bool = False,
) -> tuple[pd.DataFrame, PTBXLArrays]:
    """Load labeled metadata and all waveforms, optionally through a cache.

    Parsing ~21k WFDB records takes a minute or two; the `.npz` cache brings
    that down to a couple of seconds. The cache stores raw (unfiltered)
    signals, so changing the preprocessing config never requires rebuilding it.

    Returns:
        `(meta, arrays)`: the labeled metadata (see `attach_labels`) and the
        arrays, with rows in the same order as `meta`.
    """
    meta = attach_labels(load_metadata(root), load_diagnostic_map(root))
    labels = meta[list(SUPERCLASSES)].to_numpy(dtype=np.float32)
    ecg_ids = meta.index.to_numpy(dtype=np.int64)

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.is_file():
            with np.load(cache_path) as cached:
                if np.array_equal(cached["ecg_ids"], ecg_ids):
                    return meta, PTBXLArrays(cached["signals"], labels, ecg_ids)

    signals = load_signals(meta, root, sampling_rate=sampling_rate, progress=progress)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, so an interrupted run never leaves a truncated
        # cache behind that the next run would trust.
        partial = cache_path.with_name(cache_path.name + ".partial")
        with partial.open("wb") as handle:
            np.savez(handle, signals=signals, ecg_ids=ecg_ids)
        partial.replace(cache_path)
    return meta, PTBXLArrays(signals, labels, ecg_ids)

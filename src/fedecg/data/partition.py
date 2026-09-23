"""Split the training set across simulated hospitals.

How the data is divided *is* the experiment in the federated phases, so every
partitioner takes an explicit NumPy generator and returns positions into the
training arrays (not `ecg_id`s), sorted within each client.

Three schemes, from least to most heterogeneous in label mix:

**IID** (`iid`). Records are shuffled and dealt out in near-equal shares.
Every hospital sees the same label mix; the only cost measured is that of
decentralization itself.

**Metadata groups** (`site`, `device`). Each value of a metadata column becomes
one hospital. This is the realistic kind of heterogeneity: PTB-XL's recording
sites and devices differ in patient mix (the `CS-12   E` device records 79%
normal ECGs, `CS100    3` only 30%) and in the acquisition hardware itself.
Values with fewer than `min_records` training records are pooled into one
`other` client, since a hospital with 20 ECGs cannot train anything.

**Dirichlet label skew** (`dirichlet`). The standard synthetic benchmark
(Hsu et al., 2019): for each class, the class's records are divided across
hospitals in proportions drawn from `Dirichlet(alpha)`. Small `alpha` gives
each hospital a few dominant classes; large `alpha` approaches IID. PTB-XL is
multi-label, so each record is assigned by its *rarest* positive superclass,
which keeps the rare classes (`HYP`, `CD`) from being swamped by `NORM`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fedecg.data.constants import SUPERCLASSES

PARTITIONS: tuple[str, ...] = ("iid", "site", "device", "dirichlet")


@dataclass(frozen=True)
class Partition:
    """Which training records each simulated hospital holds."""

    scheme: str
    clients: list[np.ndarray]
    """One sorted array of positions into the training arrays per client."""
    names: list[str]
    """A readable name per client, e.g. `site 0` or `hospital 3`."""

    def __len__(self) -> int:
        return len(self.clients)

    def sizes(self) -> np.ndarray:
        """Number of records held by each client."""
        return np.array([len(c) for c in self.clients])


def iid_partition(n_records: int, n_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle positions `0..n_records-1` and deal them into `n_clients` shares."""
    if not 1 <= n_clients <= n_records:
        raise ValueError(f"Need 1 <= n_clients <= n_records, got {n_clients}, {n_records}")
    shares = np.array_split(rng.permutation(n_records), n_clients)
    return [np.sort(s) for s in shares]


def group_partition(
    values: pd.Series | np.ndarray, *, min_records: int = 1, other: str = "other"
) -> tuple[list[np.ndarray], list[str]]:
    """One client per distinct value; values rarer than `min_records` pooled.

    Missing values count as their own group (`missing`). Clients are ordered
    from largest to smallest, with the pooled client last.

    Returns:
        `(clients, names)`.
    """
    labels = pd.Series(np.asarray(values, dtype=object)).map(
        lambda v: "missing" if pd.isna(v) else str(v).strip()
    )
    counts = labels.value_counts()
    kept = [v for v, n in counts.items() if n >= min_records]
    pooled = labels.isin(counts.index[counts < min_records])

    clients = [np.flatnonzero(labels.to_numpy() == v) for v in kept]
    names = list(kept)
    if pooled.any():
        clients.append(np.flatnonzero(pooled.to_numpy()))
        names.append(other)
    if len(clients) < 2:
        raise ValueError("Grouping produced fewer than two clients; lower min_records")
    return clients, names


def primary_label(labels: np.ndarray) -> np.ndarray:
    """Index of each record's rarest positive class (by frequency in `labels`).

    Records without any positive class (none exist after PTB-XL's labeling,
    but a caller may pass others) get the most frequent class.
    """
    frequency = labels.sum(axis=0)
    rarity_order = np.argsort(frequency, kind="stable")  # rarest first
    ranked = labels[:, rarity_order] > 0
    first = ranked.argmax(axis=1)
    first[~ranked.any(axis=1)] = len(rarity_order) - 1
    return rarity_order[first]


def dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
    *,
    min_records: int = 1,
    max_attempts: int = 1000,
) -> list[np.ndarray]:
    """Label-skewed split: per class, shares drawn from `Dirichlet(alpha)`.

    Draws are repeated until every client holds at least `min_records`
    records, the usual guard against empty hospitals at small `alpha`.

    Raises:
        RuntimeError: If no valid draw is found within `max_attempts`.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if n_clients * min_records > len(labels):
        raise ValueError("Not enough records for n_clients * min_records")
    primary = primary_label(labels)
    for _ in range(max_attempts):
        buckets: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
        for k in np.unique(primary):
            members = rng.permutation(np.flatnonzero(primary == k))
            shares = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(shares)[:-1] * len(members)).astype(int)
            for client, part in enumerate(np.split(members, cuts)):
                buckets[client].append(part)
        clients = [np.sort(np.concatenate(b)) for b in buckets]
        if min(len(c) for c in clients) >= min_records:
            return clients
    raise RuntimeError(
        f"No Dirichlet(alpha={alpha}) draw gave every client >= {min_records} records "
        f"in {max_attempts} attempts; raise alpha or lower min_client_records"
    )


def partition_from_config(
    federated_config: Mapping[str, Any],
    meta_train: pd.DataFrame,
    y_train: np.ndarray,
    rng: np.random.Generator,
) -> Partition:
    """Build the partition described by a config's `federated` section.

    Args:
        federated_config: Uses `partition`, and depending on the scheme
            `n_clients`, `dirichlet_alpha` and `min_client_records`.
        meta_train: Metadata of the training records, aligned with `y_train`.
        y_train: Multi-hot training labels.
        rng: Generator that fully determines random partitions.
    """
    scheme = federated_config.get("partition", "iid")
    min_records = int(federated_config.get("min_client_records", 1))
    if len(meta_train) != len(y_train):
        raise ValueError("meta_train and y_train must describe the same records")

    if scheme == "iid":
        n = int(federated_config["n_clients"])
        clients = iid_partition(len(y_train), n, rng)
        names = [f"hospital {i + 1}" for i in range(n)]
    elif scheme in ("site", "device"):
        values = meta_train[scheme]
        # `site` is stored as floats (0.0, 1.0, ...) because it has missing
        # values; a nullable integer column names hospitals `site 0`, not `site 0.0`.
        if pd.api.types.is_float_dtype(values) and (values.dropna() % 1 == 0).all():
            values = values.astype("Int64")
        clients, raw_names = group_partition(values, min_records=min_records)
        names = [n if n == "other" else f"{scheme} {n}" for n in raw_names]
    elif scheme == "dirichlet":
        n = int(federated_config["n_clients"])
        alpha = float(federated_config["dirichlet_alpha"])
        clients = dirichlet_partition(y_train, n, alpha, rng, min_records=min_records)
        names = [f"hospital {i + 1}" for i in range(n)]
    else:
        raise ValueError(f"Unknown partition {scheme!r}; use one of {PARTITIONS}")
    return Partition(scheme=scheme, clients=clients, names=names)


def partition_summary(partition: Partition, y_train: np.ndarray) -> pd.DataFrame:
    """Records and label prevalence (% of the client's records) per client.

    The same shape as `records_by` in `fedecg.data.stats`, one row per client,
    so hospitals can be compared side by side with the dataset-level mix.
    """
    rows = []
    for name, idx in zip(partition.names, partition.clients, strict=True):
        prevalence = y_train[idx].mean(axis=0) * 100
        rows.append(
            {"client": name, "n_records": len(idx)}
            | {c: round(float(p), 2) for c, p in zip(SUPERCLASSES, prevalence, strict=True)}
        )
    return pd.DataFrame(rows).set_index("client")

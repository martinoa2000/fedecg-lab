# fedecg-lab

[![CI](https://github.com/martinoa2000/fedecg-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/martinoa2000/fedecg-lab/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Multi-label classification of 12-lead ECGs, used as a testbed for the question
that decides whether a clinical ML model can actually be built: **what does it
cost to train on data that never leaves the hospital?**

The experiments compare a centralized baseline against federated training
across simulated hospitals, then add realistic non-IID data, differential
privacy, and saliency-based explanations — measuring the accuracy given up at
each step.

> **Not a medical device.** Research and educational code only. See
> [Limitations](#limitations).

## The question

A hospital cannot usually ship patient ECGs to a central server. Federated
learning promises a way around that: train locally, share only model updates.
But that promise comes with costs that are rarely quantified side by side:

1. **Decentralization cost.** How much AUROC is lost by replacing one training
   set with N hospitals running FedAvg?
2. **Heterogeneity cost.** Real hospitals do not hold random samples. They have
   different recording devices and different patient mixes. How far does that
   push performance down, and does FedProx recover any of it?
3. **Privacy cost.** Model updates still leak. Adding DP-SGD buys a formal
   guarantee — at what epsilon does the model stop being useful?
4. **Trust.** When the model does fire on a myocardial infarction, is it looking
   at the ST segment, or at noise?

## Dataset

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (PhysioNet): 21,799
clinical 12-lead ECGs from 18,869 patients, 10 seconds each. This project uses
the 100 Hz version so everything runs on a laptop.

- **Task:** multi-label classification over the 5 diagnostic superclasses —
  `NORM`, `MI`, `STTC`, `CD`, `HYP`. Records can carry several at once.
- **Split:** the authors' recommended `strat_fold` split — folds 1–8 train,
  fold 9 validation, fold 10 test. Folds 9 and 10 were human-validated. Using
  this split is what makes the numbers here comparable to published baselines.
- **Primary metric:** macro AUROC, with per-class F1 reported alongside.

The data is **never committed**. `scripts/download_data.py` fetches it from
PhysioNet and verifies every file against the official `SHA256SUMS.txt`.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/martinoa2000/fedecg-lab.git
cd fedecg-lab
uv sync
```

Download PTB-XL (~1.8 GB download, ~1 GB on disk after extraction):

```bash
uv run python scripts/download_data.py
```

Produce the exploration tables and figures (label balance, co-occurrence,
per-site/device breakdown, example signals) into `results/`:

```bash
uv run python scripts/explore_data.py
```

The first training run caches all waveforms in `data/cache/` as a single
`.npz`, so later runs skip WFDB parsing.

Run the test suite:

```bash
uv run pytest
```

Enable the git hooks (lint, formatting, notebook output stripping):

```bash
uv run pre-commit install
```

## Roadmap

- [x] **1. Setup** — project structure, pinned dependencies, pre-commit, CI, reproducible data download.
- [ ] **2. Exploration and preprocessing** — label distribution, per-class ECG visualization, filtering and normalization.
  *Code, tests and notebook `01_exploration` done (`fedecg.data.ptbxl`, `preprocess`, `stats`); committed tables pending a run on the full dataset.*
- [ ] **3. Centralized baseline** — 1D ResNet, early stopping, MLflow tracking, comparison against published PTB-XL results.
- [ ] **4. Federated (IID)** — Flower simulation of N hospitals with random splits, FedAvg vs. centralized.
- [ ] **5. Federated (non-IID)** — partitions by recording `site`/`device` and Dirichlet label skew; FedAvg vs. FedProx.
- [ ] **6. Differential privacy** — DP-SGD with Opacus, local and federated; the performance/epsilon trade-off curve.
- [ ] **7. Explainability** — Integrated Gradients / Grad-CAM 1D saliency overlaid on the raw signal.
- [ ] **8. Write-up** — architecture diagram, comparative results table, reproduction instructions.

## Repository layout

```
fedecg-lab/
├── configs/          # Experiment configs (YAML, with `extends` inheritance)
├── notebooks/        # 01..07, numbered; concepts explained before code
├── scripts/          # Data download and experiment entry points
├── src/fedecg/       # The reusable package
│   ├── data/         # PTB-XL loading, labels, preprocessing, partitioning
│   ├── models/       # 1D ResNet
│   ├── training/     # Training loop, metrics, MLflow
│   ├── federated/    # Flower clients and strategies
│   ├── privacy/      # Opacus / DP-SGD
│   └── explain/      # Saliency over raw waveforms
├── tests/            # Fast unit tests (no dataset, no training)
└── results/          # Committed tables and figures
```

Notebooks are for learning and exploration; once a piece of code works it moves
into `src/fedecg/`, gets a test, and is imported back into the notebook.

## Design notes

**GroupNorm, not BatchNorm.** Opacus cannot privatize BatchNorm, because its
batch statistics mix information across samples and break the per-sample
gradient accounting DP-SGD relies on. Rather than train one architecture for
the baseline and a different one under DP, the model uses GroupNorm everywhere
so all phases stay directly comparable.

**Reproducibility.** Fixed seeds, versioned configs, and an exact `uv.lock`.
Data partitioning uses explicitly-passed NumPy generators rather than global
random state, since the split across hospitals *is* the experiment.

**Zero-phase filtering, train-only statistics.** The 0.5–40 Hz band-pass runs
forwards and backwards so ST segments do not shift relative to the QRS, and the
per-lead standardizer is fitted on the training split only. It is serializable,
so the federated phases can compare global against per-hospital statistics.

**Tests without the dataset.** `tests/conftest.py` writes a miniature PTB-XL
tree (same CSV columns, same WFDB record format) so the loading, labeling and
splitting code is exercised in CI without the 1.8 GB download.

**Laptop-sized.** No GPU required. Every config exposes a `subsample` option for
fast smoke runs, and CI never trains anything.

## Limitations

- **Simulated federation.** Clients run in one process via Flower's simulation
  backend. This reproduces the *statistical* consequences of decentralization,
  not real network conditions, stragglers, or systems failures.
- **Single-country data.** PTB-XL was collected in Germany between 1989 and
  1996 by a single provider. Partitioning it into synthetic "hospitals" cannot
  reproduce genuine cross-institution distribution shift.
- **Record-level, not hospital-level privacy.** DP-SGD inside each client
  protects individual records within that client. It does not hide a hospital's
  participation, which needs noise at aggregation time instead.
- **Not clinically validated.** Superclass labels are a coarse simplification of
  the full diagnostic hierarchy. Nothing here is fit for clinical use.

## References

- Wagner et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data 7, 154.
- Strodthoff et al. (2021). *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL.* IEEE JBHI 25(5).
- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Li et al. (2020). *Federated Optimization in Heterogeneous Networks (FedProx).* MLSys.
- Abadi et al. (2016). *Deep Learning with Differential Privacy.* CCS.

## License

MIT for the code in this repository. PTB-XL is distributed by PhysioNet under
its own license and is not redistributed here.

# Changelog

All notable changes to this project. Numbers are test macro AUROC on PTB-XL's
fold 10 unless stated otherwise.

## 1.0.0 — 2026-09-24

The first public release: all eight roadmap phases.

- **Data** (phases 1-2): reproducible PTB-XL download with checksum
  verification, superclass labels, the official fold split, zero-phase
  band-pass filtering and per-lead standardization fitted on training data only.
- **Centralized baseline** (phase 3): a 1D ResNet with GroupNorm, trained on
  random 2.5 s crops with augmentation and a cosine schedule, selected on
  validation: 0.917 (0.922 averaged over three seeds; published
  `resnet1d_wang`: 0.930).
- **Baseline tuning**: `scripts/tune.py` compares training options on the
  validation fold only (network width, augmentation, class-weighted and focal
  losses, seed ensembles); the winning setting became the baseline.
- **Federated learning** (phases 4-5): Flower FedAvg and FedProx across 4 to 20
  simulated hospitals, split at random, by recording site, by device, or by
  Dirichlet label skew. Decentralization costs 0.007 to 0.026; realistic
  heterogeneity adds almost nothing on top.
- **Differential privacy** (phase 6): DP-SGD with Opacus, centrally and inside
  each hospital, at ε = 1, 3 and 8 against no-privacy controls. Privacy costs
  0.024 to 0.066 centrally and up to 0.112 per hospital.
- **Explainability** (phase 7): Integrated Gradients and Grad-CAM, beat-segment
  enrichment and the model-randomization test. Infarction is recognized from
  the QRS complex, not the ST segment.
- **Write-up** (phase 8): notebooks 01-06, a README with every result,
  `scripts/reproduce.sh` for the whole study.
- **Dashboard**: one page per experiment, and a Tuning and training page that
  launches and follows training runs through a local training server
  (`scripts/serve.py`). `scripts/publish_dashboard.sh` publishes a static copy to
  GitHub Pages.

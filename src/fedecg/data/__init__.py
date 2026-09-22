"""PTB-XL loading, labels, preprocessing and federated partitioning.

Planned contents:
    constants: Diagnostic superclasses and the official fold split.
    ptbxl:     Read `ptbxl_database.csv`, map SCP codes to superclasses,
               load waveforms with `wfdb`.
    preprocess: Baseline-wander removal, filtering, per-lead normalization.
    partition: Split the training set across simulated hospitals (IID,
               metadata-based, and Dirichlet label skew).
"""

"""PTB-XL loading, labels, preprocessing and federated partitioning.

Contents:
    constants:  Diagnostic superclasses, lead names and the official fold split.
    ptbxl:      Read `ptbxl_database.csv`, map SCP codes to superclasses,
                load waveforms with `wfdb`, cache them as one `.npz`.
    preprocess: Zero-phase band-pass filtering and per-lead standardization.
    stats:      Label distribution, co-occurrence and cardinality tables.
    pipeline:   Config -> preprocessed train / validation / test arrays, shared
                by every experiment script.
    partition:  Split the training set across simulated hospitals (IID,
                by recording site or device, and Dirichlet label skew).
"""

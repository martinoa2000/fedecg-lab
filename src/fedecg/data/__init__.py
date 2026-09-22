"""PTB-XL loading, labels, preprocessing and federated partitioning.

Contents:
    constants:  Diagnostic superclasses, lead names and the official fold split.
    ptbxl:      Read `ptbxl_database.csv`, map SCP codes to superclasses,
                load waveforms with `wfdb`, cache them as one `.npz`.
    preprocess: Zero-phase band-pass filtering and per-lead standardization.
    stats:      Label distribution, co-occurrence and cardinality tables.

Planned:
    partition:  Split the training set across simulated hospitals (IID,
                metadata-based, and Dirichlet label skew).
"""

"""Training loops, evaluation metrics and experiment tracking.

Planned contents:
    loop:    Epoch-level train/validate loop with early stopping.
    metrics: Macro AUROC (the headline metric) and per-class F1.
    tracking: MLflow run helpers.

The metric choice follows the PTB-XL benchmark paper so that results here can
be compared against published numbers.
"""

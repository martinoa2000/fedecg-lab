"""Training loops, evaluation metrics and experiment tracking.

Contents:
    loop:     Epoch-level train/validate loop with early stopping.
    metrics:  Macro AUROC (the headline metric), per-class F1 and thresholds.
    tracking: MLflow run helpers that become no-ops when tracking is off.
    evaluate: The final validation-thresholds-then-test evaluation.
    results:  The shared `results/tables/experiments.csv` summary table.

The metric choice follows the PTB-XL benchmark paper so that results here can
be compared against published numbers.
"""

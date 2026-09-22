"""fedecg: federated deep learning on 12-lead ECG.

Reusable logic for the experiments in this repository. Notebooks under
`notebooks/` are for exploration and explanation; once a piece of code works
there it is moved here, covered by a test, and imported back into the notebook.

Subpackages:
    data:      PTB-XL loading, label construction, preprocessing, partitioning.
    models:    PyTorch architectures (1D ResNet).
    training:  Training loops, early stopping, metrics, MLflow logging.
    federated: Flower clients, server strategies, simulation entry points.
    privacy:   Opacus / DP-SGD integration.
    explain:   Saliency methods over raw ECG signals.
"""

__version__ = "0.1.0"

"""PyTorch architectures for 12-lead ECG classification.

Planned contents:
    resnet1d: A small 1D residual network sized for CPU training.

Design constraint: normalization layers must be compatible with Opacus, which
cannot privatize `BatchNorm` because its statistics mix information across
samples in a batch, breaking the per-sample gradient accounting that DP-SGD
depends on. The models here use `GroupNorm` throughout so that the exact same
architecture is used for the centralized, federated and DP experiments.
"""

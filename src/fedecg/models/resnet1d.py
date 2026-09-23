"""A small 1D residual network for 12-lead ECG classification.

The layout follows the `resnet1d_wang` family that performs well on PTB-XL in
Strodthoff et al. (2021), scaled down so that a full training run fits on a
laptop CPU:

    stem:    Conv(k=7) -> GroupNorm -> ReLU -> MaxPool(2)
    stage i: `blocks_per_stage[i]` residual blocks with `base_channels * 2**i`
             channels; every stage after the first halves the time axis
    head:    concatenated global average + max pooling -> dropout -> linear

Concatenating average and max pooling lets the head see both the typical beat
(average) and a single abnormal one (max), which matters when an arrhythmia or
an ectopic beat appears only once in ten seconds.

The head returns **logits**, one per superclass. The labels are multi-hot, so
each logit goes through its own sigmoid (inside `BCEWithLogitsLoss` during
training); there is deliberately no softmax.

Every normalization layer is `GroupNorm`: see `fedecg.models` for why.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from fedecg.data.constants import NUM_CLASSES, NUM_LEADS


def _norm(channels: int, groups: int) -> nn.GroupNorm:
    if channels % groups:
        raise ValueError(f"norm_groups={groups} must divide the channel count {channels}")
    return nn.GroupNorm(groups, channels)


class ResidualBlock1d(nn.Module):
    """Two convolutions with a skip connection.

    When the block changes the channel count or downsamples, the skip path gets
    a 1x1 convolution so that the two paths can be added.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        kernel_size: int = 5,
        norm_groups: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.norm1 = _norm(out_channels, norm_groups)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.norm2 = _norm(out_channels, norm_groups)
        self.relu = nn.ReLU()

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                _norm(out_channels, norm_groups),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1d(nn.Module):
    """1D ResNet mapping `(batch, leads, samples)` to `(batch, num_classes)` logits."""

    def __init__(
        self,
        *,
        in_channels: int = NUM_LEADS,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
        blocks_per_stage: Sequence[int] = (2, 2, 2),
        norm_groups: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        if not blocks_per_stage or min(blocks_per_stage) < 1:
            raise ValueError("blocks_per_stage needs at least one stage with >= 1 block")

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 7, padding=3, bias=False),
            _norm(base_channels, norm_groups),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        blocks: list[nn.Module] = []
        channels = base_channels
        for stage, n_blocks in enumerate(blocks_per_stage):
            out_channels = base_channels * 2**stage
            for i in range(n_blocks):
                stride = 2 if stage > 0 and i == 0 else 1
                blocks.append(
                    ResidualBlock1d(channels, out_channels, stride=stride, norm_groups=norm_groups)
                )
                channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        features = self.blocks(self.stem(x))
        pooled = torch.cat([features.mean(dim=-1), features.amax(dim=-1)], dim=1)
        return self.head(pooled)


def build_model(model_config: Mapping[str, Any]) -> nn.Module:
    """Instantiate the model described by a config's `model` section."""
    name = model_config.get("name", "resnet1d")
    if name != "resnet1d":
        raise ValueError(f"Unknown model {name!r}; only 'resnet1d' is implemented")
    return ResNet1d(
        base_channels=model_config.get("base_channels", 32),
        blocks_per_stage=tuple(model_config.get("blocks_per_stage", (2, 2, 2))),
        norm_groups=model_config.get("norm_groups", 8),
        dropout=model_config.get("dropout", 0.2),
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

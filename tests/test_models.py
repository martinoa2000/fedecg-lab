"""Tests for the 1D ResNet."""

from __future__ import annotations

import pytest
import torch
from opacus.validators import ModuleValidator
from torch import nn

from fedecg.config import load_config
from fedecg.data.constants import NUM_CLASSES, NUM_LEADS, SIGNAL_LENGTH
from fedecg.models.resnet1d import ResNet1d, build_model, count_parameters


def small_model(**kwargs) -> ResNet1d:
    return ResNet1d(base_channels=8, blocks_per_stage=(1, 1), norm_groups=4, **kwargs)


class TestResNet1d:
    def test_maps_a_batch_of_ecgs_to_one_logit_per_class(self):
        out = small_model()(torch.randn(3, NUM_LEADS, SIGNAL_LENGTH))
        assert out.shape == (3, NUM_CLASSES)

    def test_accepts_other_signal_lengths(self):
        # Global pooling makes the head independent of the input length.
        assert small_model()(torch.randn(2, NUM_LEADS, 500)).shape == (2, NUM_CLASSES)

    def test_uses_no_batchnorm(self):
        model = build_model(load_config("default.yaml")["model"])
        assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())

    def test_is_accepted_by_opacus(self):
        model = build_model(load_config("default.yaml")["model"])
        assert ModuleValidator.validate(model, strict=False) == []

    def test_records_in_a_batch_do_not_influence_each_other(self):
        # With BatchNorm this would fail in train mode; GroupNorm is per-record.
        model = small_model(dropout=0.0).train()
        x = torch.randn(4, NUM_LEADS, SIGNAL_LENGTH)
        alone = model(x[:1])
        together = model(torch.cat([x[:1], 10 * x[1:]]))[:1]
        torch.testing.assert_close(alone, together)

    def test_rejects_groups_that_do_not_divide_channels(self):
        with pytest.raises(ValueError, match="norm_groups"):
            ResNet1d(base_channels=10, norm_groups=4)

    def test_default_config_stays_laptop_sized(self):
        n = count_parameters(build_model(load_config("default.yaml")["model"]))
        assert 100_000 < n < 2_000_000

    def test_unknown_model_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model({"name": "transformer"})

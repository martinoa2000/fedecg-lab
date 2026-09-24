"""Tests for training augmentation, the loss options and ensembles."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fedecg.training.augment import augment_record, is_enabled
from fedecg.training.loop import FocalLoss, make_loader, make_loss

X = torch.ones(12, 200)


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestAugment:
    def test_nothing_enabled_returns_an_equal_copy(self):
        out = augment_record(X, {}, gen())
        torch.testing.assert_close(out, X)
        assert out is not X

    def test_amplitude_scales_each_lead_by_one_factor(self):
        out = augment_record(X, {"amplitude": 0.2}, gen())
        per_lead = out / X
        torch.testing.assert_close(per_lead, per_lead[:, :1].expand_as(per_lead))
        assert per_lead[:, 0].std() > 0

    def test_noise_and_wander_keep_the_shape_and_change_values(self):
        for name in ("noise", "wander"):
            out = augment_record(X, {name: 0.3}, gen())
            assert out.shape == X.shape and not torch.equal(out, X)

    def test_lead_dropout_zeroes_whole_leads_but_never_all(self):
        out = augment_record(X, {"lead_dropout": 0.5}, gen(3))
        zeroed = (out == 0).all(dim=1)
        assert zeroed.any()
        assert (out[~zeroed] == 1).all()
        everything = augment_record(X, {"lead_dropout": 0.9999}, gen())
        assert (everything != 0).any(dim=1).sum() == 1

    def test_depends_only_on_the_generator(self):
        config = {"amplitude": 0.1, "noise": 0.1, "wander": 0.1, "lead_dropout": 0.2}
        torch.testing.assert_close(
            augment_record(X, config, gen(5)), augment_record(X, config, gen(5))
        )

    def test_rejects_unknown_augmentations(self):
        with pytest.raises(ValueError, match="Unknown"):
            augment_record(X, {"mixup": 0.2}, gen())

    def test_loader_augments_training_batches(self):
        signals = np.ones((8, 12, 100), dtype=np.float32)
        labels = np.zeros((8, 5), dtype=np.float32)
        plain = next(iter(make_loader(signals, labels, batch_size=8, shuffle=False)))[0]
        noisy = next(
            iter(make_loader(signals, labels, batch_size=8, shuffle=False, augment={"noise": 0.1}))
        )[0]
        assert torch.equal(plain, torch.ones_like(plain))
        assert not torch.equal(noisy, plain)
        assert (
            is_enabled({"noise": 0.1}) and not is_enabled({"noise": 0.0}) and not is_enabled(None)
        )


class TestLosses:
    labels = np.array([[1, 0], [1, 0], [1, 1], [1, 0]], dtype=np.float32)

    def test_weighted_bce_weights_rare_positives_up(self):
        loss = make_loss({"loss": "weighted_bce"}, self.labels)
        # Class 0 is always positive (weight 0), class 1 once in four (weight 3).
        torch.testing.assert_close(loss.pos_weight, torch.tensor([0.0, 3.0]))

    def test_focal_loss_down_weights_easy_examples(self):
        focal, bce = FocalLoss(2.0), torch.nn.BCEWithLogitsLoss()
        easy_logits, targets = torch.tensor([[6.0]]), torch.tensor([[1.0]])
        hard_logits = torch.tensor([[-2.0]])
        assert focal(easy_logits, targets) < 0.01 * bce(easy_logits, targets)
        assert focal(hard_logits, targets) > 0.5 * bce(hard_logits, targets)

    def test_default_is_plain_bce_and_unknown_raises(self):
        assert isinstance(make_loss({}, self.labels), torch.nn.BCEWithLogitsLoss)
        assert isinstance(make_loss({"loss": "focal"}, self.labels), FocalLoss)
        with pytest.raises(ValueError, match="Unknown loss"):
            make_loss({"loss": "mse"}, self.labels)

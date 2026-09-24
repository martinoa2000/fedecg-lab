"""Tests for DP-SGD calibration, wrapping and accounting (tiny, CPU-only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fedecg.models.resnet1d import ResNet1d
from fedecg.privacy.dp_sgd import PrivateTraining, private_training_from_config
from fedecg.seed import new_generator, set_seed
from fedecg.training.loop import fit, make_loader

CPU = torch.device("cpu")
PRIVACY = {"target_epsilon": 8.0, "delta": 1e-5, "max_grad_norm": 1.0, "accountant": "prv"}


def separable_data(n: int = 128, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = new_generator(seed)
    labels = rng.integers(0, 2, size=(n, 5)).astype(np.float32)
    signals = rng.standard_normal((n, 12, 100)).astype(np.float32) * 0.1
    signals[:, :5, :] += 2.0 * labels[:, :, None]
    return signals, labels


def tiny_model() -> ResNet1d:
    return ResNet1d(base_channels=8, blocks_per_stage=(1,), norm_groups=4, dropout=0.0)


def hook_count(model: torch.nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._backward_hooks) for m in model.modules())


class TestCalibration:
    def test_a_larger_budget_needs_less_noise(self):
        def noise(eps):
            config = {**PRIVACY, "target_epsilon": eps}
            return PrivateTraining(config, n_records=1000, batch_size=50, epochs=5).noise_multiplier

        assert noise(1.0) > noise(3.0) > noise(8.0) > 0

    def test_more_epochs_need_more_noise_for_the_same_budget(self):
        short = PrivateTraining(PRIVACY, n_records=1000, batch_size=50, epochs=2)
        long = PrivateTraining(PRIVACY, n_records=1000, batch_size=50, epochs=20)
        assert long.noise_multiplier > short.noise_multiplier

    def test_needs_more_records_than_a_batch(self):
        with pytest.raises(ValueError, match="batch_size"):
            PrivateTraining(PRIVACY, n_records=10, batch_size=50, epochs=1)

    def test_from_config_is_none_when_disabled(self):
        config = {"privacy": {**PRIVACY, "enabled": False}, "training": {"batch_size": 8}}
        assert private_training_from_config(config, n_records=100, epochs=1) is None
        config["privacy"]["enabled"] = True
        assert isinstance(
            private_training_from_config(config, n_records=100, epochs=1), PrivateTraining
        )


class TestTraining:
    def test_spends_about_the_target_budget_over_the_planned_epochs(self):
        set_seed(0)
        x, y = separable_data()
        epochs = 3
        private = PrivateTraining(PRIVACY, n_records=len(x), batch_size=16, epochs=epochs)
        assert private.epsilon() == 0.0
        loader = make_loader(x, y, batch_size=16, shuffle=True, seed=0)
        cfg = {"epochs": epochs, "learning_rate": 0.01, "early_stopping_patience": epochs}
        model = tiny_model()
        fit(
            model,
            loader,
            make_loader(x, y, batch_size=128, shuffle=False),
            cfg,
            CPU,
            wrap=private.wrap,
        )
        private.unwrap()
        # The accountant's bound for the run it was calibrated for.
        assert private.epsilon() == pytest.approx(8.0, rel=0.05)

    def test_learns_a_separable_problem_at_a_loose_budget(self):
        set_seed(0)
        x, y = separable_data(256)
        config = {**PRIVACY, "target_epsilon": 50.0}
        private = PrivateTraining(config, n_records=len(x), batch_size=32, epochs=10)
        loader = make_loader(x, y, batch_size=32, shuffle=True, seed=0)
        cfg = {"epochs": 10, "learning_rate": 0.01, "early_stopping_patience": 10}
        result = fit(
            tiny_model(),
            loader,
            make_loader(x, y, batch_size=256, shuffle=False),
            cfg,
            CPU,
            wrap=private.wrap,
        )
        private.unwrap()
        assert result.best_score > 0.9

    def test_unwrap_removes_the_per_sample_hooks_and_keeps_the_weights(self):
        x, y = separable_data(64)
        model = tiny_model()
        before = hook_count(model)
        private = PrivateTraining(PRIVACY, n_records=len(x), batch_size=16, epochs=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        wrapped, optimizer, loader = private.wrap(
            model, optimizer, make_loader(x, y, batch_size=16, shuffle=True)
        )
        assert hook_count(model) > before
        xb, yb = next(iter(loader))
        torch.nn.functional.binary_cross_entropy_with_logits(wrapped(xb), yb).backward()
        optimizer.step()
        trained = [p.detach().clone() for p in model.parameters()]
        private.unwrap()
        assert hook_count(model) == before
        for p, q in zip(model.parameters(), trained, strict=True):
            torch.testing.assert_close(p, q)

    def test_accountant_keeps_counting_across_wraps(self):
        """A hospital wraps its model every round; its budget must accumulate."""
        x, y = separable_data(64)
        model = tiny_model()
        private = PrivateTraining(PRIVACY, n_records=len(x), batch_size=16, epochs=4)
        spent = []
        for _ in range(2):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            wrapped, optimizer, loader = private.wrap(
                model, optimizer, make_loader(x, y, batch_size=16, shuffle=True)
            )
            for xb, yb in loader:
                optimizer.zero_grad()
                torch.nn.functional.binary_cross_entropy_with_logits(wrapped(xb), yb).backward()
                optimizer.step()
            private.unwrap()
            spent.append(private.epsilon())
        assert 0 < spent[0] < spent[1] <= 8.0 * 1.05

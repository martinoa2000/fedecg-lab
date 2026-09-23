"""Tests for the training loop and tracking helpers (tiny, CPU-only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fedecg.models.resnet1d import ResNet1d
from fedecg.seed import new_generator, set_seed
from fedecg.training.loop import fit, make_loader, predict, resolve_device
from fedecg.training.tracking import flatten, start_run

CPU = torch.device("cpu")
TRAIN_CFG = {
    "epochs": 15,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "early_stopping_patience": 15,
}


def separable_data(n: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Class k is present iff lead k carries a large offset."""
    rng = new_generator(seed)
    labels = rng.integers(0, 2, size=(n, 5)).astype(np.float32)
    signals = rng.standard_normal((n, 12, 100)).astype(np.float32) * 0.1
    signals[:, :5, :] += 2.0 * labels[:, :, None]
    return signals, labels


def tiny_model() -> ResNet1d:
    return ResNet1d(base_channels=8, blocks_per_stage=(1,), norm_groups=4, dropout=0.0)


class TestFit:
    def test_learns_a_separable_problem(self):
        set_seed(0)
        x, y = separable_data()
        loader = make_loader(x, y, batch_size=16, shuffle=True, seed=0)
        model = tiny_model()
        result = fit(model, loader, make_loader(x, y, batch_size=64, shuffle=False), TRAIN_CFG, CPU)
        assert result.best_score > 0.95
        assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]

    def test_stops_early_and_restores_the_best_weights(self):
        set_seed(0)
        x, y = separable_data()
        # Validation labels are shuffled noise, so AUROC cannot keep improving.
        y_noise = new_generator(1).permutation(y)
        loader = make_loader(x, y, batch_size=16, shuffle=True, seed=0)
        val = make_loader(x, y_noise, batch_size=64, shuffle=False)
        model = tiny_model()
        cfg = {**TRAIN_CFG, "epochs": 30, "early_stopping_patience": 2}
        result = fit(model, loader, val, cfg, CPU)

        assert result.stopped_early
        assert len(result.history) == result.best_epoch + 2
        from fedecg.training.metrics import macro_auroc

        prob, true = predict(model, val, CPU)
        assert macro_auroc(true, prob) == pytest.approx(result.best_score, abs=1e-6)

    def test_rejects_unknown_stopping_metric(self):
        x, y = separable_data(8)
        loader = make_loader(x, y, batch_size=8, shuffle=False)
        with pytest.raises(ValueError, match="early_stopping_metric"):
            fit(tiny_model(), loader, loader, {**TRAIN_CFG, "early_stopping_metric": "acc"}, CPU)


class TestLoader:
    def test_shuffle_order_depends_only_on_the_seed(self):
        x, y = separable_data(32)

        def first_batch(seed):
            torch.manual_seed(123 + seed)  # global RNG must not matter
            return next(iter(make_loader(x, y, batch_size=8, shuffle=True, seed=seed)))[1]

        torch.testing.assert_close(first_batch(0), first_batch(0))


def test_resolve_device():
    assert resolve_device("cpu") == CPU
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_flatten_uses_dotted_keys():
    assert flatten({"a": {"b": 1, "c": {"d": [1, 2]}}, "e": None}) == {
        "a.b": "1",
        "a.c.d": "[1, 2]",
        "e": "None",
    }


def test_disabled_tracking_is_a_no_op(tmp_path):
    with start_run({"enabled": False}, tracking_dir=tmp_path) as tracker:
        tracker.log_params({"a": 1})
        tracker.log_metrics({"m": 1.0}, step=1)
    assert list(tmp_path.iterdir()) == []


def test_enabled_tracking_writes_a_run(tmp_path):
    with start_run({"enabled": True, "experiment_name": "t"}, tracking_dir=tmp_path) as tracker:
        tracker.log_params({"training": {"epochs": 2}})
        tracker.log_metrics({"val_macro_auroc": 0.5}, step=1)
    assert any(tmp_path.iterdir())

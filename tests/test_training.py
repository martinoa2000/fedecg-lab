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


class TestCrops:
    def test_crops_have_the_requested_length_and_come_from_the_record(self):
        x, y = separable_data(8)
        x[:] = np.arange(100, dtype=np.float32)  # sample value == its time index
        batch, _ = next(iter(make_loader(x, y, batch_size=8, shuffle=False, crop_samples=25)))
        assert batch.shape == (8, 12, 25)
        starts = batch[:, 0, 0]
        torch.testing.assert_close(batch[:, 0, :], starts[:, None] + torch.arange(25.0))
        assert len(set(starts.tolist())) > 1  # records get different offsets

    def test_crop_sequence_depends_only_on_the_seed(self):
        x, y = separable_data(16)

        def crops(seed):
            torch.manual_seed(seed + 99)
            return next(
                iter(make_loader(x, y, batch_size=16, shuffle=True, seed=seed, crop_samples=10))
            )[0]

        torch.testing.assert_close(crops(3), crops(3))

    def test_rejects_crops_longer_than_the_record(self):
        x, y = separable_data(4)
        with pytest.raises(ValueError, match="crop length"):
            make_loader(x, y, batch_size=4, shuffle=False, crop_samples=101)


class TestWindowedPredict:
    def test_windows_cover_the_whole_record(self):
        from fedecg.training.loop import _windows

        x = torch.arange(10.0).view(1, 1, 10)
        windows = _windows(x, 4, 3)
        # Starts 0, 3, 6: the last window ends exactly at the final sample.
        assert windows.shape == (1, 3, 1, 4)
        assert windows[0, -1, 0].tolist() == [6.0, 7.0, 8.0, 9.0]

    def test_aggregates_window_probabilities_per_record(self):
        class LastSample(torch.nn.Module):
            """Logit = the window's last sample, so each window scores differently."""

            def forward(self, x):
                return x[:, :1, -1].repeat(1, 5)

        x = np.zeros((2, 12, 8), dtype=np.float32)
        x[:, 0, :] = np.linspace(-2, 2, 8)
        y = np.zeros((2, 5), dtype=np.float32)
        loader = make_loader(x, y, batch_size=2, shuffle=False)
        mean, _ = predict(LastSample(), loader, CPU, window=4, stride=4, aggregate="mean")
        top, _ = predict(LastSample(), loader, CPU, window=4, stride=4, aggregate="max")
        last = torch.sigmoid(torch.tensor(x[0, 0, [3, 7]]))
        assert mean[0, 0] == pytest.approx(float(last.mean()), abs=1e-6)
        assert top[0, 0] == pytest.approx(float(last.max()), abs=1e-6)

    def test_crop_training_learns_and_is_scored_with_windows(self):
        set_seed(0)
        x, y = separable_data()
        cfg = {**TRAIN_CFG, "crop_samples": 40, "eval_window": 40, "eval_stride": 20}
        loader = make_loader(x, y, batch_size=16, shuffle=True, seed=0, crop_samples=40)
        model = tiny_model()
        result = fit(model, loader, make_loader(x, y, batch_size=64, shuffle=False), cfg, CPU)
        assert result.best_score > 0.95


class TestSchedule:
    def test_cosine_warms_up_then_decays_to_zero(self):
        from fedecg.training.loop import lr_factor

        assert lr_factor(0.0, "cosine", 0.1) == 0.0
        assert lr_factor(0.05, "cosine", 0.1) == pytest.approx(0.5)
        assert lr_factor(0.1, "cosine", 0.1) == pytest.approx(1.0)
        assert lr_factor(0.55, "cosine", 0.1) == pytest.approx(0.5)
        assert lr_factor(1.0, "cosine", 0.1) == pytest.approx(0.0, abs=1e-12)
        assert lr_factor(0.3, "constant") == 1.0
        with pytest.raises(ValueError):
            lr_factor(0.3, "step")

    def test_fit_records_the_scheduled_learning_rate(self):
        x, y = separable_data(32)
        loader = make_loader(x, y, batch_size=8, shuffle=True, seed=0)
        cfg = {**TRAIN_CFG, "epochs": 4, "lr_schedule": "cosine", "warmup_fraction": 0.25}
        result = fit(
            tiny_model(), loader, make_loader(x, y, batch_size=32, shuffle=False), cfg, CPU
        )
        lrs = [row["lr"] for row in result.history]
        assert lrs[0] == 0.0
        assert lrs[1] == pytest.approx(TRAIN_CFG["learning_rate"])
        assert lrs[1] > lrs[2] > lrs[3]


class TestProximal:
    def test_proximal_term_pulls_weights_toward_the_anchor(self):
        from fedecg.training.loop import train_one_epoch

        x, y = separable_data(32)
        loader = make_loader(x, y, batch_size=8, shuffle=True, seed=0)

        def drift(mu):
            set_seed(0)
            model = tiny_model()
            anchor = [p.detach().clone() for p in model.parameters()]
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            for _ in range(3):
                train_one_epoch(
                    model, loader, optimizer, torch.nn.BCEWithLogitsLoss(), CPU,
                    proximal_mu=mu, proximal_anchor=anchor,
                )  # fmt: skip
            return sum(
                float(((p.detach() - a) ** 2).sum())
                for p, a in zip(model.parameters(), anchor, strict=True)
            )

        assert drift(10.0) < drift(0.0)

    def test_needs_an_anchor(self):
        from fedecg.training.loop import train_one_epoch

        x, y = separable_data(8)
        model = tiny_model()
        with pytest.raises(ValueError, match="proximal_anchor"):
            train_one_epoch(
                model,
                make_loader(x, y, batch_size=8, shuffle=False),
                torch.optim.SGD(model.parameters(), lr=0.1),
                torch.nn.BCEWithLogitsLoss(),
                CPU,
                proximal_mu=0.1,
            )

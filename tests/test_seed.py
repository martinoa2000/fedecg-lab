"""Tests for reproducible seeding."""

from __future__ import annotations

import random

import numpy as np
import torch

from fedecg.seed import new_generator, set_seed


class TestSetSeed:
    def test_same_seed_gives_the_same_torch_draw(self):
        set_seed(123)
        first = torch.randn(8)
        set_seed(123)
        second = torch.randn(8)

        assert torch.equal(first, second)

    def test_different_seeds_give_different_draws(self):
        set_seed(1)
        first = torch.randn(8)
        set_seed(2)
        second = torch.randn(8)

        assert not torch.equal(first, second)

    def test_seeds_numpy_and_python_too(self):
        set_seed(7)
        first = (np.random.rand(4).tolist(), random.random())
        set_seed(7)
        second = (np.random.rand(4).tolist(), random.random())

        assert first == second


class TestNewGenerator:
    def test_is_reproducible(self):
        assert new_generator(0).integers(0, 1000, 5).tolist() == (
            new_generator(0).integers(0, 1000, 5).tolist()
        )

    def test_is_isolated_from_global_state(self):
        """The whole point: unrelated global draws must not shift a partition."""
        generator = new_generator(99)
        expected = generator.integers(0, 1000, 5).tolist()

        np.random.seed(0)
        np.random.rand(1000)  # noise from some unrelated library call
        random.random()

        assert new_generator(99).integers(0, 1000, 5).tolist() == expected

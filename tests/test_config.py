"""Tests for YAML config loading and inheritance."""

from __future__ import annotations

import re

import pytest
import yaml

from fedecg.config import deep_merge, load_config
from fedecg.paths import CONFIG_DIR


def write_config(directory, name: str, payload: dict) -> None:
    (directory / name).write_text(yaml.safe_dump(payload), encoding="utf-8")


class TestDeepMerge:
    def test_override_wins_on_scalars(self):
        assert deep_merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}

    def test_nested_dicts_merge_key_by_key(self):
        base = {"training": {"epochs": 30, "lr": 0.001}}
        override = {"training": {"epochs": 5}}
        assert deep_merge(base, override) == {"training": {"epochs": 5, "lr": 0.001}}

    def test_lists_are_replaced_not_concatenated(self):
        assert deep_merge({"folds": [1, 2, 3]}, {"folds": [9]}) == {"folds": [9]}

    def test_inputs_are_not_mutated(self):
        base = {"nested": {"value": 1}}
        deep_merge(base, {"nested": {"value": 2}})
        assert base == {"nested": {"value": 1}}


class TestLoadConfig:
    def test_loads_a_plain_config(self, tmp_path):
        write_config(tmp_path, "plain.yaml", {"seed": 7})
        assert load_config(tmp_path / "plain.yaml") == {"seed": 7}

    def test_resolves_extends_and_drops_the_key(self, tmp_path):
        write_config(tmp_path, "base.yaml", {"seed": 42, "training": {"epochs": 30}})
        write_config(tmp_path, "child.yaml", {"extends": "base.yaml", "training": {"epochs": 3}})

        config = load_config(tmp_path / "child.yaml")

        assert "extends" not in config
        assert config == {"seed": 42, "training": {"epochs": 3}}

    def test_resolves_a_multi_level_chain(self, tmp_path):
        write_config(tmp_path, "a.yaml", {"x": 1, "y": 1, "z": 1})
        write_config(tmp_path, "b.yaml", {"extends": "a.yaml", "y": 2})
        write_config(tmp_path, "c.yaml", {"extends": "b.yaml", "z": 3})

        assert load_config(tmp_path / "c.yaml") == {"x": 1, "y": 2, "z": 3}

    def test_detects_circular_inheritance(self, tmp_path):
        write_config(tmp_path, "loop_a.yaml", {"extends": "loop_b.yaml"})
        write_config(tmp_path, "loop_b.yaml", {"extends": "loop_a.yaml"})

        with pytest.raises(ValueError, match="Circular"):
            load_config(tmp_path / "loop_a.yaml")

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_missing_parent_names_both_the_child_and_the_parent(self, tmp_path):
        write_config(tmp_path, "orphan.yaml", {"extends": "ghost.yaml"})

        with pytest.raises(FileNotFoundError, match=re.escape("ghost.yaml")):
            load_config(tmp_path / "orphan.yaml")

    def test_empty_file_loads_as_empty_dict(self, tmp_path):
        (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
        assert load_config(tmp_path / "empty.yaml") == {}

    def test_bare_name_resolves_against_the_configs_directory(self):
        config = load_config("default.yaml")
        assert config["seed"] == 42
        assert (CONFIG_DIR / "default.yaml").is_file()


class TestDefaultConfig:
    """The shipped default config must stay consistent with the code."""

    def test_uses_the_official_ptbxl_split(self):
        from fedecg.data.constants import TEST_FOLD, TRAIN_FOLDS, VAL_FOLD

        config = load_config("default.yaml")

        assert tuple(config["data"]["train_folds"]) == TRAIN_FOLDS
        assert config["data"]["val_fold"] == VAL_FOLD
        assert config["data"]["test_fold"] == TEST_FOLD

    def test_train_val_test_folds_are_disjoint(self):
        config = load_config("default.yaml")
        folds = config["data"]
        assert folds["val_fold"] not in folds["train_folds"]
        assert folds["test_fold"] not in folds["train_folds"]
        assert folds["val_fold"] != folds["test_fold"]

    def test_sampling_rate_matches_constants(self):
        from fedecg.data.constants import SAMPLING_RATE_HZ

        assert load_config("default.yaml")["data"]["sampling_rate_hz"] == SAMPLING_RATE_HZ

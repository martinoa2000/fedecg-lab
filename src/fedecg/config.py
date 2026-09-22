"""YAML experiment configuration with single-parent inheritance.

Every experiment in this repository is defined by a file in `configs/`, and
results are only meaningful if that file is versioned alongside the numbers it
produced. Configs share a common base through an `extends` key so that, for
example, the federated and centralized experiments provably use the same data
split and the same model width -- the diff between two config files is the
experimental variable.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from fedecg.paths import CONFIG_DIR


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict.

    Nested dicts are merged key by key; every other type (including lists) is
    replaced wholesale. Lists are replaced rather than concatenated because a
    config that half-inherits a list is almost never what the author meant.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, *, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a YAML config, resolving its `extends` chain.

    A config may declare `extends: <filename>` to inherit from another config.
    The parent is resolved relative to the child's directory first, then to
    `configs/`. The child's own keys win.

    Args:
        path: Config file path. A bare name (no directory part) is looked up in
            `configs/`, so `load_config("centralized.yaml")` works from anywhere.
        _seen: Internal. Tracks the inheritance chain to detect cycles.

    Returns:
        The fully merged config, without the `extends` key.

    Raises:
        FileNotFoundError: If the config or one of its parents does not exist.
        ValueError: If the `extends` chain contains a cycle.
    """
    path = Path(path)
    if not path.parent.name and not path.is_absolute():
        path = CONFIG_DIR / path
    path = path.resolve()

    if path in _seen:
        chain = " -> ".join(p.name for p in [*_seen, path])
        raise ValueError(f"Circular 'extends' in config chain: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle) or {}

    parent_name = config.pop("extends", None)
    if parent_name is None:
        return config

    candidates = [path.parent / parent_name, CONFIG_DIR / parent_name]
    parent_path = next((c for c in candidates if c.is_file()), None)
    if parent_path is None:
        raise FileNotFoundError(
            f"Config {path.name} extends '{parent_name}', which was not found in "
            f"{path.parent} or {CONFIG_DIR}"
        )

    parent = load_config(parent_path, _seen=_seen | {path})
    return deep_merge(parent, config)

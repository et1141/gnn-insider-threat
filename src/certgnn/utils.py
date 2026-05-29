"""Project-wide helpers: locating the repo root and loading config.

Configuration is split across ``configs/*.yaml`` by concern
(``paths.yaml``, ``preprocessing.yaml``, ``training.yaml``). They are
deep-merged here into a single dict, so callers keep using
``config["paths"]``, ``config["preprocessing"]``, ``config["training"]``
as before — the split is purely organisational.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return path to project root folder."""
    return Path(__file__).resolve().parent.parent.parent


def _deep_merge(base: dict, incoming: dict) -> dict:
    """Recursively merge ``incoming`` into ``base`` (in place) and return it."""
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(configs_dir: Path | None = None) -> dict:
    """Load and deep-merge every ``*.yaml`` under ``configs/``.

    Files are merged in sorted filename order; top-level keys are disjoint by
    design (``paths``/``download``/``preprocessing``/``training``), so order
    only matters if a key is intentionally overridden across files.
    """
    configs_dir = configs_dir or (get_project_root() / "configs")
    config: dict = {}
    yaml_files = sorted(configs_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No config files found in {configs_dir}")
    for path in yaml_files:
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        _deep_merge(config, loaded)
    return config

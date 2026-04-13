"""User-level split helpers for the CERT baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split


def _safe_split(
    items: list[str],
    labels: list[int],
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if len(items) <= 1 or test_size <= 0:
        return items, []

    stratify = labels if len(set(labels)) > 1 else None
    try:
        train_items, heldout_items = train_test_split(
            items,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_items, heldout_items = train_test_split(
            items,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    return list(train_items), list(heldout_items)


def build_user_splits(
    user_ids: Iterable[str],
    user_labels: dict[str, int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split users into train/val/test with light stratification."""

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0.0, 1.0)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    users = sorted(dict.fromkeys(user_ids))
    labels = [int(user_labels.get(user, 0)) for user in users]

    train_users, test_users = _safe_split(users, labels, test_ratio, seed)
    remaining_labels = [int(user_labels.get(user, 0)) for user in train_users]
    adjusted_val_ratio = val_ratio / max(1.0 - test_ratio, 1e-8)
    train_users, val_users = _safe_split(
        train_users,
        remaining_labels,
        adjusted_val_ratio,
        seed + 1,
    )

    splits = {
        "train": sorted(train_users),
        "val": sorted(val_users),
        "test": sorted(test_users),
    }

    if not splits["train"] and users:
        splits["train"] = [users[0]]
        if users[0] in splits["val"]:
            splits["val"].remove(users[0])
        if users[0] in splits["test"]:
            splits["test"].remove(users[0])

    return splits


def save_user_split_manifest(
    processed_dir: Path,
    splits: dict[str, list[str]],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Path:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "splits": splits,
    }
    path = processed_dir / "user_split_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def load_user_split_manifest(processed_dir: Path) -> dict:
    path = Path(processed_dir) / "user_split_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    return json.loads(path.read_text())

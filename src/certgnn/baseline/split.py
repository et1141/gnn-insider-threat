"""User-level split helpers for the CERT baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import numpy as np


def build_user_splits(
    user_ids: Iterable[str],
    user_labels: dict[str, int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Split users into train/val/test with strict scenario-level stratification.
    
    Groups users by their specific label (0 for normal, 1, 2, 3... for specific
    malicious scenarios). Then, it calculates exact capacities for each split
    within that group to ensure every scenario is represented as equally as possible.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0.0, 1.0)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    rng = np.random.RandomState(seed)
    users = sorted(dict.fromkeys(user_ids))
    
    # Group users by their label/scenario
    label_to_users = {}
    for user in users:
        lbl = int(user_labels.get(user, 0))
        if lbl not in label_to_users:
            label_to_users[lbl] = []
        label_to_users[lbl].append(user)

    splits = {"train": [], "val": [], "test": []}

    # Distribute users for each scenario group separately
    for lbl, group in label_to_users.items():
        group_copy = list(group)
        # Shuffle to ensure random assignment before splitting
        rng.shuffle(group_copy)
        
        n = len(group_copy)
        
        # Calculate standard capacities based on ratios
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        n_train = n - n_val - n_test
        
        # -------------------------------------------------------------
        # SAFEGUARD 1: Train set MUST have at least 1 user (if possible)
        # Without training data for a scenario, testing on it is useless.
        # -------------------------------------------------------------
        if n_train <= 0 and n > 0:
            n_train = 1
            # Steal a sample back from val or test to give to train
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

        # -------------------------------------------------------------
        # SAFEGUARD 2: Force val/test representation for Malicious users
        # (Only if we have enough users to safely populate train as well)
        # -------------------------------------------------------------
        if lbl != 0 and n >= 3:
            if n_val == 0 and val_ratio > 0:
                n_val = 1
                n_train -= 1
            if n_test == 0 and test_ratio > 0:
                n_test = 1
                n_train -= 1

        # Assign sliced groups to respective splits EXACTLY using n_train
        # Train gets the first chunk, then Val gets the next, Test gets the rest
        splits["train"].extend(group_copy[:n_train])
        splits["val"].extend(group_copy[n_train : n_train + n_val])
        splits["test"].extend(group_copy[n_train + n_val :])

    # Sort the final lists for deterministic output
    splits["train"] = sorted(splits["train"])
    splits["val"] = sorted(splits["val"])
    splits["test"] = sorted(splits["test"])

    # Fallback if train is completely empty
    if not splits["train"] and users:
        splits["train"] = [users[0]]
        if users[0] in splits["val"]: splits["val"].remove(users[0])
        if users[0] in splits["test"]: splits["test"].remove(users[0])

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
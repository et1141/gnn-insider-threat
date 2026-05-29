"""User-level scenario-stratified preprocessing variant.

User selection: fixed-count (``num_normal_users``, ``num_malicious_users``)
with optional per-scenario balancing.

Split strategy: per-user stratified by scenario class so every malicious
scenario appears in train/val/test with safeguards against degenerate
splits. The split assignment is materialised at preprocessing time as
``train_/val_/test_graph_chunk_*.pt`` files plus a ``user_split_manifest.json``
audit trail. Eliminates the per-graph leak that the paper-faithful flow
suffers from (sub-session variants of the same activity scattering across
folds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from certgnn.preprocessing.common import (
    build_combined_dataframe,
    create_graph,
    iter_subsessions,
    load_user_df_and_malicious_ids,
    save_processed,
)
from certgnn.utils import get_project_root, load_config

CHUNK_SIZE = 20_000  # graphs per split-prefixed chunk file


# ---------------------------------------------------------------------------
# User selection (fixed-count, scenario-aware)
# ---------------------------------------------------------------------------
def select_users(
    extract_dir: Path,
    answers_dir: Path,
    dataset_version: str,
    num_normal_users: int,
    num_malicious_users: int,
    balance_malicious_by_scenario: bool = False,
    strict_equal_scenarios: bool = True,
    seed: int = 42,
) -> set | None:
    """Select fixed counts of normal and malicious users.

    With ``balance_malicious_by_scenario=True``, malicious users are sampled
    equally per scenario; ``strict_equal_scenarios=True`` raises if the
    requested per-scenario count cannot be met from any scenario.
    """
    if num_normal_users < 0 or num_malicious_users < 0:
        raise ValueError("Number of users to select cannot be negative.")

    rng = np.random.RandomState(seed)
    all_users = set(
        pd.read_csv(extract_dir / "logon.csv", usecols=["user"])["user"].unique()
    )

    insiders = pd.read_csv(answers_dir / "insiders.csv")
    insiders = insiders[
        insiders["dataset"].astype(str).str.startswith(str(dataset_version))
    ].copy()
    insiders = insiders[insiders["user"].isin(all_users)].copy()

    mal_users_all = set(insiders["user"].unique())
    non_mal = sorted(all_users - mal_users_all)

    n_non = min(num_normal_users, len(non_mal))
    sampled_non_mal: set = set() if n_non == 0 else set(rng.choice(non_mal, size=n_non, replace=False))

    if num_malicious_users == 0 or len(mal_users_all) == 0:
        sampled_mal: set = set()
    elif not balance_malicious_by_scenario:
        mal_list = sorted(mal_users_all)
        n_mal = min(num_malicious_users, len(mal_list))
        sampled_mal = set(rng.choice(mal_list, size=n_mal, replace=False))
    else:
        scenario_users: dict[str, list[str]] = {}
        for scenario, g in insiders.groupby(insiders["scenario"].astype(str)):
            users = sorted(set(g["user"]))
            if users:
                scenario_users[scenario] = users

        if not scenario_users:
            sampled_mal = set()
        else:
            n_scen = len(scenario_users)
            per_scen = num_malicious_users // n_scen
            if per_scen == 0 and num_malicious_users > 0 and strict_equal_scenarios:
                raise ValueError(
                    "num_malicious_users too small for equal-per-scenario sampling. "
                    f"You need at least {n_scen} malicious users to cover all scenarios."
                )

            sampled: set = set()
            for scenario, users in sorted(scenario_users.items(), key=lambda x: x[0]):
                if strict_equal_scenarios and len(users) < per_scen:
                    raise ValueError(
                        f"Scenario {scenario} has only {len(users)} users, "
                        f"cannot sample {per_scen} equally from all scenarios."
                    )
                take = min(per_scen, len(users))
                if take > 0:
                    sampled.update(rng.choice(users, size=take, replace=False))
            sampled_mal = sampled

    selected = sampled_non_mal | sampled_mal
    if len(selected) == len(all_users):
        print(f"  All users selected ({len(all_users)} total) — disabling filter.")
        return None

    print(
        f"  Selected users: total={len(selected)} | "
        f"malicious={len(sampled_mal)} | non-malicious={len(sampled_non_mal)}"
    )
    return selected


# ---------------------------------------------------------------------------
# User-level scenario-stratified split
# ---------------------------------------------------------------------------
def build_user_splits(
    user_ids: Iterable[str],
    user_labels: dict[str, int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split users into train/val/test stratified per scenario class.

    Users are grouped by their scenario label (0 for normal, 1..N for each
    malicious scenario). For each group, val/test capacities are computed
    and adjusted to guarantee:

    * Every group keeps at least 1 train user (otherwise testing on an
      unseen scenario is meaningless).
    * Every malicious group of size ≥3 gets at least 1 val and 1 test user.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0.0, 1.0)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    rng = np.random.RandomState(seed)
    users = sorted(dict.fromkeys(user_ids))

    label_to_users: dict[int, list[str]] = {}
    for user in users:
        lbl = int(user_labels.get(user, 0))
        label_to_users.setdefault(lbl, []).append(user)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for lbl, group in label_to_users.items():
        group_copy = list(group)
        rng.shuffle(group_copy)
        n = len(group_copy)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        n_train = n - n_val - n_test

        # Ensure at least 1 train user per group.
        if n_train <= 0 and n > 0:
            n_train = 1
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

        # Force val/test representation for malicious groups when possible.
        if lbl != 0 and n >= 3:
            if n_val == 0 and val_ratio > 0:
                n_val = 1
                n_train -= 1
            if n_test == 0 and test_ratio > 0:
                n_test = 1
                n_train -= 1

        splits["train"].extend(group_copy[:n_train])
        splits["val"].extend(group_copy[n_train : n_train + n_val])
        splits["test"].extend(group_copy[n_train + n_val :])

    splits["train"] = sorted(splits["train"])
    splits["val"] = sorted(splits["val"])
    splits["test"] = sorted(splits["test"])

    if not splits["train"] and users:
        splits["train"] = [users[0]]
        for k in ("val", "test"):
            if users[0] in splits[k]:
                splits[k].remove(users[0])

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
    path = processed_dir / "user_split_manifest.json"
    path.write_text(
        json.dumps(
            {"seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio, "splits": splits},
            indent=2,
        )
    )
    return path


def _build_user_to_scenario(answers_dir: Path, dataset_version: str) -> dict[str, int]:
    """Map insider user_id → scenario_id (1..N). Normal users are absent."""
    insiders_df = pd.read_csv(answers_dir / "insiders.csv")
    insiders_df = insiders_df[
        insiders_df["dataset"].astype(str).str.startswith(str(dataset_version))
    ]
    out: dict[str, int] = {}
    for _, row in insiders_df.iterrows():
        u_id = str(row["user"])
        out.setdefault(u_id, int(row["scenario"]))
    return out


# ---------------------------------------------------------------------------
# Graph creation (split-prefixed chunks)
# ---------------------------------------------------------------------------
def create_all_graphs(
    combined: pd.DataFrame,
    activity_types: dict,
    min_session_size: int,
    max_session_size: int,
    processed_dir: Path,
    user_to_split: dict[str, str],
    stream: bool = False,
) -> int:
    """Bucket graphs into ``train``/``val``/``test`` chunk streams by user split."""
    from certgnn.data.chunk_store import DvcChunkStore  # lazy: only needed with --stream

    buffers: dict[str, list] = {"train": [], "val": [], "test": []}
    chunk_idx: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    total_graphs = 0
    store = DvcChunkStore(processed_dir) if stream else None

    processed_dir.mkdir(parents=True, exist_ok=True)

    def _flush(split_name: str) -> None:
        graph_list = buffers[split_name]
        if not graph_list:
            return
        chunk_path = processed_dir / f"{split_name}_graph_chunk_{chunk_idx[split_name]}.pt"
        torch.save(graph_list, chunk_path)
        if store is not None:
            store.push_chunk(chunk_path, num_graphs=len(graph_list), delete_local=True)
        chunk_idx[split_name] += 1
        graph_list.clear()

    for user, user_data in tqdm(combined.groupby("user_id"), desc="  Graphs"):
        split_name = user_to_split.get(str(user), "train")
        if split_name not in buffers:
            split_name = "train"

        for _sub, feat_dict, seq, labels in iter_subsessions(user_data, min_session_size, max_session_size):
            for mask_idx in range(len(seq)):
                remaining = seq[:mask_idx] + seq[mask_idx + 1 :]
                if len(remaining) < 2:
                    continue
                graph = create_graph(
                    remaining, seq[mask_idx], labels[mask_idx],
                    feat_dict, activity_types,
                    user_id=str(user), split_name=split_name,
                )
                buffers[split_name].append(graph)
                total_graphs += 1

        # Flush whichever buffer crossed the chunk threshold.
        for name in buffers:
            if len(buffers[name]) >= CHUNK_SIZE:
                _flush(name)

    for name in buffers:
        _flush(name)

    return total_graphs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="User-level-split preprocessing variant")
    parser.add_argument("--stream", action="store_true",
                        help="Push each chunk to GDrive via DVC and delete it locally after saving.")
    args = parser.parse_args()

    config = load_config()
    root = get_project_root()
    prep = config.get("preprocessing", {})
    variant_cfg = prep.get("user_level_split", {}) or {}
    common_cfg = prep.get("common", {}) or {}

    num_normal_users = variant_cfg.get("num_normal_users", 100)
    num_malicious_users = variant_cfg.get("num_malicious_users", 90)
    balance_malicious = variant_cfg.get("balance_malicious_by_scenario", False)
    strict_equal_scenarios = variant_cfg.get("strict_equal_scenarios", True)
    split_val_ratio = float(variant_cfg.get("val_ratio", 0.15))
    split_test_ratio = float(variant_cfg.get("test_ratio", 0.15))
    dataset_version = common_cfg.get("dataset_version", "5.2")
    min_session = common_cfg.get("min_session_size", 5)
    max_session = common_cfg.get("max_session_size", 50)
    seed = common_cfg.get("seed", 42)

    extract_dir = root / config["paths"]["extract_dir"]
    processed_dir = root / config["paths"]["processed_dir"]
    answers_dir = root / config["paths"]["raw_dir"] / "answers"

    np.random.seed(seed)

    user_df, mal_ids = load_user_df_and_malicious_ids(extract_dir, answers_dir, dataset_version)

    print(
        f"Selecting users (num_normal={num_normal_users}, num_malicious={num_malicious_users}, "
        f"balance_by_scenario={balance_malicious})..."
    )
    selected = select_users(
        extract_dir=extract_dir, answers_dir=answers_dir,
        dataset_version=dataset_version,
        num_normal_users=num_normal_users, num_malicious_users=num_malicious_users,
        balance_malicious_by_scenario=balance_malicious,
        strict_equal_scenarios=strict_equal_scenarios, seed=seed,
    )

    selected_users = (
        sorted(selected) if selected is not None
        else sorted(user_df["user_id"].astype(str).unique())
    )

    user_to_scenario = _build_user_to_scenario(answers_dir, dataset_version)
    user_labels = {uid: user_to_scenario.get(str(uid), 0) for uid in selected_users}

    user_split_map = build_user_splits(
        user_ids=selected_users, user_labels=user_labels,
        val_ratio=split_val_ratio, test_ratio=split_test_ratio, seed=seed,
    )
    save_user_split_manifest(
        processed_dir=processed_dir, splits=user_split_map,
        seed=seed, val_ratio=split_val_ratio, test_ratio=split_test_ratio,
    )
    user_to_split = {
        uid: split_name
        for split_name, users in user_split_map.items()
        for uid in users
    }

    combined, encoder, act_types = build_combined_dataframe(
        extract_dir, processed_dir, user_df, mal_ids, selected, min_session,
    )

    stream_msg = " (streaming to GDrive)" if args.stream else ""
    print(f"Creating split-aware graphs{stream_msg}...")
    total_graphs = create_all_graphs(
        combined, act_types, min_session, max_session, processed_dir,
        user_to_split=user_to_split, stream=args.stream,
    )

    print("Saving metadata...")
    save_processed(total_graphs, processed_dir, encoder, act_types, user_splits=user_split_map)
    print(f"\nDone! Successfully created and saved {total_graphs} graphs.")


if __name__ == "__main__":
    main()

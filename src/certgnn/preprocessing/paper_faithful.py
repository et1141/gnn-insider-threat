"""Paper-faithful preprocessing variant.

User selection: fractional (``frac_normal_users``, ``frac_malicious_users``).
Split strategy: NONE here — each sub-session contributes one graph per
masked activity, all graphs are written to a single ``graph_chunk_*.pt``
stream. Train/val/test slicing happens at training time via a per-graph
``RandomSplit`` strategy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from certgnn.preprocessing.common import (
    aggregate_features,
    build_activity_types_dict,
    build_user_pc_mapping,
    collect_malicious_ids,
    combine_and_encode_parquet,
    create_graph,
    iter_subsessions,
    process_and_dump_to_parquet,
    save_processed,
    update_hours,
)
from certgnn.utils import get_project_root, load_config


def select_users(
    extract_dir: Path,
    answers_dir: Path,
    dataset_version: str,
    frac_normal_users: float,
    seed: int = 42,
    frac_malicious_users: float = 1.0,
) -> set | None:
    """Select a fractional subset of users.

    Returns ``None`` when the fractions are 1.0 (i.e. select everyone) so
    downstream SQL can skip the filter join entirely.
    """
    if frac_normal_users >= 1.0 and frac_malicious_users >= 1.0:
        return None
    rng = np.random.RandomState(seed)
    con = duckdb.connect(":memory:")
    all_users = set(
        con.execute(
            f"SELECT DISTINCT user FROM read_csv_auto('{extract_dir}/logon.csv')"
        ).df()["user"]
    )
    insiders = pd.read_csv(answers_dir / "insiders.csv")
    mal_users = set(
        insiders[insiders["dataset"].astype(str).str.startswith(dataset_version)]["user"].unique()
    )

    if frac_malicious_users < 1.0 and len(mal_users) > 0:
        mal_count = max(1, int(len(mal_users) * frac_malicious_users))
        mal_users = set(rng.choice(sorted(mal_users), size=mal_count, replace=False))

    non_mal = sorted(all_users - mal_users)
    n = max(1, int(len(non_mal) * frac_normal_users))
    sampled = set(rng.choice(non_mal, size=n, replace=False))
    selected = sampled | mal_users
    print(f"  {len(selected)} users: {len(mal_users)} malicious + {len(sampled)} non-malicious")
    return selected


def estimate_chunk_count(
    combined: pd.DataFrame, min_session_size: int, max_session_size: int,
) -> tuple[int, int, int]:
    """Estimate (chunks, graphs, activities) used in pre-flight logging."""
    CHUNK_SIZE = 10_000
    total_activities = len(combined)
    total_subsessions = 0
    for _, user_data in combined.groupby("user_id"):
        for _ in iter_subsessions(user_data, min_session_size, max_session_size):
            total_subsessions += 1
    estimated_graphs = total_activities
    estimated_chunks = (estimated_graphs + CHUNK_SIZE - 1) // CHUNK_SIZE
    return estimated_chunks, estimated_graphs, total_activities


def create_all_graphs(
    combined: pd.DataFrame,
    activity_types: dict,
    min_session_size: int,
    max_session_size: int,
    processed_dir: Path,
    stream: bool = False,
    keep_local: bool = False,
) -> int:
    """Create one chunk stream of masked-activity graphs.

    Chunks are flushed at ``CHUNK_SIZE`` graphs to stay well under the 2 GB
    zip64 ceiling of ``torch.save`` (~55 KB per Data object → ~550 MB files).
    """
    from certgnn.data.chunk_store import DvcChunkStore  # lazy: only needed with --stream

    CHUNK_SIZE = 10_000
    graph_list: list = []
    chunk_idx = 0
    total_graphs = 0
    total_activities = 0
    total_sessions = 0
    total_subsessions = 0
    store = DvcChunkStore(processed_dir) if stream else None

    processed_dir.mkdir(parents=True, exist_ok=True)
    grouped_users = combined.groupby("user_id")

    def _flush() -> None:
        nonlocal chunk_idx
        if not graph_list:
            return
        chunk_path = processed_dir / f"graph_chunk_{chunk_idx}.pt"
        torch.save(graph_list, chunk_path)
        if store is not None:
            store.push_chunk(chunk_path, num_graphs=len(graph_list), delete_local=not keep_local)
        chunk_idx += 1
        graph_list.clear()

    for _, user_data in tqdm(grouped_users, total=len(grouped_users), desc="  Graphs"):
        for sub, feat_dict, seq, labels in iter_subsessions(user_data, min_session_size, max_session_size):
            total_subsessions += 1
            total_sessions += 1
            total_activities += len(sub)

            for mask_idx in range(len(seq)):
                remaining = seq[:mask_idx] + seq[mask_idx + 1 :]
                if len(remaining) < 2:
                    continue
                graph = create_graph(remaining, seq[mask_idx], labels[mask_idx], feat_dict, activity_types)
                graph_list.append(graph)
                total_graphs += 1

            if len(graph_list) >= CHUNK_SIZE:
                _flush()
    _flush()

    print(f"\n{'='*70}")
    print("Graph Creation Summary")
    print(f"{'='*70}")
    print(f"  Total activities processed:    {total_activities:>12,}")
    print(f"  Total sub-sessions:            {total_subsessions:>12,}")
    print(f"  Total graphs created:          {total_graphs:>12,}")
    print(f"  Chunks created:                {chunk_idx:>12,}")
    print(f"  Approx disk usage:             {chunk_idx * 0.55:>12.1f} GB")
    print(f"{'='*70}\n")
    return total_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-faithful preprocessing variant")
    parser.add_argument("--stream", action="store_true",
                        help="Push each chunk to GDrive via DVC and delete it locally after saving.")
    parser.add_argument("--keep-local", action="store_true",
                        help="When used with --stream, keep local .pt files after pushing.")
    args = parser.parse_args()

    config = load_config()
    root = get_project_root()
    prep = config.get("preprocessing", {})
    variant_cfg = prep.get("paper_faithful", {}) or {}
    common_cfg = prep.get("common", {}) or {}

    # Backward-compat: accept both nested (preferred) and flat keys.
    frac_normal_users = variant_cfg.get("frac_normal_users", prep.get("frac_normal_users", 1.0))
    frac_malicious_users = variant_cfg.get("frac_malicious_users", prep.get("frac_malicious_users", 1.0))
    dataset_version = common_cfg.get("dataset_version", prep.get("dataset_version", "5.2"))
    min_session = common_cfg.get("min_session_size", prep.get("min_session_size", 5))
    max_session = common_cfg.get("max_session_size", prep.get("max_session_size", 50))
    seed = common_cfg.get("seed", prep.get("seed", 42))
    keep_local = args.keep_local or variant_cfg.get("keep_local", prep.get("keep_local", False))

    extract_dir = root / config["paths"]["extract_dir"]
    processed_dir = root / config["paths"]["processed_dir"]
    answers_dir = root / config["paths"]["raw_dir"] / "answers"

    np.random.seed(seed)

    print("[1/8] Building user-PC mapping...")
    user_df = build_user_pc_mapping(extract_dir)
    print(f"  {len(user_df)} users")

    print("[2/8] Collecting malicious activity IDs...")
    mal_ids = collect_malicious_ids(answers_dir, dataset_version)
    print(f"  {len(mal_ids)} malicious IDs")

    mal_str = f", frac_malicious={frac_malicious_users}" if frac_malicious_users < 1.0 else ""
    print(f"[3/8] Selecting users (frac_normal={frac_normal_users}{mal_str})...")
    selected = select_users(
        extract_dir, answers_dir, dataset_version,
        frac_normal_users, seed, frac_malicious_users=frac_malicious_users,
    )

    print("[4/8 & 5/8] Extracting features via DuckDB SQL to Parquet...")
    parquet_path = process_and_dump_to_parquet(
        extract_dir, processed_dir, user_df, mal_ids, selected,
    )

    print("[6/8] Loading from Parquet and encoding...")
    combined, encoder = combine_and_encode_parquet(parquet_path)

    print("[7/8] Hour merging and feature aggregation...")
    combined["update_hour"] = combined["hour"]
    combined, _ = update_hours(combined, min_session)
    combined["date"] = combined["timestamp"].dt.date
    combined = aggregate_features(combined)

    stream_msg = " (streaming to GDrive)" if args.stream else ""
    if args.stream and keep_local:
        stream_msg += " + keeping local"
    print(f"[8/8] Creating graphs (saving in chunks){stream_msg}...")

    est_chunks, est_graphs, total_acts = estimate_chunk_count(combined, min_session, max_session)
    print(f"  → Estimated {est_chunks} chunks (~{est_graphs:,} graphs from {total_acts:,} activities)")

    act_types = build_activity_types_dict(combined)
    total_graphs = create_all_graphs(
        combined, act_types, min_session, max_session, processed_dir,
        stream=args.stream, keep_local=keep_local,
    )

    print("Saving metadata...")
    save_processed(total_graphs, processed_dir, encoder, act_types)
    print(f"\nDone! Successfully created and saved {total_graphs} graphs.")


if __name__ == "__main__":
    main()

"""Preprocessing pipeline for GNN Insider Threat Detection.

Loads raw CMU CERT data, extracts features, creates session graphs,
and saves processed data ready for training.

Adapted from the source paper's 01_feature_extraction.ipynb.

Usage:
    uv run preprocess           # save chunks locally
    uv run preprocess --stream  # push each chunk to GDrive and delete locally
"""

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from tqdm import tqdm

from certgnn.feature_extraction import GetFeature
from certgnn.utils import get_project_root, load_config

# Feature vector layout: [logon(4) | device(6) | file(16) | email(22) | http(6)] = 54
FEATURE_DIMS = {"logon": 4, "device": 6, "file": 16, "email": 22, "http": 6}
TOTAL_FEATURE_DIM = sum(FEATURE_DIMS.values())  # 54

# Pre-computed padding (before, after) for each activity type
_PADDING: dict[str, tuple[int, int]] = {}
_offset = 0
for _name, _dim in FEATURE_DIMS.items():
    _PADDING[_name] = (_offset, TOTAL_FEATURE_DIM - _offset - _dim)
    _offset += _dim
# logon=(0,50), device=(4,44), file=(10,28), email=(26,6), http=(48,0)


# ---------------------------------------------------------------------------
# Step 1: User-PC mapping
# ---------------------------------------------------------------------------

def build_user_pc_mapping(extract_dir: Path) -> pd.DataFrame:
    """Build user DataFrame with employee info and most-frequent PC.

    Combines LDAP monthly snapshots with logon data to determine each
    user's primary PC (most frequently used).
    """
    ldap_dir = extract_dir / "LDAP"
    ldap_dfs = [pd.read_csv(f) for f in sorted(ldap_dir.glob("*.csv"))]
    ldap_df = pd.concat(ldap_dfs).drop_duplicates(subset="user_id", keep="last")

    logon = pd.read_csv(extract_dir / "logon.csv", usecols=["user", "pc"])
    pc_counts = logon.groupby(["user", "pc"]).size().reset_index(name="count")
    most_freq = pc_counts.loc[pc_counts.groupby("user")["count"].idxmax()]

    user_df = ldap_df.merge(
        most_freq[["user", "pc"]], left_on="user_id", right_on="user", how="left",
    ).drop(columns=["user"])
    return user_df


# ---------------------------------------------------------------------------
# Step 2: Malicious activity IDs
# ---------------------------------------------------------------------------

def collect_malicious_ids(answers_dir: Path, dataset_version: str) -> set:
    """Collect all malicious activity IDs from answer files."""
    malicious_ids = set()
    for scenario_dir in sorted(answers_dir.glob(f"r{dataset_version}-*")):
        if not scenario_dir.is_dir():
            continue
        for answer_file in scenario_dir.glob("*.csv"):
            with open(answer_file) as f:
                for line in f:
                    match = re.search(r"\{[^}]+\}", line)
                    if match:
                        malicious_ids.add(match.group(0))
    return malicious_ids


# ---------------------------------------------------------------------------
# Step 3: User selection
# ---------------------------------------------------------------------------

def select_users(
    extract_dir: Path,
    answers_dir: Path,
    dataset_version: str,
    data_fraction: float,
    seed: int = 42,
) -> set | None:
    """Select a subset of users. Malicious users are always included."""
    if data_fraction >= 1.0:
        return None

    rng = np.random.RandomState(seed)
    all_users = set(
        pd.read_csv(extract_dir / "logon.csv", usecols=["user"])["user"].unique(),
    )
    insiders = pd.read_csv(answers_dir / "insiders.csv")
    mal_users = set(
        insiders[
            insiders["dataset"].astype(str).str.startswith(dataset_version)
        ]["user"].unique(),
    )
    non_mal = sorted(all_users - mal_users)
    n = max(1, int(len(non_mal) * data_fraction))
    sampled = set(rng.choice(non_mal, size=n, replace=False))
    selected = sampled | mal_users
    print(f"  {len(selected)} users: {len(mal_users)} malicious + {len(sampled)} non-malicious")
    return selected


# ---------------------------------------------------------------------------
# Step 4: Load raw sessions
# ---------------------------------------------------------------------------

def _load_filtered(
    filepath: Path,
    selected_users: set | None,
    chunksize: int = 100_000,
    **kwargs,
):
    """Load CSV, optionally filtering by user set via chunked reading."""
    if selected_users is None:
        return pd.read_csv(filepath, **kwargs)

    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize, **kwargs):
        filtered = chunk[chunk["user"].isin(selected_users)]
        if len(filtered) > 0:
            chunks.append(filtered)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_sessions(
    extract_dir: Path,
    malicious_ids: set,
    selected_users: set | None,
) -> dict[str, pd.DataFrame]:
    """Load raw CSVs into session DataFrames with parsed timestamps and labels."""
    sessions = {}
    sources = [
        ("logon", "logon.csv"),
        ("device", "device.csv"),
        ("file", "file.csv"),
        ("email", "email.csv"),
        ("http", "http.csv"),
    ]

    for name, filename in sources:
        print(f"  Loading {name}...")
        df = _load_filtered(extract_dir / filename, selected_users)
        if df.empty:
            sessions[name] = df
            print(f"    {name}: 0 rows")
            continue

        df["timestamp"] = pd.to_datetime(df["date"])
        df["hour"] = df["timestamp"].dt.hour
        df["is_malicious"] = df["id"].isin(malicious_ids).astype(int)
        df["user_id"] = df["user"]

        # HTTP has no 'activity' column in raw data
        if "activity" in df.columns:
            df["activity_type"] = df["activity"]
        else:
            df["activity_type"] = "WWW Visit"

        # File: ensure boolean columns are proper bools
        if name == "file":
            for col in ["to_removable_media", "from_removable_media"]:
                if df[col].dtype == object:
                    df[col] = df[col] == "True"

        sessions[name] = df
        n_mal = int(df["is_malicious"].sum())
        print(f"    {name}: {len(df):,} rows ({n_mal} malicious)")

    return sessions


# ---------------------------------------------------------------------------
# Step 5: Feature extraction
# ---------------------------------------------------------------------------

def extract_features(sessions: dict, extractor: GetFeature) -> dict:
    """Extract and pad features for each activity type."""
    funcs = {
        "logon": extractor.get_logon_feature,
        "device": extractor.get_device_feature,
        "file": extractor.get_file_feature,
        "email": extractor.get_email_feature,
        "http": extractor.get_http_feature,
    }

    for name, df in sessions.items():
        if df.empty:
            continue
        func = funcs[name]
        before, after = _PADDING[name]
        features = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {name}"):
            feat = func(row)
            features.append([0] * before + feat + [0] * after)

        df["feature"] = features
        sessions[name] = df.sort_values(by=["user", "timestamp"])

    return sessions


# ---------------------------------------------------------------------------
# Step 6: Combine and encode
# ---------------------------------------------------------------------------

def combine_and_encode(sessions: dict) -> tuple[pd.DataFrame, LabelEncoder]:
    """Combine all session types, encode activity types, compute activity codes."""
    for name, df in sessions.items():
        if not df.empty:
            df["source"] = name

    combined = pd.concat(
        [sessions[n] for n in FEATURE_DIMS if not sessions[n].empty],
        ignore_index=True,
    )

    encoder = LabelEncoder()
    combined["activity_type_id"] = encoder.fit_transform(combined["activity_type"])
    combined["activity_code"] = combined["activity_type_id"] * 24 + combined["hour"]
    combined = combined.sort_values(by=["user", "timestamp"])

    return combined, encoder


# ---------------------------------------------------------------------------
# Step 7: Hour merging and feature aggregation
# ---------------------------------------------------------------------------

def update_hours(df: pd.DataFrame, min_size: int = 5):
    """Merge hours with too few activities into nearest larger hour.

    Faithfully adapted from the source notebook's update_hours function.
    """
    changed = False
    counts = df.groupby(["user", "update_hour"]).size().reset_index(name="count")

    for _, row in counts.iterrows():
        if row["count"] < min_size:
            user = row["user"]
            current = row["update_hour"]
            possible = counts[
                (counts["user"] == user) & (counts["count"] >= min_size)
            ]["update_hour"]

            above = possible[possible > current]
            below = possible[possible < current]
            target = above.min() if above.any() else (below.max() if below.any() else None)

            if pd.notna(target):
                df.loc[
                    (df["user"] == user) & (df["update_hour"] == current),
                    "update_hour",
                ] = target
                changed = True

    return df, changed


def _sum_lists(lists):
    """Element-wise sum of a list of lists."""
    return [sum(x) for x in zip(*lists)]


def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (user, date, update_hour, activity_code) and sum features."""
    grouped = (
        df.groupby(["user", "date", "update_hour", "activity_code"])["feature"]
        .apply(lambda x: _sum_lists(x.tolist()))
        .reset_index(name="updated_feature_final")
    )
    return df.merge(
        grouped, on=["user", "date", "update_hour", "activity_code"], how="left",
    )


# ---------------------------------------------------------------------------
# Step 8: Graph creation
# ---------------------------------------------------------------------------

def build_activity_types_dict(combined: pd.DataFrame) -> dict[str, set]:
    """Map each activity category to its set of activity codes.

    Used to create implicit (activity-type) edges in the graph.
    """
    return {
        source: set(combined[combined["source"] == source]["activity_code"].unique())
        for source in FEATURE_DIMS
        if source in combined["source"].unique()
    }


def _create_graph(sequence, masked_activity, masked_label, features_dict, activity_types):
    """Create a PyG Data object from a masked activity session.

    Edges:
        1. Sequential: bidirectional edges between consecutive nodes
        2. Activity-type: fully connect nodes of the same activity category

    Adapted from the source notebook's create_graph function.
    """
    num_nodes = len(sequence)
    edges = []

    # Sequential connections (bidirectional)
    edges.extend((i, i + 1) for i in range(num_nodes - 1))
    edges.extend((i + 1, i) for i in range(num_nodes - 1))

    # Activity-type connections: fully connect nodes of the same category
    activity_nodes = defaultdict(list)
    for idx, code in enumerate(sequence):
        for act_type, codes in activity_types.items():
            if code in codes:
                activity_nodes[act_type].append(idx)

    for nodes in activity_nodes.values():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append((nodes[i], nodes[j]))
                edges.append((nodes[j], nodes[i]))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    features = np.array([features_dict[code] for code in sequence])
    feature_matrix = torch.tensor(features, dtype=torch.float)

    data = Data(x=feature_matrix, edge_index=edge_index)
    data.y_act = torch.tensor(masked_activity, dtype=torch.long)
    data.y_label = torch.tensor(masked_label, dtype=torch.long)
    return data


def create_all_graphs(
    combined: pd.DataFrame,
    activity_types: dict,
    min_session_size: int,
    max_session_size: int,
    processed_dir: Path,
    stream: bool = False,
) -> int:
    """Create graph training examples and save them in chunks to save RAM.

    Args:
        stream: If True, each chunk is pushed to GDrive via DvcChunkStore and
                deleted locally immediately after saving. Requires DVC remote
                to be configured and authenticated.
    """
    from certgnn.chunk_store import DvcChunkStore  # lazy import — not needed without --stream

    # Keep chunks well under the 2 GB zip64 ceiling of torch.save. At ~55 KB
    # per PyG Data object this yields ~550 MB files.
    CHUNK_SIZE = 10_000

    graph_list = []
    chunk_idx = 0
    total_graphs = 0
    store = DvcChunkStore(processed_dir) if stream else None

    processed_dir.mkdir(parents=True, exist_ok=True)
    grouped_users = combined.groupby("user_id")

    def _flush():
        nonlocal chunk_idx
        if not graph_list:
            return
        chunk_path = processed_dir / f"graph_chunk_{chunk_idx}.pt"
        torch.save(graph_list, chunk_path)
        if store is not None:
            store.push_chunk(chunk_path, num_graphs=len(graph_list), delete_local=True)
        chunk_idx += 1
        graph_list.clear()

    for user, user_data in tqdm(grouped_users, total=len(grouped_users), desc="  Graphs"):
        # note: user_data = combined[...] deleted to improve performance

        for session_hour in user_data["update_hour"].unique():
            sess = user_data[user_data["update_hour"] == session_hour].copy()
            sess["activity_code"] = sess["activity_code"].astype(int)

            if len(sess) < min_session_size:
                continue

            # Split into sub-sessions
            subs = [
                sess.iloc[i : i + max_session_size]
                for i in range(0, len(sess), max_session_size)
            ]

            # Merge small sub-sessions (faithful to original logic)
            i = 0
            while i < len(subs):
                if len(subs[i]) < min_session_size and i + 1 < len(subs):
                    subs[i] = pd.concat([subs[i], subs.pop(i + 1)])
                elif len(subs[i]) < min_session_size and i > 0:
                    subs[i - 1] = pd.concat([subs[i - 1], subs.pop(i)])
                else:
                    i += 1

            for sub in subs:
                if len(sub) < min_session_size:
                    continue

                # Build activity_code -> aggregated feature mapping
                unique = sub.drop_duplicates(subset="activity_code")
                feat_dict = dict(
                    zip(
                        unique["activity_code"].astype(int),
                        unique["updated_feature_final"],
                    ),
                )

                seq = sub["activity_code"].astype(int).tolist()
                labels = sub["is_malicious"].tolist()

                # Create one graph per masked position
                for mask_idx in range(len(seq)):
                    remaining = seq[:mask_idx] + seq[mask_idx + 1 :]
                    if len(remaining) < 2:
                        continue

                    graph = _create_graph(
                        remaining,
                        seq[mask_idx],
                        labels[mask_idx],
                        feat_dict,
                        activity_types,
                    )
                    graph_list.append(graph)
                    total_graphs += 1

                # Flush inside the session loop so a single high-activity
                # user can't balloon a chunk past the 2 GB torch.save limit.
                if len(graph_list) >= CHUNK_SIZE:
                    _flush()

    _flush()
    return total_graphs


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_processed(
    total_graphs: int,
    processed_dir: Path,
    encoder: LabelEncoder,
    activity_types: dict,
):
    """Save metadata and label encoder."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "num_graphs": total_graphs,
        "feature_dim": TOTAL_FEATURE_DIM,
        "num_activity_types": len(encoder.classes_),
        "num_classes": len(encoder.classes_) * 24,
        "activity_types": list(encoder.classes_),
        "activity_types_dict": {k: sorted(v) for k, v in activity_types.items()},
    }
    
    with open(processed_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    with open(processed_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print(f"  Saved metadata to {processed_dir}")
    print(
        f"  Metadata: feature_dim={TOTAL_FEATURE_DIM}, "
        f"num_classes={metadata['num_classes']}, "
        f"total_graphs={total_graphs}"
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the full preprocessing pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Push each chunk to GDrive via DVC and delete it locally after saving.",
    )
    args = parser.parse_args()

    config = load_config()
    root = get_project_root()

    prep = config.get("preprocessing", {})
    data_fraction = prep.get("data_fraction", 1.0)
    dataset_version = prep.get("dataset_version", "5.2")
    min_session = prep.get("min_session_size", 5)
    max_session = prep.get("max_session_size", 50)
    seed = prep.get("seed", 42)

    extract_dir = root / config["paths"]["extract_dir"]
    processed_dir = root / config["paths"]["processed_dir"]
    answers_dir = root / config["paths"]["raw_dir"] / "answers"

    np.random.seed(seed)

    # 1. User-PC mapping
    print("[1/8] Building user-PC mapping...")
    user_df = build_user_pc_mapping(extract_dir)
    print(f"  {len(user_df)} users")

    # 2. Malicious IDs
    print("[2/8] Collecting malicious activity IDs...")
    mal_ids = collect_malicious_ids(answers_dir, dataset_version)
    print(f"  {len(mal_ids)} malicious IDs")

    # 3. User selection
    print(f"[3/8] Selecting users (data_fraction={data_fraction})...")
    selected = select_users(
        extract_dir, answers_dir, dataset_version, data_fraction, seed,
    )

    # 4. Load sessions
    print("[4/8] Loading raw data...")
    sessions = load_sessions(extract_dir, mal_ids, selected)

    # 5. Feature extraction
    print("[5/8] Extracting features...")
    extractor = GetFeature(user_df)
    sessions = extract_features(sessions, extractor)

    # 6. Combine and encode
    print("[6/8] Combining and encoding...")
    combined, encoder = combine_and_encode(sessions)
    print(
        f"  {len(combined):,} activities, "
        f"{len(encoder.classes_)} types: {list(encoder.classes_)}",
    )

    # 7. Update hours and aggregate
    print("[7/8] Hour merging and feature aggregation...")
    combined["update_hour"] = combined["hour"]
    combined, _ = update_hours(combined, min_session)
    combined["date"] = combined["timestamp"].dt.date
    combined = aggregate_features(combined)

    # 8. Graph creation
    stream_msg = " (streaming to GDrive)" if args.stream else ""
    print(f"[8/8] Creating graphs (saving in chunks){stream_msg}...")
    act_types = build_activity_types_dict(combined)
    total_graphs = create_all_graphs(
        combined, act_types, min_session, max_session, processed_dir, stream=args.stream,
    )

    # Save
    print("Saving metadata...")
    save_processed(total_graphs, processed_dir, encoder, act_types)

    print(f"\nDone! Successfully created and saved {total_graphs} graphs.")


if __name__ == "__main__":
    main()
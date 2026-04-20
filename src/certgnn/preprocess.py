"""Preprocessing pipeline for GNN Insider Threat Detection using DuckDB.

Adapted from the source paper's 01_feature_extraction.ipynb
but uses DuckDB for efficient feature extraction.
Loads raw CMU CERT data, extracts features efficiently with SQL,
creates session graphs, and saves processed data ready for training.

Usage:
    uv run preprocess           # save chunks locally
    uv run preprocess --stream  # push each chunk to GDrive and delete locally
"""

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from tqdm import tqdm

from certgnn.utils import get_project_root, load_config
from certgnn.duckdb_features import (
    initialize_duckdb,
    LOGON_SQL,
    DEVICE_SQL,
    FILE_SQL,
    EMAIL_SQL,
    HTTP_SQL,
)

# Feature vector layout: [logon(4) | device(6) | file(16) | email(22) | http(6)] = 54
FEATURE_DIMS = {"logon": 4, "device": 6, "file": 16, "email": 22, "http": 6}
TOTAL_FEATURE_DIM = sum(FEATURE_DIMS.values())

# Pre-computed padding (before, after) for each activity type
_PADDING = {}
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
    # Use DuckDB for performance
    con = duckdb.connect(":memory:")
    query = f"""
    WITH RankedLDAP AS (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY filename DESC) as rn
        FROM read_csv_auto('{extract_dir}/LDAP/*.csv', filename=true)
    ),
    LatestLDAP AS (
        SELECT * EXCLUDE(rn, filename) FROM RankedLDAP WHERE rn = 1
    ),
    LogonCounts AS (
        SELECT user, pc, COUNT(*) as cnt
        FROM read_csv_auto('{extract_dir}/logon.csv')
        GROUP BY user, pc
    ),
    RankedLogons AS (
        SELECT user, pc, cnt, ROW_NUMBER() OVER(PARTITION BY user ORDER BY cnt DESC) as rn
        FROM LogonCounts
    ),
    MostFreqPC AS (
        SELECT user, pc FROM RankedLogons WHERE rn = 1
    )
    SELECT l.*, m.pc as pc_most_freq
    FROM LatestLDAP l
    LEFT JOIN MostFreqPC m ON l.user_id = m.user
    """
    user_df = con.execute(query).df()
    user_df = user_df.rename(columns={"pc_most_freq": "pc"})
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
    frac_normal_users: float,
    seed: int = 42,
    frac_malicious_users: float = 1.0,
) -> set | None:
    """Select a subset of users.

    Args:
        frac_normal_users: Fraction of non-malicious users to include (0.0-1.0).
        frac_malicious_users: Fraction of malicious users to sample (0.0-1.0). 1.0 = all.
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
        insiders[insiders["dataset"].astype(str).str.startswith(dataset_version)][
            "user"
        ].unique()
    )

    # Sample malicious users based on fraction
    if frac_malicious_users < 1.0 and len(mal_users) > 0:
        mal_count = max(1, int(len(mal_users) * frac_malicious_users))
        mal_users = set(
            rng.choice(sorted(mal_users), size=mal_count, replace=False)
        )

    non_mal = sorted(all_users - mal_users)
    n = max(1, int(len(non_mal) * frac_normal_users))
    sampled = set(rng.choice(non_mal, size=n, replace=False))
    selected = sampled | mal_users
    print(
        f"  {len(selected)} users: {len(mal_users)} malicious + {len(sampled)} non-malicious"
    )
    return selected


# ---------------------------------------------------------------------------
# Step 4: Load raw sessions & Step 5: Extract features with DuckDB SQL
# ---------------------------------------------------------------------------
def create_padded_feature_sql(base_sql: str, source: str) -> str:
    before, after = _PADDING[source]
    before_pad = (
        f"CAST([{', '.join(['0.0'] * before)}] AS DOUBLE[]) || " if before > 0 else ""
    )
    after_pad = (
        f" || CAST([{', '.join(['0.0'] * after)}] AS DOUBLE[])" if after > 0 else ""
    )

    fields = ["super_pc_acess", "other_pc_acess", "after_hour", "week"]
    if source == "device":
        fields.extend(["conn", "dis_conn"])
    elif source == "file":
        fields.extend(
            [
                "to_rem",
                "f_rem",
                "f_open",
                "f_write",
                "f_delete",
                "f_copy",
                "f_type_comp",
                "f_type_img",
                "f_type_doc",
                "f_type_file",
                "f_type_exec",
                "f_type_other",
            ]
        )
    elif source == "email":
        fields.extend(
            [
                "to_out_count",
                "to_in_count",
                "bcc_out_count",
                "bcc_in_count",
                "cc_out_count",
                "cc_in_count",
            ]
        )
    elif source == "http":
        fields.extend(["flag_url", "flag_word"])

    cast_fields = [f"CAST({f} AS DOUBLE)" for f in fields]

    feature_calc = f"""
        {before_pad} 
        [{", ".join(cast_fields)}]
        {"" if source != "email" else " || CAST(attachment_features AS DOUBLE[])"}
        {after_pad}
    """

    return f"""
    SELECT id, timestamp, user_id, pc, activity_type, source, is_malicious, 
        {feature_calc} AS feature
    FROM (
        {base_sql}
    ) sub
    """


def process_and_dump_to_parquet(
    extract_dir: Path,
    processed_dir: Path,
    user_df: pd.DataFrame,
    malicious_ids: set,
    selected_users: set | None,
):
    print("  Initializing DuckDB ETL...")
    con = duckdb.connect(":memory:")
    con.register("user_df", user_df)
    con.execute("CREATE TABLE users AS SELECT * FROM user_df")

    mal_df = pd.DataFrame({"id": list(malicious_ids)})
    con.register("mal_df", mal_df)
    con.execute("CREATE TABLE mal_ids AS SELECT id FROM mal_df")

    sel_filter = ""
    if selected_users is not None:
        sel_df = pd.DataFrame({"user": list(selected_users)})
        con.register("sel_df", sel_df)
        con.execute("CREATE TABLE sel_users AS SELECT user FROM sel_df")
        sel_filter = "WHERE e.user IN (SELECT user FROM sel_users)"

    initialize_duckdb(con)
    parquet_path = processed_dir / "all_features.parquet"

    # HTTP activity doesn't have an activity column, so we default to 'WWW Visit'
    sources = [
        ("logon", "logon.csv", LOGON_SQL, "e.activity"),
        ("device", "device.csv", DEVICE_SQL, "e.activity"),
        ("file", "file.csv", FILE_SQL, "e.activity"),
        ("email", "email.csv", EMAIL_SQL, "e.activity"),
        ("http", "http.csv", HTTP_SQL, "'WWW Visit'"),
    ]

    for source_name, filename, base_sql, activity_col in sources:
        filepath = extract_dir / filename
        if not filepath.exists():
            continue

        # Read the file as events
        con.execute(
            f"CREATE OR REPLACE VIEW events_{source_name} AS SELECT * FROM read_csv_auto('{filepath}', all_varchar=true, ignore_errors=true) e {sel_filter}"
        )

        modified_base_sql = base_sql.replace(
            "SELECT",
            f"""SELECT 
                e.id,
                strptime(e.date, '%m/%d/%Y %H:%M:%S') AS timestamp,
                e.user AS user_id,
                e.pc,
                {activity_col} AS activity_type,
                '{source_name}' AS source,
                CASE WHEN e.id IN (SELECT id FROM mal_ids) THEN 1 ELSE 0 END AS is_malicious,
            """,
            1,
        ).replace("FROM events e", f"FROM events_{source_name} e")

        final_query = create_padded_feature_sql(modified_base_sql, source_name)

        print(f"  Processing {source_name} and dumping to Parquet: {parquet_path} ...")
        # DuckDB 0.10+ supports COPY (query) TO ... with APPEND or we can just insert into a table and then copy.
        # Let's write to a table first, then we dump the table at the end.
        con.execute(f"CREATE TABLE IF NOT EXISTS all_features AS {final_query} LIMIT 0")
        con.execute(f"INSERT INTO all_features {final_query}")

    print(f"  Saving Parquet to {parquet_path} ...")
    con.execute(
        f"COPY (SELECT id, timestamp, user_id, pc, activity_type, source, is_malicious, feature FROM all_features) TO '{parquet_path}' (FORMAT PARQUET)"
    )
    return parquet_path


# ---------------------------------------------------------------------------
# Step 6: Combine and encode
# ---------------------------------------------------------------------------
def combine_and_encode_parquet(parquet_path: Path) -> tuple[pd.DataFrame, LabelEncoder]:
    """Read Parquet, encode activity types, and compute activity codes."""
    con = duckdb.connect(":memory:")
    df = con.execute(f"SELECT * FROM '{parquet_path}' ORDER BY user_id, timestamp").df()

    encoder = LabelEncoder()
    df["activity_type_id"] = encoder.fit_transform(df["activity_type"])
    df["hour"] = df["timestamp"].dt.hour
    df["activity_code"] = df["activity_type_id"] * 24 + df["hour"]
    return df, encoder


# ---------------------------------------------------------------------------
# Step 7: Hour merging and feature aggregation
# ---------------------------------------------------------------------------
def update_hours(df: pd.DataFrame, min_size: int = 5):
    """Merge hours with too few activities into nearest larger hour.

    Faithfully adapted from the source notebook's update_hours function.
    """
    changed = False
    counts = df.groupby(["user_id", "update_hour"]).size().reset_index(name="count")

    for _, row in counts.iterrows():
        if row["count"] < min_size:
            user = row["user_id"]
            current = row["update_hour"]
            possible = counts[
                (counts["user_id"] == user) & (counts["count"] >= min_size)
            ]["update_hour"]

            above = possible[possible > current]
            below = possible[possible < current]
            target = (
                above.min() if above.any() else (below.max() if below.any() else None)
            )

            if pd.notna(target):
                df.loc[
                    (df["user_id"] == user) & (df["update_hour"] == current),
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
        df.groupby(["user_id", "date", "update_hour", "activity_code"])["feature"]
        .apply(lambda x: _sum_lists(x.tolist()))
        .reset_index(name="updated_feature_final")
    )
    return df.merge(
        grouped,
        on=["user_id", "date", "update_hour", "activity_code"],
        how="left",
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


def _create_graph(
    sequence, masked_activity, masked_label, features_dict, activity_types
):
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
    keep_local: bool = False,
) -> int:
    """Create graph training examples and save them in chunks to save RAM.

    Args:
        stream: If True, each chunk is pushed to GDrive via DvcChunkStore.
                Requires DVC remote to be configured and authenticated.
        keep_local: If True (with --stream), keeps local .pt files after push.
                    If False, deletes them after push to save disk space.
    """
    from certgnn.chunk_store import (
        DvcChunkStore,
    )  # lazy import — not needed without --stream

    # Keep chunks well under the 2 GB zip64 ceiling of torch.save. At ~55 KB
    # per PyG Data object this yields ~550 MB files.
    CHUNK_SIZE = 10_000

    graph_list = []
    chunk_idx = 0
    total_graphs = 0
    total_activities = 0
    total_sessions = 0
    total_subsessions = 0
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
            store.push_chunk(chunk_path, num_graphs=len(graph_list), delete_local=not keep_local)
        chunk_idx += 1
        graph_list.clear()

    for user, user_data in tqdm(
        grouped_users, total=len(grouped_users), desc="  Graphs"
    ):
        for session_hour in user_data["update_hour"].unique():
            sess = user_data[user_data["update_hour"] == session_hour].copy()
            sess["activity_code"] = sess["activity_code"].astype(int)
            if len(sess) < min_session_size:
                continue

            total_sessions += 1
            total_activities += len(sess)

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

                total_subsessions += 1

                # Build activity_code -> aggregated feature mapping
                unique = sub.drop_duplicates(subset="activity_code")
                feat_dict = dict(
                    zip(
                        unique["activity_code"].astype(int),
                        unique["updated_feature_final"],
                    )
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

    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"Graph Creation Summary")
    print(f"{'='*70}")
    print(f"  Total activities processed:    {total_activities:>12,}")
    print(f"  Total sessions (by hour):      {total_sessions:>12,}")
    print(f"  Total sub-sessions:            {total_subsessions:>12,}")
    print(f"  Total graphs created:          {total_graphs:>12,}")
    print(f"  Avg activities per session:    {total_activities/max(1, total_sessions):>12.1f}")
    print(f"  Avg activities per subsession: {total_activities/max(1, total_subsessions):>12.1f}")
    print(f"  Avg graphs per subsession:     {total_graphs/max(1, total_subsessions):>12.1f}")
    print(f"  Chunks created:                {chunk_idx:>12,}")
    print(f"  Approx disk usage:             {chunk_idx * 0.55:>12.1f} GB")
    print(f"{'='*70}\n")

    return total_graphs


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_processed(
    total_graphs: int, processed_dir: Path, encoder: LabelEncoder, activity_types: dict
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


def main():
    """Run the full preprocessing pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Push each chunk to GDrive via DVC and delete it locally after saving.",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="When used with --stream, keeps local .pt files after pushing to GDrive.",
    )
    args = parser.parse_args()

    config = load_config()
    root = get_project_root()
    prep = config.get("preprocessing", {})
    frac_normal_users = prep.get("frac_normal_users", 1.0)
    frac_malicious_users = prep.get("frac_malicious_users", 1.0)
    dataset_version = prep.get("dataset_version", "5.2")
    min_session = prep.get("min_session_size", 5)
    max_session = prep.get("max_session_size", 50)
    seed = prep.get("seed", 42)
    keep_local = args.keep_local or prep.get("keep_local", False)

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
        extract_dir,
        answers_dir,
        dataset_version,
        frac_normal_users,
        seed,
        frac_malicious_users=frac_malicious_users,
    )

    print("[4/8 & 5/8] Extracting features via DuckDB SQL to Parquet...")
    parquet_path = process_and_dump_to_parquet(
        extract_dir, processed_dir, user_df, mal_ids, selected
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
    act_types = build_activity_types_dict(combined)
    total_graphs = create_all_graphs(
        combined, act_types, min_session, max_session, processed_dir, stream=args.stream, keep_local=keep_local
    )

    print("Saving metadata...")
    save_processed(total_graphs, processed_dir, encoder, act_types)
    print(f"\nDone! Successfully created and saved {total_graphs} graphs.")


if __name__ == "__main__":
    main()

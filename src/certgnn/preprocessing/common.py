"""Shared preprocessing pipeline pieces used by both variants.

Both ``paper_faithful`` and ``user_level_split`` consume the same upstream
DuckDB pipeline:

    raw CSVs ──> user_pc map + malicious IDs ──> SQL feature extraction
        ──> all_features.parquet ──> combine/encode/hour-merge/aggregate
        ──> ``combined`` DataFrame ready for graph creation.

The variants only diverge in:
  * user selection (fraction vs fixed-count + scenario balancing),
  * whether per-user split assignment is computed,
  * how chunks are buffered and named (single vs train/val/test).
"""

from __future__ import annotations

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

from certgnn.preprocessing.features_duckdb import (
    DEVICE_SQL,
    EMAIL_SQL,
    FILE_SQL,
    HTTP_SQL,
    LOGON_SQL,
    initialize_duckdb,
)

# Feature vector layout: [logon(4) | device(6) | file(16) | email(22) | http(6)] = 54
FEATURE_DIMS = {"logon": 4, "device": 6, "file": 16, "email": 22, "http": 6}
TOTAL_FEATURE_DIM = sum(FEATURE_DIMS.values())

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
    return user_df.rename(columns={"pc_most_freq": "pc"})


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
# Steps 4 & 5: SQL feature extraction → all_features.parquet
# ---------------------------------------------------------------------------
def _create_padded_feature_sql(base_sql: str, source: str) -> str:
    """Wrap a per-source feature SQL with the global 54-dim padding."""
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
        fields.extend([
            "to_rem", "f_rem", "f_open", "f_write", "f_delete", "f_copy",
            "f_type_comp", "f_type_img", "f_type_doc", "f_type_file",
            "f_type_exec", "f_type_other",
        ])
    elif source == "email":
        fields.extend([
            "to_out_count", "to_in_count", "bcc_out_count", "bcc_in_count",
            "cc_out_count", "cc_in_count",
        ])
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
) -> Path:
    """Extract per-source features via DuckDB SQL and dump to a single Parquet."""
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

    # Activity-type labels follow the paper (sec. IV-A) and the authors'
    # reference preprocessing (preprocess_df): logon/device/file keep their
    # raw activity, but email and http each collapse to a single type, and
    # File Delete/Copy merge into File Write. This yields 8 distinct activity
    # types -> 8 * 24 = 192 activity codes (the model's output dimension).
    # Using the raw email activity ('Send'/'Receive') instead would add a 9th
    # type and inflate the class space to 216, diverging from the paper.
    sources = [
        ("logon", "logon.csv", LOGON_SQL, "e.activity"),
        ("device", "device.csv", DEVICE_SQL, "e.activity"),
        ("file", "file.csv", FILE_SQL,
         "CASE WHEN e.activity IN ('File Delete', 'File Copy') THEN 'File Write' ELSE e.activity END"),
        ("email", "email.csv", EMAIL_SQL, "'email'"),
        ("http", "http.csv", HTTP_SQL, "'http'"),
    ]

    for source_name, filename, base_sql, activity_col in sources:
        filepath = extract_dir / filename
        if not filepath.exists():
            continue

        con.execute(
            f"CREATE OR REPLACE VIEW events_{source_name} AS "
            f"SELECT * FROM read_csv_auto('{filepath}', all_varchar=true, ignore_errors=true) e {sel_filter}"
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

        final_query = _create_padded_feature_sql(modified_base_sql, source_name)

        print(f"  Processing {source_name} -> {parquet_path.name} ...")
        con.execute(f"CREATE TABLE IF NOT EXISTS all_features AS {final_query} LIMIT 0")
        con.execute(f"INSERT INTO all_features {final_query}")

    print(f"  Saving Parquet to {parquet_path} ...")
    con.execute(
        f"COPY (SELECT id, timestamp, user_id, pc, activity_type, source, is_malicious, feature "
        f"FROM all_features) TO '{parquet_path}' (FORMAT PARQUET)"
    )
    return parquet_path


# ---------------------------------------------------------------------------
# Step 6: Combine and encode
# ---------------------------------------------------------------------------
def combine_and_encode_parquet(parquet_path: Path) -> tuple[pd.DataFrame, LabelEncoder]:
    """Read Parquet, encode activity types, compute activity codes."""
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
def update_hours(df: pd.DataFrame, min_size: int = 5) -> tuple[pd.DataFrame, bool]:
    """Merge hours with too few activities into the nearest larger hour.

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
            target = above.min() if above.any() else (below.max() if below.any() else None)

            if pd.notna(target):
                df.loc[
                    (df["user_id"] == user) & (df["update_hour"] == current),
                    "update_hour",
                ] = target
                changed = True
    return df, changed


def _sum_lists(lists):
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
# Step 8: Graph creation primitives
# ---------------------------------------------------------------------------
def build_activity_types_dict(combined: pd.DataFrame) -> dict[str, set]:
    """Map each activity category to its set of activity codes."""
    return {
        source: set(combined[combined["source"] == source]["activity_code"].unique())
        for source in FEATURE_DIMS
        if source in combined["source"].unique()
    }


def create_graph(
    sequence,
    masked_activity,
    masked_label,
    features_dict,
    activity_types,
    user_id: str | None = None,
    split_name: str | None = None,
) -> Data:
    """Create a PyG Data object from a masked activity session.

    Edges:
        1. Sequential: bidirectional edges between consecutive nodes
        2. Activity-type: fully connect nodes of the same activity category

    ``user_id`` and ``split_name`` are stored as Data attributes when supplied
    (used by the user-level-split variant for traceability and debugging).
    """
    num_nodes = len(sequence)
    edges = []

    edges.extend((i, i + 1) for i in range(num_nodes - 1))
    edges.extend((i + 1, i) for i in range(num_nodes - 1))

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
    if user_id is not None:
        data.user_id = user_id
    if split_name is not None:
        data.split = split_name
    return data


def iter_subsessions(
    user_data: pd.DataFrame,
    min_session_size: int,
    max_session_size: int,
):
    """Yield ``(sub, feat_dict, seq, labels)`` tuples per processable sub-session.

    Splits the user's activity into per-hour sessions, divides each into
    sub-sessions of up to ``max_session_size``, merges short tails forward
    or backward, and skips anything that ends below ``min_session_size``.
    """
    for session_hour in user_data["update_hour"].unique():
        sess = user_data[user_data["update_hour"] == session_hour].copy()
        sess["activity_code"] = sess["activity_code"].astype(int)
        if len(sess) < min_session_size:
            continue

        subs = [
            sess.iloc[i : i + max_session_size]
            for i in range(0, len(sess), max_session_size)
        ]
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
            unique = sub.drop_duplicates(subset="activity_code")
            feat_dict = dict(
                zip(
                    unique["activity_code"].astype(int),
                    unique["updated_feature_final"],
                )
            )
            seq = sub["activity_code"].astype(int).tolist()
            labels = sub["is_malicious"].tolist()
            yield sub, feat_dict, seq, labels


# ---------------------------------------------------------------------------
# High-level pipeline orchestration shared by both variants
# ---------------------------------------------------------------------------
def load_user_df_and_malicious_ids(
    extract_dir: Path,
    answers_dir: Path,
    dataset_version: str,
) -> tuple[pd.DataFrame, set]:
    """Run the variant-independent prologue: user-PC map + malicious IDs."""
    print("Building user-PC mapping...")
    user_df = build_user_pc_mapping(extract_dir)
    print(f"  {len(user_df)} users")

    print("Collecting malicious activity IDs...")
    mal_ids = collect_malicious_ids(answers_dir, dataset_version)
    print(f"  {len(mal_ids)} malicious IDs")
    return user_df, mal_ids


def build_combined_dataframe(
    extract_dir: Path,
    processed_dir: Path,
    user_df: pd.DataFrame,
    malicious_ids: set,
    selected_users: set | None,
    min_session_size: int,
) -> tuple[pd.DataFrame, LabelEncoder, dict[str, set]]:
    """Run the shared DuckDB → Parquet → encode → aggregate pipeline.

    Returns the ``combined`` DataFrame ready for graph creation, the fitted
    ``LabelEncoder``, and the activity-types lookup. Identical for both
    variants — only the upstream user selection differs.
    """
    print("Extracting features via DuckDB SQL to Parquet...")
    parquet_path = process_and_dump_to_parquet(
        extract_dir, processed_dir, user_df, malicious_ids, selected_users,
    )

    print("Loading from Parquet and encoding...")
    combined, encoder = combine_and_encode_parquet(parquet_path)
    print(f"  {len(combined):,} activities, {len(encoder.classes_)} activity types")

    print("Hour merging and feature aggregation...")
    combined["update_hour"] = combined["hour"]
    combined, _ = update_hours(combined, min_session_size)
    combined["date"] = combined["timestamp"].dt.date
    combined = aggregate_features(combined)

    activity_types = build_activity_types_dict(combined)
    return combined, encoder, activity_types


# ---------------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------------
def save_processed(
    total_graphs: int,
    processed_dir: Path,
    encoder: LabelEncoder,
    activity_types: dict,
    user_splits: dict[str, list[str]] | None = None,
) -> None:
    """Save metadata.pkl and label_encoder.pkl."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "num_graphs": total_graphs,
        "feature_dim": TOTAL_FEATURE_DIM,
        "num_activity_types": len(encoder.classes_),
        "num_classes": len(encoder.classes_) * 24,
        "activity_types": list(encoder.classes_),
        "activity_types_dict": {k: sorted(v) for k, v in activity_types.items()},
    }
    if user_splits is not None:
        metadata["user_splits"] = {key: len(value) for key, value in user_splits.items()}

    with open(processed_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    with open(processed_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print(f"  Saved metadata to {processed_dir}")
    print(
        f"  Metadata: feature_dim={TOTAL_FEATURE_DIM}, "
        f"num_classes={metadata['num_classes']}, total_graphs={total_graphs}"
    )

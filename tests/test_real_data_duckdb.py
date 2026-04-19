import pandas as pd
import duckdb
import numpy as np
from certgnn.utils import get_project_root
from certgnn.feature_extraction import GetFeature
from certgnn.duckdb_features import (
    initialize_duckdb,
    LOGON_SQL,
    DEVICE_SQL,
    FILE_SQL,
    EMAIL_SQL,
    HTTP_SQL,
)
from certgnn.preprocess import build_user_pc_mapping


def test_real_data_equivalence():
    root = get_project_root()
    extract_dir = root / "data/raw/cmu_cert_r5.2/r5.2"

    print("Loading user mapping...")
    user_df = build_user_pc_mapping(extract_dir)
    get_feature = GetFeature(user_df)

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE users AS SELECT * FROM user_df")
    initialize_duckdb(con)

    files = {
        "logon": ("logon.csv", get_feature.get_logon_feature, LOGON_SQL),
        "device": ("device.csv", get_feature.get_device_feature, DEVICE_SQL),
        "file": ("file.csv", get_feature.get_file_feature, FILE_SQL),
        "email": ("email.csv", get_feature.get_email_feature, EMAIL_SQL),
        "http": ("http.csv", get_feature.get_http_feature, HTTP_SQL),
    }

    for name, (filename, py_func, sql_query) in files.items():
        print(f"\n--- Testing {name} on real data (1000 rows) ---")
        filepath = extract_dir / filename
        if not filepath.exists():
            print(f"Skipping {name}, file not found.")
            continue

        # Load 1000 rows
        df = pd.read_csv(filepath, nrows=1000)
        df = df.sort_values(by="id").reset_index(drop=True)
        df["timestamp"] = pd.to_datetime(df["date"])

        if "activity" in df.columns:
            df["activity_type"] = df["activity"]
        else:
            df["activity_type"] = "WWW Visit"

        if name == "file":
            for col in ["to_removable_media", "from_removable_media"]:
                if df[col].dtype == object:
                    df[col] = df[col] == "True"

        # 1. Pandas Extraction
        py_features = []
        for _, row in df.iterrows():
            py_features.append(py_func(row))

        # 2. DuckDB Extraction
        # Create events table
        con.execute("DROP TABLE IF EXISTS events")
        con.execute("CREATE TABLE events AS SELECT * FROM df")

        # Execute SQL
        db_res = con.execute(sql_query + " ORDER BY e.id").fetchall()

        if name == "email":
            # Flatten UDF output for email
            db_features = [list(row[:10]) + row[10] for row in db_res]
        else:
            db_features = [list(row) for row in db_res]

        # 3. Compare
        assert len(py_features) == len(db_features), (
            f"Length mismatch: {len(py_features)} vs {len(db_features)}"
        )

        mismatches = 0
        for i, (py_feat, db_feat) in enumerate(zip(py_features, db_features)):
            # Use np.allclose for numeric comparison to handle potential float differences
            if not np.allclose(py_feat, db_feat, rtol=1e-5, atol=1e-5):
                mismatches += 1
                if mismatches <= 3:
                    print(f"Mismatch at row {i}:")
                    print(f"  Py: {py_feat}")
                    print(f"  DB: {db_feat}")
                    print(f"  Row data: {df.iloc[i].to_dict()}")

        assert mismatches == 0, f"{name}: {mismatches} mismatches found!"
        print(f"✅ {name}: 1000/1000 rows match perfectly!")


if __name__ == "__main__":
    test_real_data_equivalence()

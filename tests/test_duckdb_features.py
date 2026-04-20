import pandas as pd
from datetime import datetime
import duckdb

# Importujemy starą implementację Pythona dla zapewnienia identyczności wyników z nową
from certgnn.feature_extraction import GetFeature
from certgnn.duckdb_features import (
    initialize_duckdb,
    extract_logon_features,
    extract_device_features,
    extract_file_features,
    extract_email_features,
    extract_http_features,
)

mock_users = [
    {"user_id": "U1", "pc": "PC-1", "supervisor": "Boss", "employee_name": "User One"},
    {
        "user_id": "BOSS1",
        "pc": "PC-BOSS",
        "supervisor": "BossBoss",
        "employee_name": "Boss",
    },
]
user_df = pd.DataFrame(mock_users)


def test_duckdb_vs_pandas():
    get_feature = GetFeature(user_df)
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE users AS SELECT * FROM user_df")
    initialize_duckdb(con)

    events = [
        {
            "id": 1,
            "user": "U1",
            "pc": "PC-BOSS",
            "date": "10/24/2023 07:30:00",
            "timestamp": datetime(2023, 10, 24, 7, 30),
            "activity": "Logon",
            "activity_type": "Logon",
            "to_removable_media": False,
            "from_removable_media": False,
            "filename": None,
            "to": None,
            "bcc": None,
            "cc": None,
            "attachments": None,
            "url": None,
            "content": None,
        },
        {
            "id": 2,
            "user": "U1",
            "pc": "PC-UNKNOWN",
            "date": "10/28/2023 12:00:00",
            "timestamp": datetime(2023, 10, 28, 12, 00),
            "activity": "Connect",
            "activity_type": "Connect",
            "to_removable_media": False,
            "from_removable_media": False,
            "filename": None,
            "to": None,
            "bcc": None,
            "cc": None,
            "attachments": None,
            "url": None,
            "content": None,
        },
        {
            "id": 3,
            "user": "U1",
            "pc": "PC-1",
            "date": "10/24/2023 10:00:00",
            "timestamp": datetime(2023, 10, 24, 10, 00),
            "activity": "File Copy",
            "activity_type": "File Copy",
            "to_removable_media": True,
            "from_removable_media": False,
            "filename": "C:\\path\\secret.zip",
            "to": None,
            "bcc": None,
            "cc": None,
            "attachments": None,
            "url": None,
            "content": None,
        },
        {
            "id": 4,
            "user": "U1",
            "pc": "PC-1",
            "date": "10/24/2023 10:00:00",
            "timestamp": datetime(2023, 10, 24, 10, 00),
            "activity": "Email",
            "activity_type": "Email",
            "to_removable_media": False,
            "from_removable_media": False,
            "filename": None,
            "to": "boss@dtaa.com;hacker@evil.com",
            "bcc": "hidden@dtaa.com",
            "cc": "nan",
            "attachments": "file.zip(100);image.png(50)",
            "url": None,
            "content": None,
        },
        {
            "id": 5,
            "user": "U1",
            "pc": "PC-1",
            "date": "10/24/2023 10:00:00",
            "timestamp": datetime(2023, 10, 24, 10, 00),
            "activity": "Http",
            "activity_type": "Http",
            "to_removable_media": False,
            "from_removable_media": False,
            "filename": None,
            "to": None,
            "bcc": None,
            "cc": None,
            "attachments": None,
            "url": "http://wikileaks.org/secret",
            "content": "Looking for a new job with keylogging skills",
        },
    ]

    events_df = pd.DataFrame(events)
    con.execute("CREATE TABLE events_all AS SELECT * FROM events_df")
    con.execute("CREATE VIEW events AS SELECT * FROM events_all")

    logon_row = events_df.iloc[0]
    py_logon = get_feature.get_logon_feature(logon_row)
    con.execute(
        "CREATE OR REPLACE VIEW events AS SELECT * FROM events_all WHERE id = 1"
    )
    db_logon = extract_logon_features(con)[0]
    assert py_logon == db_logon, f"Logon mismatch: {py_logon} != {db_logon}"

    device_row = events_df.iloc[1]
    py_device = get_feature.get_device_feature(device_row)
    con.execute(
        "CREATE OR REPLACE VIEW events AS SELECT * FROM events_all WHERE id = 2"
    )
    db_device = extract_device_features(con)[0]
    assert py_device == db_device, f"Device mismatch: {py_device} != {db_device}"

    file_row = events_df.iloc[2]
    py_file = get_feature.get_file_feature(file_row)
    con.execute(
        "CREATE OR REPLACE VIEW events AS SELECT * FROM events_all WHERE id = 3"
    )
    db_file = extract_file_features(con)[0]
    assert py_file == db_file, f"File mismatch: {py_file} != {db_file}"

    email_row = events_df.iloc[3]
    py_email = get_feature.get_email_feature(email_row)
    con.execute(
        "CREATE OR REPLACE VIEW events AS SELECT * FROM events_all WHERE id = 4"
    )
    db_email = extract_email_features(con)[0]
    assert py_email == db_email, f"Email mismatch: {py_email} != {db_email}"

    http_row = events_df.iloc[4]
    py_http = get_feature.get_http_feature(http_row)
    con.execute(
        "CREATE OR REPLACE VIEW events AS SELECT * FROM events_all WHERE id = 5"
    )
    db_http = extract_http_features(con)[0]
    assert py_http == db_http, f"HTTP mismatch: {py_http} != {db_http}"

    print("All tests passed successfully!")

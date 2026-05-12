import duckdb
from urllib.parse import urlparse


def initialize_duckdb(con: duckdb.DuckDBPyConnection):
    """Sets up Python UDFs inside DuckDB for logic too complex for pure SQL."""

    # 1. Attachment Size Logic (Ported exactly from ProcessFeature)
    def attachment_size_type(attachments: str) -> list:
        if not attachments or str(attachments) == "nan":
            return [0] * 12

        f_type_comp, f_type_img, f_type_doc = 0, 0, 0
        f_type_file, f_type_exec, f_type_other = 0, 0, 0
        f_type_comp_size, f_type_img_size, f_type_doc_size = 0, 0, 0
        f_type_file_size, f_type_exec_size, f_type_other_size = 0, 0, 0

        new_f_path = attachments.split(";")
        for f_path_size in new_f_path:
            try:
                split_ex_data = f_path_size.split(".")[1]
                exten = split_ex_data[0:3]
                size_str = split_ex_data[4:-1]
                size = int(size_str) if size_str.isdigit() else 0
            except IndexError:
                continue

            if exten in ["zip", "rar", "7z"]:
                f_type_comp += 1
                f_type_comp_size += size
            elif exten in ["jpg", "png", "bmp"]:
                f_type_img += 1
                f_type_img_size += size
            elif exten in ["doc", "docx", "pdf"]:
                f_type_doc += 1
                f_type_doc_size += size
            elif exten in ["txt", "cfg", "rtf"]:
                f_type_file += 1
                f_type_file_size += size
            elif exten in ["exe", "sh"]:
                f_type_exec += 1
                f_type_exec_size += size
            else:
                f_type_other += 1
                f_type_other_size += size

        type_count = [
            f_type_comp,
            f_type_img,
            f_type_doc,
            f_type_file,
            f_type_exec,
            f_type_other,
        ]
        type_size = [
            f_type_comp_size,
            f_type_img_size,
            f_type_doc_size,
            f_type_file_size,
            f_type_exec_size,
            f_type_other_size,
        ]
        return type_count + type_size

    def process_url(urldata: str) -> int:
        if not urldata or str(urldata) == "nan":
            return 0
        url_result = urlparse(urldata)
        flag_host_list = ["wikileaks.org", "dropbox.com"]
        if url_result.hostname in flag_host_list:
            return 1
        return 0

    con.create_function(
        "udf_attachment_size_type", attachment_size_type, [str], list[int]
    )
    con.create_function("udf_process_url", process_url, [str], int)


COMMON_SELECT = """
    CASE WHEN sup.pc IS NOT NULL AND e.pc = sup.pc THEN 1 ELSE 0 END AS super_pc_acess,
    CASE WHEN (sup.pc IS NULL OR e.pc != sup.pc) AND e.pc != u.pc THEN 1 ELSE 0 END AS other_pc_acess,
    CASE 
        WHEN CAST(strptime(e.date, '%m/%d/%Y %H:%M:%S') AS TIME) < TIME '08:30:00' THEN date_diff('second', CAST(strptime(e.date, '%m/%d/%Y %H:%M:%S') AS TIME), TIME '08:30:00') / 60.0
        WHEN CAST(strptime(e.date, '%m/%d/%Y %H:%M:%S') AS TIME) > TIME '17:30:00' THEN date_diff('second', TIME '17:30:00', CAST(strptime(e.date, '%m/%d/%Y %H:%M:%S') AS TIME)) / 60.0
        ELSE 0.0
    END AS after_hour,
    CASE WHEN isodow(strptime(e.date, '%m/%d/%Y %H:%M:%S')) > 5 THEN 1 ELSE 0 END AS week
"""

COMMON_JOIN = """
    LEFT JOIN users u ON e.user = u.user_id
    LEFT JOIN users sup ON u.supervisor = sup.employee_name
"""

LOGON_SQL = f"""
SELECT 
{COMMON_SELECT}
FROM events e
{COMMON_JOIN}
"""

DEVICE_SQL = f"""
SELECT 
{COMMON_SELECT},
    CASE WHEN e.activity = 'Connect' THEN 1 ELSE 0 END AS conn,
    CASE WHEN e.activity = 'Disconnect' THEN 1 ELSE 0 END AS dis_conn
FROM events e
{COMMON_JOIN}
"""

FILE_SQL = f"""
SELECT 
{COMMON_SELECT},
    CASE WHEN e.to_removable_media THEN 1 ELSE 0 END AS to_rem,
    CASE WHEN e.from_removable_media THEN 1 ELSE 0 END AS f_rem,
    CASE WHEN e.activity = 'File Open' THEN 1 ELSE 0 END AS f_open,
    CASE WHEN e.activity = 'File Write' THEN 1 ELSE 0 END AS f_write,
    CASE WHEN e.activity = 'File Delete' THEN 1 ELSE 0 END AS f_delete,
    CASE WHEN e.activity = 'File Copy' THEN 1 ELSE 0 END AS f_copy,
    CASE WHEN split_part(e.filename, '.', 2) IN ('zip', 'rar', '7z') THEN 1 ELSE 0 END AS f_type_comp,
    CASE WHEN split_part(e.filename, '.', 2) IN ('jpg', 'png', 'bmp') THEN 1 ELSE 0 END AS f_type_img,
    CASE WHEN split_part(e.filename, '.', 2) IN ('doc', 'docx', 'pdf') THEN 1 ELSE 0 END AS f_type_doc,
    CASE WHEN split_part(e.filename, '.', 2) IN ('txt', 'cfg', 'rtf') THEN 1 ELSE 0 END AS f_type_file,
    CASE WHEN split_part(e.filename, '.', 2) IN ('exe', 'sh') THEN 1 ELSE 0 END AS f_type_exec,
    CASE WHEN e.filename IS NOT NULL AND split_part(e.filename, '.', 2) NOT IN ('zip', 'rar', '7z', 'jpg', 'png', 'bmp', 'doc', 'docx', 'pdf', 'txt', 'cfg', 'rtf', 'exe', 'sh') THEN 1 ELSE 0 END AS f_type_other
FROM events e
{COMMON_JOIN}
"""

EMAIL_SQL = f"""
SELECT 
{COMMON_SELECT},
    CASE WHEN e."to" IS NULL OR e."to" = 'nan' THEN 0 ELSE len(list_filter(string_split(e."to", ';'), x -> x not like '%dtaa.com%')) END AS to_out_count,
    CASE WHEN e."to" IS NULL OR e."to" = 'nan' THEN 0 ELSE len(list_filter(string_split(e."to", ';'), x -> x like '%dtaa.com%')) END AS to_in_count,
    CASE WHEN e.bcc IS NULL OR e.bcc = 'nan' THEN 0 ELSE len(list_filter(string_split(e.bcc, ';'), x -> x not like '%dtaa.com%')) END AS bcc_out_count,
    CASE WHEN e.bcc IS NULL OR e.bcc = 'nan' THEN 0 ELSE len(list_filter(string_split(e.bcc, ';'), x -> x like '%dtaa.com%')) END AS bcc_in_count,
    CASE WHEN e.cc IS NULL OR e.cc = 'nan' THEN 0 ELSE len(list_filter(string_split(e.cc, ';'), x -> x not like '%dtaa.com%')) END AS cc_out_count,
    CASE WHEN e.cc IS NULL OR e.cc = 'nan' THEN 0 ELSE len(list_filter(string_split(e.cc, ';'), x -> x like '%dtaa.com%')) END AS cc_in_count,
    udf_attachment_size_type(COALESCE(e.attachments, '')) AS attachment_features
FROM events e
{COMMON_JOIN}
"""

HTTP_SQL = f"""
SELECT 
{COMMON_SELECT},
    udf_process_url(COALESCE(e.url, '')) AS flag_url,
    COALESCE(CASE WHEN e.content LIKE '%job%' THEN 1 ELSE 0 END, 0) + COALESCE(CASE WHEN e.content LIKE '%keylogging%' THEN 1 ELSE 0 END, 0) AS flag_word
FROM events e
{COMMON_JOIN}
"""


def extract_logon_features(con: duckdb.DuckDBPyConnection) -> list:
    return [list(row) for row in con.sql(LOGON_SQL).fetchall()]


def extract_device_features(con: duckdb.DuckDBPyConnection) -> list:
    return [list(row) for row in con.sql(DEVICE_SQL).fetchall()]


def extract_file_features(con: duckdb.DuckDBPyConnection) -> list:
    return [list(row) for row in con.sql(FILE_SQL).fetchall()]


def extract_email_features(con: duckdb.DuckDBPyConnection) -> list:
    # Need to unpack the UDF list output
    results = []
    for row in con.sql(EMAIL_SQL).fetchall():
        results.append(list(row[:10]) + row[10])
    return results


def extract_http_features(con: duckdb.DuckDBPyConnection) -> list:
    return [list(row) for row in con.sql(HTTP_SQL).fetchall()]

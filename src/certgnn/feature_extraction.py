"""Feature extraction for insider threat detection.

Adapted from the source paper's get_feature.py and process_feature.py.
Preserves the original feature extraction logic exactly.
"""

from datetime import datetime
from urllib.parse import urlparse

import pandas as pd


class ProcessFeature:
    """Low-level feature processing utilities.

    Original: context/source_code/process_feature.py
    """

    # Parse once at class level instead of on every call
    _WORK_START = datetime.strptime("8:30", "%H:%M").time()
    _WORK_END = datetime.strptime("17:30", "%H:%M").time()

    def __init__(self, user_df: pd.DataFrame):
        self.user_df = user_df
        # Cache (own_pc, super_pc_or_None) per user_id to avoid repeated
        # O(n) DataFrame scans — critical for large datasets.
        self._pc_cache: dict = {}

    def process_url(self, urldata):
        if not isinstance(urldata, str):
            return 0
        url_result = urlparse(urldata)
        flag_host_list = ["wikileaks.org", "dropbox.com"]
        if url_result.hostname in flag_host_list:
            return 1
        else:
            return 0

    def process_content(self, content):
        if not isinstance(content, str):
            return 0
        flag_words_count = 0
        flag_word_list = ["job", "keylogging"]
        for flag_word in flag_word_list:
            if flag_word in content:
                flag_words_count += 1
        return flag_words_count

    def get_outside_inside_email_count(self, email_string):
        if str(email_string) != "nan":
            num_def_inside_email = 0
            num_def_outside_email = 0
            def_email_single_list = email_string.split(";")
            for single_email_def in def_email_single_list:
                if "dtaa.com" in single_email_def:
                    num_def_inside_email += 1
                else:
                    num_def_outside_email += 1
            return num_def_outside_email, num_def_inside_email
        else:
            return 0, 0

    def attachment_size_type(self, attachments):
        if str(attachments) != "nan":
            f_type_comp, f_type_img, f_type_doc = 0, 0, 0
            f_type_file, f_type_exec, f_type_other = 0, 0, 0
            f_type_comp_size, f_type_img_size, f_type_doc_size = 0, 0, 0
            f_type_file_size, f_type_exec_size, f_type_other_size = 0, 0, 0
            new_f_path = attachments.split(";")
            for f_path_size in new_f_path:
                split_ex_data = f_path_size.split(".")[1]
                exten = split_ex_data[0:3]
                size = split_ex_data[4:-1]
                if exten in ["zip", "rar", "7z"]:
                    f_type_comp += 1
                    f_type_comp_size += int(size)
                elif exten in ["jpg", "png", "bmp"]:
                    f_type_img += 1
                    f_type_img_size += int(size)
                elif exten in ["doc", "docx", "pdf"]:
                    f_type_doc += 1
                    f_type_doc_size += int(size)
                elif exten in ["txt", "cfg", "rtf"]:
                    f_type_file += 1
                    f_type_file_size += int(size)
                elif exten in ["exe", "sh"]:
                    f_type_exec += 1
                    f_type_exec_size += int(size)
                else:
                    f_type_other += 1
                    f_type_other_size += int(size)
            type_count = [
                f_type_comp, f_type_img, f_type_doc,
                f_type_file, f_type_exec, f_type_other,
            ]
            type_size = [
                f_type_comp_size, f_type_img_size, f_type_doc_size,
                f_type_file_size, f_type_exec_size, f_type_other_size,
            ]
            return type_count + type_size
        else:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # feature len 16
    def process_file_feature(self, row):
        to_rem = 1 if row["to_removable_media"] == True else 0  # noqa: E712
        f_rem = 1 if row["from_removable_media"] == True else 0  # noqa: E712
        f_open = 1 if row["activity_type"] == "File Open" else 0
        f_write = 1 if row["activity_type"] == "File Write" else 0
        f_delete = 1 if row["activity_type"] == "File Delete" else 0
        f_copy = 1 if row["activity_type"] == "File Copy" else 0

        f_path = row["filename"]
        exten = f_path.split(".")[1]
        f_type_comp, f_type_img, f_type_doc = 0, 0, 0
        f_type_file, f_type_exec, f_type_other = 0, 0, 0
        if exten in ["zip", "rar", "7z"]:
            f_type_comp = 1
        elif exten in ["jpg", "png", "bmp"]:
            f_type_img = 1
        elif exten in ["doc", "docx", "pdf"]:
            f_type_doc = 1
        elif exten in ["txt", "cfg", "rtf"]:
            f_type_file = 1
        elif exten in ["exe", "sh"]:
            f_type_exec = 1
        else:
            f_type_other = 1

        return (
            to_rem, f_rem, f_open, f_write, f_delete, f_copy,
            f_type_comp, f_type_img, f_type_doc, f_type_file,
            f_type_exec, f_type_other,
        )

    def conn_disconn_status(self, status):
        conn = 1 if status == "Connect" else 0
        dis_conn = 1 if status == "Disconnect" else 0
        return conn, dis_conn

    def is_after_whour(self, dt):
        """Check if timestamp is outside work hours (8:30-17:30)."""
        weekno_d1 = dt.weekday()
        week = 0 if weekno_d1 < 5 else 1

        dt_date = dt.date()
        d1_start = datetime.combine(dt_date, self._WORK_START)
        d1_end = datetime.combine(dt_date, self._WORK_END)
        if dt < d1_start or dt > d1_end:
            if dt < d1_start:
                diff = d1_start - dt
                return diff.total_seconds() / 60, week
            else:
                diff = dt - d1_end
                return diff.total_seconds() / 60, week
        return 0, week

    def _get_user_pc_info(self, user_id):
        """Return (own_pc, super_pc_or_None), cached per user_id."""
        if user_id not in self._pc_cache:
            try:
                user_row = self.user_df.loc[self.user_df["user_id"] == user_id]
                own_pc = user_row["pc"].iloc[0]
                sup_name = user_row["supervisor"].iloc[0]
                super_pc = self.user_df.loc[
                    self.user_df["employee_name"] == sup_name
                ]["pc"].iloc[0]
                self._pc_cache[user_id] = (own_pc, super_pc)
            except Exception:
                own_pc = self.user_df.loc[
                    self.user_df["user_id"] == user_id
                ]["pc"].iloc[0]
                self._pc_cache[user_id] = (own_pc, None)
        return self._pc_cache[user_id]

    def pc_details(self, user_id, pc_acess):
        own_pc, super_pc = self._get_user_pc_info(user_id)
        if super_pc is not None and pc_acess == super_pc:
            return own_pc, 1, 0
        else:
            if own_pc == pc_acess:
                return own_pc, 0, 0
            else:
                return own_pc, 0, 1


class GetFeature:
    """High-level feature extraction per activity type.

    Feature dimensions:
        logon=4, device=6, file=16, email=22, http=6 -> total=54
    """

    def __init__(self, user_df: pd.DataFrame):
        self.extract_feature = ProcessFeature(user_df)

    def get_logon_feature(self, row):
        own_pc, super_pc_acess, other_pc_acess = self.extract_feature.pc_details(
            row["user"], row["pc"],
        )
        after_hour, week = self.extract_feature.is_after_whour(row["timestamp"])
        return [super_pc_acess, other_pc_acess, after_hour, week]

    def get_device_feature(self, row):
        own_pc, super_pc_acess, other_pc_acess = self.extract_feature.pc_details(
            row["user"], row["pc"],
        )
        after_hour, week = self.extract_feature.is_after_whour(row["timestamp"])
        conn, dis_conn = self.extract_feature.conn_disconn_status(row["activity_type"])
        return [super_pc_acess, other_pc_acess, after_hour, week, conn, dis_conn]

    def get_file_feature(self, row):
        own_pc, super_pc_acess, other_pc_acess = self.extract_feature.pc_details(
            row["user"], row["pc"],
        )
        after_hour, week = self.extract_feature.is_after_whour(row["timestamp"])
        (
            to_rem, f_rem, f_open, f_write, f_delete, f_copy,
            f_type_comp, f_type_img, f_type_doc, f_type_file,
            f_type_exec, f_type_other,
        ) = self.extract_feature.process_file_feature(row)
        return [
            super_pc_acess, other_pc_acess, after_hour, week,
            to_rem, f_rem, f_open, f_write, f_delete, f_copy,
            f_type_comp, f_type_img, f_type_doc, f_type_file,
            f_type_exec, f_type_other,
        ]

    def get_email_feature(self, row):
        own_pc, super_pc_acess, other_pc_acess = self.extract_feature.pc_details(
            row["user"], row["pc"],
        )
        after_hour, week = self.extract_feature.is_after_whour(row["timestamp"])
        to_out_count, to_in_count = self.extract_feature.get_outside_inside_email_count(
            row["to"],
        )
        bcc_out_count, bcc_in_count = self.extract_feature.get_outside_inside_email_count(
            row["bcc"],
        )
        cc_out_count, cc_in_count = self.extract_feature.get_outside_inside_email_count(
            row["cc"],
        )
        email_file_feature = self.extract_feature.attachment_size_type(row["attachments"])
        return (
            [super_pc_acess, other_pc_acess, after_hour, week]
            + [to_out_count, to_in_count, bcc_out_count, bcc_in_count,
               cc_out_count, cc_in_count]
            + email_file_feature
        )

    def get_http_feature(self, row):
        own_pc, super_pc_acess, other_pc_acess = self.extract_feature.pc_details(
            row["user"], row["pc"],
        )
        after_hour, week = self.extract_feature.is_after_whour(row["timestamp"])
        flag_url = self.extract_feature.process_url(row["url"])
        flag_word = self.extract_feature.process_content(row["content"])
        return [super_pc_acess, other_pc_acess, after_hour, week, flag_url, flag_word]

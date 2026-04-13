# CERT r5.2 Dataset Structure Report

Summary report of the dataset structure. Extended descriptions are available in `readme.txt` files in the raw dataset.

## 1. High-Level Dataset Organization
The dataset package is split into two logical groups:

1. Activity data (`r5.2/`)
- User and machine activity logs across multiple channels.
- Organizational metadata (LDAP).
- Psychometric metadata.

2. Ground-truth and scenario definitions (`answers/`)
- Master list of true positives (`insiders.csv`).
- Per-incident observables files referenced by the master list.
- Human-readable scenario descriptions.

## 2. Core Activity Tables

### 2.1 logon.csv
Fields:
- `id`
- `date`
- `user`
- `pc`
- `activity` (`Logon` or `Logoff`)

Semantics and constraints from release notes:
- Logoff requires a preceding logon.
- Some daily logons are intentionally missing (dirty-data simulation).
- After-hours logins are present and are intended to be meaningful.
- Weekends and holidays are included with lower activity.
- Users have assigned PCs, plus a set of shared machines exists.
- Some users can access other users' machines.

### 2.2 device.csv
Fields:
- `id`
- `date`
- `user`
- `pc`
- `file_tree`
- `activity` (`Connect` or `Disconnect`)

Semantics:
- Represents removable media usage.
- Connect may exist without a matching disconnect.
- Per-user removable-media baseline behavior is expected.

### 2.3 http.csv
Fields:
- `id`
- `date`
- `user`
- `pc`
- `url`
- `content`

Semantics:
- `content` contains space-separated keywords.
- URL/path words are related to page topics.

### 2.4 email.csv
Fields:
- `id`
- `date`
- `user`
- `pc`
- `to`
- `cc`
- `bcc`
- `from`
- `activity`
- `size`
- `attachments`
- `content`

Semantics:
- Multi-recipient emails are possible.
- Internal vs external recipients are represented by address domains.
- Message `content` is keyword-based and can include multiple topics.
- Attachment names and sizes are included in this release.

### 2.5 file.csv
Fields:
- `id`
- `date`
- `user`
- `pc`
- `filename`
- `content`

Semantics:
- Represents copies to removable media.
- `content` starts with hex file header plus keyword tokens.
- Deviations from each user's normal copy volume are meaningful signals.

## 3. Metadata Tables

### 3.1 LDAP monthly files
Key fields include:
- `employee_name`, `user_id`, `email`, `role`, `projects`
- Organizational hierarchy:
  - `business_unit`
  - `functional_unit`
  - `department`
  - `team`
  - `supervisor`

Role in project:
- Defines organizational context for graph-aware modeling.
- Enables user-to-user and user-to-org feature engineering.

### 3.2 psychometric.csv
Fields:
- `employee_name`, `user_id`, `O`, `C`, `E`, `A`, `N`

Semantics:
- Big Five psychometric scores.
- In synthetic generation logic, personality affects behavioral tendencies.

### 3.3 decoy_file.csv
Contains decoy filenames and host locations.

Role in project:
- Can support additional suspicious-behavior indicators.

## 4. Ground Truth and Incident Files

### 4.1 insiders.csv (master labels)
This is the master true-positive file. It links:
- dataset/release,
- red-team scenario index,
- observables filename,
- username,
- incident start and end times.

### 4.2 Per-incident observables files
Important note:
- These files are variable-length row structures and are not strict CSV tables.
- First column denotes record type; rows of different types are interleaved chronologically.

Implication:
- They require custom parsing logic if used directly.

### 4.3 scenarios.txt
Provides short natural-language definitions for scenarios 1-5, such as:
- after-hours behavior shifts,
- removable-media escalation,
- credential abuse,
- exfiltration via email or cloud-sharing websites.

## 5. Keys and Cross-Table Linking Strategy
Primary practical linking keys in this release:
- `user`
- `pc`
- `date`/timestamp

Important caveat from release notes:
- `id` values are unique only within a given CSV file and are not globally unique across all tables.

Therefore:
- Multi-table alignment must be done using user, machine, and time logic.
- Any global-join design based purely on `id` is invalid.

## 6. Data Quality and Modeling Caveats
Documented caveats that affect downstream pipelines:
- Intentionally missing logon events.
- Possible connect without disconnect in removable-media logs.
- Mixed internal/external communication in email lists.
- Synthetic but behaviorally structured generation process.
- Non-tabular observables format in incident detail files.

## 7. Why This Structure Matters for Graph Construction
Graph construction is not a direct table-to-graph conversion. It requires staged transformations:
1. Normalize timestamps and entity identifiers across channels.
2. Align activity records by user-machine-time context.
3. Derive graph entities (users, devices, event types, or time windows).
4. Derive edges from interactions and temporal adjacency.
5. Attach labels using `insiders.csv` plus temporal boundaries.

Because the largest channels are very large, this process must be implemented as a memory-aware pipeline rather than a single in-memory merge.

import os
import re
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.config.timestamps import to_datetime64_ms

HANG_TIME_SAMPLING_FREQ = 50
HANG_TIME_GAP_THRESHOLD_MS = (1e3 / HANG_TIME_SAMPLING_FREQ) * 3.0

HANG_TIME_ACTIVITY_NAMES_BY_TIER: Dict[str, List[str]] = {
    "basketball": ["dribbling", "shot", "layup", "pass", "rebound", "not_labeled"],
    "locomotion": [
        "sitting",
        "standing",
        "walking",
        "running",
        "jumping",
        "not_labeled",
    ],
    "in_out": ["in", "out", "not_labeled"],
}

HANG_TIME_LABEL_ALIASES: Dict[str, Dict[str, str]] = {
    "basketball": {
        "dribble": "dribbling",
        "dribbling": "dribbling",
        "shoot": "shot",
        "shooting": "shot",
        "shot": "shot",
        "lay_up": "layup",
        "layup": "layup",
        "pass": "pass",
        "passing": "pass",
        "rebound": "rebound",
        "none": "not_labeled",
        "unknown": "not_labeled",
        "not_labeled": "not_labeled",
        "notlabelled": "not_labeled",
        "not_labeled_2": "not_labeled",
    },
    "locomotion": {
        "sit": "sitting",
        "sitting": "sitting",
        "stand": "standing",
        "standing": "standing",
        "walk": "walking",
        "walking": "walking",
        "run": "running",
        "running": "running",
        "jump": "jumping",
        "jumping": "jumping",
        "none": "not_labeled",
        "unknown": "not_labeled",
        "not_labeled": "not_labeled",
        "notlabelled": "not_labeled",
        "not_labeled_2": "not_labeled",
    },
    "in_out": {
        "in": "in",
        "inside": "in",
        "out": "out",
        "outside": "out",
        "none": "not_labeled",
        "unknown": "not_labeled",
        "not_labeled": "not_labeled",
        "notlabelled": "not_labeled",
        "not_labeled_2": "not_labeled",
    },
}


def _normalize_token(token: str) -> str:
    normalized = token.strip().lower()
    normalized = normalized.replace("/", "_")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _resolve_label_tier(activity_id_col: str) -> str:
    requested = _normalize_token(activity_id_col)
    if requested in {"activity_id", "activity", "label", ""}:
        return "basketball"

    if "basketball" in requested:
        return "basketball"
    if "locomotion" in requested:
        return "locomotion"
    if requested in {"in_out", "inout"} or "court" in requested:
        return "in_out"
    if "coarse" in requested:
        return "coarse"

    return "basketball"


def _resolve_label_col(df: pd.DataFrame, label_tier: str) -> str:
    col_norm: Dict[str, str] = {col: _normalize_token(col) for col in df.columns}
    normalized_to_col = {norm: col for col, norm in col_norm.items()}

    priority_by_tier = {
        "basketball": ["basketball", "tier_3", "tier3", "label_3", "label3"],
        "locomotion": ["locomotion", "tier_2", "tier2", "label_2", "label2"],
        "in_out": ["in_out", "inout", "tier_4", "tier4", "label_4", "label4"],
        "coarse": ["coarse", "tier_1", "tier1", "label_1", "label1"],
    }

    for token in priority_by_tier.get(label_tier, []):
        if token in normalized_to_col:
            return normalized_to_col[token]

    for col, normalized in col_norm.items():
        if "label" not in normalized and "tier" not in normalized:
            continue
        if label_tier == "basketball" and "basketball" in normalized:
            return col
        if label_tier == "locomotion" and "locomotion" in normalized:
            return col
        if label_tier == "in_out" and (
            "inout" in normalized or "in_out" in normalized or "court" in normalized
        ):
            return col
        if label_tier == "coarse" and "coarse" in normalized:
            return col

    for token in ("label", "tier"):
        for col, normalized in col_norm.items():
            if token in normalized:
                return col

    raise ValueError(
        "Could not determine activity label column in Hang Time dataset CSV."
    )


def _resolve_timestamp_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "unix_timestamp",
        "epoch",
        "ts",
    ]
    col_norm: Dict[str, str] = {col: _normalize_token(col) for col in df.columns}
    normalized_to_col = {norm: col for col, norm in col_norm.items()}

    for candidate in candidates:
        if candidate in normalized_to_col:
            return normalized_to_col[candidate]

    for col, normalized in col_norm.items():
        if "time" in normalized:
            return col

    return None


def _resolve_accel_cols(df: pd.DataFrame, blocked_cols: set[str]) -> List[str]:
    col_norm: Dict[str, str] = {col: _normalize_token(col) for col in df.columns}
    normalized_to_col = {norm: col for col, norm in col_norm.items()}

    explicit_triplets = [
        ["x", "y", "z"],
        ["acc_x", "acc_y", "acc_z"],
        ["accelerometer_x", "accelerometer_y", "accelerometer_z"],
        ["a_x", "a_y", "a_z"],
    ]
    for triplet in explicit_triplets:
        if all(name in normalized_to_col for name in triplet):
            cols = [normalized_to_col[name] for name in triplet]
            if all(col not in blocked_cols for col in cols):
                return cols

    numeric_candidates: List[str] = []
    for col in df.columns:
        if col in blocked_cols:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.5:
            numeric_candidates.append(col)

    if len(numeric_candidates) < 3:
        raise ValueError(
            "Could not identify three accelerometer channels for Hang Time dataset."
        )

    return numeric_candidates[:3]


def _canonicalize_label(raw_label: object, label_tier: str) -> str:
    token = _normalize_token(str(raw_label))
    if token == "":
        token = "not_labeled"

    aliases = HANG_TIME_LABEL_ALIASES.get(label_tier)
    if aliases is not None:
        return aliases.get(token, "not_labeled")

    return token


def _to_timestamp_series(values: pd.Series, fallback_length: int) -> pd.Series:
    maybe_datetime = pd.to_datetime(values, errors="coerce")
    if maybe_datetime.notna().mean() >= 0.8:
        return maybe_datetime.astype("datetime64[ms]")

    numeric = pd.to_numeric(values, errors="coerce")
    converted = to_datetime64_ms(numeric, default_unit="s")
    if converted.notna().mean() >= 0.8:
        return converted

    return to_datetime64_ms(
        pd.Series(range(fallback_length), dtype="float64")
        * (1.0 / HANG_TIME_SAMPLING_FREQ),
        default_unit="s",
    )


def parse_hang_time(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    label_tier = _resolve_label_tier(activity_id_col)

    csv_files: List[str] = []
    for root, _dirs, files in os.walk(dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            csv_files.append(os.path.join(root, file))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under '{dir}'.")

    csv_files = sorted(csv_files)
    subject_tokens = [os.path.basename(path).split(".")[0] for path in csv_files]
    subject_map = {token: idx for idx, token in enumerate(sorted(set(subject_tokens)))}

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int]] = []
    label_rows: List[str] = []
    global_session_id = 0

    loop = tqdm(csv_files)
    loop.set_description("Parsing Hang Time")

    for file_path in loop:
        file_df = pd.read_csv(file_path)
        if file_df.empty:
            continue

        label_col = _resolve_label_col(file_df, label_tier)
        timestamp_col = _resolve_timestamp_col(file_df)
        blocked_cols = {label_col}
        if timestamp_col is not None:
            blocked_cols.add(timestamp_col)

        accel_cols = _resolve_accel_cols(file_df, blocked_cols)

        work_df = file_df[
            [*([timestamp_col] if timestamp_col else []), label_col, *accel_cols]
        ].copy()
        work_df = work_df.rename(
            columns={
                label_col: "activity_name",
                accel_cols[0]: "acc_x",
                accel_cols[1]: "acc_y",
                accel_cols[2]: "acc_z",
            }
        )

        if timestamp_col is None:
            work_df["timestamp"] = to_datetime64_ms(
                pd.Series(range(len(work_df)), dtype="float64")
                * (1.0 / HANG_TIME_SAMPLING_FREQ),
                default_unit="s",
            )
        else:
            raw_timestamp = work_df[timestamp_col].copy()
            work_df["timestamp"] = _to_timestamp_series(raw_timestamp, len(work_df))
            if timestamp_col != "timestamp":
                work_df = work_df.drop(columns=[timestamp_col])

        work_df["activity_name"] = work_df["activity_name"].map(
            lambda x: _canonicalize_label(x, label_tier)
        )

        for sensor_col in ["acc_x", "acc_y", "acc_z"]:
            work_df[sensor_col] = pd.to_numeric(work_df[sensor_col], errors="coerce")

        work_df = work_df.dropna(
            subset=["timestamp", "acc_x", "acc_y", "acc_z", "activity_name"]
        )
        if work_df.empty:
            continue

        work_df = work_df.sort_values("timestamp").reset_index(drop=True)
        time_diffs_ms = work_df["timestamp"].diff().dt.total_seconds().fillna(0.0) * 1e3
        split_mask = (
            (work_df["activity_name"] != work_df["activity_name"].shift(1))
            | (time_diffs_ms > HANG_TIME_GAP_THRESHOLD_MS)
            | (time_diffs_ms <= 0)
        ).fillna(True)
        split_mask.iloc[0] = True

        local_session_ids = split_mask.cumsum() - 1
        work_df["local_session_id"] = local_session_ids.astype("int64")

        subject_token = os.path.basename(file_path).split(".")[0]
        subject_id = subject_map[subject_token]

        for local_id, session_chunk in work_df.groupby("local_session_id"):
            del local_id
            activity_name = str(session_chunk["activity_name"].iloc[0])
            label_rows.append(activity_name)

            session_rows.append(
                {
                    "session_id": global_session_id,
                    "subject_id": subject_id,
                    "activity_name": activity_name,
                }
            )

            session_df = session_chunk[["timestamp", "acc_x", "acc_y", "acc_z"]].copy()
            session_df = session_df.astype(
                {
                    "timestamp": "datetime64[ms]",
                    "acc_x": "float32",
                    "acc_y": "float32",
                    "acc_z": "float32",
                }
            )
            session_df[["acc_x", "acc_y", "acc_z"]] = session_df[
                ["acc_x", "acc_y", "acc_z"]
            ].round(6)

            sessions[global_session_id] = session_df.reset_index(drop=True)
            global_session_id += 1

    if not sessions:
        raise ValueError("No valid Hang Time sessions were parsed.")

    if label_tier in HANG_TIME_ACTIVITY_NAMES_BY_TIER:
        activity_names = HANG_TIME_ACTIVITY_NAMES_BY_TIER[label_tier]
        name_to_id = {name: idx for idx, name in enumerate(activity_names)}
    else:
        unique_names = sorted(set(label_rows))
        name_to_id = {name: idx for idx, name in enumerate(unique_names)}
        activity_names = unique_names

    activity_metadata = pd.DataFrame(
        {
            "activity_id": list(range(len(activity_names))),
            "activity_name": activity_names,
        }
    ).astype({"activity_id": "int32", "activity_name": "string"})

    session_metadata = pd.DataFrame(session_rows)
    session_metadata["activity_id"] = session_metadata["activity_name"].map(name_to_id)
    session_metadata = session_metadata.drop(columns=["activity_name"])
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_hang_time = WHARConfig(
    dataset_id="hang_time",
    download_url="https://zenodo.org/records/7920485/files/hangtime_har.zip?download=1",
    sampling_freq=HANG_TIME_SAMPLING_FREQ,
    num_of_subjects=24,
    num_of_activities=6,
    num_of_channels=3,
    parallelize=True,
    parse=parse_hang_time,
    activity_id_col="locomotion",
    activity_names=[
        "sitting",
        "standing",
        "walking",
        "running",
        "jumping",
        "not_labeled",
    ],
    sensor_channels=["acc_x", "acc_y", "acc_z"],
    window_time=2.56,
    window_overlap=0.5,
)

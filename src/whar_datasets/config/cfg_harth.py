from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

HARTH_ACTIVITY_NAMES: List[str] = [
    "walking",
    "running",
    "shuffling",
    "stairs (ascending)",
    "stairs (descending)",
    "standing",
    "sitting",
    "lying",
    "cycling (sit)",
    "cycling (stand)",
    "cycling (sit, inactive)",
    "cycling (stand, inactive)",
]

HARTH_ACTIVITY_MAP: Dict[int, str] = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs (ascending)",
    5: "stairs (descending)",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycling (sit)",
    14: "cycling (stand)",
    130: "cycling (sit, inactive)",
    140: "cycling (stand, inactive)",
}

HARTH_SENSOR_CHANNELS: List[str] = [
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
]

HARTH_MAX_STEP_MULTIPLIER = 3.0
_SUBJECT_PATTERN = re.compile(r"^S(\d+)", re.IGNORECASE)


def _extract_subject_id(path: Path) -> int:
    match = _SUBJECT_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Could not parse subject id from '{path.name}'.")
    return int(match.group(1))


def parse_harth(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    data_dir = Path(dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"HARTH data directory not found at '{data_dir}'.")

    csv_paths = sorted(data_dir.rglob("S*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No HARTH recordings found inside '{data_dir}'.")

    session_dfs: List[pd.DataFrame] = []
    global_session_id = 0

    required_columns = ["timestamp", *HARTH_SENSOR_CHANNELS, "label"]
    expected_step_seconds = 1.0 / 50.0
    max_allowed_gap_seconds = expected_step_seconds * HARTH_MAX_STEP_MULTIPLIER

    for csv_path in tqdm(csv_paths, desc="Parsing HARTH"):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns {sorted(missing_cols)} in '{csv_path.name}'."
            )
        df = df[required_columns].reset_index(drop=True).copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        df[activity_id_col] = df["label"].astype("int32")
        df["raw_activity_id"] = df[activity_id_col]
        df["subject_id"] = _extract_subject_id(csv_path)
        df["activity_name"] = df[activity_id_col].map(HARTH_ACTIVITY_MAP)
        df["activity_name"] = df["activity_name"].fillna("unknown")

        time_diff = df["timestamp"].diff().dt.total_seconds().fillna(0.0)
        activity_change = df[activity_id_col] != df[activity_id_col].shift(1)
        session_gap = time_diff > max_allowed_gap_seconds
        session_start = activity_change | session_gap
        session_start.iloc[0] = True
        local_session_ids = session_start.cumsum().astype("int32")
        df["session_id"] = local_session_ids + global_session_id
        global_session_id = int(df["session_id"].max()) + 1

        session_dfs.append(df)

    df = pd.concat(session_dfs, ignore_index=True)

    df["activity_id"] = pd.factorize(df["raw_activity_id"])[0]
    df["subject_id"] = pd.factorize(df["subject_id"])[0]
    df["session_id"] = pd.factorize(df["session_id"])[0]

    activity_metadata = (
        df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"])
        .sort_values("activity_id")
        .reset_index(drop=True)
    )

    session_metadata = (
        df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates(subset=["session_id"])
        .reset_index(drop=True)
    )

    sessions: Dict[int, pd.DataFrame] = {}
    loop = tqdm(session_metadata["session_id"].unique(), desc="Creating sessions")
    for session_id in loop:
        session_df = df[df["session_id"] == session_id]
        session_df = session_df.drop(
            columns=[
                "session_id",
                "subject_id",
                "activity_id",
                "activity_name",
                "label",
                activity_id_col,
                "raw_activity_id",
            ],
            errors="ignore",
        ).reset_index(drop=True)

        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"])
        float_cols = [col for col in session_df.columns if col != "timestamp"]
        session_df[float_cols] = session_df[float_cols].astype("float32")
        session_df[float_cols] = session_df[float_cols].round(6)
        sessions[int(session_id)] = session_df

    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_harth = WHARConfig(
    dataset_id="harth",
    dataset_url="https://archive.ics.uci.edu/dataset/779/harth",
    download_url="https://archive.ics.uci.edu/static/public/779/harth.zip",
    sampling_freq=50,
    num_of_subjects=22,
    num_of_activities=len(HARTH_ACTIVITY_NAMES),
    num_of_channels=len(HARTH_SENSOR_CHANNELS),
    datasets_dir="./datasets",
    parse=parse_harth,
    activity_id_col="activity_id",
    activity_names=HARTH_ACTIVITY_NAMES,
    sensor_channels=HARTH_SENSOR_CHANNELS,
    window_time=3,
    window_overlap=0.5,
    parallelize=True,
)

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

HAR70_ACTIVITY_MAP: Dict[int, str] = {
    1: "walking",
    3: "shuffling",
    4: "stairs (ascending)",
    5: "stairs (descending)",
    6: "standing",
    7: "sitting",
    8: "lying",
}

HAR70_ACTIVITY_NAMES: List[str] = [
    "walking",
    "shuffling",
    "stairs (ascending)",
    "stairs (descending)",
    "standing",
    "sitting",
    "lying",
]

HAR70_SENSOR_CHANNELS: List[str] = [
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
]

# Use the same continuity constraint as downstream validation:
# at 50 Hz, splits are introduced once the gap exceeds 3 * 20ms.
HAR70_SESSION_GAP_SECONDS = 0.06


def _resolve_har70_root(data_dir: str) -> Path:
    base = Path(data_dir)
    direct = base / "har70plus"
    if direct.is_dir() and any(direct.glob("*.csv")):
        return direct

    nested = base / "har70" / "har70plus"
    if nested.is_dir() and any(nested.glob("*.csv")):
        return nested

    for candidate in base.rglob("*"):
        if candidate.is_dir() and any(candidate.glob("*.csv")):
            return candidate

    raise FileNotFoundError(f"Could not locate HAR70 CSV files under '{data_dir}'.")


def _extract_subject_raw_id(csv_path: Path) -> int:
    if not csv_path.stem.isdigit():
        raise ValueError(
            f"Could not infer HAR70 subject identifier from '{csv_path.name}'."
        )
    return int(csv_path.stem)


def parse_har70(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    root = _resolve_har70_root(dir)
    csv_paths = sorted(root.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No HAR70 recordings found inside '{root}'.")

    required_columns = ["timestamp", *HAR70_SENSOR_CHANNELS, "label"]
    session_records: List[Dict[str, int | str]] = []
    sessions: Dict[int, pd.DataFrame] = {}
    next_session_id = 0

    loop = tqdm(csv_paths, desc="Parsing HAR70")
    for csv_path in loop:
        subject_raw_id = _extract_subject_raw_id(csv_path)
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns {sorted(missing_cols)} in '{csv_path.name}'."
            )

        df = df[required_columns].copy()
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["timestamp", "label"]).reset_index(drop=True)
        if df.empty:
            continue

        df["label"] = df["label"].astype("int32")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df[activity_id_col] = df["label"].astype("int32")
        df["activity_name"] = (
            df[activity_id_col]
            .map(HAR70_ACTIVITY_MAP)
            .fillna("unknown")
            .astype("string")
        )

        time_diff = df["timestamp"].diff().dt.total_seconds().fillna(0.0)
        activity_change = df[activity_id_col] != df[activity_id_col].shift(1)
        session_gap = time_diff > HAR70_SESSION_GAP_SECONDS
        session_start = activity_change | session_gap
        session_start.iloc[0] = True
        local_session_ids = session_start.cumsum().astype("int32")

        for _, chunk in df.groupby(local_session_ids):
            session_df = chunk.reset_index(drop=True)[
                ["timestamp", *HAR70_SENSOR_CHANNELS]
            ]
            session_df["timestamp"] = session_df["timestamp"].astype("datetime64[ms]")
            session_df[HAR70_SENSOR_CHANNELS] = session_df[
                HAR70_SENSOR_CHANNELS
            ].astype("float32")

            if session_df.empty:
                continue

            raw_activity = int(chunk[activity_id_col].iloc[0])
            activity_name = str(chunk["activity_name"].iloc[0])

            sessions[next_session_id] = session_df
            session_records.append(
                {
                    "session_id": next_session_id,
                    "subject_raw_id": subject_raw_id,
                    "raw_activity_id": raw_activity,
                    "activity_name": activity_name,
                }
            )
            next_session_id += 1

    session_metadata = pd.DataFrame(session_records)
    if session_metadata.empty:
        raise ValueError("No HAR70 sessions could be parsed.")

    session_metadata["subject_id"] = pd.factorize(
        session_metadata["subject_raw_id"], sort=True
    )[0]
    session_metadata["activity_id"] = pd.factorize(
        session_metadata["raw_activity_id"], sort=False
    )[0]
    session_metadata = session_metadata[["session_id", "subject_id", "activity_id"]]
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    activity_metadata = (
        pd.DataFrame(session_records)[["raw_activity_id", "activity_name"]]
        .drop_duplicates(subset=["raw_activity_id"])
        .sort_values("raw_activity_id")
        .reset_index(drop=True)
    )
    activity_metadata["activity_id"] = pd.factorize(
        activity_metadata["raw_activity_id"], sort=False
    )[0]
    activity_metadata = activity_metadata[["activity_id", "activity_name"]]
    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )

    return activity_metadata, session_metadata, sessions


SELECTED_ACTIVITIES = HAR70_ACTIVITY_NAMES

cfg_har70 = WHARConfig(
    # Info + common
    dataset_id="har70",
    dataset_url="https://archive.ics.uci.edu/dataset/780/har70",
    download_url="https://archive.ics.uci.edu/static/public/780/har70.zip",
    sampling_freq=50,
    num_of_subjects=18,
    num_of_activities=len(HAR70_ACTIVITY_NAMES),
    num_of_channels=len(HAR70_SENSOR_CHANNELS),
    datasets_dir="./datasets",
    # Parsing
    parse=parse_har70,
    activity_id_col="activity_id",
    # Preprocessing (selections + sliding window)
    available_activities=canonicalize_activity_name_list(HAR70_ACTIVITY_NAMES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=HAR70_SENSOR_CHANNELS,
    selected_channels=HAR70_SENSOR_CHANNELS,
    window_time=3,
    window_overlap=0.5,
    parallelize=True,
    # Training (split info)
)

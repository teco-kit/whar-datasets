from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

UP_FALL_ACTIVITY_NAMES: List[str] = [f"activity_{idx}" for idx in range(1, 12)]

UP_FALL_SENSOR_CHANNELS: List[str] = [
    "ankle_acc_x",
    "ankle_acc_y",
    "ankle_acc_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_lux",
    "right_pocket_acc_x",
    "right_pocket_acc_y",
    "right_pocket_acc_z",
    "right_pocket_gyro_x",
    "right_pocket_gyro_y",
    "right_pocket_gyro_z",
    "right_pocket_lux",
    "belt_acc_x",
    "belt_acc_y",
    "belt_acc_z",
    "belt_gyro_x",
    "belt_gyro_y",
    "belt_gyro_z",
    "belt_lux",
    "neck_acc_x",
    "neck_acc_y",
    "neck_acc_z",
    "neck_gyro_x",
    "neck_gyro_y",
    "neck_gyro_z",
    "neck_lux",
    "wrist_acc_x",
    "wrist_acc_y",
    "wrist_acc_z",
    "wrist_gyro_x",
    "wrist_gyro_y",
    "wrist_gyro_z",
    "wrist_lux",
    "eeg_raw",
    "infrared_1",
    "infrared_2",
    "infrared_3",
    "infrared_4",
    "infrared_5",
    "infrared_6",
]

UP_FALL_SESSION_GAP_SECONDS = 60.0
UP_FALL_STEP_MS = 50


def _resolve_up_fall_csv(data_dir: str) -> Path:
    root = Path(data_dir)
    candidates = [
        root,
        root / "data",
        root / "up_fall" / "data",
    ]

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        csvs = sorted(candidate.glob("*.csv"))
        if csvs:
            return csvs[0]
        merged = sorted(candidate.glob("uc?id=*export=download"))
        if merged:
            return merged[0]

    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".csv":
            return candidate
        if "uc?id=" in candidate.name and "export=download" in candidate.name:
            return candidate

    raise FileNotFoundError(f"Could not locate UP-Fall CSV data under '{data_dir}'.")


def _load_up_fall_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        raise ValueError(f"UP-Fall file '{csv_path}' is empty.")

    required_meta = {"TimeStamps", "Subject", "Activity", "Trial"}
    missing_meta = required_meta.difference(df.columns)
    if missing_meta:
        raise ValueError(
            f"UP-Fall file '{csv_path.name}' is missing required columns: {sorted(missing_meta)}"
        )

    if len(df.columns) < 47:
        raise ValueError(
            f"Unexpected UP-Fall schema in '{csv_path.name}': expected >=47 columns, got {len(df.columns)}."
        )

    keep_cols = (
        ["TimeStamps"]
        + list(df.columns[1:43])
        + ["Subject", "Activity", "Trial"]
        + (["Tag"] if "Tag" in df.columns else [])
    )
    df = df.loc[:, keep_cols].copy()

    df["timestamp"] = pd.to_datetime(df["TimeStamps"], errors="coerce")
    df["subject_raw_id"] = pd.to_numeric(df["Subject"], errors="coerce")
    df["activity_raw_id"] = pd.to_numeric(df["Activity"], errors="coerce")
    df["trial_raw_id"] = pd.to_numeric(df["Trial"], errors="coerce")

    valid_mask = df[["timestamp", "subject_raw_id", "activity_raw_id", "trial_raw_id"]]
    df = df[~valid_mask.isna().any(axis=1)].copy()
    if df.empty:
        raise ValueError(
            f"No valid UP-Fall rows with timestamp/subject/activity/trial in '{csv_path.name}'."
        )

    channel_cols = list(df.columns[1:43])
    df[channel_cols] = df[channel_cols].apply(pd.to_numeric, errors="coerce")
    df = df[~df[channel_cols].isna().all(axis=1)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No valid UP-Fall sensor rows left after numeric conversion for '{csv_path.name}'."
        )

    return df


def _split_session_by_gap(session_df: pd.DataFrame) -> List[pd.DataFrame]:
    if session_df.empty:
        return []

    deltas = session_df["timestamp"].diff().dt.total_seconds().fillna(0.0)
    split_markers = (deltas > UP_FALL_SESSION_GAP_SECONDS).astype("int32")
    group_ids = split_markers.cumsum()

    chunks: List[pd.DataFrame] = []
    for _, group in session_df.groupby(group_ids):
        chunk = group.reset_index(drop=True)
        if not chunk.empty:
            chunks.append(chunk)
    return chunks


def _build_session_frame(rows: pd.DataFrame) -> pd.DataFrame:
    sensor_values = rows.iloc[:, 1:43].copy()
    sensor_values.columns = UP_FALL_SENSOR_CHANNELS
    sensor_values = sensor_values.interpolate(
        method="linear",
        limit_direction="both",
        axis=0,
    ).fillna(0.0)
    sensor_values = sensor_values.astype("float32")

    start_time = rows["timestamp"].iloc[0]
    elapsed_ms = pd.Series(range(len(rows)), dtype="int64") * UP_FALL_STEP_MS

    session = pd.DataFrame(
        {"timestamp": start_time + pd.to_timedelta(elapsed_ms, unit="ms")}
    )
    session["timestamp"] = session["timestamp"].astype("datetime64[ms]")
    session = pd.concat([session, sensor_values.reset_index(drop=True)], axis=1)
    return session


def parse_up_fall(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    csv_path = _resolve_up_fall_csv(dir)
    raw = _load_up_fall_raw(csv_path)

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[dict] = []
    next_session_id = 0

    grouped = raw.groupby(
        ["subject_raw_id", "activity_raw_id", "trial_raw_id"], sort=True
    )
    loop = tqdm(grouped, total=grouped.ngroups)
    loop.set_description("Creating sessions")

    for (subject_raw, activity_raw, trial_raw), rows in loop:
        rows = rows.sort_values("timestamp").reset_index(drop=True)
        chunks = _split_session_by_gap(rows)
        for chunk in chunks:
            sessions[next_session_id] = _build_session_frame(chunk)
            session_rows.append(
                {
                    "session_id": next_session_id,
                    "subject_raw_id": int(subject_raw),  # type: ignore[union-attr]
                    "activity_raw_id": int(activity_raw),  # type: ignore[union-attr]
                    "trial_raw_id": int(trial_raw),  # type: ignore[union-attr]
                }
            )
            next_session_id += 1

    if not session_rows:
        raise ValueError("No UP-Fall sessions could be parsed.")

    session_metadata = pd.DataFrame(session_rows)
    session_metadata["subject_id"] = pd.factorize(
        session_metadata["subject_raw_id"], sort=True
    )[0]
    session_metadata["activity_id"] = pd.factorize(
        session_metadata["activity_raw_id"], sort=True
    )[0]
    session_metadata = session_metadata[["session_id", "subject_id", "activity_id"]]

    activity_metadata = (
        session_metadata[["activity_id"]].drop_duplicates().sort_values("activity_id")
    )
    activity_metadata["activity_name"] = activity_metadata["activity_id"].map(
        lambda idx: (
            UP_FALL_ACTIVITY_NAMES[int(idx)]
            if int(idx) < len(UP_FALL_ACTIVITY_NAMES)
            else f"activity_{int(idx) + 1}"
        )
    )
    activity_metadata = activity_metadata[["activity_id", "activity_name"]]

    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_up_fall = WHARConfig(
    dataset_id="up_fall",
    dataset_url="https://sites.google.com/up.edu.mx/har-up/",
    download_url="https://drive.usercontent.google.com/u/0/uc?id=1JBGU5W2uq9rl8h7bJNt2lN4SjfZnFxmQ&export=download",
    sampling_freq=20,
    num_of_subjects=17,
    num_of_activities=11,
    num_of_channels=42,
    datasets_dir="./datasets",
    parse=parse_up_fall,
    activity_id_col="activity_id",
    activity_names=UP_FALL_ACTIVITY_NAMES,
    sensor_channels=UP_FALL_SENSOR_CHANNELS,
    window_time=2,
    window_overlap=0.5,
    parallelize=True,
)

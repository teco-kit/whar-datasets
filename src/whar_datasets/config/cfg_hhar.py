from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

HHAR_ACTIVITY_NAMES: List[str] = [
    "bike",
    "sit",
    "stand",
    "walk",
    "stairsup",
    "stairsdown",
]

HHAR_SENSOR_CHANNELS: List[str] = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]

# HHAR activity data was captured with multiple devices per user in overlapping
# time windows. We therefore treat each (user, activity, device stream) as an
# independent session source and do not merge across devices.
HHAR_GAP_MULTIPLIER = 3.0
HHAR_MIN_GAP_MS = 50.0
HHAR_MAX_GAP_MS = 150.0


def _resolve_hhar_root(data_dir: str) -> Path:
    base = Path(data_dir)
    candidates = [
        base / "Activity recognition exp" / "Activity recognition exp",
        base / "Activity recognition exp",
        base / "hhar" / "data" / "Activity recognition exp" / "Activity recognition exp",
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "Phones_accelerometer.csv").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate HHAR activity files under '{data_dir}'.")


def _load_sensor_csv(file_path: Path, prefix: str) -> pd.DataFrame:
    cols = ["Creation_Time", "Arrival_Time", "x", "y", "z", "User", "Device", "gt"]
    df = pd.read_csv(
        file_path,
        usecols=cols,
        dtype={
            "Creation_Time": "int64",
            "Arrival_Time": "int64",
            "x": "float32",
            "y": "float32",
            "z": "float32",
            "User": "string",
            "Device": "string",
            "gt": "string",
        },
    )

    df = df.rename(
        columns={
            "Creation_Time": "creation_time",
            "Arrival_Time": "arrival_time",
            "User": "subject_raw",
            "Device": "device_raw",
            "gt": "activity_name",
            "x": f"{prefix}_x",
            "y": f"{prefix}_y",
            "z": f"{prefix}_z",
        }
    )
    df["subject_raw"] = df["subject_raw"].str.strip().str.lower()
    df["device_raw"] = df["device_raw"].str.strip().str.lower()
    df["activity_name"] = df["activity_name"].str.strip().str.lower()
    df = df.dropna(subset=["subject_raw", "device_raw", "activity_name"])
    df = df[df["activity_name"] != "null"]
    df = df.drop_duplicates(
        subset=["subject_raw", "device_raw", "activity_name", "creation_time"],
        keep="first",
    )
    df = df.sort_values(
        ["subject_raw", "device_raw", "activity_name", "creation_time"]
    ).reset_index(drop=True)
    return df


def _estimate_tolerance(df: pd.DataFrame) -> int:
    diffs = df.groupby(["subject_raw", "device_raw", "activity_name"], sort=False)[
        "creation_time"
    ].diff()
    diffs = diffs[(diffs > 0) & diffs.notna()]
    if diffs.empty:
        return 1_000_000
    median_step = int(diffs.median())
    return max(median_step * 2, 1)


def _merge_accel_gyro(
    accel_df: pd.DataFrame, gyro_df: pd.DataFrame, modality: str
) -> pd.DataFrame:
    tolerance = max(_estimate_tolerance(accel_df), _estimate_tolerance(gyro_df))
    # `merge_asof` requires both inputs to be globally sorted by the merge key.
    # We keep grouping columns as secondary keys for deterministic matching.
    merged = pd.merge_asof(
        accel_df.sort_values(
            ["creation_time", "subject_raw", "device_raw", "activity_name"]
        ),
        gyro_df.sort_values(
            ["creation_time", "subject_raw", "device_raw", "activity_name"]
        ),
        on="creation_time",
        by=["subject_raw", "device_raw", "activity_name"],
        direction="nearest",
        tolerance=tolerance,
        suffixes=("", "_gyro"),
    )
    merged = merged.dropna(
        subset=["gyro_x", "gyro_y", "gyro_z", "arrival_time", "arrival_time_gyro"]
    ).copy()
    merged["arrival_time"] = (
        (merged["arrival_time"].astype("int64") + merged["arrival_time_gyro"].astype("int64"))
        // 2
    )
    merged["modality"] = modality
    merged = merged[
        [
            "subject_raw",
            "device_raw",
            "activity_name",
            "modality",
            "arrival_time",
            *HHAR_SENSOR_CHANNELS,
        ]
    ]
    # Collapse exact timestamp collisions to avoid zero-delta fragments.
    merged = (
        merged.groupby(
            ["subject_raw", "device_raw", "activity_name", "modality", "arrival_time"],
            as_index=False,
        )[HHAR_SENSOR_CHANNELS]
        .mean()
        .sort_values(
            ["subject_raw", "device_raw", "activity_name", "modality", "arrival_time"]
        )
        .reset_index(drop=True)
    )
    return merged


def _split_sessions(df: pd.DataFrame) -> pd.Series:
    group_cols = ["subject_raw", "device_raw", "activity_name", "modality"]
    time_diff_ms = df.groupby(group_cols, sort=False)["arrival_time"].diff()
    positive_diff = time_diff_ms.where(time_diff_ms > 0)
    median_step_ms = positive_diff.groupby(
        [df[c] for c in group_cols], sort=False
    ).transform("median")
    gap_threshold_ms = (median_step_ms * HHAR_GAP_MULTIPLIER).clip(
        lower=HHAR_MIN_GAP_MS, upper=HHAR_MAX_GAP_MS
    )
    session_breaks = (
        (df["subject_raw"] != df["subject_raw"].shift(1))
        | (df["activity_name"] != df["activity_name"].shift(1))
        | (df["device_raw"] != df["device_raw"].shift(1))
        | (df["modality"] != df["modality"].shift(1))
        | time_diff_ms.isna()
        | (time_diff_ms < 0)
        | (time_diff_ms > gap_threshold_ms)
    )
    session_breaks.iloc[0] = True
    return pd.Series(
        np.cumsum(session_breaks.to_numpy(dtype=bool)),
        index=df.index,
        dtype="int64",
    )


def parse_hhar(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col
    root = _resolve_hhar_root(dir)

    phone_acc = _load_sensor_csv(root / "Phones_accelerometer.csv", "accel")
    phone_gyro = _load_sensor_csv(root / "Phones_gyroscope.csv", "gyro")
    watch_acc = _load_sensor_csv(root / "Watch_accelerometer.csv", "accel")
    watch_gyro = _load_sensor_csv(root / "Watch_gyroscope.csv", "gyro")

    phone_df = _merge_accel_gyro(phone_acc, phone_gyro, modality="phone")
    watch_df = _merge_accel_gyro(watch_acc, watch_gyro, modality="watch")

    df = pd.concat([phone_df, watch_df], ignore_index=True)
    if df.empty:
        raise ValueError("No HHAR rows remained after accel/gyro alignment.")

    unknown_activity = sorted(
        set(df["activity_name"].dropna().astype(str)) - set(HHAR_ACTIVITY_NAMES)
    )
    if unknown_activity:
        raise ValueError(
            "Found HHAR activities not covered by configured labels: "
            + ", ".join(unknown_activity)
        )

    df = df.sort_values(
        ["subject_raw", "activity_name", "modality", "device_raw", "arrival_time"]
    ).reset_index(drop=True)
    local_session_ids = _split_sessions(df)

    activity_id_map = {name: idx for idx, name in enumerate(HHAR_ACTIVITY_NAMES)}
    subject_raw_unique = sorted(df["subject_raw"].dropna().astype(str).unique().tolist())
    subject_id_map = {name: idx for idx, name in enumerate(subject_raw_unique)}

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int]] = []
    next_session_id = 0

    loop = tqdm(df.groupby(local_session_ids, sort=False), desc="Creating sessions")
    for _, chunk in loop:
        session_df = chunk[["arrival_time", *HHAR_SENSOR_CHANNELS]].copy()
        session_df = session_df.rename(columns={"arrival_time": "timestamp"})
        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"], unit="ms")
        session_df = session_df.astype(
            {"timestamp": "datetime64[ms]", **{col: "float32" for col in HHAR_SENSOR_CHANNELS}}
        )
        session_df[HHAR_SENSOR_CHANNELS] = session_df[HHAR_SENSOR_CHANNELS].round(6)
        if session_df.empty:
            continue

        subject_raw = str(chunk["subject_raw"].iloc[0])
        activity_name = str(chunk["activity_name"].iloc[0])

        sessions[next_session_id] = session_df.reset_index(drop=True)
        session_rows.append(
            {
                "session_id": next_session_id,
                "subject_id": subject_id_map[subject_raw],
                "activity_id": activity_id_map[activity_name],
            }
        )
        next_session_id += 1

    if not sessions:
        raise ValueError("No HHAR sessions were produced.")

    activity_metadata = pd.DataFrame(
        {"activity_id": list(range(len(HHAR_ACTIVITY_NAMES))), "activity_name": HHAR_ACTIVITY_NAMES}
    ).astype({"activity_id": "int32", "activity_name": "string"})
    session_metadata = pd.DataFrame(session_rows).astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_hhar = WHARConfig(
    # Info + common
    dataset_id="hhar",
    dataset_url="https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition",
    download_url="https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip",
    sampling_freq=20,
    num_of_subjects=9,
    num_of_activities=6,
    num_of_channels=6,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_hhar,
    activity_id_col="activity_id",
    # Preprocessing (selections + sliding window)
    activity_names=HHAR_ACTIVITY_NAMES,
    sensor_channels=HHAR_SENSOR_CHANNELS,
    window_time=5,
    window_overlap=0.5,
    parallelize=True,
    # Training (split info)
)

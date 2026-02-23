from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

SENSOR_POSITIONS: List[str] = [
    "rla",
    "rua",
    "back",
    "lua",
    "lla",
    "rc",
    "rt",
    "lt",
    "lc",
]

SENSOR_MODALITIES: List[str] = ["acc", "gyro", "mag", "quat"]
SENSOR_AXES: Dict[str, List[str]] = {
    "acc": ["x", "y", "z"],
    "gyro": ["x", "y", "z"],
    "mag": ["x", "y", "z"],
    "quat": ["1", "2", "3", "4"],
}


def _build_sensor_columns() -> List[str]:
    cols: List[str] = []
    for position in SENSOR_POSITIONS:
        for modality in SENSOR_MODALITIES:
            for axis in SENSOR_AXES[modality]:
                cols.append(f"{modality}_{position}_{axis}")
    return cols


SENSOR_COLUMNS: List[str] = _build_sensor_columns()

ACTIVITY_NAMES: List[str] = [
    "no_activity",
    "walking",
    "jogging",
    "running",
    "jump_up",
    "jump_front_back",
    "jump_sideways",
    "jump_legs_arms_open_closed",
    "jump_rope",
    "trunk_twist_arms_outstretched",
    "trunk_twist_elbows_bended",
    "waist_bends_forward",
    "waist_rotation",
    "waist_bends_reach_opposite_foot",
    "reach_heels_backwards",
    "lateral_bend",
    "lateral_bend_arm_up",
    "repetitive_forward_stretching",
    "upper_trunk_lower_body_opposite_twist",
    "arms_lateral_elevation",
    "arms_frontal_elevation",
    "frontal_hand_claps",
    "arms_frontal_crossing",
    "shoulders_high_amplitude_rotation",
    "shoulders_low_amplitude_rotation",
    "arms_inner_rotation",
    "knees_alternatively_to_breast",
    "heels_alternatively_to_backside",
    "knees_bending_crouching",
    "knees_alternatively_bend_forward",
    "rotation_on_the_knees",
    "rowing",
    "elliptic_bike",
    "cycling",
]

REAL_DISP_GAP_MULTIPLIER: float = 3.0


def _extract_subject_id(log_path: Path) -> int:
    match = re.match(r"^subject(\d+)_", log_path.name)
    if match is None:
        raise ValueError(
            f"Could not parse subject identifier from filename: '{log_path.name}'."
        )
    return int(match.group(1))


def parse_real_disp(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    data_dir = Path(dir)
    log_paths = sorted(data_dir.glob("subject*_*.log"))
    if not log_paths:
        raise FileNotFoundError(f"No REALDISP log files found in '{data_dir}'.")

    sampling_step_us = int(1e6 / 50)
    max_gap_us = int(sampling_step_us * REAL_DISP_GAP_MULTIPLIER)

    session_rows: List[Dict[str, int]] = []
    sessions: Dict[int, pd.DataFrame] = {}
    observed_activity_ids: set[int] = set()
    session_id = 0

    loop = tqdm(log_paths, desc="Parsing REALDISP")
    for log_path in loop:
        subject_from_name = _extract_subject_id(log_path)
        loop.set_postfix(file=log_path.name, refresh=False)
        expected_cols = 2 + len(SENSOR_COLUMNS) + 1
        csv_columns = ["timestamp_s", "timestamp_us", *SENSOR_COLUMNS, activity_id_col]
        dtype_map: Dict[str, str] = {
            "timestamp_s": "int64",
            "timestamp_us": "int64",
            activity_id_col: "int32",
            **{col: "float32" for col in SENSOR_COLUMNS},
        }
        try:
            df = pd.read_csv(
                log_path,
                sep="\t",
                header=None,
                names=csv_columns,
                dtype=dtype_map,
                engine="c",
            )
        except ValueError:
            # Fallback for malformed rows: parse loosely, then coerce.
            df = pd.read_csv(log_path, sep="\t", header=None, names=csv_columns)
            numeric_columns = [
                "timestamp_s",
                "timestamp_us",
                activity_id_col,
                *SENSOR_COLUMNS,
            ]
            df[numeric_columns] = df[numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            df = df.dropna(subset=["timestamp_s", "timestamp_us", activity_id_col])
            if df.empty:
                continue
            df = df.astype(dtype_map)

        if df.shape[1] != len(csv_columns):
            raise ValueError(
                f"Unexpected column count in '{log_path.name}': "
                f"expected {expected_cols}, got {df.shape[1]}."
            )

        df[activity_id_col] = df[activity_id_col].astype("int32")
        timestamp_total_us = (
            df["timestamp_s"].astype("int64") * 1_000_000
            + df["timestamp_us"].astype("int64")
        )
        observed_activity_ids.update(df[activity_id_col].unique().tolist())

        timestamp_diff = timestamp_total_us.diff()
        positive_step_ratio = float((timestamp_diff > 0).mean())
        has_timestamp_variation = bool(timestamp_total_us.nunique() > 1)
        has_valid_timestamps = has_timestamp_variation and positive_step_ratio > 0.1

        if has_valid_timestamps:
            session_breaks = (
                (df[activity_id_col] != df[activity_id_col].shift(1))
                | (timestamp_diff <= 0)
                | (timestamp_diff > max_gap_us)
            )
        else:
            # Some files (e.g., subject13_self.log) have zeroed timestamps.
            # Fall back to activity-contiguous sessions and synthesize a 50Hz timeline.
            session_breaks = df[activity_id_col] != df[activity_id_col].shift(1)
            timestamp_total_us = pd.Series(
                np.arange(len(df), dtype=np.int64) * sampling_step_us,
                index=df.index,
            )

        local_session_id = session_breaks.cumsum()
        sensor_values = df[SENSOR_COLUMNS].copy()
        if sensor_values.isna().any().any():
            sensor_values = sensor_values.groupby(local_session_id, sort=False).transform(
                lambda g: g.interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )
        sensor_values = sensor_values.astype("float32")
        timestamps = pd.to_datetime(timestamp_total_us, unit="us").astype("datetime64[ms]")
        grouped_indices = df.groupby(local_session_id, sort=False).indices

        for idx in grouped_indices.values():
            idx = np.asarray(idx)
            if idx.size == 0:
                continue

            session_df = pd.DataFrame(
                sensor_values.iloc[idx].to_numpy(copy=False),
                columns=SENSOR_COLUMNS,
            )
            session_df.insert(
                0,
                "timestamp",
                timestamps.iloc[idx].to_numpy(),
            )
            session_df = session_df.dropna()

            if session_df.empty:
                continue

            session_df = session_df.astype(
                {
                    **{col: "float32" for col in SENSOR_COLUMNS},
                    "timestamp": "datetime64[ms]",
                }
            )

            subject_id = int(subject_from_name) - 1
            activity_id = int(df[activity_id_col].iloc[idx[0]])

            session_rows.append(
                {
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "activity_id": activity_id,
                }
            )
            sessions[session_id] = session_df
            session_id += 1

    if not sessions:
        raise ValueError("No valid sessions were parsed from REALDISP logs.")

    activity_df = pd.DataFrame(
        {
            "activity_id": sorted(observed_activity_ids),
        }
    )
    activity_df["activity_name"] = activity_df["activity_id"].map(
        lambda idx: (
            ACTIVITY_NAMES[idx]
            if 0 <= idx < len(ACTIVITY_NAMES)
            else f"activity_{idx:02d}"
        )
    )
    activity_df = activity_df.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )

    session_df = pd.DataFrame(session_rows)
    session_df = session_df.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_df, session_df, sessions


cfg_real_disp = WHARConfig(
    # Info + common
    dataset_id="real_disp",
    dataset_url="https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset",
    download_url="https://archive.ics.uci.edu/static/public/305/realdisp+activity+recognition+dataset.zip",
    sampling_freq=50,
    num_of_subjects=17,
    num_of_activities=34,
    num_of_channels=117,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_real_disp,
    activity_id_col="activity_id",
    # Preprocessing (selections + sliding window)
    activity_names=ACTIVITY_NAMES,
    sensor_channels=SENSOR_COLUMNS,
    window_time=3,
    window_overlap=0.5,
)

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig
from whar_datasets.config.timestamps import to_datetime64_ms

BMHAD_ACTIVITY_NAMES: List[str] = [
    "jumping_in_place",
    "jumping_jacks",
    "bending",
    "punching",
    "waving_two_hands",
    "waving_one_hand",
    "clapping_hands",
    "throwing_a_ball",
    "sit_down",
    "stand_up",
    "sit_down_and_stand_up",
]

BMHAD_SENSOR_CHANNELS: List[str] = [
    f"mocap_marker_{marker:02d}_{axis}"
    for marker in range(43)
    for axis in ("x", "y", "z")
]

_ACTION_FILE_PATTERN = re.compile(r"^moc_s(\d{2})_a(\d{2})_r(\d{2})\.txt$")


def _resolve_bmhad_data_dir(data_dir: str) -> Path:
    base = Path(data_dir)
    candidates = [base, base / "bmhad"]

    # Case 1: parser receives the action-data directory directly.
    if any(base.glob("moc_s*_a*_r*.txt")):
        return base

    # Case 2: parser receives dataset root that contains a `data/` directory.
    for candidate in candidates:
        candidate_data = candidate / "data"
        if candidate_data.is_dir():
            return candidate_data

    raise FileNotFoundError(
        f"Could not locate BMHAD data directory under '{data_dir}'. "
        "Expected '<root>/data/moc_sXX_aYY_rZZ.txt' files."
    )


def parse_bmhad(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    data_dir = _resolve_bmhad_data_dir(dir)

    file_records: List[Tuple[int, int, int, Path]] = []
    for file_path in data_dir.iterdir():
        if not file_path.is_file():
            continue
        match = _ACTION_FILE_PATTERN.match(file_path.name)
        if match is None:
            continue

        subject_1b = int(match.group(1))
        activity_1b = int(match.group(2))
        repetition_1b = int(match.group(3))
        file_records.append((subject_1b, activity_1b, repetition_1b, file_path))

    if not file_records:
        raise FileNotFoundError(
            f"No BMHAD motion capture files found in '{data_dir}'. "
            "Expected filenames like 'moc_s01_a01_r01.txt'."
        )

    file_records.sort(key=lambda row: (row[0], row[1], row[2], row[3].name))

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int]] = []

    loop = tqdm(file_records)
    loop.set_description("Parsing BMHAD sessions")

    for session_id, (subject_1b, activity_1b, _rep, file_path) in enumerate(loop):
        raw_df = pd.read_csv(file_path, sep=r"\s+", header=None)
        expected_cols = len(BMHAD_SENSOR_CHANNELS) + 2
        if raw_df.shape[1] < expected_cols:
            raise ValueError(
                f"Unexpected column count in '{file_path.name}': {raw_df.shape[1]} "
                f"(expected at least {expected_cols})."
            )

        timestamp = to_datetime64_ms(raw_df.iloc[:, 130], default_unit="s")
        sensor_values = raw_df.iloc[:, : len(BMHAD_SENSOR_CHANNELS)].apply(
            pd.to_numeric, errors="coerce"
        )
        sensor_values.columns = BMHAD_SENSOR_CHANNELS

        # BMHAD uses exact (0,0,0) marker triplets for occlusions/dropout.
        # Convert these to missing values so interpolation can remove step-like artifacts.
        for marker in range(43):
            cols = [
                f"mocap_marker_{marker:02d}_x",
                f"mocap_marker_{marker:02d}_y",
                f"mocap_marker_{marker:02d}_z",
            ]
            zero_mask = (sensor_values[cols] == 0.0).all(axis=1)
            if zero_mask.any():
                sensor_values.loc[zero_mask, cols] = np.nan

        sensor_values = sensor_values.interpolate(
            method="linear",
            axis=0,
            limit_direction="both",
        )
        sensor_values = sensor_values.fillna(0.0)

        session_df = pd.concat([timestamp.rename("timestamp"), sensor_values], axis=1)
        session_df = session_df.dropna(subset=["timestamp"]).reset_index(drop=True)
        if session_df.empty:
            raise ValueError(f"No valid timestamps parsed in '{file_path.name}'.")

        session_df = session_df.sort_values("timestamp").reset_index(drop=True)
        session_df["timestamp"] = session_df["timestamp"].astype("datetime64[ms]")
        session_df[BMHAD_SENSOR_CHANNELS] = session_df[BMHAD_SENSOR_CHANNELS].astype(
            "float32"
        )
        session_df[BMHAD_SENSOR_CHANNELS] = session_df[BMHAD_SENSOR_CHANNELS].round(6)

        sessions[session_id] = session_df
        session_rows.append(
            {
                "session_id": session_id,
                "subject_id": subject_1b - 1,
                activity_id_col: activity_1b - 1,
            }
        )

    session_metadata = pd.DataFrame(session_rows).astype(
        {"session_id": "int32", "subject_id": "int32", activity_id_col: "int32"}
    )
    if activity_id_col != "activity_id":
        session_metadata["activity_id"] = session_metadata[activity_id_col].astype(
            "int32"
        )

    activity_metadata = pd.DataFrame(
        [
            {"activity_id": activity_id, "activity_name": name}
            for activity_id, name in enumerate(BMHAD_ACTIVITY_NAMES)
        ]
    ).astype({"activity_id": "int32", "activity_name": "string"})

    return activity_metadata, session_metadata, sessions


SELECTED_ACTIVITIES = BMHAD_ACTIVITY_NAMES

cfg_bmhad = WHARConfig(
    # Info + common
    dataset_id="bmhad",
    dataset_url="https://www.kaggle.com/datasets/dasmehdixtr/berkeley-multimodal-human-action-database",
    download_url="https://www.kaggle.com/api/v1/datasets/download/dasmehdixtr/berkeley-multimodal-human-action-database",
    sampling_freq=500,
    num_of_subjects=12,
    num_of_activities=len(BMHAD_ACTIVITY_NAMES),
    num_of_channels=len(BMHAD_SENSOR_CHANNELS),
    datasets_dir="./datasets",
    # Parsing
    parse=parse_bmhad,
    # Preprocessing (selections + sliding window)
    available_activities=canonicalize_activity_name_list(BMHAD_ACTIVITY_NAMES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=BMHAD_SENSOR_CHANNELS,
    selected_channels=BMHAD_SENSOR_CHANNELS,
    window_time=2,
    window_overlap=0.5,
    parallelize=True,
    # Training (split info)
)

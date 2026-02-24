import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig


ID_TO_ACTIVITY = {
    0: "Swipe Left",
    1: "Swipe Right",
    2: "Wave",
    3: "Clap",
    4: "Throw",
    5: "Arm Cross",
    6: "Basketball Shoot",
    7: "Draw X",
    8: "Draw Circle Clockwise",
    9: "Draw Circle Counterclockwise",
    10: "Draw Triangle",
    11: "Bowling",
    12: "Boxing",
    13: "Baseball Swing",
    14: "Tennis Swing",
    15: "Arm Curl",
    16: "Tennis Serve",
    17: "Two-Hand Push",
    18: "Knock",
    19: "Catch",
    20: "Pick Up and Throw",
    21: "Jog",
    22: "Walk",
    23: "Sit to Stand",
    24: "Stand to Sit",
    25: "Lunge",
    26: "Squat",
}


def parse_utd_mhad(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    root_path = os.path.join(dir, "Inertial")
    pattern = re.compile(r"^a(\d+)_s(\d+)_t(\d+)_inertial\.mat$")
    sensor_cols = ["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ"]
    sampling_rate_hz = 50.0

    file_records: List[Tuple[int, int, int, str]] = []
    for root, _, files in os.walk(root_path):
        for file_name in files:
            match = pattern.match(file_name)
            if match is None:
                continue
            activity_1b = int(match.group(1))
            subject_1b = int(match.group(2))
            trial = int(match.group(3))
            file_path = os.path.join(root, file_name)
            file_records.append((subject_1b, activity_1b, trial, file_path))

    if not file_records:
        raise ValueError(
            "No inertial .mat files found for UTD-MHAD. Expected files like "
            "'a1_s1_t1_inertial.mat' in '<data>/Inertial'."
        )

    file_records.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int]] = []
    session_id = 0

    loop = tqdm(file_records)
    loop.set_description("Parsing UTD-MHAD sessions")

    for subject_1b, activity_1b, _trial, file_path in loop:
        mat = scipy.io.loadmat(file_path)
        if "d_iner" not in mat:
            raise ValueError(f"Missing 'd_iner' in MAT file: {file_path}")

        arr = np.asarray(mat["d_iner"], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected shape for 'd_iner' in {file_path}: {arr.shape}")

        if arr.shape[1] == len(sensor_cols):
            data = arr
        elif arr.shape[0] == len(sensor_cols):
            data = arr.T
        else:
            raise ValueError(
                f"Expected 6 inertial channels in {file_path}, got shape {arr.shape}"
            )

        time_sec = np.arange(data.shape[0], dtype=np.float64) / sampling_rate_hz
        session_df = pd.DataFrame(data, columns=sensor_cols)
        session_df["timestamp"] = pd.to_datetime(time_sec, unit="s").astype(
            "datetime64[ms]"
        )

        for col in sensor_cols:
            session_df[col] = session_df[col].astype("float32").round(6)

        sessions[session_id] = session_df
        session_rows.append(
            {
                "session_id": session_id,
                "subject_id": subject_1b - 1,
                activity_id_col: activity_1b - 1,
            }
        )
        session_id += 1

    session_metadata = pd.DataFrame(session_rows).astype(
        {"session_id": "int32", "subject_id": "int32", activity_id_col: "int32"}
    )

    if activity_id_col != "activity_id":
        session_metadata["activity_id"] = session_metadata[activity_id_col].astype(
            "int32"
        )

    activity_metadata = pd.DataFrame(
        [
            {"activity_id": activity_id, "activity_name": activity_name}
            for activity_id, activity_name in ID_TO_ACTIVITY.items()
        ]
    ).astype({"activity_id": "int32", "activity_name": "string"})

    return activity_metadata, session_metadata, sessions


cfg_utd_mhad = WHARConfig(
    # Info + common
    dataset_id="utd_mhad",
    dataset_url="https://personal.utdallas.edu/~kehtar/UTD-MHAD.html",
    download_url="https://drive.usercontent.google.com/u/0/uc?id=16qcl5I5NSQ54aBeye284szQqjjuCY9jm&export=download",
    sampling_freq=50,
    num_of_subjects=8,
    num_of_activities=27,
    num_of_channels=6,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_utd_mhad,
    # Preprocessing (selections + sliding window)
    activity_names=list(ID_TO_ACTIVITY.values()),
    sensor_channels=["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ"],
    window_time=2,
    window_overlap=0.5,
    # Training (split info)
)

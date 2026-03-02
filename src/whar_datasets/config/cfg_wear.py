from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

WEAR_SENSOR_CHANNELS: List[str] = [
    "right_arm_acc_x",
    "right_arm_acc_y",
    "right_arm_acc_z",
    "right_leg_acc_x",
    "right_leg_acc_y",
    "right_leg_acc_z",
    "left_leg_acc_x",
    "left_leg_acc_y",
    "left_leg_acc_z",
    "left_arm_acc_x",
    "left_arm_acc_y",
    "left_arm_acc_z",
]

WEAR_ACTIVITY_NAMES: List[str] = [
    "bench-dips",
    "burpees",
    "jogging",
    "jogging (butt-kicks)",
    "jogging (rotating arms)",
    "jogging (sidesteps)",
    "jogging (skipping)",
    "lunges",
    "lunges (complex)",
    "push-ups",
    "push-ups (complex)",
    "sit-ups",
    "sit-ups (complex)",
    "stretching (hamstrings)",
    "stretching (lumbar rotation)",
    "stretching (lunging)",
    "stretching (shoulders)",
    "stretching (triceps)",
]


def parse_wear(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    data_root = Path(dir)
    file_paths = sorted(data_root.glob("sbj_*.csv"))
    if not file_paths:
        raise FileNotFoundError(f"No WEAR CSV files found under '{dir}'.")

    activity_map = {
        activity_name: idx for idx, activity_name in enumerate(WEAR_ACTIVITY_NAMES)
    }
    step_ms = int(1e3 / 50)

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int]] = []
    next_session_id = 0

    loop = tqdm(file_paths, desc="Parsing WEAR")
    for file_path in loop:
        frame = pd.read_csv(
            file_path,
            usecols=["sbj_id", *WEAR_SENSOR_CHANNELS, "label"],
            low_memory=False,
        )

        missing_cols = [
            col
            for col in ["sbj_id", "label", *WEAR_SENSOR_CHANNELS]
            if col not in frame.columns
        ]
        if missing_cols:
            raise ValueError(
                f"WEAR file '{file_path.name}' is missing required columns: {missing_cols}"
            )

        frame["label"] = frame["label"].astype("string").str.strip()
        frame["label"] = frame["label"].replace(
            {"null": pd.NA, "None": pd.NA, "nan": pd.NA, "": pd.NA}  # type: ignore
        )
        frame["sbj_id"] = pd.to_numeric(frame["sbj_id"], errors="coerce")
        for sensor_channel in WEAR_SENSOR_CHANNELS:
            frame[sensor_channel] = pd.to_numeric(
                frame[sensor_channel], errors="coerce"
            )

        frame = frame.dropna(subset=["sbj_id", "label", *WEAR_SENSOR_CHANNELS]).copy()
        if frame.empty:
            continue

        subject_values = frame["sbj_id"].astype(int).unique().tolist()
        if len(subject_values) != 1:
            raise ValueError(
                f"WEAR file '{file_path.name}' has mixed subject ids: {subject_values[:8]}"
            )

        subject_from_column = int(subject_values[0])
        subject_from_filename = int(file_path.stem.split("_")[1])
        if subject_from_column != subject_from_filename:
            raise ValueError(
                f"Subject mismatch in '{file_path.name}': column={subject_from_column}, "
                f"filename={subject_from_filename}."
            )

        unknown_labels = sorted(
            set(frame["label"].dropna().astype(str).unique().tolist())
            - set(WEAR_ACTIVITY_NAMES)
        )
        if unknown_labels:
            raise ValueError(
                "Found WEAR activity labels not covered by config: "
                + ", ".join(unknown_labels)
            )

        frame["activity_id"] = frame["label"].map(activity_map).astype("int32")
        frame["subject_id"] = frame["sbj_id"].astype("int32")

        session_breaks = (frame["subject_id"] != frame["subject_id"].shift(1)) | (
            frame["activity_id"] != frame["activity_id"].shift(1)
        )
        session_breaks.iloc[0] = True
        local_session_ids = session_breaks.astype("int64").cumsum() - 1

        for _, chunk in frame.groupby(local_session_ids, sort=False):
            if chunk.empty:
                continue

            timestamps = pd.to_datetime(
                np.arange(len(chunk), dtype="int64") * step_ms, unit="ms"
            )
            session = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    **{col: chunk[col].to_numpy() for col in WEAR_SENSOR_CHANNELS},
                }
            )
            session = session.astype(
                {
                    "timestamp": "datetime64[ms]",
                    **{col: "float32" for col in WEAR_SENSOR_CHANNELS},
                }
            )
            session[WEAR_SENSOR_CHANNELS] = session[WEAR_SENSOR_CHANNELS].round(6)
            sessions[next_session_id] = session.reset_index(drop=True)

            session_rows.append(
                {
                    "session_id": next_session_id,
                    "subject_id": int(chunk["subject_id"].iloc[0]),
                    "activity_id": int(chunk["activity_id"].iloc[0]),
                }
            )
            next_session_id += 1

    if not sessions:
        raise ValueError(
            "No WEAR sessions were created. Activity labels or subject identifiers are missing."
        )

    activity_df = pd.DataFrame(
        {
            "activity_id": list(range(len(WEAR_ACTIVITY_NAMES))),
            "activity_name": WEAR_ACTIVITY_NAMES,
        }
    ).astype({"activity_id": "int32", "activity_name": "string"})

    session_df = pd.DataFrame(session_rows).astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    if session_df["subject_id"].nunique() == 0:
        raise ValueError("WEAR subject identifiers are missing after parsing.")
    if session_df["activity_id"].nunique() == 0:
        raise ValueError("WEAR activity labels are missing after parsing.")

    return activity_df, session_df, sessions


cfg_wear = WHARConfig(
    # Info + common
    dataset_id="wear",
    dataset_url="https://mariusbock.github.io/wear/",
    download_url=[
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_0.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_1.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_2.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_3.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_4.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_5.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_6.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_7.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_8.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_9.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_10.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_11.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_12.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_13.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_14.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_15.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_16.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_17.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_18.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_19.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_20.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_21.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_22.csv",
        "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/inertial/50hz/sbj_23.csv",
    ],
    sampling_freq=50,
    num_of_subjects=24,
    num_of_activities=18,
    num_of_channels=12,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_wear,
    activity_id_col="activity_id",
    # Preprocessing (selections + sliding window)
    activity_names=WEAR_ACTIVITY_NAMES,
    sensor_channels=WEAR_SENSOR_CHANNELS,
    window_time=2,
    window_overlap=0.5,
    parallelize=True,
    # Training (split info)
)

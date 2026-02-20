import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
from whar_datasets.config.config import WHARConfig


def parse_sad(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:

    root = (
        Path(dir)
        / "sad"
        / "data"
        / "sensors-activity-recognition-dataset-shoaib"
        / "DataSet"
    )

    def load_sensor_csv(file_path, participant_counter=0):
        sensors = ["Left_pocket", "Right_pocket", "Wrist", "Upper_arm", "Belt"]
        features = [
            "time_stamp",
            "Ax",
            "Ay",
            "Az",
            "Lx",
            "Ly",
            "Lz",
            "Gx",
            "Gy",
            "Gz",
            "Mx",
            "My",
            "Mz",
        ]

        column_names = []

        for i, sensor in enumerate(sensors):
            for feature in features:
                column_names.append(f"{sensor}_{feature}")
            if i < len(sensors) - 1:
                column_names.append(f"empty_{i}")
            else:
                column_names.append("activity_name")

        df = pd.read_csv(file_path, skiprows=2, names=column_names)

        empty_cols = [col for col in df.columns if col.startswith("empty_")]
        df = df.drop(columns=empty_cols)
        df["subject_id"] = participant_counter
        return df

    all_dfs = []
    for i, participant_file in enumerate(root.glob("*.csv")):
        tmp_df = load_sensor_csv(participant_file, i)
        all_dfs.append(tmp_df)

    df = pd.concat(all_dfs, ignore_index=True)

    timestamp_cols_to_drop = [
        "Right_pocket_time_stamp",
        "Wrist_time_stamp",
        "Upper_arm_time_stamp",
        "Belt_time_stamp",
    ]
    df = df.drop(columns=timestamp_cols_to_drop)
    df = df.rename(columns={"Left_pocket_time_stamp": "start_timestamp_ms"})

    df["activity_id"] = pd.factorize(df["activity_name"])[0]

    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )
    df["session_id"] = changes.cumsum() - 1

    activity_metadata = (
        df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"], keep="first")
        .reset_index(drop=True)
    )

    session_metadata = (
        df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates(subset=["session_id"], keep="first")
        .reset_index(drop=True)
    )

    session_metadata["activity_id"] = session_metadata["activity_id"].apply(
        lambda x: [int(x)]
    )

    sessions: Dict[int, pd.DataFrame] = {}
    loop = tqdm(session_metadata["session_id"].unique())
    loop.set_description("Creating sessions")

    for session_id in loop:
        session_df = df[df["session_id"] == session_id].copy()

        start_time_ms = session_df["start_timestamp_ms"].iloc[0]

        start_datetime = pd.to_datetime(start_time_ms, unit="ms")

        session_df["timestamp"] = pd.date_range(
            start=start_datetime, periods=len(session_df), freq="20ms"
        )

        cols_to_drop = [
            "session_id",
            "subject_id",
            "activity_id",
            "activity_name",
            "start_timestamp_ms",
        ]
        session_df = session_df.drop(columns=cols_to_drop).reset_index(
            drop=True
        )

        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        session_df = session_df.astype(dtypes)

        sessions[session_id] = session_df

    return activity_metadata, session_metadata, sessions


cfg_sad = WHARConfig(
    # Info + common
    dataset_id="sad",
    download_url="https://www.utwente.nl/en/eemcs/ps/dataset-folder/sensors-activity-recognition-dataset-shoaib.rar",
    sampling_freq=50,
    num_of_subjects=10,
    num_of_activities=8,
    num_of_channels=60,
    datasets_dir="./datasets",
    parse=parse_sad,
    activity_names=[
        "walking",
        "sitting",
        "standing",
        "jogging",
        "biking",
        "upstairs",  # im readme steht walking upstairs, im csv nur upstairs
        "downstairs",  # gleiches für downstairs
    ],
    sensor_channels=[
        "Left_pocket_Ax",
        "Left_pocket_Ay",
        "Left_pocket_Az",
        "Left_pocket_Lx",
        "Left_pocket_Ly",
        "Left_pocket_Lz",
        "Left_pocket_Gx",
        "Left_pocket_Gy",
        "Left_pocket_Gz",
        "Left_pocket_Mx",
        "Left_pocket_My",
        "Left_pocket_Mz",
        "Right_pocket_Ax",
        "Right_pocket_Ay",
        "Right_pocket_Az",
        "Right_pocket_Lx",
        "Right_pocket_Ly",
        "Right_pocket_Lz",
        "Right_pocket_Gx",
        "Right_pocket_Gy",
        "Right_pocket_Gz",
        "Right_pocket_Mx",
        "Right_pocket_My",
        "Right_pocket_Mz",
        "Wrist_Ax",
        "Wrist_Ay",
        "Wrist_Az",
        "Wrist_Lx",
        "Wrist_Ly",
        "Wrist_Lz",
        "Wrist_Gx",
        "Wrist_Gy",
        "Wrist_Gz",
        "Wrist_Mx",
        "Wrist_My",
        "Wrist_Mz",
        "Upper_arm_Ax",
        "Upper_arm_Ay",
        "Upper_arm_Az",
        "Upper_arm_Lx",
        "Upper_arm_Ly",
        "Upper_arm_Lz",
        "Upper_arm_Gx",
        "Upper_arm_Gy",
        "Upper_arm_Gz",
        "Upper_arm_Mx",
        "Upper_arm_My",
        "Upper_arm_Mz",
        "Belt_Ax",
        "Belt_Ay",
        "Belt_Az",
        "Belt_Lx",
        "Belt_Ly",
        "Belt_Lz",
        "Belt_Gx",
        "Belt_Gy",
        "Belt_Gz",
        "Belt_Mx",
        "Belt_My",
        "Belt_Mz",
    ],
    window_time=2.56,
    window_overlap=0.5,
)

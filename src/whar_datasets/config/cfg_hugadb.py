import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

ID_TO_ACTIVITY = {
    0: "walking",  # ID 0 (Original 1)
    1: "running",  # ID 1 (Original 2)
    2: "going_up",  # ID 2
    3: "going_down",  # ID 3
    4: "sitting",  # ID 4
    5: "sitting down",  # ID 5 (mit Leerzeichen laut Bild)
    6: "standing up",  # ID 6 (mit Leerzeichen laut Bild)
    7: "standing",  # ID 7
    8: "bicycling",  # ID 8
    9: "up_by_elevator",  # ID 9
    10: "down_by_elevator",  # ID 10
    11: "sitting in car",
}


def parse_hugadb(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    all_dfs = []
    base_path = os.path.join(dir, "Data")

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if not file.startswith("HuGaDB") or not file.endswith(".txt"):
                continue

            full_path = os.path.join(root, file)

            df = pd.read_csv(full_path, sep="\s+", skiprows=3)  # type: ignore

            df.rename(columns={"act": "activity_id"}, inplace=True)

            df["activity_id"] = pd.to_numeric(df["activity_id"], errors="coerce")
            df["activity_id"] = df["activity_id"] - 1

            pr_id = file.replace(".txt", "").split("_")[-2]

            ##nochmal reingucken

            sampling_rate = 60.0

            activity_groups = (df["activity_id"] != df["activity_id"].shift()).cumsum()
            samples_since_change = df.groupby(activity_groups).cumcount()

            time_sec = samples_since_change * (1.0 / sampling_rate)

            df["timestamp"] = pd.to_datetime(time_sec, unit="s")
            df["subject_id"] = pr_id

            df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
            df["subject_id"] = df["subject_id"] - 1

            all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)

    changes = (
        (df["activity_id"] != df["activity_id"].shift(1))
        | (df["subject_id"] != df["subject_id"].shift(1))
        | (
            df["timestamp"] < df["timestamp"].shift(1)
        )  # Zeit springt zurück -> Neue Datei/Session
        | (df["timestamp"] == pd.Timestamp("1970-01-01"))
    ).fillna(True)

    df["session_id"] = changes.cumsum() - 1  # damit mit 0 anfängt

    metadata_cols = ["session_id", "subject_id", "activity_id"]

    # hier sind noch scenerio, trial... drin
    session_metadata = (
        df.groupby("session_id")[metadata_cols].first().reset_index(drop=True)
    )

    # create sessions
    sessions: Dict[int, pd.DataFrame] = {}

    loop = tqdm(session_metadata["session_id"].unique())
    loop.set_description("Creating sessions")

    for session_id in loop:
        # get session df
        session_df = df[df["session_id"] == session_id]

        # drop nan rows
        session_df = session_df.dropna()

        # drop metadata cols
        session_df = session_df.drop(
            columns=[
                "session_id",
                "subject_id",
                "activity_id",
            ]
        ).reset_index(drop=True)

        # set types
        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"], unit="s")
        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        dtypes["timestamp"] = "datetime64[ms]"
        session_df = session_df.round(6)
        session_df = session_df.astype(dtypes)

        # add to sessions
        sessions[session_id] = session_df

    activity_metadata = pd.DataFrame(
        list(ID_TO_ACTIVITY.items()), columns=["activity_id", "activity_name"]
    )

    # set metadata types
    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


# toDO !!Split noch nicht angepasst

cfg_hugadb = WHARConfig(
    # Info + common
    dataset_id="hugadb",
    download_url="https://github.com/romanchereshnev/HuGaDB/raw/refs/heads/master/HumanGaitDataBase.zip?download=",
    sampling_freq=60,
    num_of_subjects=18,
    num_of_activities=12,
    num_of_channels=38,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_hugadb,
    # Preprocessing (selections + sliding window)
    activity_names=[
        "walking",
        "running",
        "going_up",
        "going_down",
        "sitting",
        "sitting down",
        "standing up",
        "standing",
        "bicycling",
        "up_by_elevator",
        "down_by_elevator",
        "sitting in car",
    ],
    sensor_channels=[
        "RF_acc_x",
        "RF_acc_y",
        "RF_acc_z",
        "RF_gyro_x",
        "RF_gyro_y",
        "RF_gyro_z",
        "RS_acc_x",
        "RS_acc_y",
        "RS_acc_z",
        "RS_gyro_x",
        "RS_gyro_y",
        "RS_gyro_z",
        "RT_acc_x",
        "RT_acc_y",
        "RT_acc_z",
        "RT_gyro_x",
        "RT_gyro_y",
        "RT_gyro_z",
        "LF_acc_x",
        "LF_acc_y",
        "LF_acc_z",
        "LF_gyro_x",
        "LF_gyro_y",
        "LF_gyro_z",
        "LS_acc_x",
        "LS_acc_y",
        "LS_acc_z",
        "LS_gyro_x",
        "LS_gyro_y",
        "LS_gyro_z",
        "LT_acc_x",
        "LT_acc_y",
        "LT_acc_z",
        "LT_gyro_x",
        "LT_gyro_y",
        "LT_gyro_z",
        "R_EMG",
        "L_EMG",
    ],
    window_time=2,
    window_overlap=0.5,
)

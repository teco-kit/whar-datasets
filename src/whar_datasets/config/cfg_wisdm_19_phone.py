import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

LETTER_TO_INT = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "O": 13,
    "P": 14,
    "Q": 15,
    "R": 16,
    "S": 17,
}

ID_TO_ACTIVITY = {
    0: "walking",
    1: "jogging",
    2: "stairs",
    3: "sitting",
    4: "standing",
    5: "typing",
    6: "teeth",
    7: "soup",
    8: "chips",
    9: "pasta",
    10: "drinking",
    11: "sandwich",
    12: "kicking",
    13: "catch",
    14: "dribbling",
    15: "writing",
    16: "clapping",
    17: "folding",
}


def parse_wisdm_19_phone(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    all_subjects_list: list[pd.DataFrame] = []
    base_path = os.path.join(dir, "wisdm-dataset/wisdm-dataset/raw/phone")

    for subject in range(1600, 1651):
        accel_path = os.path.join(base_path, "accel", f"data_{subject}_accel_phone.txt")
        gyro_path = os.path.join(base_path, "gyro", f"data_{subject}_gyro_phone.txt")
        if not os.path.exists(accel_path) or not os.path.exists(gyro_path):
            continue

        accel_cols = [
            "subject_id",
            "activity_id",
            "timestamp",
            "accel_phone_x",
            "accel_phone_y",
            "accel_phone_z",
        ]
        gyro_cols = [
            "subject_id",
            "activity_id",
            "timestamp",
            "gyro_phone_x",
            "gyro_phone_y",
            "gyro_phone_z",
        ]

        accel_df = pd.read_csv(accel_path, header=None, names=accel_cols)
        gyro_df = pd.read_csv(gyro_path, header=None, names=gyro_cols)

        accel_df["accel_phone_z"] = (
            accel_df["accel_phone_z"].astype(str).str.replace(";", "", regex=False)
        )
        gyro_df["gyro_phone_z"] = (
            gyro_df["gyro_phone_z"].astype(str).str.replace(";", "", regex=False)
        )

        for col in ["timestamp", "accel_phone_x", "accel_phone_y", "accel_phone_z"]:
            accel_df[col] = pd.to_numeric(accel_df[col], errors="coerce")
        for col in ["timestamp", "gyro_phone_x", "gyro_phone_y", "gyro_phone_z"]:
            gyro_df[col] = pd.to_numeric(gyro_df[col], errors="coerce")

        accel_df = accel_df.dropna().sort_values("timestamp")
        gyro_df = gyro_df.dropna().sort_values("timestamp")

        accel_df = accel_df.groupby(
            ["subject_id", "activity_id", "timestamp"], as_index=False
        ).mean()
        gyro_df = gyro_df.groupby(
            ["subject_id", "activity_id", "timestamp"], as_index=False
        ).mean()

        merged = accel_df.merge(
            gyro_df[["timestamp", "gyro_phone_x", "gyro_phone_y", "gyro_phone_z"]],
            on="timestamp",
            how="inner",
        )
        if not merged.empty:
            all_subjects_list.append(merged)

    if not all_subjects_list:
        raise ValueError("No WISDM 2019 phone data found for parsing.")

    complete_df = pd.concat(all_subjects_list, ignore_index=True)
    complete_df["subject_id"] = complete_df["subject_id"] - 1600
    complete_df["activity_id"] = (
        complete_df["activity_id"].astype(str).str.strip().map(LETTER_TO_INT)
    )

    complete_df = complete_df.dropna(subset=["subject_id", "activity_id", "timestamp"]).copy()
    complete_df["timestamp"] = pd.to_numeric(complete_df["timestamp"], errors="coerce")
    complete_df = complete_df.dropna(subset=["timestamp"])
    complete_df["subject_id"] = complete_df["subject_id"].astype("int32")
    complete_df["activity_id"] = complete_df["activity_id"].astype("int32")
    complete_df = complete_df.sort_values(by=["subject_id", "activity_id", "timestamp"]).reset_index(drop=True)
    step_ms = int(1e3 / 20)
    complete_df["timestamp"] = (
        complete_df.groupby(["subject_id", "activity_id"]).cumcount().astype("int64")
        * step_ms
    )
    complete_df["timestamp"] = pd.to_datetime(complete_df["timestamp"], unit="ms")
    changes = (complete_df["activity_id"] != complete_df["activity_id"].shift(1)) | (
        complete_df["subject_id"] != complete_df["subject_id"].shift(1)
    )
    complete_df["session_id"] = changes.cumsum() - 1

    metadata_cols = ["session_id", "subject_id", "activity_id"]

    session_metadata = (
        complete_df.groupby("session_id")[metadata_cols].first().reset_index(drop=True)
    )

    sessions: Dict[int, pd.DataFrame] = {}

    loop = tqdm(session_metadata["session_id"].unique())
    loop.set_description("Creating sessions")

    for session_id in loop:
        # get session df
        session_df = complete_df[complete_df["session_id"] == session_id]

        # drop nan rows
        session_df = session_df.dropna()

        # drop metadata cols
        session_df = session_df.drop(
            columns=["session_id", "subject_id", "activity_id"]
        ).reset_index(drop=True)

        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"])
        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        dtypes["timestamp"] = "datetime64[ms]"
        float_cols = [col for col in session_df.columns if col != "timestamp"]
        session_df[float_cols] = session_df[float_cols].round(6)
        session_df = session_df.astype(dtypes)

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

cfg_wisdm_19_phone = WHARConfig(
    # Info + common
    dataset_id="wisdm_19_phone",
    download_url="https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
    sampling_freq=20,
    num_of_subjects=51,
    num_of_activities=18,
    num_of_channels=6,
    # Parsing
    parse=parse_wisdm_19_phone,
    # Preprocessing (selections + sliding window)
    activity_names=[
        "walking",
        "jogging",
        "stairs",
        "sitting",
        "standing",
        "typing",
        "teeth",
        "soup",
        "chips",
        "pasta",
        "drinking",
        "sandwich",
        "kicking",
        "catch",
        "dribbling",
        "writing",
        "clapping",
        "folding",
    ],
    sensor_channels=[
        "accel_phone_x",
        "accel_phone_y",
        "accel_phone_z",
        "gyro_phone_x",
        "gyro_phone_y",
        "gyro_phone_z",
    ],
    window_time=5,
    window_overlap=0.5,
)

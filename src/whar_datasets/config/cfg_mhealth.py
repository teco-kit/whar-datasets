import os
import re
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

SENSOR_COLS = [
    "chest_acc_x",  # acceleration from the chest sensor (X axis)
    "chest_acc_y",  # acceleration from the chest sensor (Y axis)
    "chest_acc_z",  # acceleration from the chest sensor (Z axis)
    "ecg_1",  # electrocardiogram signal (lead 1)
    "ecg_2",  # electrocardiogram signal (lead 2)
    "lankle_acc_x",  # acceleration from the left-ankle sensor (X axis)
    "lankle_acc_y",  # acceleration from the left-ankle sensor (Y axis)
    "lankle_acc_z",  # acceleration from the left-ankle sensor (Z axis)
    "lankle_gyro_x",  # gyro from the left-ankle sensor (X axis)
    "lankle_gyro_y",  # gyro from the left-ankle sensor (Y axis)
    "lankle_gyro_z",  # gyro from the left-ankle sensor (Z axis)
    "lankle_mag_x",  # magnetometer from the left-ankle sensor (X axis)
    "lankle_mag_y",  # magnetometer from the left-ankle sensor (Y axis)
    "lankle_mag_z",  # magnetometer from the left-ankle sensor (Z axis)
    "rarm_acc_x",  # acceleration from the right-lower-arm sensor (X axis)
    "rarm_acc_y",  # acceleration from the right-lower-arm sensor (Y axis)
    "rarm_acc_z",  # acceleration from the right-lower-arm sensor (Z axis)
    "rarm_gyro_x",  # gyro from the right-lower-arm sensor (X axis)
    "rarm_gyro_y",  # gyro from the right-lower-arm sensor (Y axis)
    "rarm_gyro_z",  # gyro from the right-lower-arm sensor (Z axis)
    "rarm_mag_x",  # magnetometer from the right-lower-arm sensor (X axis)
    "rarm_mag_y",  # magnetometer from the right-lower-arm sensor (Y axis)
    "rarm_mag_z",  # magnetometer from the right-lower-arm sensor (Z axis)
]

LABEL_COLS = [
    "activity_id",  # Label (0 for the null class)
]

ACTIVITY_MAP = {
    0: "Unknown",
    1: "Standing still",
    2: "Sitting and relaxing",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bends forward",
    7: "Frontal elevation of arms",
    8: "Knees bending (crouching)",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump front and back",
}


def parse_mhealth(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    dir = os.path.join(dir, "MHEALTHDATASET/")

    files = [file for file in os.listdir(dir) if file.endswith(".log")]

    sub_dfs = []

    for file in files:
        # get subject id from filename
        match = re.search(r"\d+", file)
        assert match
        subject_id = int(match.group(0))

        # use body sensors and labels
        sub_df = pd.read_table(
            os.path.join(dir, file),
            header=None,
            sep="\\s+",
            names=[*SENSOR_COLS, *LABEL_COLS],
        )

        # fill nans in sensor cols with linear interpolation
        sub_df.loc[:, SENSOR_COLS] = sub_df[SENSOR_COLS].interpolate(method="linear")

        # # fills nans in label cols with backward fill
        sub_df.loc[:, LABEL_COLS] = sub_df[LABEL_COLS].bfill()

        # add subject id
        sub_df["subject_id"] = subject_id

        # append to list
        sub_dfs.append(sub_df)

    # concat all sub_dfs
    df = pd.concat(sub_dfs)

    # identify where activity or subject changes or chnage in nan entries
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # add timestamp column per session
    sampling_interval = 1 / 50 * 1e3  # 50 Hz → 0.02 seconds -> to ms
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # add activity name
    df["activity_name"] = df["activity_id"].map(ACTIVITY_MAP)

    # factorize
    df["activity_id"] = df["activity_id"].factorize()[0]
    df["subject_id"] = df["subject_id"].factorize()[0]
    df["session_id"] = df["session_id"].factorize()[0]

    # create activity index
    activity_metadata = (
        df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"], keep="first")
        .reset_index(drop=True)
    )

    # create session_metadata
    session_metadata = (
        df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates(subset=["session_id"], keep="first")
        .reset_index(drop=True)
    )

    # create sessions
    sessions: Dict[int, pd.DataFrame] = {}

    # loop over sessions
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
                "activity_name",
            ]
        ).reset_index(drop=True)

        # set types
        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"], unit="ms")
        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        dtypes["timestamp"] = "datetime64[ms]"
        float_cols = [col for col in session_df.columns if col != "timestamp"]
        session_df[float_cols] = session_df[float_cols].round(6)
        session_df = session_df.astype(dtypes)

        # add to sessions
        sessions[session_id] = session_df

    # set metadata types
    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_mhealth = WHARConfig(
    # Info fields + common
    dataset_id="mhealth",
    download_url="https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
    sampling_freq=50,
    num_of_subjects=10,
    num_of_activities=13,
    num_of_channels=23,
    datasets_dir="./datasets",
    # Parsing fields
    parse=parse_mhealth,
    activity_id_col="activity_id",
    # Preprocessing fields (flatten selections + sliding_window)
    activity_names=[
        "Unknown",
        "Standing still",
        "Sitting and relaxing",
        "Lying down",
        "Walking",
        "Climbing stairs",
        "Waist bends forward",
        "Frontal elevation of arms",
        "Knees bending (crouching)",
        "Cycling",
        "Jogging",
        "Running",
        "Jump front and back",
    ],
    sensor_channels=[
        "chest_acc_x",
        "chest_acc_y",
        "chest_acc_z",
        "ecg_1",
        "ecg_2",
        "lankle_acc_x",
        "lankle_acc_y",
        "lankle_acc_z",
        "lankle_gyro_x",
        "lankle_gyro_y",
        "lankle_gyro_z",
        "lankle_mag_x",
        "lankle_mag_y",
        "lankle_mag_z",
        "rarm_acc_x",
        "rarm_acc_y",
        "rarm_acc_z",
        "rarm_gyro_x",
        "rarm_gyro_y",
        "rarm_gyro_z",
        "rarm_mag_x",
        "rarm_mag_y",
        "rarm_mag_z",
    ],
    window_time=2.56,
    window_overlap=0.5,
)

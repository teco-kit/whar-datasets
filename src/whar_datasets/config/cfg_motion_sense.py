import os
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

ACTIVITY_MAP = {
    "dws": "downstairs",
    "ups": "upstairs",
    "sit": "sitting",
    "std": "standing",
    "wlk": "walking",
    "jog": "jogging",
}


def get_sub_dfs(dir: str, names: List[str] | None) -> List[pd.DataFrame]:
    sub_dfs: List[pd.DataFrame] = []

    for sub_dir in os.listdir(dir):
        # get activity from filename
        activity_id = sub_dir.split("_")[0]

        sub_dir = os.path.join(dir, sub_dir)

        # go through all csv files
        for file in [f for f in os.listdir(sub_dir) if f.endswith(".csv")]:
            file_path = os.path.join(sub_dir, file)

            # get subject id from filename between _ and . but multiple
            subject_id = file.split("_")[1].split(".")[0]

            # read file as df
            sub_df = (
                pd.read_csv(file_path, names=names, index_col=0, header=0)
                if names is not None
                else pd.read_csv(file_path, index_col=0, header=0)
            )

            # add subject id and activity id
            sub_df["subject_id"] = subject_id
            sub_df["activity_id"] = activity_id

            # append to list
            sub_dfs.append(sub_df)

    return sub_dfs


def parse_motion_sense(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    dir = os.path.join(dir, "motion-sense-master/data/")
    motion_dir = os.path.join(dir, "A_DeviceMotion_data/A_DeviceMotion_data/")
    accel_dir = os.path.join(dir, "B_Accelerometer_data/B_Accelerometer_data/")
    gyro_dir = os.path.join(dir, "C_Gyroscope_data/C_Gyroscope_data/")

    # get dfs for each sensor type
    motion_dfs = get_sub_dfs(motion_dir, names=None)
    accel_dfs = get_sub_dfs(accel_dir, names=["accel_x", "accel_y", "accel_z"])
    gyro_dfs = get_sub_dfs(gyro_dir, names=["gyro_x", "gyro_y", "gyro_z"])

    # concatenate dfs
    sub_dfs = [
        pd.concat([m_df, a_df, g_df], axis=1)
        for m_df, a_df, g_df in zip(motion_dfs, accel_dfs, gyro_dfs)
    ]

    # remove duplicate cols
    sub_dfs = [df.loc[:, ~df.columns.duplicated()] for df in sub_dfs]

    # concatenate dfs
    df = pd.concat(sub_dfs)

    # identify where activity or subject changes or change in nan entries
    changes = (
        (df["activity_id"] != df["activity_id"].shift(1))
        | (df["subject_id"] != df["subject_id"].shift(1))
        | (df.isnull().any(axis=1) != df.isnull().any(axis=1).shift(1))
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # remove nan rows
    df = df.dropna()

    # map activity_id to activity_name
    df["activity_name"] = df["activity_id"].map(ACTIVITY_MAP)

    # factorize activity_id
    df["activity_id"] = pd.factorize(df["activity_name"])[0]

    # add timestamp column per session
    sampling_interval = 1 / 50 * 1e3  # 50 Hz → 0.02 seconds -> to ms
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

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


cfg_motion_sense = WHARConfig(
    # Info + common
    dataset_id="motion_sense",
    download_url="https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip",
    sampling_freq=50,
    num_of_subjects=24,
    num_of_activities=6,
    num_of_channels=18,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_motion_sense,
    activity_id_col="activity_id",
    # Preprocessing (selections + sliding window)
    activity_names=[
        "downstairs",
        "upstairs",
        "walking",
        "jogging",
        "sitting",
        "standing",
    ],
    sensor_channels=[
        "attitude.roll",
        "attitude.pitch",
        "attitude.yaw",
        "gravity.x",
        "gravity.y",
        "gravity.z",
        "rotationRate.x",
        "rotationRate.y",
        "rotationRate.z",
        "userAcceleration.x",
        "userAcceleration.y",
        "userAcceleration.z",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ],
    window_time=2.56,
    window_overlap=0.5,
    # Training (split info)
)

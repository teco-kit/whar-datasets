import os
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

ACTIVITY_MAP = {
    "dws": "downstairs",
    "ups": "upstairs",
    "sit": "sitting",
    "std": "standing",
    "wlk": "walking",
    "jog": "jogging",
}


def get_sub_dfs(
    dir: str, names: List[str] | None
) -> Dict[Tuple[str, str], pd.DataFrame]:
    # The directory suffix is a repetition/recording identifier. It must be
    # part of the key: the same subject performs several recordings of some
    # activities (for example dws_1, dws_2, and dws_11).
    sub_dfs: Dict[Tuple[str, str], pd.DataFrame] = {}

    for sub_dir_name in sorted(os.listdir(dir)):
        if not os.path.isdir(os.path.join(dir, sub_dir_name)):
            continue
        # get activity from filename
        activity_id = sub_dir_name.split("_")[0]

        sub_dir = os.path.join(dir, sub_dir_name)

        # go through all csv files
        for file in sorted(f for f in os.listdir(sub_dir) if f.endswith(".csv")):
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

            key = (sub_dir_name, subject_id)
            if key in sub_dfs:
                raise ValueError(
                    f"Duplicate MotionSense recording for directory={sub_dir_name}, "
                    f"subject={subject_id}."
                )
            sub_df["source_recording_id"] = sub_dir_name
            sub_dfs[key] = sub_df

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

    keys = set(motion_dfs).intersection(accel_dfs, gyro_dfs)
    all_keys = set(motion_dfs).union(accel_dfs, gyro_dfs)
    if keys != all_keys:
        missing = sorted(all_keys.difference(keys))
        raise ValueError(
            "MotionSense sensor files do not have matching activity/subject keys: "
            + ", ".join(f"{recording}/{subject}" for recording, subject in missing[:10])
        )

    sub_dfs: List[pd.DataFrame] = []
    for key in sorted(keys):
        m_df, a_df, g_df = motion_dfs[key], accel_dfs[key], gyro_dfs[key]
        # MotionSense stores no physical timestamp; the shared row index is
        # the only available time base. Align the simultaneous streams by
        # that index and retain their common prefix explicitly. The source
        # contains occasional trailing samples in only one stream.
        sample_count = min(len(m_df), len(a_df), len(g_df))
        if sample_count == 0:
            raise ValueError(
                f"MotionSense recording {key[0]}/{key[1]} is empty."
            )
        sub_dfs.append(
            pd.concat(
                [
                    m_df.iloc[:sample_count].reset_index(drop=True),
                    a_df.iloc[:sample_count].reset_index(drop=True),
                    g_df.iloc[:sample_count].reset_index(drop=True),
                ],
                axis=1,
            )
        )

    # remove duplicate cols
    sub_dfs = [df.loc[:, ~df.columns.duplicated()] for df in sub_dfs]

    # concatenate dfs
    df = pd.concat(sub_dfs)

    # identify where activity or subject changes or change in nan entries
    changes = (
        (df["activity_id"] != df["activity_id"].shift(1))
        | (df["subject_id"] != df["subject_id"].shift(1))
        | (df["source_recording_id"] != df["source_recording_id"].shift(1))
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
                "source_recording_id",
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


ALL_ACTIVITIES = [
    "downstairs",
    "upstairs",
    "walking",
    "jogging",
    "sitting",
    "standing",
]

ALL_CHANNELS = [
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
]


SELECTED_ACTIVITIES = ALL_ACTIVITIES

cfg_motion_sense = WHARConfig(
    # Info + common
    dataset_id="motion_sense",
    dataset_url="https://github.com/mmalekzadeh/motion-sense",
    download_url="https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip",
    sampling_freq=50,
    num_of_subjects=24,
    num_of_activities=6,
    num_of_channels=18,
    # Parsing
    parse=parse_motion_sense,
    activity_id_col="activity_id",
    # Preprocessing (selections + sliding window)
    available_activities=canonicalize_activity_name_list(ALL_ACTIVITIES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=ALL_CHANNELS,
    selected_channels=ALL_CHANNELS,
    # Training (split info)
)

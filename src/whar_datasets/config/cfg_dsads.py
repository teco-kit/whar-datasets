import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

SENSOR_COLS = [
    # Trunk (T)
    "T_xacc",
    "T_yacc",
    "T_zacc",
    "T_xgyro",
    "T_ygyro",
    "T_zgyro",
    "T_xmag",
    "T_ymag",
    "T_zmag",
    # Right Arm (RA)
    "RA_xacc",
    "RA_yacc",
    "RA_zacc",
    "RA_xgyro",
    "RA_ygyro",
    "RA_zgyro",
    "RA_xmag",
    "RA_ymag",
    "RA_zmag",
    # Left Arm (LA)
    "LA_xacc",
    "LA_yacc",
    "LA_zacc",
    "LA_xgyro",
    "LA_ygyro",
    "LA_zgyro",
    "LA_xmag",
    "LA_ymag",
    "LA_zmag",
    # Right Leg (RL)
    "RL_xacc",
    "RL_yacc",
    "RL_zacc",
    "RL_xgyro",
    "RL_ygyro",
    "RL_zgyro",
    "RL_xmag",
    "RL_ymag",
    "RL_zmag",
    # Left Leg (LL)
    "LL_xacc",
    "LL_yacc",
    "LL_zacc",
    "LL_xgyro",
    "LL_ygyro",
    "LL_zgyro",
    "LL_xmag",
    "LL_ymag",
    "LL_zmag",
]

ACTIVITY_MAP = {
    1: "sitting",
    2: "standing",
    3: "lying on back",
    4: "lying on right side",
    5: "ascending stairs",
    6: "descending stairs",
    7: "standing in an elevator still",
    8: "moving around in an elevator",
    9: "walking in a parking lot",
    10: "walking on treadmill (flat, 4 km/h)",
    11: "walking on treadmill (15° incline, 4 km/h)",
    12: "running on treadmill (8 km/h)",
    13: "exercising on a stepper",
    14: "exercising on a cross trainer",
    15: "cycling on exercise bike (horizontal)",
    16: "cycling on exercise bike (vertical)",
    17: "rowing",
    18: "jumping",
    19: "playing basketball",
}


def parse_dsads(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    dir = os.path.join(dir, "data/")

    sub_dfs = []

    for activity_dir in os.listdir(dir):
        # get activity from dirname
        activity_id = int(activity_dir[1:])

        activity_dir = os.path.join(dir, activity_dir)

        for subject_dir in os.listdir(activity_dir):
            # get subject id from dirname
            subject_id = int(subject_dir[1:])

            subject_dir = os.path.join(activity_dir, subject_dir)

            sub_df = pd.concat(
                [
                    pd.read_csv(
                        os.path.join(subject_dir, file),
                        header=None,
                        names=SENSOR_COLS,
                    )
                    for file in [
                        f for f in os.listdir(subject_dir) if f.endswith(".txt")
                    ]
                ]
            )

            # fill nans in sensor cols with linear interpolation
            sub_df.loc[:, SENSOR_COLS] = sub_df[SENSOR_COLS].interpolate(
                method="linear"
            )

            sub_df["subject_id"] = subject_id
            sub_df["activity_id"] = activity_id

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
    sampling_interval = 1 / 25 * 1e9  # 25 Hz → 0.02 seconds -> to ns
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")

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


cfg_dsads = WHARConfig(
    # Info fields + common
    dataset_id="dsads",
    download_url="https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip",
    sampling_freq=25,
    num_of_subjects=8,
    num_of_activities=19,
    num_of_channels=45,
    datasets_dir="./datasets",
    # Parsing fields
    parse=parse_dsads,
    activity_id_col="activity_id",
    # Preprocessing fields (flatten selections + sliding_window)
    activity_names=[
        "sitting",  # A1
        "standing",  # A2
        "lying on back",  # A3
        "lying on right side",  # A4
        "ascending stairs",  # A5
        "descending stairs",  # A6
        "standing in an elevator still",  # A7
        "moving around in an elevator",  # A8
        "walking in a parking lot",  # A9
        "walking on treadmill (flat, 4 km/h)",  # A10
        "walking on treadmill (15° incline, 4 km/h)",  # A11
        "running on treadmill (8 km/h)",  # A12
        "exercising on a stepper",  # A13
        "exercising on a cross trainer",  # A14
        "cycling on exercise bike (horizontal)",  # A15
        "cycling on exercise bike (vertical)",  # A16
        "rowing",  # A17
        "jumping",  # A18
        "playing basketball",  # A19
    ],
    sensor_channels=[
        # Trunk (T)
        "T_xacc",
        "T_yacc",
        "T_zacc",
        "T_xgyro",
        "T_ygyro",
        "T_zgyro",
        "T_xmag",
        "T_ymag",
        "T_zmag",
        # Right Arm (RA)
        "RA_xacc",
        "RA_yacc",
        "RA_zacc",
        "RA_xgyro",
        "RA_ygyro",
        "RA_zgyro",
        "RA_xmag",
        "RA_ymag",
        "RA_zmag",
        # Left Arm (LA)
        "LA_xacc",
        "LA_yacc",
        "LA_zacc",
        "LA_xgyro",
        "LA_ygyro",
        "LA_zgyro",
        "LA_xmag",
        "LA_ymag",
        "LA_zmag",
        # Right Leg (RL)
        "RL_xacc",
        "RL_yacc",
        "RL_zacc",
        "RL_xgyro",
        "RL_ygyro",
        "RL_zgyro",
        "RL_xmag",
        "RL_ymag",
        "RL_zmag",
        # Left Leg (LL)
        "LL_xacc",
        "LL_yacc",
        "LL_zacc",
        "LL_xgyro",
        "LL_ygyro",
        "LL_zgyro",
        "LL_xmag",
        "LL_ymag",
        "LL_zmag",
    ],
    window_time=5.0,
    window_overlap=0.5,
)

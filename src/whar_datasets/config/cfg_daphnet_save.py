import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

ACTIVITY_MAP = {
    0: "Unknown",
    1: "No Freeze",
    2: "Freeze",
}


def parse_daphnet(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    dir = os.path.join(dir, "dataset_fog_release/dataset/")

    files = [f for f in os.listdir(dir) if f.endswith(".txt")]

    sub_dfs = []

    for file in files:
        # read file
        sub_df = pd.read_table(
            os.path.join(dir, file),
            sep="\\s+",
            header=None,
            names=[
                "timestamp",
                "shank_acc_x",
                "shank_acc_y",
                "shank_acc_z",
                "thigh_acc_x",
                "thigh_acc_y",
                "thigh_acc_z",
                "trunk_acc_x",
                "trunk_acc_y",
                "trunk_acc_z",
                "activity_id",
            ],
        )

        # add subject id from filename
        sub_df["subject_id"] = int(file[1:3])

        sub_dfs.append(sub_df)

    global_session_id = 0

    for sub_df in sub_dfs:
        # identify where activity or subject changes
        changes = (sub_df["activity_id"] != sub_df["activity_id"].shift(1)) | (
            sub_df["subject_id"] != sub_df["subject_id"].shift(1)
        )

        # compute local session ids
        local_session_ids = changes.cumsum()

        # offset by global counter
        sub_df["session_id"] = local_session_ids + global_session_id

        # update global counter
        global_session_id = sub_df["session_id"].max() + 1

    # concat all sub_dfs
    df = pd.concat(sub_dfs)

    # convert timestamp to datetime in ns
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")

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
        session_df = session_df.round(6)
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


cfg_daphnet = WHARConfig(
    # Info fields + common
    dataset_id="daphnet",
    download_url="https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip",
    sampling_freq=64,
    num_of_subjects=10,
    num_of_activities=3,
    num_of_channels=9,
    datasets_dir="./datasets",
    # Parsing fields
    parse=parse_daphnet,
    activity_id_col="activity_id",
    # Preprocessing fields (flatten selections + sliding_window)
    activity_names=[
        "Unknown",
        "No Freeze",
        "Freeze",
    ],
    sensor_channels=[
        "shank_acc_x",
        "shank_acc_y",
        "shank_acc_z",
        "thigh_acc_x",
        "thigh_acc_y",
        "thigh_acc_z",
        "trunk_acc_x",
        "trunk_acc_y",
        "trunk_acc_z",
    ],
    window_time=1.0,
    window_overlap=0.5,
    # Training fields (flattened splits)
)

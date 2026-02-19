import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig


def parse_har_sense(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    files = [f for f in os.listdir(dir) if f.endswith(".csv") and f != "har_sense.csv"]

    sub_dfs = []

    for file in files:
        sub_df = pd.read_csv(os.path.join(dir, file))
        sub_df["subject_id"] = int(file.split("b")[1].split("_")[0])

        # if it has col Axx-Y, rename it to Acc-Y
        if "Axx-Y" in sub_df.columns:
            sub_df = sub_df.rename(columns={"Axx-Y": "Acc-Y"})

        sub_dfs.append(sub_df)

    # concat all sub_dfs
    df = pd.concat(sub_dfs)

    # rename activity column
    df = df.rename(columns={"activity": "activity_name"})

    # replace downstaires activtiy name with Downstairs, use map
    df["activity_name"] = df["activity_name"].replace(
        {
            "downstaires": "Downstairs",
            "upstaires": "Upstairs",
            "sitting": "Sitting",
            "walking": "Walking",
        }
    )

    # get activity id from activity name
    df["activity_id"] = df["activity_name"].factorize()[0]

    # identify where activity or subject changes
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # create session id
    df["session_id"] = changes.cumsum()

    # add timestamp column per session
    sampling_interval = 1 / 50 * 1e3  # 50 Hz → 0.02 seconds -> to ms
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )

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


cfg_har_sense = WHARConfig(
    # Info fields + common
    dataset_id="har_sense",
    download_url="https://www.kaggle.com/api/v1/datasets/download/nurulaminchoudhury/harsense-datatset",
    sampling_freq=50,
    num_of_subjects=12,
    num_of_activities=7,
    num_of_channels=16,
    datasets_dir="./datasets",
    # Parsing fields
    parse=parse_har_sense,
    activity_id_col="activity_id",
    # Preprocessing fields (flatten selections + sliding_window)
    activity_names=[
        "Walking",
        "Standing",
        "Upstairs",
        "Downstairs",
        "Running",
        "Sitting",
        "Sleeping",
    ],
    sensor_channels=[
        "AG-X",
        "AG-Y",
        "AG-Z",
        "Acc-X",
        "Acc-Y",
        "Acc-Z",
        "Gravity-X",
        "Gravity-Y",
        "Gravity-Z",
        "RR-X",
        "RR-Y",
        "RR-Z",
        "RV-X",
        "RV-Y",
        "RV-Z",
        "cos",
    ],
    window_time=2.56,
    window_overlap=0.0,
    # Training fields (flattened splits)
)

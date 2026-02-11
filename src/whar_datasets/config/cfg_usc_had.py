import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

ID_TO_ACTIVITY = {
    0: "Walking Forward",
    1: "Walking Left",
    2: "Walking Right",
    3: "Walking Upstairs",
    4: "Walking Downstairs",
    5: "Running Forward",
    6: "Jumping Up",
    7: "Sitting",
    8: "Standing",
    9: "Sleeping",
    10: "Elevator Up",
    11: "Elevator Down",
}


def parse_usc_had(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:

    root_path = os.path.join(dir, r"USC-HAD")
    all_dfs = []
    cols = ["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ"]

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".mat"):
                file_path = os.path.join(root, file)

                try:
                    mat = scipy.io.loadmat(file_path)

                    df = pd.DataFrame(mat["sensor_readings"], columns=cols)

                    df["subject_id"] = mat["subject"].item()
                    try:
                        df["activity_id"] = mat["activity_number"].item()
                    except KeyError:
                        df["activity_id"] = mat["activity_numbr"].item()

                    sampling_rate = 100  # Hz

                    time_sec = np.arange(len(df)) * (1.0 / sampling_rate)

                    # timestamps existier vorher noch nicht?
                    df["timestamp"] = pd.to_timedelta(time_sec, unit="s")

                    all_dfs.append(df)

                except Exception as e:
                    # bei subject 13 ist "activity_numbr statt number"
                    print(f"Error in {file}: {e}")

    big_df = pd.concat(all_dfs, ignore_index=True)
    df = big_df

    df["activity_id"] = pd.to_numeric(df["activity_id"], errors="coerce")
    df["activity_id"] = df["activity_id"] - 1

    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
    df["subject_id"] = df["subject_id"] - 1

    changes = (
        (df["activity_id"] != df["activity_id"].shift(1))
        | (df["subject_id"] != df["subject_id"].shift(1))
        | (df["timestamp"] == 0)
        | (df["timestamp"] == pd.to_timedelta(0, unit="s"))
    )

    df["session_id"] = changes.cumsum() - 1

    df = df.sort_values(by=["session_id", "timestamp"])

    metadata_cols = ["session_id", "subject_id", "activity_id"]

    session_metadata = (
        df.groupby("session_id")[metadata_cols].first().reset_index(drop=True)
    )

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
            columns=["session_id", "subject_id", "activity_id"]
        ).reset_index(drop=True)

        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"], unit="ns")
        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        dtypes["timestamp"] = "datetime64[ms]"
        session_df = session_df.round(6)
        session_df = session_df.astype(dtypes)

        sessions[session_id] = session_df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")
    dtypes = {}
    for col in df.columns:
        if col == "timestamp":
            dtypes[col] = "datetime64[ms]"  # Hier wird von ns auf ms gekürzt
        elif col in ["subject_id", "activity_id"]:
            dtypes[col] = "int32"
        else:
            dtypes[col] = "float32"

    float_cols = [c for c, t in dtypes.items() if t == "float32"]
    df[float_cols] = df[float_cols].round(6)

    df = df.astype(dtypes)

    activity_metadata = pd.DataFrame(
        list(ID_TO_ACTIVITY.items()), columns=["activity_id", "activity_name"]
    )

    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


# config Zeugs
cfg_usc_had = WHARConfig(
    dataset_id="usc_had",
    download_url="https://sipi.usc.edu/had/USC-HAD.zip",
    sampling_freq=100,
    num_of_subjects=14,
    num_of_activities=12,
    num_of_channels=6,
    datasets_dir="./datasets/",
    # Parsing
    parse=parse_usc_had,
    # Preprocessing (selections + sliding window)
    # verschiedene Aktivitäten
    activity_names=[
        "Walking Forward",
        "Walking Left",
        "Walking Right",
        "Walking Upstairs",
        "Walking Downstairs",
        "Running Forward",
        "Jumping Up",
        "Sitting",
        "Standing",
        "Sleeping",
        "Elevator Up",
        "Elevator Down",
    ],
    sensor_channels=["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ"],
    window_time=1.28,
    window_overlap=0.5,
)

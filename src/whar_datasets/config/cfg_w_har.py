import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.config.timestamps import to_datetime64_ms


def parse_w_har(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    file_path = os.path.join(dir, "motion_data_22_users.csv")

    with open(file_path, "r") as file:
        broken_header = file.readline().strip()

    current_cols = broken_header.split(",")
    current_cols.append("activity_id")

    df = pd.read_csv(file_path, header=0, names=current_cols)

    df.rename(columns={"User": "subject_id"}, inplace=True)
    df.rename(columns={"Time (s)": "timestamp"}, inplace=True)

    # subject id muss mit 0 anfangen
    df["subject_id"] = pd.factorize(df["subject_id"], sort=True)[0]

    # codes ist activity spalte jz in ids, uniques sind Namen sodass 0.te ist walk, 1. transition...
    codes, uniques = pd.factorize(df["activity_id"])

    df["activity_id"] = codes

    activity_metadata = pd.DataFrame(
        {"activity_id": range(len(uniques)), "activity_name": uniques}
    )

    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (df["timestamp"] == 0)

    df["session_id"] = changes.cumsum() - 1  # damit mit 0 anfängt

    metadata_cols = ["session_id", "subject_id", "Scenerio", "Trial", "activity_id"]

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
            columns=["session_id", "subject_id", "activity_id", "Trial", "Scenerio"]
        ).reset_index(drop=True)

        # set types
        session_df["timestamp"] = to_datetime64_ms(
            session_df["timestamp"], default_unit="s"
        )
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

    # print("min subject_id: " + df["subject_id"].min())

    return activity_metadata, session_metadata, sessions


# config Zeugs
cfg_w_har = WHARConfig(
    dataset_id="w_har",
    download_url="https://github.com/gmbhat/human-activity-recognition/raw/refs/heads/master/datasets/raw_data/motion_data_22_users.csv",
    sampling_freq=250,  # ist das pro Sekunde? dann ist 250 richtig
    num_of_subjects=22,
    num_of_activities=9,
    num_of_channels=6,
    # Parsing
    parse=parse_w_har,
    # Preprocessing (selections + sliding window)
    # verschiedene Aktivitäten
    activity_names=[
        "walk",
        "transition",
        "sit",
        "stand",
        "jumpundefined",
        "liedown",
        "stairsup",
        "stairsdown",
    ],
    sensor_channels=["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ"],
    window_time=1.28,
    window_overlap=0.5,
)

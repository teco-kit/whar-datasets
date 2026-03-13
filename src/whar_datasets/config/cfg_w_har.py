import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig
from whar_datasets.config.timestamps import to_datetime64_ms


def _canonical_activity_label(raw_label: str) -> str:
    token = "".join(ch for ch in str(raw_label).strip().lower() if ch.isalnum())
    mapping = {
        "walk": "Walking",
        "walking": "Walking",
        "transition": "Transition",
        "sit": "Sitting",
        "sitting": "Sitting",
        "stand": "Standing",
        "standing": "Standing",
        "jumpundefined": "Jumping",
        "jump": "Jumping",
        "jumping": "Jumping",
        "liedown": "Lying",
        "laydown": "Lying",
        "lie": "Lying",
        "lying": "Lying",
        "stairsup": "Upstairs",
        "upstairs": "Upstairs",
        "walkingupstairs": "Upstairs",
        "stairsdown": "Downstairs",
        "downstairs": "Downstairs",
        "walkingdownstairs": "Downstairs",
        "unknown": "Unknown",
        "unkown": "Unknown",
    }
    return mapping.get(
        token, (token[:1].upper() + token[1:].lower()) if token else "Unknown"
    )


def parse_w_har(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col
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

    # Map raw labels to a standardized, single-word naming scheme.
    df["activity_name"] = df["activity_id"].astype(str).map(_canonical_activity_label)
    codes, uniques = pd.factorize(df["activity_name"], sort=False)

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
            columns=[
                "session_id",
                "subject_id",
                "activity_id",
                "activity_name",
                "Trial",
                "Scenerio",
            ]
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
ALL_ACTIVITIES = [
    "Walking",
    "Transition",
    "Sitting",
    "Standing",
    "Jumping",
    "Lying",
    "Upstairs",
    "Downstairs",
    "Unknown",
]
SELECTED_ACTIVITIES = [
    "Walking",
    "Transition",
    "Sitting",
    "Standing",
    "Jumping",
    "Lying",
    "Upstairs",
    "Downstairs",
]

ALL_CHANNELS = ["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ"]


cfg_w_har = WHARConfig(
    dataset_id="w_har",
    dataset_url="https://github.com/gmbhat/human-activity-recognition",
    download_url="https://github.com/gmbhat/human-activity-recognition/raw/refs/heads/master/datasets/raw_data/motion_data_22_users.csv",
    sampling_freq=250,  # ist das pro Sekunde? dann ist 250 richtig
    num_of_subjects=22,
    num_of_activities=9,
    num_of_channels=6,
    # Parsing
    parse=parse_w_har,
    available_activities=canonicalize_activity_name_list(ALL_ACTIVITIES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=ALL_CHANNELS,
    selected_channels=ALL_CHANNELS,
    window_time=1.28,
    window_overlap=0.5,
)

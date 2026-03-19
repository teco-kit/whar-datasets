import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig
from whar_datasets.config.timestamps import to_datetime64_ms

W_HAR_MOTION_CHANNELS = ["Ax", "Ay", "Az", "GyroX", "GyroY", "GyroZ"]
W_HAR_STRETCH_RAW_COLUMN = "Stretch Value"
W_HAR_STRETCH_CHANNEL = "StretchValue"


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


def _read_w_har_broken_csv(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as file:
        broken_header = file.readline().strip()

    cols = broken_header.split(",")
    cols.append("activity_id")
    return pd.read_csv(file_path, header=0, names=cols)


def _estimate_stretch_tolerance_seconds(stretch_df: pd.DataFrame) -> float:
    group_cols = ["subject_raw", "Scenerio", "Trial", "activity_name"]
    diffs = stretch_df.groupby(group_cols, sort=False)["timestamp"].diff()
    positive_diffs = diffs[(diffs > 0) & diffs.notna()]
    if positive_diffs.empty:
        return 0.05
    return max(float(positive_diffs.median()) * 2.5, 0.02)


def parse_w_har(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col
    motion_path = os.path.join(dir, "motion_data_22_users.csv")
    stretch_path = os.path.join(dir, "stretch_data_22_users.csv")

    if not os.path.exists(motion_path):
        raise FileNotFoundError(f"W-HAR motion file not found: {motion_path}")
    if not os.path.exists(stretch_path):
        raise FileNotFoundError(
            f"W-HAR stretch file not found: {stretch_path}. "
            "Both motion and stretch files are required."
        )

    motion_df = _read_w_har_broken_csv(motion_path)
    stretch_df = _read_w_har_broken_csv(stretch_path)

    motion_df = motion_df.rename(
        columns={"User": "subject_raw", "Time (s)": "timestamp"}
    )
    stretch_df = stretch_df.rename(
        columns={
            "User": "subject_raw",
            "Time (s)": "timestamp",
            W_HAR_STRETCH_RAW_COLUMN: W_HAR_STRETCH_CHANNEL,
        }
    )

    if W_HAR_STRETCH_CHANNEL not in stretch_df.columns:
        raise ValueError(
            f"Expected stretch column '{W_HAR_STRETCH_RAW_COLUMN}' in {stretch_path}."
        )

    for frame in (motion_df, stretch_df):
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame["subject_raw"] = pd.to_numeric(frame["subject_raw"], errors="coerce")
        frame["Scenerio"] = pd.to_numeric(frame["Scenerio"], errors="coerce")
        frame["Trial"] = pd.to_numeric(frame["Trial"], errors="coerce")
        frame["activity_name"] = (
            frame["activity_id"].astype(str).map(_canonical_activity_label)
        )

    for col in W_HAR_MOTION_CHANNELS:
        motion_df[col] = pd.to_numeric(motion_df[col], errors="coerce")
    stretch_df[W_HAR_STRETCH_CHANNEL] = pd.to_numeric(
        stretch_df[W_HAR_STRETCH_CHANNEL], errors="coerce"
    )

    merge_keys = ["subject_raw", "Scenerio", "Trial", "activity_name"]
    motion_df = motion_df[[*merge_keys, "timestamp", *W_HAR_MOTION_CHANNELS]].dropna(
        subset=[*merge_keys, "timestamp"]
    )
    stretch_df = stretch_df[[*merge_keys, "timestamp", W_HAR_STRETCH_CHANNEL]].dropna(
        subset=[*merge_keys, "timestamp"]
    )

    motion_df = motion_df.sort_values([*merge_keys, "timestamp"]).drop_duplicates(
        subset=[*merge_keys, "timestamp"], keep="first"
    )
    stretch_df = stretch_df.sort_values([*merge_keys, "timestamp"]).drop_duplicates(
        subset=[*merge_keys, "timestamp"], keep="first"
    )

    tolerance_s = _estimate_stretch_tolerance_seconds(stretch_df)
    df = pd.merge_asof(
        motion_df.sort_values(["timestamp", *merge_keys]),
        stretch_df.sort_values(["timestamp", *merge_keys]),
        on="timestamp",
        by=merge_keys,
        direction="nearest",
        tolerance=tolerance_s,  # type: ignore
    )
    df[W_HAR_STRETCH_CHANNEL] = df.groupby(merge_keys, sort=False)[
        W_HAR_STRETCH_CHANNEL
    ].transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
    df[W_HAR_STRETCH_CHANNEL] = df.groupby(merge_keys, sort=False)[
        W_HAR_STRETCH_CHANNEL
    ].transform(lambda s: s.ffill().bfill().fillna(0.0))
    df = df.sort_values([*merge_keys, "timestamp"]).reset_index(drop=True)

    # subject id muss mit 0 anfangen
    df["subject_id"] = pd.factorize(df["subject_raw"], sort=True)[0]

    # Map raw labels to a standardized, single-word naming scheme.
    codes, uniques = pd.factorize(df["activity_name"], sort=False)
    df["activity_id"] = codes

    activity_metadata = pd.DataFrame(
        {"activity_id": range(len(uniques)), "activity_name": uniques}
    )

    time_diff = df.groupby(merge_keys, sort=False)["timestamp"].diff()
    positive_diffs = time_diff[(time_diff > 0) & time_diff.notna()]
    gap_threshold = (
        float(positive_diffs.median() * 10) if not positive_diffs.empty else 1.0
    )
    changes = (
        (df["subject_raw"] != df["subject_raw"].shift(1))
        | (df["Scenerio"] != df["Scenerio"].shift(1))
        | (df["Trial"] != df["Trial"].shift(1))
        | (df["activity_id"] != df["activity_id"].shift(1))
        | time_diff.isna()
        | (time_diff < 0)
        | (time_diff > gap_threshold)
    )

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
                "subject_raw",
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

ALL_CHANNELS = [*W_HAR_MOTION_CHANNELS, W_HAR_STRETCH_CHANNEL]


cfg_w_har = WHARConfig(
    dataset_id="w_har",
    dataset_url="https://github.com/gmbhat/human-activity-recognition",
    download_url=[
        "https://github.com/gmbhat/human-activity-recognition/raw/refs/heads/master/datasets/raw_data/motion_data_22_users.csv",
        "https://github.com/gmbhat/human-activity-recognition/raw/refs/heads/master/datasets/raw_data/stretch_data_22_users.csv",
    ],
    sampling_freq=250,  # ist das pro Sekunde? dann ist 250 richtig
    num_of_subjects=22,
    num_of_activities=9,
    num_of_channels=7,
    # Parsing
    parse=parse_w_har,
    available_activities=canonicalize_activity_name_list(ALL_ACTIVITIES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=ALL_CHANNELS,
    selected_channels=ALL_CHANNELS,
    window_time=1.28,
    window_overlap=0.5,
)

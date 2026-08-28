import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

WISDM_MAX_GAP_SECONDS = 1.0


def parse_wisdm_12(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    dir = os.path.join(dir, "WISDM_ar_v1.1/")
    file_path = os.path.join(dir, "WISDM_ar_v1.1_raw.txt")

    # Read the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse all entries into a list of lists
    data = []

    for line in lines:
        # Remove whitespace and newline characters
        line = line.strip()

        if not line:
            continue

        # Split by semicolon to get individual entries
        entries = line.split(";")

        for entry in entries:
            # Skip empty entries
            if len(entry) == 0:
                continue
            # Some entries have a trailing comma
            if entry[-1] == ",":
                entry = entry[:-1]

            # Split each entry by comma
            fields = entry.split(",")

            # Skip entries with too many or too few entries
            if len(fields) != 6:
                continue

            data.append(fields)

    # Create a DataFrame
    df = pd.DataFrame(
        data,
        columns=[
            "subject_id",
            "activity_name",
            "timestamp",
            "accel_x",
            "accel_y",
            "accel_z",
        ],
    )

    # Keep source row order: it is the only available recording chronology.
    # Sorting by activity would merge separate repetitions of the same label.
    df["subject_raw"] = df["subject_id"].astype(str).str.strip()
    df["activity_raw"] = df["activity_name"].astype(str).str.strip().str.lower()
    activity_map = {
        name.lower(): (activity_id, name)
        for activity_id, name in enumerate(ALL_ACTIVITIES)
    }
    unknown_activities = sorted(set(df["activity_raw"]).difference(activity_map))
    if unknown_activities:
        raise ValueError(
            "Found WISDM activity labels not covered by the configured mapping: "
            + ", ".join(unknown_activities)
        )

    df["activity_id"] = df["activity_raw"].map(
        lambda name: activity_map[name][0]
    ).astype("int32")
    df["activity_name"] = df["activity_raw"].map(
        lambda name: activity_map[name][1]
    )

    # Parse timestamps as full-precision integer ns to avoid precision loss.
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()]
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df[df["timestamp"] != 0]

    # drop nan rows
    df = df.dropna()

    # change timestamp to datetime in ns
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")

    df["timestamp_raw"] = df["timestamp"]
    value_cols = ["accel_x", "accel_y", "accel_z"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["subject_raw", "activity_raw", *value_cols]).copy()
    df = df.drop_duplicates(
        subset=["subject_raw", "activity_id", "timestamp_raw"], keep="first"
    ).reset_index(drop=True)

    raw_subjects = df["subject_raw"].unique().tolist()
    numeric_subjects = pd.to_numeric(pd.Series(raw_subjects), errors="coerce")
    if numeric_subjects.isna().any() or (numeric_subjects % 1 != 0).any():
        raise ValueError(
            "WISDM subject identifiers must be integer-like; found "
            + ", ".join(map(str, raw_subjects))
        )
    subject_order = [
        raw_subject
        for _, raw_subject in sorted(
            zip(numeric_subjects.astype("int64"), raw_subjects),
            key=lambda item: item[0],
        )
    ]
    subject_map = {
        raw_subject: subject_id for subject_id, raw_subject in enumerate(subject_order)
    }
    df["subject_id"] = df["subject_raw"].map(subject_map).astype("int32")

    raw_time = pd.to_datetime(df["timestamp_raw"], unit="ns")
    time_diff = raw_time.diff().dt.total_seconds()
    session_start = (
        df["subject_id"].ne(df["subject_id"].shift(1))
        | df["activity_id"].ne(df["activity_id"].shift(1))
        | time_diff.isna()
        | time_diff.le(0.0)
        | time_diff.gt(WISDM_MAX_GAP_SECONDS)
    )
    df["session_id"] = session_start.astype("int64").cumsum() - 1

    # Normalize each source-contiguous session to a stable 20 Hz timeline.
    step_ms = int(1e3 / 20)
    df["timestamp"] = (
        df.groupby("session_id").cumcount().astype("int64") * step_ms
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

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

        # drop metadata cols
        session_df = session_df.drop(
            columns=[
                "session_id",
                "subject_id",
                "activity_id",
                "activity_name",
                "subject_raw",
                "activity_raw",
                "timestamp_raw",
            ]
        ).reset_index(drop=True)

        # set types
        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"])
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
    "Walking",
    "Jogging",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing",
]

ALL_CHANNELS = [
    "accel_x",
    "accel_y",
    "accel_z",
]


SELECTED_ACTIVITIES = ALL_ACTIVITIES

cfg_wisdm = WHARConfig(
    # Info + common
    dataset_id="wisdm",
    dataset_url="https://www.cis.fordham.edu/wisdm/dataset.php",
    download_url="https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
    sampling_freq=20,
    num_of_subjects=36,
    num_of_activities=6,
    num_of_channels=3,
    # Parsing
    parse=parse_wisdm_12,
    # Preprocessing (selections + sliding window)
    available_activities=canonicalize_activity_name_list(ALL_ACTIVITIES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=ALL_CHANNELS,
    selected_channels=ALL_CHANNELS,
    # Training (split info)
)

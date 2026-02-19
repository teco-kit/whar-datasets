import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig


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

    # parse timestamps as full-precision integer ns to avoid precision loss
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()]
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df[df["timestamp"] != 0]

    # drop nan rows
    df = df.dropna()

    # change timestamp to datetime in ns
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")

    # add activity_id
    df["activity_id"] = pd.factorize(df["activity_name"])[0]
    id_to_activity = (
        df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"], keep="first")
        .set_index("activity_id")["activity_name"]
        .to_dict()
    )

    # Deduplicate exact timestamp collisions and normalize to a stable 20Hz cadence.
    # WISDM raw timestamps are noisy and may include resets; stable synthetic timing
    # preserves sequence order while enforcing continuous sessions.
    value_cols = ["accel_x", "accel_y", "accel_z"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=value_cols)
    df = (
        df.groupby(["subject_id", "activity_id", "timestamp"], as_index=False)[value_cols]
        .mean()
    )
    df["activity_name"] = df["activity_id"].map(id_to_activity)

    df = df.sort_values(by=["subject_id", "activity_id", "timestamp"]).reset_index(drop=True)
    step_ms = int(1e3 / 20)
    df["timestamp"] = (
        df.groupby(["subject_id", "activity_id"]).cumcount().astype("int64") * step_ms
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )
    df["session_id"] = changes.cumsum() - 1

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


cfg_wisdm = WHARConfig(
    # Info + common
    dataset_id="wisdm",
    download_url="https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
    sampling_freq=20,
    num_of_subjects=36,
    num_of_activities=6,
    num_of_channels=3,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_wisdm_12,
    # Preprocessing (selections + sliding window)
    activity_names=[
        "Walking",
        "Jogging",
        "Upstairs",
        "Downstairs",
        "Sitting",
        "Standing",
    ],
    sensor_channels=[
        "accel_x",
        "accel_y",
        "accel_z",
    ],
    window_time=5,
    window_overlap=0.5,
    # Training (split info)
)

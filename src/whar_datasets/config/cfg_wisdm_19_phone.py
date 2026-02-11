import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

LETTER_TO_INT = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "O": 13,
    "P": 14,
    "Q": 15,
    "R": 16,
    "S": 17,
}

ID_TO_ACTIVITY = {
    0: "walking",
    1: "jogging",
    2: "stairs",
    3: "sitting",
    4: "standing",
    5: "typing",
    6: "teeth",
    7: "soup",
    8: "chips",
    9: "pasta",
    10: "drinking",
    11: "sandwich",
    12: "kicking",
    13: "catch",
    14: "dribbling",
    15: "writing",
    16: "clapping",
    17: "folding",
}


def parse_wisdm_19_phone(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    all_subjects_list = []
    print(str(dir))

    base_path = os.path.join(dir, "wisdm-dataset/wisdm-dataset/raw/phone")
    measures = ["accel_phone", "gyro_phone"]
    leit_measure = "accel_phone"

    for i in range(1600, 1651):
        dfs = {}
        current_subject_failed = False

        for measure in measures:
            parts = measure.split("_")
            sensor = parts[0]
            filename = f"data_{i}_{measure}.txt"
            full_path = os.path.join(base_path, sensor, filename)

            header = [
                "subject_id",
                "activity_id",
                "timestamp",
                f"{measure}_x",
                f"{measure}_y",
                f"{measure}_z",
            ]

            try:
                df = pd.read_csv(full_path, header=None, names=header)

                if df[f"{measure}_z"].dtype == object:
                    df[f"{measure}_z"] = (
                        df[f"{measure}_z"]
                        .astype(str)
                        .str.replace(";", "", regex=False)
                        .astype(float)
                    )

                df = df.sort_values("timestamp")
                dfs[measure] = df

            except FileNotFoundError:
                print(f"File not found: {full_path} - Skipping Subject {i}")
                current_subject_failed = True
                break

        if current_subject_failed or leit_measure not in dfs:
            continue

        df_final_subj = dfs[leit_measure].copy()

        for measure in measures:
            if measure == leit_measure:
                continue

            cols_to_use = ["timestamp", f"{measure}_x", f"{measure}_y", f"{measure}_z"]

            df_final_subj = pd.merge(
                df_final_subj, dfs[measure][cols_to_use], on="timestamp", how="inner"
            )

        # df_final_subj.dropna(inplace=True)

        all_subjects_list.append(df_final_subj)

    complete_df = pd.DataFrame(
        columns=["subject_id", "activity_id", "timestamp", "..."]
    )

    if all_subjects_list:
        complete_df = pd.concat(all_subjects_list, ignore_index=True)

        complete_df["subject_id"] = complete_df["subject_id"] - 1600

        complete_df["activity_id"] = (
            complete_df["activity_id"].astype(str).str.strip().map(LETTER_TO_INT)
        )

    else:
        print("Keine Daten gefunden.")

    changes = (complete_df["activity_id"] != complete_df["activity_id"].shift(1)) | (
        complete_df["subject_id"] != complete_df["subject_id"].shift(1)
    )

    complete_df["session_id"] = changes.cumsum() - 1

    complete_df = complete_df.sort_values(by=["session_id", "timestamp"])
    complete_df.head()

    metadata_cols = ["session_id", "subject_id", "activity_id"]

    session_metadata = (
        complete_df.groupby("session_id")[metadata_cols].first().reset_index(drop=True)
    )

    sessions: Dict[int, pd.DataFrame] = {}

    loop = tqdm(session_metadata["session_id"].unique())
    loop.set_description("Creating sessions")

    for session_id in loop:
        # get session df
        session_df = complete_df[complete_df["session_id"] == session_id]

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

    complete_df["timestamp"] = pd.to_datetime(complete_df["timestamp"], unit="ns")
    dtypes = {}
    for col in complete_df.columns:
        if col == "timestamp":
            dtypes[col] = "datetime64[ms]"  # Hier wird von ns auf ms gekürzt
        elif col in ["subject_id", "activity_id"]:
            dtypes[col] = "int32"
        else:
            dtypes[col] = "float32"

    # 3. Floats runden (Nur die Sensordaten)
    float_cols = [c for c, t in dtypes.items() if t == "float32"]
    complete_df[float_cols] = complete_df[float_cols].round(6)

    # 4. Finales Casting der Typen
    complete_df = complete_df.astype(dtypes)

    activity_metadata = pd.DataFrame(
        list(ID_TO_ACTIVITY.items()), columns=["activity_id", "activity_name"]
    )

    # set metadata types
    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


# toDO !!Split noch nicht angepasst

cfg_wisdm_19_phone = WHARConfig(
    # Info + common
    dataset_id="wisdm_19_phone",
    download_url="https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
    sampling_freq=20,
    num_of_subjects=51,
    num_of_activities=18,
    num_of_channels=6,
    # Parsing
    parse=parse_wisdm_19_phone,
    # Preprocessing (selections + sliding window)
    activity_names=[
        "walking",
        "jogging",
        "stairs",
        "sitting",
        "standing",
        "typing",
        "teeth",
        "soup",
        "chips",
        "pasta",
        "drinking",
        "sandwich",
        "kicking",
        "catch",
        "dribbling",
        "writing",
        "clapping",
        "folding",
    ],
    sensor_channels=[
        "accel_phone_x",
        "accel_phone_y",
        "accel_phone_z",
        "gyro_phone_x",
        "gyro_phone_y",
        "gyro_phone_z",
    ],
    window_time=5,
    window_overlap=0.5,
)

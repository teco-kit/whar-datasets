import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

SAD_SENSOR_POSITIONS: List[str] = [
    "Left_pocket",
    "Right_pocket",
    "Wrist",
    "Upper_arm",
    "Belt",
]

SAD_SENSOR_FEATURES: List[str] = [
    "time_stamp",
    "Ax",
    "Ay",
    "Az",
    "Lx",
    "Ly",
    "Lz",
    "Gx",
    "Gy",
    "Gz",
    "Mx",
    "My",
    "Mz",
]

SAD_ACTIVITY_NORMALIZATION: Dict[str, str] = {
    "upsatirs": "upstairs",
}

SAD_ACTIVITY_NAMES: List[str] = [
    "walking",
    "standing",
    "jogging",
    "sitting",
    "biking",
    "upstairs",
    "downstairs",
]


def _find_sad_dataset_root(data_dir: str) -> Path:
    base = Path(data_dir)
    candidates = [
        base / "DataSet",
        base / "sad" / "data" / "DataSet",
        base / "sad" / "data" / "sensors-activity-recognition-dataset-shoaib" / "DataSet",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    for candidate in base.rglob("DataSet"):
        if candidate.is_dir() and any(candidate.glob("Participant_*.csv")):
            return candidate

    raise FileNotFoundError(
        f"Could not locate SAD participant CSV files under '{data_dir}'."
    )


def _extract_participant_id(file_path: Path) -> int:
    match = re.search(r"Participant_(\d+)\.csv$", file_path.name)
    if match is None:
        raise ValueError(f"Unexpected SAD participant filename: '{file_path.name}'.")
    return int(match.group(1))


def _load_participant_csv(file_path: Path) -> pd.DataFrame:
    column_names: List[str] = []
    for i, sensor in enumerate(SAD_SENSOR_POSITIONS):
        for feature in SAD_SENSOR_FEATURES:
            column_names.append(f"{sensor}_{feature}")
        if i < len(SAD_SENSOR_POSITIONS) - 1:
            column_names.append(f"empty_{i}")
        else:
            column_names.append("activity_name")

    df = pd.read_csv(file_path, skiprows=2, names=column_names)
    empty_cols = [col for col in df.columns if col.startswith("empty_")]
    df = df.drop(columns=empty_cols)
    df["participant_raw_id"] = _extract_participant_id(file_path)
    return df


def parse_sad(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    root = _find_sad_dataset_root(dir)
    participant_files = sorted(
        root.glob("Participant_*.csv"),
        key=_extract_participant_id,
    )
    if not participant_files:
        raise FileNotFoundError(f"No participant CSV files found in '{root}'.")

    all_dfs = [_load_participant_csv(participant_file) for participant_file in participant_files]
    df = pd.concat(all_dfs, ignore_index=True)

    df["activity_name"] = (
        df["activity_name"].astype(str).str.strip().str.lower().replace(SAD_ACTIVITY_NORMALIZATION)
    )

    df["subject_id"] = pd.factorize(df["participant_raw_id"], sort=True)[0]
    timestamp_cols_to_drop = [
        "Right_pocket_time_stamp",
        "Wrist_time_stamp",
        "Upper_arm_time_stamp",
        "Belt_time_stamp",
    ]
    df = df.drop(columns=timestamp_cols_to_drop)
    df = df.rename(columns={"Left_pocket_time_stamp": "start_timestamp_ms"})

    df["activity_id"] = pd.factorize(df["activity_name"], sort=False)[0]

    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )
    df["session_id"] = (changes.cumsum() - 1).astype("int32")

    activity_metadata = (
        df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"], keep="first")
        .sort_values("activity_id")
        .reset_index(drop=True)
    )

    session_metadata = (
        df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates(subset=["session_id"], keep="first")
        .reset_index(drop=True)
    )

    sessions: Dict[int, pd.DataFrame] = {}
    loop = tqdm(session_metadata["session_id"].unique())
    loop.set_description("Creating sessions")

    for session_id in loop:
        session_df = df[df["session_id"] == session_id].copy()

        start_time_ms = session_df["start_timestamp_ms"].iloc[0]

        start_datetime = pd.to_datetime(start_time_ms, unit="ms")

        session_df["timestamp"] = pd.date_range(
            start=start_datetime, periods=len(session_df), freq="20ms"
        )

        cols_to_drop = [
            "session_id",
            "subject_id",
            "activity_id",
            "activity_name",
            "participant_raw_id",
            "start_timestamp_ms",
        ]
        session_df = session_df.drop(columns=cols_to_drop).reset_index(drop=True)

        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        dtypes["timestamp"] = "datetime64[ms]"
        session_df = session_df.astype(dtypes)

        sessions[int(session_id)] = session_df

    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_sad = WHARConfig(
    # Info + common
    dataset_id="sad",
    download_url="https://www.utwente.nl/en/eemcs/ps/dataset-folder/sensors-activity-recognition-dataset-shoaib.rar",
    sampling_freq=50,
    num_of_subjects=10,
    num_of_activities=7,
    num_of_channels=60,
    datasets_dir="./datasets",
    parse=parse_sad,
    activity_names=SAD_ACTIVITY_NAMES,
    sensor_channels=[
        "Left_pocket_Ax",
        "Left_pocket_Ay",
        "Left_pocket_Az",
        "Left_pocket_Lx",
        "Left_pocket_Ly",
        "Left_pocket_Lz",
        "Left_pocket_Gx",
        "Left_pocket_Gy",
        "Left_pocket_Gz",
        "Left_pocket_Mx",
        "Left_pocket_My",
        "Left_pocket_Mz",
        "Right_pocket_Ax",
        "Right_pocket_Ay",
        "Right_pocket_Az",
        "Right_pocket_Lx",
        "Right_pocket_Ly",
        "Right_pocket_Lz",
        "Right_pocket_Gx",
        "Right_pocket_Gy",
        "Right_pocket_Gz",
        "Right_pocket_Mx",
        "Right_pocket_My",
        "Right_pocket_Mz",
        "Wrist_Ax",
        "Wrist_Ay",
        "Wrist_Az",
        "Wrist_Lx",
        "Wrist_Ly",
        "Wrist_Lz",
        "Wrist_Gx",
        "Wrist_Gy",
        "Wrist_Gz",
        "Wrist_Mx",
        "Wrist_My",
        "Wrist_Mz",
        "Upper_arm_Ax",
        "Upper_arm_Ay",
        "Upper_arm_Az",
        "Upper_arm_Lx",
        "Upper_arm_Ly",
        "Upper_arm_Lz",
        "Upper_arm_Gx",
        "Upper_arm_Gy",
        "Upper_arm_Gz",
        "Upper_arm_Mx",
        "Upper_arm_My",
        "Upper_arm_Mz",
        "Belt_Ax",
        "Belt_Ay",
        "Belt_Az",
        "Belt_Lx",
        "Belt_Ly",
        "Belt_Lz",
        "Belt_Gx",
        "Belt_Gy",
        "Belt_Gz",
        "Belt_Mx",
        "Belt_My",
        "Belt_Mz",
    ],
    window_time=2.56,
    window_overlap=0.5,
)

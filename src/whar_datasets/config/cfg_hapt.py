import os
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND",
}


def find_hapt_root(dir: str) -> str:
    # common extracted layout: <root>/RawData/{acc..., gyro..., labels.txt}
    raw_dir_direct = os.path.join(dir, "RawData")
    if os.path.isdir(raw_dir_direct) and os.path.isfile(
        os.path.join(raw_dir_direct, "labels.txt")
    ):
        return dir

    # nested layouts after zip extraction.
    for root, dirs, files in os.walk(dir):
        if "RawData" in dirs and os.path.isfile(
            os.path.join(root, "RawData", "labels.txt")
        ):
            return root
        if (
            os.path.basename(root) == "RawData"
            and "labels.txt" in files
            and os.path.isfile(os.path.join(root, "labels.txt"))
        ):
            return os.path.dirname(root)

    raise FileNotFoundError(
        f"Could not locate HAPT data with 'RawData/labels.txt' under '{dir}'."
    )


def load_signal_df(path: str, cols: List[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=cols)


def parse_hapt(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    root_dir = find_hapt_root(dir)
    raw_dir = os.path.join(root_dir, "RawData")
    labels_path = os.path.join(raw_dir, "labels.txt")

    labels_df = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["exp_id", "user_id", "activity_raw_id", "start_idx", "end_idx"],
    )

    labels_df = labels_df[labels_df["activity_raw_id"].isin(ACTIVITY_MAP)].copy()
    labels_df = labels_df.sort_values(
        by=["user_id", "exp_id", "start_idx", "end_idx"]
    ).reset_index(drop=True)

    labels_df["subject_id"] = pd.factorize(labels_df["user_id"], sort=True)[0]
    labels_df["activity_name"] = labels_df["activity_raw_id"].map(ACTIVITY_MAP)
    labels_df["activity_id"] = pd.factorize(labels_df["activity_name"], sort=False)[0]
    labels_df["session_id"] = labels_df.index.astype("int32")

    # cache experiment/user files to avoid repeated disk reads
    signals_cache: Dict[Tuple[int, int], pd.DataFrame] = {}
    sessions: Dict[int, pd.DataFrame] = {}

    loop = tqdm(labels_df.itertuples(index=False), total=len(labels_df))
    loop.set_description("Creating sessions")

    for row in loop:
        exp_id = int(row.exp_id)
        user_id = int(row.user_id)
        session_id = int(row.session_id)
        start_idx = int(row.start_idx)
        end_idx = int(row.end_idx)

        cache_key = (exp_id, user_id)
        if cache_key not in signals_cache:
            acc_path = os.path.join(raw_dir, f"acc_exp{exp_id:02d}_user{user_id:02d}.txt")
            gyro_path = os.path.join(
                raw_dir, f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt"
            )

            acc_df = load_signal_df(acc_path, ["acc_x", "acc_y", "acc_z"])
            gyro_df = load_signal_df(gyro_path, ["gyro_x", "gyro_y", "gyro_z"])

            # Both files describe the same timeline.
            n = min(len(acc_df), len(gyro_df))
            merged_df = pd.concat(
                [acc_df.iloc[:n].reset_index(drop=True), gyro_df.iloc[:n].reset_index(drop=True)],
                axis=1,
            )
            signals_cache[cache_key] = merged_df

        full_signal_df = signals_cache[cache_key]

        # labels.txt indices are 1-based and inclusive.
        start_zero = max(start_idx - 1, 0)
        end_exclusive = min(end_idx, len(full_signal_df))

        session_df = full_signal_df.iloc[start_zero:end_exclusive].copy()
        if session_df.empty:
            continue

        sampling_interval_ms = 1 / 50 * 1e3
        session_df["timestamp"] = (
            session_df.reset_index(drop=True).index * sampling_interval_ms
        )
        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"], unit="ms")

        value_cols = [col for col in session_df.columns if col != "timestamp"]
        session_df[value_cols] = session_df[value_cols].round(6)
        dtypes = {col: "float32" for col in value_cols}
        dtypes["timestamp"] = "datetime64[ms]"
        session_df = session_df.astype(dtypes)
        sessions[session_id] = session_df.reset_index(drop=True)

    session_metadata = labels_df[["session_id", "subject_id", "activity_id"]]
    session_metadata = session_metadata[
        session_metadata["session_id"].isin(sessions.keys())
    ].reset_index(drop=True)

    if session_metadata.empty:
        raise ValueError("No HAPT sessions could be created from labels.txt intervals.")

    # Keep ids dense in case some labeled intervals were empty after slicing.
    session_metadata["session_id"] = pd.factorize(session_metadata["session_id"])[0]
    remap = dict(zip(labels_df["session_id"], session_metadata["session_id"]))
    sessions = {int(remap[k]): v for k, v in sessions.items() if k in remap}

    activity_metadata = (
        labels_df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"], keep="first")
        .sort_values("activity_id")
        .reset_index(drop=True)
    )

    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_hapt = WHARConfig(
    # Info + common
    dataset_id="hapt",
    download_url="https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip",
    sampling_freq=50,
    num_of_subjects=30,
    num_of_activities=12,
    num_of_channels=6,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_hapt,
    # Preprocessing (selections + sliding window)
    activity_names=[
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
        "STAND_TO_SIT",
        "SIT_TO_STAND",
        "SIT_TO_LIE",
        "LIE_TO_SIT",
        "STAND_TO_LIE",
        "LIE_TO_STAND",
    ],
    sensor_channels=[
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ],
    window_time=2.56,
    window_overlap=0.5,
)

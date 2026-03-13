import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

UCA_EHAR_FILENAME_PATTERN = re.compile(
    r"^(?P<activity>[A-Z]+)_T(?P<subject>\d+)(?:_(?P<part>\d+))?\.csv$"
)

UCA_EHAR_ACTIVITY_NAMES: List[str] = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "RUNNING",
    "SITTING",
    "STANDING",
    "LYING",
    "DRINKING",
    "SIT_TO_STAND",
    "STAND_TO_SIT",
    "SIT_TO_LIE",
    "LIE_TO_SIT",
]

UCA_EHAR_SENSOR_CHANNELS: List[str] = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz", "P"]

UCA_EHAR_GAP_MULTIPLIER = 5.0
UCA_EHAR_MIN_GAP_MS = 120.0
UCA_EHAR_MAX_GAP_MS = 2_000.0


def _resolve_uca_ehar_root(data_dir: str) -> Path:
    base = Path(data_dir)
    candidates = [base / "UCA-EHAR-1.0.0", base]

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any(
            UCA_EHAR_FILENAME_PATTERN.match(csv_path.name)
            for csv_path in candidate.glob("*.csv")
        ):
            return candidate

    for candidate in base.rglob("*"):
        if not candidate.is_dir():
            continue
        if any(
            UCA_EHAR_FILENAME_PATTERN.match(csv_path.name)
            for csv_path in candidate.glob("*.csv")
        ):
            return candidate

    raise FileNotFoundError(
        f"Could not locate extracted UCA-EHAR CSV files under '{data_dir}'."
    )


def parse_uca_ehar(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    root = _resolve_uca_ehar_root(dir)
    raw_frames: List[pd.DataFrame] = []

    for csv_path in sorted(root.glob("*.csv")):
        match = UCA_EHAR_FILENAME_PATTERN.match(csv_path.name)
        if match is None:
            continue

        frame = pd.read_csv(csv_path, sep=";")
        required_cols = ["T", *UCA_EHAR_SENSOR_CHANNELS, "CLASS"]
        missing_cols = [col for col in required_cols if col not in frame.columns]
        if missing_cols:
            raise ValueError(
                f"UCA-EHAR file '{csv_path.name}' is missing columns: {missing_cols}"
            )

        frame = frame[required_cols].copy()
        frame["subject_raw"] = int(match.group("subject"))
        frame["recording_id"] = csv_path.stem
        frame["activity_name"] = frame["CLASS"].astype(str).str.strip().str.upper()
        frame = frame.drop(columns=["CLASS"])

        numeric_cols = ["T", *UCA_EHAR_SENSOR_CHANNELS]
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        frame = frame.dropna(subset=["T", "activity_name", *UCA_EHAR_SENSOR_CHANNELS])
        frame = frame.sort_values("T").reset_index(drop=True)
        raw_frames.append(frame)

    if not raw_frames:
        raise ValueError("No UCA-EHAR CSV rows were loaded from the extracted dataset.")

    df = pd.concat(raw_frames, ignore_index=True)

    unknown_labels = sorted(
        set(df["activity_name"].astype(str).unique()) - set(UCA_EHAR_ACTIVITY_NAMES)
    )
    if unknown_labels:
        raise ValueError(
            "Found UCA-EHAR activity labels not covered by config: "
            + ", ".join(unknown_labels)
        )

    subject_values = sorted(df["subject_raw"].astype(int).unique().tolist())
    if not subject_values:
        raise ValueError("UCA-EHAR subject identifiers are missing.")
    subject_map = {subject_raw: idx for idx, subject_raw in enumerate(subject_values)}
    activity_map = {
        activity_name: idx for idx, activity_name in enumerate(UCA_EHAR_ACTIVITY_NAMES)
    }

    df["subject_id"] = df["subject_raw"].map(subject_map)
    df["activity_id"] = df["activity_name"].map(activity_map)
    df = df.dropna(subset=["subject_id", "activity_id"]).copy()

    df = df.sort_values(["subject_raw", "recording_id", "T"]).reset_index(drop=True)

    group_cols = ["subject_raw", "recording_id"]
    time_diff_ms = df.groupby(group_cols, sort=False)["T"].diff()
    positive_diff = time_diff_ms.where(time_diff_ms > 0)
    median_step_ms = positive_diff.groupby(
        [df[c] for c in group_cols], sort=False
    ).transform("median")
    gap_threshold_ms = (median_step_ms * UCA_EHAR_GAP_MULTIPLIER).clip(
        lower=UCA_EHAR_MIN_GAP_MS,
        upper=UCA_EHAR_MAX_GAP_MS,
    )

    session_breaks = (
        (df["subject_raw"] != df["subject_raw"].shift(1))
        | (df["recording_id"] != df["recording_id"].shift(1))
        | (df["activity_name"] != df["activity_name"].shift(1))
        | time_diff_ms.isna()
        | (time_diff_ms < 0)
        | (time_diff_ms > gap_threshold_ms)
    )
    session_breaks.iloc[0] = True
    local_session_ids = session_breaks.astype("int64").cumsum() - 1

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int]] = []
    next_session_id = 0

    loop = tqdm(df.groupby(local_session_ids, sort=False), desc="Creating sessions")
    for _, chunk in loop:
        session = chunk[["T", *UCA_EHAR_SENSOR_CHANNELS]].copy()
        if session.empty:
            continue

        session = session.rename(columns={"T": "timestamp"})
        session["timestamp"] = pd.to_datetime(session["timestamp"], unit="ms")
        session = session.astype(
            {
                "timestamp": "datetime64[ms]",
                **{col: "float32" for col in UCA_EHAR_SENSOR_CHANNELS},
            }
        )
        session[UCA_EHAR_SENSOR_CHANNELS] = session[UCA_EHAR_SENSOR_CHANNELS].round(6)
        sessions[next_session_id] = session.reset_index(drop=True)

        session_rows.append(
            {
                "session_id": next_session_id,
                "subject_id": int(chunk["subject_id"].iloc[0]),
                "activity_id": int(chunk["activity_id"].iloc[0]),
            }
        )
        next_session_id += 1

    if not sessions:
        raise ValueError("No UCA-EHAR sessions were produced.")

    activity_df = pd.DataFrame(
        {
            "activity_id": list(range(len(UCA_EHAR_ACTIVITY_NAMES))),
            "activity_name": UCA_EHAR_ACTIVITY_NAMES,
        }
    ).astype({"activity_id": "int32", "activity_name": "string"})

    session_df = pd.DataFrame(session_rows).astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    if session_df["subject_id"].nunique() == 0:
        raise ValueError("UCA-EHAR subject identifiers are missing after parsing.")
    if session_df["activity_id"].nunique() == 0:
        raise ValueError("UCA-EHAR activity labels are missing after parsing.")

    return activity_df, session_df, sessions


SELECTED_ACTIVITIES = UCA_EHAR_ACTIVITY_NAMES

cfg_uca_ehar = WHARConfig(
    # Info + common
    dataset_id="uca_ehar",
    dataset_url="https://zenodo.org/records/5659336",
    download_url="https://zenodo.org/records/5659336/files/UCA-EHAR-1.0.0.zip?download=1",
    sampling_freq=25,
    num_of_subjects=20,
    num_of_activities=12,
    num_of_channels=7,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_uca_ehar,
    activity_id_col="activity_id",
    # Preprocessing (selections + sliding window)
    available_activities=canonicalize_activity_name_list(UCA_EHAR_ACTIVITY_NAMES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=UCA_EHAR_SENSOR_CHANNELS,
    selected_channels=UCA_EHAR_SENSOR_CHANNELS,
    window_time=2,
    window_overlap=0.5,
    parallelize=True,
    # Training (split info)
)

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig
from whar_datasets.config.timestamps import to_datetime64_ms

FALLDET_SENSOR_CHANNELS: List[str] = ["acc_x", "acc_y", "acc_z"]

FALLDET_ACTIVITY_ALIASES: Dict[str, str] = {
    "downsit": "downSit",
    "freefall": "freeFall",
    "runfall": "runFall",
    "runsit": "runSit",
    "walkfall": "walkFall",
    "walksit": "walkSit",
}

FALLDET_FINE_ACTIVITY_NAMES: List[str] = [
    "downSit",
    "freeFall",
    "runFall",
    "runSit",
    "walkFall",
    "walkSit",
]

FALLDET_BINARY_ACTIVITY_NAMES: List[str] = ["fall", "non_fall"]
FALLDET_SAMPLING_FREQ_HZ = 50.0
FALLDET_MAX_STEP_MULTIPLIER = 3.0
FALLDET_SESSION_GAP_SECONDS = FALLDET_MAX_STEP_MULTIPLIER / FALLDET_SAMPLING_FREQ_HZ


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.strip().lower())


def _resolve_falldet_root(data_dir: str) -> Path:
    base = Path(data_dir)
    candidates = [base, base / "falldet", base / "falldet" / "data"]

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any(candidate.glob("*/*.csv")):
            return candidate

    for candidate in base.rglob("*"):
        if candidate.is_dir() and any(candidate.glob("*/*.csv")):
            return candidate

    raise FileNotFoundError(f"Could not locate FallDet CSV files under '{data_dir}'.")


def _select_activity_scheme(activity_id_col: str) -> str:
    token = _normalize_token(activity_id_col)
    if token in {"binary", "binaryactivity", "binarylabel", "fallbinary"}:
        return "binary"
    return "fine"


def _extract_subject_raw_id(file_path: Path) -> int:
    match = re.search(r"(\d+)$", file_path.stem)
    if match is None:
        raise ValueError(
            f"Could not infer subject identifier from filename '{file_path.name}'."
        )
    return int(match.group(1))


def _canonical_activity_from_folder(folder_name: str) -> str:
    token = _normalize_token(folder_name)
    if token not in FALLDET_ACTIVITY_ALIASES:
        raise ValueError(f"Unexpected FallDet activity folder '{folder_name}'.")
    return FALLDET_ACTIVITY_ALIASES[token]


def _activity_name_for_scheme(canonical_activity: str, scheme: str) -> str:
    if scheme == "binary":
        return "fall" if "fall" in canonical_activity.lower() else "non_fall"
    return canonical_activity


def _load_session_from_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=";")
    if df.empty:
        raise ValueError(f"File '{file_path}' is empty.")

    missing_cols = {
        "Timestamp",
        "AccelerationX",
        "AccelerationY",
        "AccelerationZ",
    }.difference(df.columns)
    if missing_cols:
        raise ValueError(
            f"File '{file_path.name}' is missing required columns: {sorted(missing_cols)}"
        )

    timestamp = to_datetime64_ms(df["Timestamp"], default_unit="s")
    if timestamp.isna().all():
        raise ValueError(f"Could not parse timestamps in '{file_path.name}'.")

    session_df = pd.DataFrame({"timestamp": timestamp})
    session_df["acc_x"] = pd.to_numeric(df["AccelerationX"], errors="coerce")
    session_df["acc_y"] = pd.to_numeric(df["AccelerationY"], errors="coerce")
    session_df["acc_z"] = pd.to_numeric(df["AccelerationZ"], errors="coerce")

    session_df = session_df.dropna().reset_index(drop=True)
    if session_df.empty:
        raise ValueError(f"No valid accelerometer rows left in '{file_path.name}'.")

    session_df = session_df.sort_values("timestamp").reset_index(drop=True)
    # Collapse millisecond collisions to guarantee strictly increasing timestamps.
    session_df = (
        session_df.groupby("timestamp", as_index=False)[FALLDET_SENSOR_CHANNELS]
        .mean()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    session_df["timestamp"] = session_df["timestamp"].astype("datetime64[ms]")
    session_df[FALLDET_SENSOR_CHANNELS] = session_df[FALLDET_SENSOR_CHANNELS].astype(
        "float32"
    )
    session_df = session_df[["timestamp"] + FALLDET_SENSOR_CHANNELS]

    return session_df


def _split_by_timestamp_gap(session_df: pd.DataFrame) -> List[pd.DataFrame]:
    if session_df.empty:
        return []

    deltas = session_df["timestamp"].diff().dt.total_seconds().fillna(0.0)
    split_markers = (deltas > FALLDET_SESSION_GAP_SECONDS).astype("int32")
    group_ids = split_markers.cumsum()

    split_sessions: List[pd.DataFrame] = []
    for _, group in session_df.groupby(group_ids):
        chunk = group.reset_index(drop=True)
        if not chunk.empty:
            split_sessions.append(chunk)

    return split_sessions


def parse_falldet(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    root = _resolve_falldet_root(dir)
    scheme = _select_activity_scheme(activity_id_col)

    session_files = sorted(root.glob("*/*.csv"))
    if not session_files:
        raise FileNotFoundError(f"No FallDet CSV files found in '{root}'.")

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int | str]] = []

    loop = tqdm(session_files)
    loop.set_description("Creating sessions")

    next_session_id = 0

    for file_path in loop:
        canonical_activity = _canonical_activity_from_folder(file_path.parent.name)
        activity_name = _activity_name_for_scheme(canonical_activity, scheme)
        subject_raw_id = _extract_subject_raw_id(file_path)

        split_sessions = _split_by_timestamp_gap(_load_session_from_csv(file_path))
        for split_session in split_sessions:
            sessions[next_session_id] = split_session
            session_rows.append(
                {
                    "session_id": next_session_id,
                    "subject_raw_id": subject_raw_id,
                    "activity_name": activity_name,
                }
            )
            next_session_id += 1

    session_metadata = pd.DataFrame(session_rows)
    if session_metadata.empty:
        raise ValueError("No FallDet sessions could be parsed.")

    session_metadata["subject_id"] = pd.factorize(
        session_metadata["subject_raw_id"], sort=True
    )[0]
    session_metadata["activity_id"] = pd.factorize(
        session_metadata["activity_name"], sort=False
    )[0]
    session_metadata = session_metadata[["session_id", "subject_id", "activity_id"]]

    activity_metadata = pd.DataFrame(session_rows)[["activity_name"]]
    activity_metadata = activity_metadata.drop_duplicates().reset_index(drop=True)
    activity_metadata["activity_id"] = pd.factorize(
        activity_metadata["activity_name"], sort=False
    )[0]
    activity_metadata = activity_metadata[["activity_id", "activity_name"]]

    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


SELECTED_ACTIVITIES = FALLDET_FINE_ACTIVITY_NAMES

cfg_falldet = WHARConfig(
    dataset_id="falldet",
    dataset_url="https://www.kaggle.com/datasets/harnoor343/fall-detection-accelerometer-data",
    download_url="https://www.kaggle.com/api/v1/datasets/download/harnoor343/fall-detection-accelerometer-data?datasetVersionNumber=1",
    sampling_freq=50,
    num_of_subjects=48,
    num_of_activities=6,
    num_of_channels=3,
    datasets_dir="./datasets",
    parse=parse_falldet,
    activity_id_col="activity_id",
    available_activities=canonicalize_activity_name_list(FALLDET_FINE_ACTIVITY_NAMES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=FALLDET_SENSOR_CHANNELS,
    selected_channels=FALLDET_SENSOR_CHANNELS,
    window_time=3,
    window_overlap=0.5,
    parallelize=True,
)

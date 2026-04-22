import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import scipy.io
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

ACTRECTUT_WALKING_ACTIVITY_NAMES: List[str] = [
    "null",
    "sitting",
    "standing",
    "walk_horizontal",
    "walk_down",
    "walk_up",
]

SELECTED_ACTIVITIES: List[str] = [
    a for a in ACTRECTUT_WALKING_ACTIVITY_NAMES if a != "null"
]

# The walking representation exposes 24 synchronized channels.
# The local `subject*_walk/data.mat` has 24 channels (4 sensor units with
# 3-ax accelerometer + 3-ax gyroscope each).
ACTRECTUT_WALKING_CHANNELS: List[str] = [
    channel_name
    for sensor_idx in range(1, 5)
    for channel_name in (
        f"acc_{sensor_idx}_x",
        f"acc_{sensor_idx}_y",
        f"acc_{sensor_idx}_z",
        f"gyr_{sensor_idx}_x",
        f"gyr_{sensor_idx}_y",
        f"gyr_{sensor_idx}_z",
    )
]

ACTRECTUT_WALKING_SAMPLING_HZ = 32.0
ACTRECTUT_WALKING_GAP_THRESHOLD_SECONDS = 5.0
SUBJECT_DIR_PATTERN = re.compile(r"^subject(?P<subject>\d+)_walk$", re.IGNORECASE)


def _resolve_walking_subject_dirs(data_dir: str) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"ActRecTut walking data directory does not exist: '{data_dir}'."
        )

    candidate_roots = [root, root / "Data"]
    subject_dirs: List[Path] = []

    for candidate in candidate_roots:
        if not candidate.is_dir():
            continue
        for child in sorted(candidate.iterdir()):
            if not child.is_dir():
                continue
            if SUBJECT_DIR_PATTERN.match(child.name):
                subject_dirs.append(child)

    unique_subject_dirs = sorted(set(subject_dirs), key=lambda path: path.name.lower())
    if not unique_subject_dirs:
        raise FileNotFoundError(
            "Could not locate any 'subject*_walk' directories under "
            f"'{data_dir}' or '{Path(data_dir) / 'Data'}'."
        )

    return unique_subject_dirs


def _load_walking_mat(mat_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing required file: '{mat_path}'.")

    mat = scipy.io.loadmat(mat_path)
    if "data" not in mat or "labels" not in mat:
        raise ValueError(f"Expected 'data' and 'labels' in '{mat_path}'.")

    data = mat["data"]
    labels = mat["labels"]

    if data.ndim != 2:
        raise ValueError(f"Unexpected data shape in '{mat_path}': {data.shape}")
    if labels.ndim not in (1, 2):
        raise ValueError(f"Unexpected labels shape in '{mat_path}': {labels.shape}")
    if int(data.shape[1]) != len(ACTRECTUT_WALKING_CHANNELS):
        raise ValueError(
            "Unexpected channel count in ActRecTut walking data matrix. "
            f"Expected {len(ACTRECTUT_WALKING_CHANNELS)}, got {int(data.shape[1])} "
            f"for '{mat_path}'."
        )

    labels_series = pd.Series(labels.reshape(-1), dtype="int64")
    data_df = pd.DataFrame(data, columns=ACTRECTUT_WALKING_CHANNELS)

    if len(data_df) != len(labels_series):
        raise ValueError(
            "Data/label length mismatch for ActRecTut walking source "
            f"'{mat_path}': data={len(data_df)}, labels={len(labels_series)}."
        )
    if labels_series.isna().any():
        raise ValueError(
            f"Missing activity labels in '{mat_path}'. LOSOCV metadata requires labels."
        )

    return data_df, labels_series


def parse_actrectut_walking(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    subject_dirs = _resolve_walking_subject_dirs(dir)

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int]] = []
    subject_raw_ids: List[int] = []
    next_session_id = 0

    loop = tqdm(subject_dirs, desc="Parsing ActRecTut Walking")
    for subject_dir in loop:
        match = SUBJECT_DIR_PATTERN.match(subject_dir.name)
        if match is None:
            continue
        subject_raw = int(match.group("subject"))
        subject_raw_ids.append(subject_raw)

        data_df, labels = _load_walking_mat(subject_dir / "data.mat")

        expected_label_ids = set(range(1, len(ACTRECTUT_WALKING_ACTIVITY_NAMES) + 1))
        observed_label_ids = set(int(value) for value in labels.unique().tolist())
        unknown_label_ids = sorted(observed_label_ids.difference(expected_label_ids))
        if unknown_label_ids:
            raise ValueError(
                "Found unexpected ActRecTut walking label IDs in "
                f"'{subject_dir / 'data.mat'}': {unknown_label_ids}. "
                "Cannot safely assign activities."
            )

        data_df = data_df.astype(
            {channel: "float32" for channel in ACTRECTUT_WALKING_CHANNELS}
        )
        data_df[ACTRECTUT_WALKING_CHANNELS] = data_df[ACTRECTUT_WALKING_CHANNELS].round(
            6
        )
        data_df["activity_id"] = labels.astype("int64") - 1

        sample_count = len(data_df)
        timestamps = pd.to_datetime(
            pd.Series(range(sample_count), dtype="float64")
            / ACTRECTUT_WALKING_SAMPLING_HZ,
            unit="s",
        ).astype("datetime64[ms]")
        data_df["timestamp"] = timestamps

        timestamp_gaps = data_df["timestamp"].diff().dt.total_seconds().fillna(0.0)
        local_session_breaks = (
            data_df["activity_id"] != data_df["activity_id"].shift(1)
        ) | (timestamp_gaps > ACTRECTUT_WALKING_GAP_THRESHOLD_SECONDS)
        local_session_ids = local_session_breaks.astype("int64").cumsum() - 1

        for _, chunk in data_df.groupby(local_session_ids, sort=False):
            if chunk.empty:
                continue

            activity_id = int(chunk["activity_id"].iloc[0])
            session_df = chunk[["timestamp", *ACTRECTUT_WALKING_CHANNELS]].reset_index(
                drop=True
            )
            session_df = session_df.astype(
                {
                    "timestamp": "datetime64[ms]",
                    **{channel: "float32" for channel in ACTRECTUT_WALKING_CHANNELS},
                }
            )

            sessions[next_session_id] = session_df
            session_rows.append(
                {
                    "session_id": next_session_id,
                    "subject_raw_id": subject_raw,
                    "activity_id": activity_id,
                }
            )
            next_session_id += 1

    if not sessions:
        raise ValueError("No ActRecTut walking sessions were parsed.")
    if not subject_raw_ids:
        raise ValueError(
            "No subject identifiers found in ActRecTut walking directory names. "
            "LOSOCV metadata requires subject identifiers."
        )

    session_metadata = pd.DataFrame(session_rows)
    session_metadata["subject_id"] = pd.factorize(
        session_metadata["subject_raw_id"], sort=True
    )[0]
    session_metadata = session_metadata[["session_id", "subject_id", "activity_id"]]
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    activity_metadata = pd.DataFrame(
        [
            {"activity_id": activity_id, "activity_name": activity_name}
            for activity_id, activity_name in enumerate(
                ACTRECTUT_WALKING_ACTIVITY_NAMES
            )
        ]
    ).astype({"activity_id": "int32", "activity_name": "string"})

    return activity_metadata, session_metadata, sessions


cfg_actrectut_walking = WHARConfig(
    # Info + common
    dataset_id="actrectut_walking",
    dataset_url="https://github.com/andreas-bulling/ActRecTut/tree/master",
    download_url="https://downgit.github.io/#/home?url=https:%2F%2Fgithub.com%2Fandreas-bulling%2FActRecTut%2Ftree%2Fe122877362f388a9a2a8d1d4b50fb4148c0794f3%2FData",
    sampling_freq=32,
    num_of_subjects=2,
    num_of_activities=6,
    num_of_channels=len(ACTRECTUT_WALKING_CHANNELS),
    datasets_dir="./datasets",
    parallelize=True,
    # Parsing
    parse=parse_actrectut_walking,
    # Preprocessing (selections + sliding window)
    available_activities=canonicalize_activity_name_list(
        ACTRECTUT_WALKING_ACTIVITY_NAMES
    ),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=ACTRECTUT_WALKING_CHANNELS,
    selected_channels=ACTRECTUT_WALKING_CHANNELS,
    window_time=2,
    window_overlap=0.5,
    # Training (split info)
)

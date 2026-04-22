import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import scipy.io
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

HAND_GESTURES_ACTIVITY_NAMES: List[str] = [
    "null",
    "open_window",
    "drink",
    "water_plant",
    "close_window",
    "cut",
    "chop",
    "stir",
    "turn_book_pages",
    "tennis_forehand",
    "tennis_backhand",
    "tennis_smash",
]

SELECTED_ACTIVITIES: List[str] = [
    a for a in HAND_GESTURES_ACTIVITY_NAMES if a != "null"
]

HAND_GESTURES_CHANNELS: List[str] = [
    "acc_1_x",
    "acc_1_y",
    "acc_1_z",
    "gyr_1_x",
    "gyr_1_y",
    "acc_2_x",
    "acc_2_y",
    "acc_2_z",
    "gyr_2_x",
    "gyr_2_y",
    "acc_3_x",
    "acc_3_y",
    "acc_3_z",
    "gyr_3_x",
    "gyr_3_y",
]

HAND_GESTURES_SAMPLING_HZ = 32.0
HAND_GESTURES_GAP_THRESHOLD_SECONDS = 5.0
SUBJECT_DIR_PATTERN = re.compile(r"^subject(?P<subject>\d+)_gesture$", re.IGNORECASE)


def _resolve_gesture_subject_dirs(data_dir: str) -> List[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Hand Gestures data directory does not exist: '{data_dir}'."
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
            "Could not locate any 'subject*_gesture' directories under "
            f"'{data_dir}' or '{Path(data_dir) / 'Data'}'."
        )

    return unique_subject_dirs


def _load_gesture_mat(mat_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
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
    if int(data.shape[1]) != len(HAND_GESTURES_CHANNELS):
        raise ValueError(
            "Unexpected channel count in hand gestures data matrix. "
            f"Expected {len(HAND_GESTURES_CHANNELS)}, got {int(data.shape[1])} "
            f"for '{mat_path}'."
        )

    labels_series = pd.Series(labels.reshape(-1), dtype="int64")
    data_df = pd.DataFrame(data, columns=HAND_GESTURES_CHANNELS)

    if len(data_df) != len(labels_series):
        raise ValueError(
            "Data/label length mismatch for hand gestures session source "
            f"'{mat_path}': data={len(data_df)}, labels={len(labels_series)}."
        )
    if labels_series.isna().any():
        raise ValueError(
            f"Missing activity labels in '{mat_path}'. LOSOCV metadata requires labels."
        )

    return data_df, labels_series


def parse_actrectut_gestures(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    subject_dirs = _resolve_gesture_subject_dirs(dir)

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int]] = []
    subject_raw_ids: List[int] = []
    next_session_id = 0

    loop = tqdm(subject_dirs, desc="Parsing Hand Gestures")
    for subject_dir in loop:
        match = SUBJECT_DIR_PATTERN.match(subject_dir.name)
        if match is None:
            continue
        subject_raw = int(match.group("subject"))
        subject_raw_ids.append(subject_raw)

        data_df, labels = _load_gesture_mat(subject_dir / "data.mat")

        expected_label_ids = set(range(1, len(HAND_GESTURES_ACTIVITY_NAMES) + 1))
        observed_label_ids = set(int(value) for value in labels.unique().tolist())
        unknown_label_ids = sorted(observed_label_ids.difference(expected_label_ids))
        if unknown_label_ids:
            raise ValueError(
                "Found unexpected Hand Gestures label IDs in "
                f"'{subject_dir / 'data.mat'}': {unknown_label_ids}. "
                "Cannot safely assign activities."
            )

        data_df = data_df.astype(
            {channel: "float32" for channel in HAND_GESTURES_CHANNELS}
        )
        data_df[HAND_GESTURES_CHANNELS] = data_df[HAND_GESTURES_CHANNELS].round(6)
        data_df["activity_id"] = labels.astype("int64") - 1

        sample_count = len(data_df)
        timestamps = pd.to_datetime(
            pd.Series(range(sample_count), dtype="float64") / HAND_GESTURES_SAMPLING_HZ,
            unit="s",
        ).astype("datetime64[ms]")
        data_df["timestamp"] = timestamps

        timestamp_gaps = data_df["timestamp"].diff().dt.total_seconds().fillna(0.0)
        local_session_breaks = (
            data_df["activity_id"] != data_df["activity_id"].shift(1)
        ) | (timestamp_gaps > HAND_GESTURES_GAP_THRESHOLD_SECONDS)
        local_session_ids = local_session_breaks.astype("int64").cumsum() - 1

        for _, chunk in data_df.groupby(local_session_ids, sort=False):
            if chunk.empty:
                continue

            activity_id = int(chunk["activity_id"].iloc[0])
            session_df = chunk[["timestamp", *HAND_GESTURES_CHANNELS]].reset_index(
                drop=True
            )
            session_df = session_df.astype(
                {
                    "timestamp": "datetime64[ms]",
                    **{channel: "float32" for channel in HAND_GESTURES_CHANNELS},
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
        raise ValueError("No Hand Gestures sessions were parsed.")
    if not subject_raw_ids:
        raise ValueError(
            "No subject identifiers found in Hand Gestures directory names. "
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
            for activity_id, activity_name in enumerate(HAND_GESTURES_ACTIVITY_NAMES)
        ]
    ).astype({"activity_id": "int32", "activity_name": "string"})

    return activity_metadata, session_metadata, sessions


cfg_actrectut_gestures = WHARConfig(
    # Info + common
    dataset_id="actrectut_gestures",
    dataset_url="https://github.com/andreas-bulling/ActRecTut/tree/master",
    download_url="https://downgit.github.io/#/home?url=https:%2F%2Fgithub.com%2Fandreas-bulling%2FActRecTut%2Ftree%2Fe122877362f388a9a2a8d1d4b50fb4148c0794f3%2FData",
    sampling_freq=32,
    num_of_subjects=2,
    num_of_activities=12,
    num_of_channels=len(HAND_GESTURES_CHANNELS),
    datasets_dir="./datasets",
    parallelize=True,
    # Parsing
    parse=parse_actrectut_gestures,
    # Preprocessing (selections + sliding window)
    available_activities=canonicalize_activity_name_list(HAND_GESTURES_ACTIVITY_NAMES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=HAND_GESTURES_CHANNELS,
    selected_channels=HAND_GESTURES_CHANNELS,
    window_time=2,
    window_overlap=0.5,
    # Training (split info)
)

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

SKODA_ACTIVITY_NAMES: List[str] = [
    "write on notepad",
    "open hood",
    "close hood",
    "check gaps on front door",
    "open left front door",
    "close left front door",
    "close both left doors",
    "check trunk gaps",
    "open and close trunk",
    "check steering wheel",
]

LEFT_SENSOR_IDS: List[int] = [3, 17, 19, 20, 23, 25, 26, 28, 30, 31]
RIGHT_SENSOR_IDS: List[int] = [1, 2, 14, 16, 18, 21, 22, 24, 27, 29]
AXES: Tuple[str, str, str] = ("x", "y", "z")
SKODA_SAMPLING_HZ = 98.0


def _build_arm_channels(arm_prefix: str, sensor_ids: List[int]) -> List[str]:
    return [
        f"{arm_prefix}_acc_sensor{sensor_id:02d}_{axis}"
        for sensor_id in sensor_ids
        for axis in AXES
    ]


LEFT_CHANNELS: List[str] = _build_arm_channels("left", LEFT_SENSOR_IDS)
RIGHT_CHANNELS: List[str] = _build_arm_channels("right", RIGHT_SENSOR_IDS)
ALL_CHANNELS: List[str] = LEFT_CHANNELS + RIGHT_CHANNELS


def _resolve_skoda_segmented_mat(data_dir: str) -> Path:
    root = Path(data_dir)
    explicit_candidates = [
        root / "dataset_cp_2007_12.mat",
        root / "SkodaMiniCP_2015_08" / "dataset_cp_2007_12.mat",
    ]

    for candidate in explicit_candidates:
        if candidate.is_file():
            return candidate

    recursive_matches = sorted(root.rglob("dataset_cp_2007_12.mat"))
    if recursive_matches:
        return recursive_matches[0]

    raise FileNotFoundError(
        f"Could not locate 'dataset_cp_2007_12.mat' under '{data_dir}'."
    )


def _extract_axis_vector(
    axis_cells: np.ndarray, class_idx: int, instance_idx: int
) -> np.ndarray:
    values = np.asarray(axis_cells[0, class_idx][0, instance_idx], dtype=np.float64)
    return values.reshape(-1)


def parse_skoda(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    segmented_mat_path = _resolve_skoda_segmented_mat(dir)
    mat = scipy.io.loadmat(segmented_mat_path)

    if "dataset_left" not in mat or "dataset_right" not in mat:
        raise ValueError(
            "Expected 'dataset_left' and 'dataset_right' in dataset_cp_2007_12.mat."
        )

    dataset_left = np.asarray(mat["dataset_left"], dtype=object)
    dataset_right = np.asarray(mat["dataset_right"], dtype=object)

    if dataset_left.ndim != 2 or dataset_right.ndim != 2:
        raise ValueError(
            f"Unexpected Skoda segmented shape: left={dataset_left.shape}, right={dataset_right.shape}"
        )

    num_left_axes = int(dataset_left.shape[1])
    num_right_axes = int(dataset_right.shape[1])
    if num_left_axes != len(LEFT_CHANNELS) or num_right_axes != len(RIGHT_CHANNELS):
        raise ValueError(
            "Unexpected number of segmented axes in Skoda dataset. "
            f"Got left={num_left_axes}, right={num_right_axes}."
        )

    class_count_left = int(np.asarray(dataset_left[0, 0]).shape[1])
    class_count_right = int(np.asarray(dataset_right[0, 0]).shape[1])
    if class_count_left != class_count_right:
        raise ValueError(
            "Skoda class count mismatch between arms: "
            f"left={class_count_left}, right={class_count_right}."
        )
    if class_count_left != len(SKODA_ACTIVITY_NAMES):
        raise ValueError(
            "Skoda segmented class count does not match configured activities: "
            f"{class_count_left} vs {len(SKODA_ACTIVITY_NAMES)}."
        )

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int]] = []
    session_id = 0

    class_loop = tqdm(range(class_count_left))
    class_loop.set_description("Parsing SKODA segmented sessions")
    for class_idx in class_loop:
        left_instances = min(
            int(np.asarray(dataset_left[0, axis_idx])[0, class_idx].shape[1])
            for axis_idx in range(num_left_axes)
        )
        right_instances = min(
            int(np.asarray(dataset_right[0, axis_idx])[0, class_idx].shape[1])
            for axis_idx in range(num_right_axes)
        )
        num_instances = min(left_instances, right_instances)

        for instance_idx in range(num_instances):
            left_axis_vectors = [
                _extract_axis_vector(
                    np.asarray(dataset_left[0, axis_idx]), class_idx, instance_idx
                )
                for axis_idx in range(num_left_axes)
            ]
            right_axis_vectors = [
                _extract_axis_vector(
                    np.asarray(dataset_right[0, axis_idx]), class_idx, instance_idx
                )
                for axis_idx in range(num_right_axes)
            ]

            left_len = min(len(values) for values in left_axis_vectors)
            right_len = min(len(values) for values in right_axis_vectors)
            sample_count = min(left_len, right_len)
            if sample_count <= 0:
                continue

            left_stack = np.column_stack(
                [values[:sample_count] for values in left_axis_vectors]
            )
            right_stack = np.column_stack(
                [values[:sample_count] for values in right_axis_vectors]
            )
            stacked_data = np.hstack([left_stack, right_stack])

            time_sec = np.arange(sample_count, dtype=np.float64) / SKODA_SAMPLING_HZ
            session_df = pd.DataFrame(stacked_data, columns=ALL_CHANNELS)
            for channel in ALL_CHANNELS:
                session_df[channel] = session_df[channel].astype("float32").round(6)
            session_df["timestamp"] = pd.to_datetime(time_sec, unit="s").astype(
                "datetime64[ms]"
            )
            session_df = session_df[["timestamp", *ALL_CHANNELS]]

            sessions[session_id] = session_df.astype(
                {"timestamp": "datetime64[ms]", **{ch: "float32" for ch in ALL_CHANNELS}}
            )
            session_rows.append(
                {
                    "session_id": session_id,
                    "subject_id": 0,
                    activity_id_col: class_idx,
                }
            )
            session_id += 1

    if not sessions:
        raise ValueError("No SKODA segmented sessions could be created.")

    session_metadata = pd.DataFrame(session_rows).astype(
        {"session_id": "int32", "subject_id": "int32", activity_id_col: "int32"}
    )
    if activity_id_col != "activity_id":
        session_metadata["activity_id"] = session_metadata[activity_id_col].astype(
            "int32"
        )

    activity_metadata = pd.DataFrame(
        [
            {"activity_id": activity_id, "activity_name": activity_name}
            for activity_id, activity_name in enumerate(SKODA_ACTIVITY_NAMES)
        ]
    ).astype({"activity_id": "int32", "activity_name": "string"})

    return activity_metadata, session_metadata, sessions


cfg_skoda = WHARConfig(
    # Info + common
    dataset_id="skoda",
    dataset_url="http://har-dataset.org/doku.php?id=wiki:dataset",
    download_url="http://har-dataset.org/lib/exe/fetch.php?media=wiki:dataset:skodaminicp:skodaminicp_2015_08.zip",
    sampling_freq=98,
    num_of_subjects=1,
    num_of_activities=10,
    num_of_channels=len(ALL_CHANNELS),
    datasets_dir="./datasets",
    parallelize=True,
    # Parsing
    parse=parse_skoda,
    # Preprocessing (selections + sliding window)
    available_activities=canonicalize_activity_name_list(SKODA_ACTIVITY_NAMES),
    selected_activities=canonicalize_activity_name_list(SKODA_ACTIVITY_NAMES),
    available_channels=ALL_CHANNELS,
    selected_channels=ALL_CHANNELS,
    window_time=2,
    window_overlap=0.5,
    # Training (split info)
)

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

GOTOV_SENSOR_CHANNELS: List[str] = [
    "ankle_x",
    "ankle_y",
    "ankle_z",
    "wrist_x",
    "wrist_y",
    "wrist_z",
    "chest_x",
    "chest_y",
    "chest_z",
]

GOTOV_ACTIVITY_NAMES_ORIGINAL: List[str] = [
    "syncJumping",
    "standing",
    "step",
    "lyingDownLeft",
    "lyingDownRight",
    "sittingSofa",
    "sittingCouch",
    "sittingChair",
    "walkingStairsUp",
    "dishwashing",
    "stakingShelves",
    "vacuumCleaning",
    "walkingSlow",
    "walkingNormal",
    "walkingFast",
    "cycling",
]

GOTOV_ACTIVITY_NAMES_PREDICTED: List[str] = [
    "sitting",
    "standing",
    "walking",
    "cycling",
    "household",
    "lyingDown",
    "jumping",
]

GOTOV_GAP_MULTIPLIER: float = 3.0


def _resolve_energy_dir(data_dir: Path) -> Path:
    candidates = [
        data_dir
        / "Energy_Expenditure_Measurements"
        / "Energy_Expenditure_Measurements",
        data_dir / "Energy_Expenditure_Measurements",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate GOTOV Energy_Expenditure_Measurements under '{data_dir}'."
    )


def _resolve_label_column(activity_id_col: str) -> str:
    requested = activity_id_col.strip().lower()
    if requested in {"activity_id", "label", "original", "original_activity_labels"}:
        return "original_activity_labels"
    if requested in {"predicted", "predicted_activity_label"}:
        return "predicted_activity_label"
    return "original_activity_labels"


def _extract_subject_token(path: Path) -> str:
    return path.stem


def parse_gotov(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    data_dir = Path(dir)
    energy_dir = _resolve_energy_dir(data_dir)
    label_col = _resolve_label_column(activity_id_col)

    subject_files = sorted(energy_dir.glob("GOTOV*.csv"))
    if not subject_files:
        raise FileNotFoundError(
            f"No GOTOV participant CSV files found in '{energy_dir}'."
        )

    subject_tokens = [_extract_subject_token(path) for path in subject_files]
    subject_id_map = {
        token: idx for idx, token in enumerate(sorted(set(subject_tokens)))
    }

    expected_step_ms = 1e3 / float(cfg_gotov.sampling_freq)
    max_gap_ms = expected_step_ms * GOTOV_GAP_MULTIPLIER

    session_rows: List[Dict[str, int | str]] = []
    sessions: Dict[int, pd.DataFrame] = {}
    observed_activity_names: set[str] = set()
    session_id = 0

    loop = tqdm(subject_files, desc="Parsing GOTOV")
    for file_path in loop:
        loop.set_postfix(file=file_path.name, refresh=False)
        df = pd.read_csv(file_path, engine="python")
        required_cols = ["time", *GOTOV_SENSOR_CHANNELS, label_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing expected GOTOV columns in '{file_path.name}': {missing_cols}"
            )
        df = df[required_cols]
        if df.empty:
            continue

        rename_map = {"time": "timestamp", label_col: "activity_name"}
        work_df = df.rename(columns=rename_map).copy()

        work_df["activity_name"] = (
            work_df["activity_name"]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "NA": pd.NA})  # type: ignore
        )
        work_df = work_df.dropna(subset=["timestamp", "activity_name"])
        if work_df.empty:
            continue

        work_df["timestamp"] = pd.to_datetime(
            work_df["timestamp"], unit="ms", errors="coerce"
        )
        work_df = work_df.dropna(subset=["timestamp"])
        if work_df.empty:
            continue

        work_df = work_df.sort_values("timestamp").reset_index(drop=True)
        work_df[GOTOV_SENSOR_CHANNELS] = (
            work_df[GOTOV_SENSOR_CHANNELS]
            .apply(pd.to_numeric, errors="coerce")
            .astype("float32")
        )
        work_df = work_df.dropna(subset=GOTOV_SENSOR_CHANNELS)
        if work_df.empty:
            continue

        observed_activity_names.update(
            work_df["activity_name"].dropna().astype(str).unique().tolist()
        )

        timestamp_diff_ms = work_df["timestamp"].diff().dt.total_seconds().mul(1e3)
        session_breaks = (
            (work_df["activity_name"] != work_df["activity_name"].shift(1))
            | timestamp_diff_ms.isna()
            | (timestamp_diff_ms <= 0)
            | (timestamp_diff_ms > max_gap_ms)
        )
        local_session_id = pd.Series(
            np.cumsum(session_breaks.to_numpy(dtype=bool)),
            index=work_df.index,
        )

        subject_token = _extract_subject_token(file_path)
        subject_id = subject_id_map[subject_token]
        grouped_indices = work_df.groupby(local_session_id, sort=False).indices

        for idx in grouped_indices.values():
            idx = np.asarray(idx)
            if idx.size == 0:
                continue

            session_chunk = work_df.iloc[idx]
            activity_name = str(session_chunk["activity_name"].iloc[0])
            session_df = session_chunk[["timestamp", *GOTOV_SENSOR_CHANNELS]].copy()
            if session_df.empty:
                continue

            sessions[session_id] = session_df.astype(
                {
                    "timestamp": "datetime64[ms]",
                    **{col: "float32" for col in GOTOV_SENSOR_CHANNELS},
                }
            )
            session_rows.append(
                {
                    "session_id": session_id,
                    "subject_id": int(subject_id),
                    "activity_name": activity_name,
                }
            )
            session_id += 1

    if not sessions:
        raise ValueError("No valid sessions were parsed from GOTOV energy files.")

    if label_col == "predicted_activity_label":
        activity_names = GOTOV_ACTIVITY_NAMES_PREDICTED
    else:
        activity_names = GOTOV_ACTIVITY_NAMES_ORIGINAL

    activity_name_to_id = {name: idx for idx, name in enumerate(activity_names)}
    unknown_activity_names = sorted(
        name for name in observed_activity_names if name not in activity_name_to_id
    )
    if unknown_activity_names:
        raise ValueError(
            "Found GOTOV activities not covered by configured activity list: "
            + ", ".join(unknown_activity_names)
        )

    activity_df = pd.DataFrame(
        {
            "activity_id": list(range(len(activity_names))),
            "activity_name": activity_names,
        }
    ).astype({"activity_id": "int32", "activity_name": "string"})

    session_metadata = pd.DataFrame(session_rows)
    session_metadata["activity_id"] = session_metadata["activity_name"].map(
        activity_name_to_id
    )
    session_metadata = session_metadata.drop(columns=["activity_name"])
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_df, session_metadata, sessions


cfg_gotov = WHARConfig(
    # Info + common
    dataset_id="gotov",
    dataset_url="https://data.4tu.nl/articles/dataset/GOTOV_Human_Physical_Activity_and_Energy_Expenditure_Dataset_on_Older_Individuals/12716081",
    download_url="https://data.4tu.nl/ndownloader/items/f9bae0cd-ec4e-4cfb-aaa5-41bd1c5554ce/versions/2",
    sampling_freq=20,
    num_of_subjects=31,
    num_of_activities=16,
    num_of_channels=9,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_gotov,
    activity_id_col="original_activity_labels",
    # Preprocessing (selections + sliding window)
    activity_names=GOTOV_ACTIVITY_NAMES_ORIGINAL,
    sensor_channels=GOTOV_SENSOR_CHANNELS,
    window_time=2,
    window_overlap=0.5,
    # Training (split info)
)

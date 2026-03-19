from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

GOTOV_GENEACTIV_CHANNELS: List[str] = [
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

GOTOV_EQUIVITAL_SOURCE_TO_CHANNEL: Dict[str, str] = {
    "x": "equivital_x",
    "y": "equivital_y",
    "z": "equivital_z",
    "ECG.Lead.1": "equivital_ecg_lead_1",
    "ECG.Lead.2": "equivital_ecg_lead_2",
    "Breathing.Wave": "equivital_breathing_wave",
}

GOTOV_COSMED_SOURCE_TO_CHANNEL: Dict[str, str] = {
    "Rf": "cosmed_rf",
    "BR": "cosmed_br",
    "VT": "cosmed_vt",
    "VE": "cosmed_ve",
    "VO2": "cosmed_vo2",
    "VCO2": "cosmed_vco2",
    "O2exp": "cosmed_o2exp",
    "CO2exp": "cosmed_co2exp",
    "FeO2": "cosmed_feo2",
    "FeCO2": "cosmed_feco2",
    "FiO2": "cosmed_fio2",
    "FiCO2": "cosmed_fico2",
    "HR": "cosmed_hr",
}

GOTOV_SENSOR_CHANNELS: List[str] = [
    *GOTOV_GENEACTIV_CHANNELS,
    *list(GOTOV_EQUIVITAL_SOURCE_TO_CHANNEL.values()),
    *list(GOTOV_COSMED_SOURCE_TO_CHANNEL.values()),
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


def _resolve_activity_dir(data_dir: Path) -> Path:
    candidates = [
        data_dir / "Activity_Measurements" / "Activity_Measurements",
        data_dir / "Activity_Measurements",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate GOTOV Activity_Measurements under '{data_dir}'."
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


def _find_equivital_file(activity_dir: Path, subject_token: str) -> Path | None:
    subject_dir = activity_dir / subject_token
    if not subject_dir.exists():
        return None

    matches = sorted(subject_dir.glob(f"{subject_token}-equivital.csv"))
    return matches[0] if matches else None


def parse_gotov(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    data_dir = Path(dir)
    energy_dir = _resolve_energy_dir(data_dir)
    activity_dir = _resolve_activity_dir(data_dir)
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
    parsed_subject_tokens: set[str] = set()
    skipped_subject_tokens: List[str] = []

    loop = tqdm(subject_files, desc="Parsing GOTOV")
    for file_path in loop:
        loop.set_postfix(file=file_path.name, refresh=False)
        df = pd.read_csv(file_path, engine="python")
        required_cols = [
            "time",
            *GOTOV_GENEACTIV_CHANNELS,
            *list(GOTOV_COSMED_SOURCE_TO_CHANNEL.keys()),
            label_col,
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing expected GOTOV columns in '{file_path.name}': {missing_cols}"
            )
        df = df[required_cols]
        if df.empty:
            continue

        rename_map = {
            "time": "timestamp",
            label_col: "activity_name",
            **GOTOV_COSMED_SOURCE_TO_CHANNEL,
        }
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

        equivital_path = _find_equivital_file(activity_dir, file_path.stem)
        if equivital_path is None:
            skipped_subject_tokens.append(file_path.stem)
            continue

        eq_df = pd.read_csv(equivital_path, engine="python")
        required_eq_cols = ["time", *list(GOTOV_EQUIVITAL_SOURCE_TO_CHANNEL.keys())]
        missing_eq_cols = [col for col in required_eq_cols if col not in eq_df.columns]
        if missing_eq_cols:
            raise ValueError(
                f"Missing expected Equivital columns in '{equivital_path.name}': {missing_eq_cols}"
            )

        eq_df = eq_df[required_eq_cols].rename(
            columns={
                "time": "timestamp",
                **GOTOV_EQUIVITAL_SOURCE_TO_CHANNEL,
            }
        )
        eq_df["timestamp"] = pd.to_datetime(
            eq_df["timestamp"], unit="ms", errors="coerce"
        )
        eq_df = eq_df.dropna(subset=["timestamp"])
        if eq_df.empty:
            skipped_subject_tokens.append(file_path.stem)
            continue

        eq_channels = list(GOTOV_EQUIVITAL_SOURCE_TO_CHANNEL.values())
        eq_df[eq_channels] = eq_df[eq_channels].apply(pd.to_numeric, errors="coerce")
        eq_df = eq_df.dropna(subset=eq_channels)
        if eq_df.empty:
            skipped_subject_tokens.append(file_path.stem)
            continue

        eq_df = (
            eq_df.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )

        work_df = pd.merge_asof(
            work_df.sort_values("timestamp"),
            eq_df[["timestamp", *eq_channels]],
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(milliseconds=100),
        )

        work_df[GOTOV_SENSOR_CHANNELS] = work_df[GOTOV_SENSOR_CHANNELS].apply(
            pd.to_numeric, errors="coerce"
        )
        # COSMED channels are lower-rate; carry nearest known value to keep
        # a synchronized multichannel timeline with dense IMU samples.
        cosmed_channels = list(GOTOV_COSMED_SOURCE_TO_CHANNEL.values())
        work_df[cosmed_channels] = work_df[cosmed_channels].ffill().bfill()
        work_df[GOTOV_SENSOR_CHANNELS] = work_df[GOTOV_SENSOR_CHANNELS].astype(
            "float32"
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
        parsed_subject_tokens.add(subject_token)
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

    if skipped_subject_tokens:
        unique_skipped = sorted(set(skipped_subject_tokens))
        # print(
        #     "Skipping GOTOV subjects with missing/invalid Equivital data: "
        #     + ", ".join(unique_skipped)
        # )

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

    parsed_subject_id_map = {
        token: idx for idx, token in enumerate(sorted(parsed_subject_tokens))
    }

    session_metadata = pd.DataFrame(session_rows)
    original_to_compact_subject_id = {
        subject_id_map[token]: parsed_subject_id_map[token]
        for token in parsed_subject_tokens
    }
    session_metadata["subject_id"] = session_metadata["subject_id"].map(
        original_to_compact_subject_id
    )
    session_metadata["activity_id"] = session_metadata["activity_name"].map(
        activity_name_to_id
    )
    session_metadata = session_metadata.drop(columns=["activity_name"])
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_df, session_metadata, sessions


SELECTED_ACTIVITIES = GOTOV_ACTIVITY_NAMES_ORIGINAL

cfg_gotov = WHARConfig(
    # Info + common
    dataset_id="gotov",
    dataset_url="https://data.4tu.nl/articles/dataset/GOTOV_Human_Physical_Activity_and_Energy_Expenditure_Dataset_on_Older_Individuals/12716081",
    download_url="https://data.4tu.nl/ndownloader/items/f9bae0cd-ec4e-4cfb-aaa5-41bd1c5554ce/versions/2",
    sampling_freq=20,
    num_of_subjects=30,
    num_of_activities=16,
    num_of_channels=len(GOTOV_SENSOR_CHANNELS),
    datasets_dir="./datasets",
    # Parsing
    parse=parse_gotov,
    activity_id_col="original_activity_labels",
    # Preprocessing (selections + sliding window)
    available_activities=canonicalize_activity_name_list(GOTOV_ACTIVITY_NAMES_ORIGINAL),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=GOTOV_SENSOR_CHANNELS,
    selected_channels=GOTOV_SENSOR_CHANNELS,
    window_time=2,
    window_overlap=0.5,
    # Training (split info)
)

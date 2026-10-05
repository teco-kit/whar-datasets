import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from tqdm import tqdm

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

HARTH_ACTIVITY_NAMES: list[str] = [
    "walking",
    "running",
    "shuffling",
    "stairs (ascending)",
    "stairs (descending)",
    "standing",
    "sitting",
    "lying",
    "cycling (sit)",
    "cycling (stand)",
    "cycling (sit, inactive)",
    "cycling (stand, inactive)",
]

HARTH_ACTIVITY_MAP: dict[int, str] = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs (ascending)",
    5: "stairs (descending)",
    6: "standing",
    7: "sitting",
    8: "lying",
    13: "cycling (sit)",
    14: "cycling (stand)",
    130: "cycling (sit, inactive)",
    140: "cycling (stand, inactive)",
}

HARTH_SENSOR_CHANNELS: list[str] = [
    "back_x",
    "back_y",
    "back_z",
    "thigh_x",
    "thigh_y",
    "thigh_z",
]

HARTH_MAX_STEP_MULTIPLIER = 3.0
_SUBJECT_PATTERN = re.compile(r"^S(\d+)", re.IGNORECASE)
HARTH_SOURCE_STEP_SECONDS = 1.0 / 100.0
HARTH_OUTPUT_STEP_SECONDS = 1.0 / 50.0


def _downsample_s006_session_to_50_hz(session_df: pd.DataFrame) -> pd.DataFrame:
    """Anti-alias the anomalous 100 Hz S006 file to HARTH's published 50 Hz."""
    timestamps = session_df["timestamp"]
    step_seconds = timestamps.diff().dt.total_seconds().dropna().to_numpy()
    if len(step_seconds) and not np.allclose(
        step_seconds,
        HARTH_SOURCE_STEP_SECONDS,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "HARTH S006 was expected to contain regular 100 Hz samples within "
            "each label/gap-delimited session before downsampling. Found a "
            "different cadence; inspect its raw timestamps instead of silently "
            "resampling across missing samples."
        )

    values = session_df[HARTH_SENSOR_CHANNELS].to_numpy(dtype=np.float64)
    if len(values) >= 3:
        # Polyphase filtering removes frequencies above the 25 Hz output
        # Nyquist limit before retaining every other 100 Hz sample.
        values = resample_poly(values, up=1, down=2, axis=0, padtype="line")
    else:
        # Very short event fragments cannot support a useful anti-alias filter.
        # Retain the first 50 Hz-grid sample without extending the source range.
        values = values[::2]

    output = pd.DataFrame(values, columns=HARTH_SENSOR_CHANNELS)
    output.insert(
        0,
        "timestamp",
        timestamps.iloc[0]
        + pd.to_timedelta(
            np.arange(len(output), dtype=np.float64) * HARTH_OUTPUT_STEP_SECONDS,
            unit="s",
        ),
    )
    return output


def _extract_subject_id(path: Path) -> int:
    match = _SUBJECT_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Could not parse subject id from '{path.name}'.")
    return int(match.group(1))


def parse_harth(
    dir: str, activity_id_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    data_dir = Path(dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"HARTH data directory not found at '{data_dir}'.")

    csv_paths = sorted(data_dir.rglob("S*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No HARTH recordings found inside '{data_dir}'.")

    session_dfs: list[pd.DataFrame] = []
    global_session_id = 0

    required_columns = ["timestamp", *HARTH_SENSOR_CHANNELS, "label"]
    expected_step_seconds = 1.0 / 50.0
    max_allowed_gap_seconds = expected_step_seconds * HARTH_MAX_STEP_MULTIPLIER

    for csv_path in tqdm(csv_paths, desc="Parsing HARTH"):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns {sorted(missing_cols)} in '{csv_path.name}'."
            )
        df = df[required_columns].reset_index(drop=True).copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        df[activity_id_col] = df["label"].astype("int32")
        df["raw_activity_id"] = df[activity_id_col]
        df["subject_id"] = _extract_subject_id(csv_path)
        df["activity_name"] = df[activity_id_col].map(HARTH_ACTIVITY_MAP)
        df["activity_name"] = df["activity_name"].fillna("unknown")
        df["source_file"] = csv_path.name

        time_diff = df["timestamp"].diff().dt.total_seconds().fillna(0.0)
        activity_change = df[activity_id_col] != df[activity_id_col].shift(1)
        session_gap = time_diff > max_allowed_gap_seconds
        session_start = activity_change | session_gap
        session_start.iloc[0] = True
        local_session_ids = session_start.cumsum().astype("int32")
        df["session_id"] = local_session_ids + global_session_id
        global_session_id = int(df["session_id"].max()) + 1

        session_dfs.append(df)

    df = pd.concat(session_dfs, ignore_index=True)

    df["activity_id"] = pd.factorize(df["raw_activity_id"])[0]
    df["subject_id"] = pd.factorize(df["subject_id"])[0]
    df["session_id"] = pd.factorize(df["session_id"])[0]

    activity_metadata = (
        df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"])
        .sort_values("activity_id")
        .reset_index(drop=True)
    )

    session_metadata = (
        df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates(subset=["session_id"])
        .reset_index(drop=True)
    )

    sessions: dict[int, pd.DataFrame] = {}
    loop = tqdm(session_metadata["session_id"].unique(), desc="Creating sessions")
    for session_id in loop:
        session_df = df[df["session_id"] == session_id]
        source_file = str(session_df["source_file"].iloc[0])
        session_df = session_df.drop(
            columns=[
                "session_id",
                "subject_id",
                "activity_id",
                "activity_name",
                "source_file",
                "label",
                activity_id_col,
                "raw_activity_id",
            ],
            errors="ignore",
        ).reset_index(drop=True)

        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"])
        if Path(source_file).stem.upper() == "S006":
            # UCI describes the distributed HARTH data as 50 Hz, and the
            # original paper says 100 Hz recordings were downsampled to 50 Hz.
            # The distributed S006.csv is the exception: its raw timestamps
            # are uniformly 10 ms apart (100 Hz), while the other 21 files are
            # 20 ms apart. Normalize this file here so HARTH remains fixed-rate.
            session_df = _downsample_s006_session_to_50_hz(session_df)
        float_cols = [col for col in session_df.columns if col != "timestamp"]
        session_df[float_cols] = session_df[float_cols].astype("float32")
        session_df[float_cols] = session_df[float_cols].round(6)
        sessions[int(session_id)] = session_df

    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


SELECTED_ACTIVITIES = HARTH_ACTIVITY_NAMES

cfg_harth = WHARConfig(
    dataset_id="harth",
    dataset_url="https://archive.ics.uci.edu/dataset/779/harth",
    download_url="https://archive.ics.uci.edu/static/public/779/harth.zip",
    sampling_freq=50,
    num_of_subjects=22,
    num_of_activities=len(HARTH_ACTIVITY_NAMES),
    num_of_channels=len(HARTH_SENSOR_CHANNELS),
    parse=parse_harth,
    activity_id_col="activity_id",
    available_activities=canonicalize_activity_name_list(HARTH_ACTIVITY_NAMES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=HARTH_SENSOR_CHANNELS,
    selected_channels=HARTH_SENSOR_CHANNELS,
)

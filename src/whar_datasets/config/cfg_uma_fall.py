import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

UMA_FILENAME_PATTERN = re.compile(
    r"^UMAFall_Subject_(?P<subject>\d+)_(?P<group>ADL|Fall)_(?P<activity>.+?)_"
    r"(?P<trial>\d+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})\.csv$"
)

UMA_ACTIVITY_NAMES: List[str] = [
    "Walking",
    "Jogging",
    "GoUpstairs",
    "GoDownstairs",
    "Bending",
    "HandsUp",
    "Hopping",
    "LyingDown_OnABed",
    "Sitting_GettingUpOnAChair",
    "MakingACall",
    "OpeningDoor",
    "Aplausing",
    "forwardFall",
    "backwardFall",
    "lateralFall",
]

UMA_SENSOR_TYPE_TO_NAME: Dict[int, str] = {
    0: "acc",
    1: "gyro",
    2: "mag",
}

UMA_SENSOR_ID_TO_NAME: Dict[int, str] = {
    0: "right_pocket_phone",
    1: "chest",
    2: "waist",
    3: "wrist",
    4: "ankle",
}


def _build_uma_sensor_channels() -> List[str]:
    channels: List[str] = []
    axes = ("x", "y", "z")

    for sensor_id in [0, 1, 2, 3, 4]:
        sensor_name = UMA_SENSOR_ID_TO_NAME[sensor_id]
        for axis in axes:
            channels.append(f"{sensor_name}_acc_{axis}")

    for sensor_type in [1, 2]:
        modality = UMA_SENSOR_TYPE_TO_NAME[sensor_type]
        for sensor_id in [1, 2, 3, 4]:
            sensor_name = UMA_SENSOR_ID_TO_NAME[sensor_id]
            for axis in axes:
                channels.append(f"{sensor_name}_{modality}_{axis}")

    return channels


UMA_SENSOR_CHANNELS: List[str] = _build_uma_sensor_channels()


def _find_uma_fall_root(data_dir: str) -> Path:
    base = Path(data_dir)
    candidates = [
        base,
        base / "data",
        base / "uma_fall" / "data",
        base / "UMAFall_Dataset",
    ]

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("UMAFall_Subject_*.csv")):
            return candidate

    for candidate in base.rglob("*"):
        if candidate.is_dir() and any(candidate.glob("UMAFall_Subject_*.csv")):
            return candidate

    raise FileNotFoundError(f"Could not locate UMAFall CSV files under '{data_dir}'.")


def _extract_file_metadata(file_path: Path) -> Tuple[int, str, pd.Timestamp]:
    match = UMA_FILENAME_PATTERN.match(file_path.name)
    if match is None:
        raise ValueError(f"Unexpected UMAFall filename format: '{file_path.name}'.")

    subject_raw = int(match.group("subject"))
    activity_name = match.group("activity")
    session_start = pd.to_datetime(
        f"{match.group('date')} {match.group('time').replace('-', ':')}"
    )

    return subject_raw, activity_name, session_start


def _load_phone_accelerometer_session(
    file_path: Path,
    session_start: pd.Timestamp,
) -> pd.DataFrame:
    columns = [
        "timestamp_ms",
        "sample_no",
        "x",
        "y",
        "z",
        "sensor_type",
        "sensor_id",
        "unused",
    ]
    df = pd.read_csv(
        file_path,
        sep=";",
        comment="%",
        header=None,
        names=columns,
        engine="python",
    )

    phone_acc = df[(df["sensor_type"] == 0) & (df["sensor_id"] == 0)].copy()
    if phone_acc.empty:
        raise ValueError(
            f"File '{file_path.name}' does not contain smartphone accelerometer rows."
        )

    phone_acc = phone_acc.sort_values("sample_no").reset_index(drop=True)
    base_step_ms = 5
    base_elapsed_ms = (
        phone_acc["sample_no"].astype("int64") - phone_acc["sample_no"].iloc[0]
    ) * base_step_ms

    session_df = pd.DataFrame(
        {"timestamp": session_start + pd.to_timedelta(base_elapsed_ms, unit="ms")}
    )
    session_df["timestamp"] = session_df["timestamp"].astype("datetime64[ms]")

    for sensor_type in [0, 1, 2]:
        for sensor_id in [0, 1, 2, 3, 4]:
            if sensor_type in {1, 2} and sensor_id == 0:
                continue

            sensor_name = UMA_SENSOR_ID_TO_NAME[sensor_id]
            modality = UMA_SENSOR_TYPE_TO_NAME[sensor_type]
            stream_cols = [
                f"{sensor_name}_{modality}_x",
                f"{sensor_name}_{modality}_y",
                f"{sensor_name}_{modality}_z",
            ]

            stream = df[
                (df["sensor_type"] == sensor_type) & (df["sensor_id"] == sensor_id)
            ][["timestamp_ms", "x", "y", "z"]].copy()
            if stream.empty:
                for col in stream_cols:
                    session_df[col] = 0.0
                continue

            stream = stream.sort_values("timestamp_ms").drop_duplicates(
                subset=["timestamp_ms"], keep="last"
            )
            stream["elapsed_ms"] = stream["timestamp_ms"].astype("int64") - int(
                phone_acc["timestamp_ms"].iloc[0]
            )
            stream = stream.rename(
                columns={
                    "x": stream_cols[0],
                    "y": stream_cols[1],
                    "z": stream_cols[2],
                }
            )
            stream = stream[["elapsed_ms"] + stream_cols]

            aligned = pd.merge_asof(
                pd.DataFrame({"elapsed_ms": base_elapsed_ms}),
                stream,
                on="elapsed_ms",
                direction="nearest",
                tolerance=150,
            )

            for col in stream_cols:
                session_df[col] = aligned[col]

    for col in UMA_SENSOR_CHANNELS:
        if col not in session_df.columns:
            session_df[col] = 0.0

    session_df[UMA_SENSOR_CHANNELS] = session_df[UMA_SENSOR_CHANNELS].interpolate(
        method="linear",
        limit_direction="both",
        axis=0,
    )
    session_df[UMA_SENSOR_CHANNELS] = session_df[UMA_SENSOR_CHANNELS].fillna(0.0)
    session_df[UMA_SENSOR_CHANNELS] = session_df[UMA_SENSOR_CHANNELS].astype("float32")
    session_df = session_df[["timestamp"] + UMA_SENSOR_CHANNELS]

    return session_df


def parse_uma_fall(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    root = _find_uma_fall_root(dir)
    files = sorted(root.glob("UMAFall_Subject_*.csv"))
    if not files:
        raise FileNotFoundError(f"No UMAFall CSV files found in '{root}'.")

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[dict] = []

    loop = tqdm(files)
    loop.set_description("Creating sessions")

    for session_id, file_path in enumerate(loop):
        subject_raw, activity_name, session_start = _extract_file_metadata(file_path)
        session_df = _load_phone_accelerometer_session(file_path, session_start)
        sessions[session_id] = session_df
        session_rows.append(
            {
                "session_id": session_id,
                "subject_raw_id": subject_raw,
                "activity_name": activity_name,
            }
        )

    if not session_rows:
        raise ValueError("No UMAFall sessions could be parsed.")

    session_metadata = pd.DataFrame(session_rows)
    session_metadata["subject_id"] = pd.factorize(
        session_metadata["subject_raw_id"], sort=True
    )[0]
    session_metadata["activity_id"] = pd.factorize(
        session_metadata["activity_name"], sort=False
    )[0]
    session_metadata = session_metadata[["session_id", "subject_id", "activity_id"]]

    activity_metadata = (
        pd.DataFrame(session_rows)[["activity_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
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


cfg_uma_fall = WHARConfig(
    # Info + common
    dataset_id="uma_fall",
    download_url="https://figshare.com/ndownloader/files/43076140",
    sampling_freq=200,
    num_of_subjects=19,
    num_of_activities=15,
    num_of_channels=39,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_uma_fall,
    # Preprocessing (selections + sliding window)
    activity_names=UMA_ACTIVITY_NAMES,
    sensor_channels=UMA_SENSOR_CHANNELS,
    window_time=3,
    window_overlap=0.5,
)

import os
import re
import zipfile
from typing import Dict, List, Tuple

import pandas as pd

from whar_datasets.config.config import WHARConfig

REAL_WORLD_ACTIVITY_NAMES = [
    "walking",
    "running",
    "sitting",
    "standing",
    "lying",
    "climbingup",
    "climbingdown",
    "jumping",
]

REAL_WORLD_IMU_POSITIONS = [
    "chest",
    "forearm",
    "head",
    "shin",
    "thigh",
    "upperarm",
    "waist",
]

REAL_WORLD_NON_FOREARM_POSITIONS = [
    "chest",
    "head",
    "shin",
    "thigh",
    "upperarm",
    "waist",
]


def get_real_world_sensor_channels() -> List[str]:
    channels: List[str] = []

    for sensor in ("acc", "gyr", "mag"):
        for position in REAL_WORLD_IMU_POSITIONS:
            channels.extend(
                [
                    f"{sensor}_{position}_x",
                    f"{sensor}_{position}_y",
                    f"{sensor}_{position}_z",
                ]
            )

    for position in REAL_WORLD_NON_FOREARM_POSITIONS:
        channels.append(f"lig_{position}")

    for position in REAL_WORLD_NON_FOREARM_POSITIONS:
        channels.extend([f"gps_{position}_lat", f"gps_{position}_lng"])

    return channels


REAL_WORLD_SENSOR_CHANNELS = get_real_world_sensor_channels()

SENSOR_VALUE_COLS = {
    "acc": ["attr_x", "attr_y", "attr_z"],
    "gyr": ["attr_x", "attr_y", "attr_z"],
    "mag": ["attr_x", "attr_y", "attr_z"],
    "lig": ["attr_light"],
    "gps": ["attr_lat", "attr_lng"],
}


def find_real_world_root(dir: str) -> str:
    proband_pattern = re.compile(r"^proband\d+$")

    root_entries = os.listdir(dir)
    if any(proband_pattern.match(entry) for entry in root_entries):
        return dir

    for entry in root_entries:
        candidate = os.path.join(dir, entry)
        if not os.path.isdir(candidate):
            continue

        child_entries = os.listdir(candidate)
        if any(proband_pattern.match(child_entry) for child_entry in child_entries):
            return candidate

    raise FileNotFoundError(
        f"Could not locate extracted RealWorld proband directories in '{dir}'."
    )


def load_csv_frames_from_source(source_path: str) -> List[Tuple[str, pd.DataFrame]]:
    frames: List[Tuple[str, pd.DataFrame]] = []

    if os.path.isdir(source_path):
        for file in sorted(os.listdir(source_path)):
            file_path = os.path.join(source_path, file)
            if not os.path.isfile(file_path) or not file.lower().endswith(".csv"):
                continue

            frames.append((file, pd.read_csv(file_path)))

        return frames

    if source_path.lower().endswith(".zip") and zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path) as zipf:
            for member in sorted(zipf.namelist()):
                if not member.lower().endswith(".csv"):
                    continue

                with zipf.open(member) as csv_file:
                    frames.append((os.path.basename(member), pd.read_csv(csv_file)))

    return frames


def get_position_from_filename(file_name: str) -> str | None:
    match = re.search(r"_([a-z]+)\.csv$", file_name.lower())
    if match is None:
        return None
    return match.group(1)


def get_sensor_channel_map(sensor: str, position: str) -> Dict[str, str]:
    if sensor in ("acc", "gyr", "mag"):
        return {
            "attr_x": f"{sensor}_{position}_x",
            "attr_y": f"{sensor}_{position}_y",
            "attr_z": f"{sensor}_{position}_z",
        }

    if sensor == "lig":
        return {"attr_light": f"lig_{position}"}

    if sensor == "gps":
        return {
            "attr_lat": f"gps_{position}_lat",
            "attr_lng": f"gps_{position}_lng",
        }

    return {}


def standardize_sensor_frame(
    raw_df: pd.DataFrame, sensor: str, position: str
) -> pd.DataFrame | None:
    if "attr_time" not in raw_df.columns:
        return None

    source_value_cols = SENSOR_VALUE_COLS[sensor]
    if any(col not in raw_df.columns for col in source_value_cols):
        return None

    channel_map = get_sensor_channel_map(sensor, position)

    df = raw_df[["attr_time"] + source_value_cols].copy()
    df["timestamp"] = pd.to_numeric(df["attr_time"], errors="coerce")

    for col in source_value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["attr_time"])
    df = df.rename(columns=channel_map)
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")

    if df.empty:
        return None

    value_cols = [channel_map[col] for col in source_value_cols]
    return df[["timestamp"] + value_cols]


def load_sensor_frames(source_path: str, sensor: str) -> List[pd.DataFrame]:
    sensor_frames: List[pd.DataFrame] = []

    for file_name, raw_df in load_csv_frames_from_source(source_path):
        position = get_position_from_filename(file_name)
        if position is None:
            continue

        standardized = standardize_sensor_frame(raw_df, sensor, position)
        if standardized is None:
            continue

        sensor_frames.append(standardized)

    return sensor_frames


def merge_real_world_session(
    sensor_sources: Dict[str, str], sampling_freq: int
) -> pd.DataFrame | None:
    frames: List[pd.DataFrame] = []
    frames_by_sensor: Dict[str, List[pd.DataFrame]] = {
        "acc": [],
        "gyr": [],
        "mag": [],
        "lig": [],
        "gps": [],
    }

    for sensor in ("acc", "gyr", "mag", "lig", "gps"):
        source = sensor_sources.get(sensor)
        if source is None:
            continue
        sensor_frames = load_sensor_frames(source, sensor)
        frames_by_sensor[sensor].extend(sensor_frames)
        frames.extend(sensor_frames)

    if not frames:
        return None

    reference_frames = frames_by_sensor["acc"] if frames_by_sensor["acc"] else frames

    min_time = min(frame["timestamp"].min() for frame in reference_frames)
    max_time = max(frame["timestamp"].max() for frame in reference_frames)

    if pd.isna(min_time) or pd.isna(max_time):
        return None

    freq_ms = int(1e3 / sampling_freq)
    timeline = pd.date_range(start=min_time, end=max_time, freq=f"{freq_ms}ms")
    if len(timeline) == 0:
        timeline = pd.DatetimeIndex([min_time])

    session_df = pd.DataFrame({"timestamp": timeline})

    for frame in frames:
        frame = frame.sort_values("timestamp")
        session_df = pd.merge_asof(
            session_df,
            frame,
            on="timestamp",
            direction="nearest",
        )

    for channel in REAL_WORLD_SENSOR_CHANNELS:
        if channel not in session_df.columns:
            session_df[channel] = 0.0

    session_df = session_df[["timestamp"] + REAL_WORLD_SENSOR_CHANNELS]

    values_df = session_df[REAL_WORLD_SENSOR_CHANNELS].apply(
        pd.to_numeric, errors="coerce"
    )
    values_df = (
        values_df.interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    session_df[REAL_WORLD_SENSOR_CHANNELS] = values_df.astype("float32")
    session_df = session_df.round(6)
    session_df = session_df.astype(
        {
            **{channel: "float32" for channel in REAL_WORLD_SENSOR_CHANNELS},
            "timestamp": "datetime64[ms]",
        }
    )

    return session_df


def parse_real_world(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    root_dir = find_real_world_root(dir)
    proband_pattern = re.compile(r"^proband(\d+)$")
    source_pattern = re.compile(
        r"^(acc|gyr|mag|lig|gps|mic)_([a-z]+)_csv(?:\.zip)?$", re.IGNORECASE
    )

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int | str]] = []
    session_id = 0

    proband_dirs = []
    for entry in os.listdir(root_dir):
        match = proband_pattern.match(entry)
        if match is None:
            continue
        proband_dirs.append((int(match.group(1)), os.path.join(root_dir, entry)))

    for subject_idx, subject_dir in sorted(proband_dirs, key=lambda item: item[0]):
        data_dir = os.path.join(subject_dir, "data")
        if not os.path.isdir(data_dir):
            continue

        activity_sources: Dict[str, Dict[str, str]] = {}

        for source_entry in os.listdir(data_dir):
            match = source_pattern.match(source_entry.lower())
            if match is None:
                continue

            sensor, activity = match.group(1), match.group(2)
            if sensor == "mic":
                continue

            source_path = os.path.join(data_dir, source_entry)
            activity_sources.setdefault(activity, {})[sensor] = source_path

        for activity in REAL_WORLD_ACTIVITY_NAMES:
            sensor_sources = activity_sources.get(activity)
            if sensor_sources is None:
                continue

            session_df = merge_real_world_session(sensor_sources, sampling_freq=50)
            if session_df is None or session_df.empty:
                continue

            sessions[session_id] = session_df
            session_rows.append(
                {
                    "session_id": session_id,
                    "subject_id": subject_idx - 1,
                    "activity_name": activity,
                }
            )
            session_id += 1

    if len(sessions) == 0:
        raise ValueError(
            f"No RealWorld sessions were parsed from extracted data under '{root_dir}'."
        )

    session_metadata = pd.DataFrame(session_rows)
    session_metadata["subject_id"] = pd.factorize(
        session_metadata["subject_id"], sort=True
    )[0]

    activity_to_id = {name: idx for idx, name in enumerate(REAL_WORLD_ACTIVITY_NAMES)}
    session_metadata["activity_id"] = session_metadata["activity_name"].map(
        activity_to_id
    )

    session_metadata = session_metadata[["session_id", "subject_id", "activity_id"]]

    activity_metadata = pd.DataFrame(
        {
            "activity_id": list(activity_to_id.values()),
            "activity_name": list(activity_to_id.keys()),
        }
    )

    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_real_world = WHARConfig(
    # Info + common
    dataset_id="real_world",
    download_url="http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip",
    sampling_freq=50,
    num_of_subjects=15,
    num_of_activities=8,
    num_of_channels=len(REAL_WORLD_SENSOR_CHANNELS),
    datasets_dir="./datasets",
    # Parsing
    parse=parse_real_world,
    # Preprocessing (selections + sliding window)
    activity_names=REAL_WORLD_ACTIVITY_NAMES,
    sensor_channels=REAL_WORLD_SENSOR_CHANNELS,
    window_time=2.56,
    window_overlap=0.5,
)

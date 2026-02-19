import os
from typing import Dict, List, Tuple

import pandas as pd

from whar_datasets.config.config import WHARConfig

REAL_LIFE_HAR_ACTIVITY_NAMES = [
    "Inactive",
    "Active",
    "Walking",
    "Driving",
]

REAL_LIFE_HAR_SENSOR_CHANNELS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "gps_lat_increment",
    "gps_long_increment",
    "gps_alt_increment",
    "gps_speed",
    "gps_bearing",
    "gps_accuracy",
]

REAL_LIFE_HAR_SENSOR_SPECS: Dict[str, Dict[str, str | List[str]]] = {
    "acc": {
        "file": "sensoringData_acc_prepared_20.csv",
        "raw_cols": ["acc_x_axis", "acc_y_axis", "acc_z_axis"],
        "std_cols": ["acc_x", "acc_y", "acc_z"],
    },
    "gyro": {
        "file": "sensoringData_gyro_prepared_20.csv",
        "raw_cols": ["gyro_x_axis", "gyro_y_axis", "gyro_z_axis"],
        "std_cols": ["gyro_x", "gyro_y", "gyro_z"],
    },
    "mag": {
        "file": "sensoringData_magn_prepared_20.csv",
        "raw_cols": ["magn_x_axis", "magn_y_axis", "magn_z_axis"],
        "std_cols": ["mag_x", "mag_y", "mag_z"],
    },
    "gps": {
        "file": "sensoringData_gps_prepared_20.csv",
        "raw_cols": [
            "gps_lat_increment",
            "gps_long_increment",
            "gps_alt_increment",
            "gps_speed",
            "gps_bearing",
            "gps_accuracy",
        ],
        "std_cols": [
            "gps_lat_increment",
            "gps_long_increment",
            "gps_alt_increment",
            "gps_speed",
            "gps_bearing",
            "gps_accuracy",
        ],
    },
}


def _find_real_life_har_root(dir: str) -> str:
    direct = os.path.join(dir, "data_cleaned_adapted_full")
    if os.path.isdir(direct):
        return direct

    for entry in os.listdir(dir):
        candidate = os.path.join(dir, entry)
        nested = os.path.join(candidate, "data_cleaned_adapted_full")
        if os.path.isdir(nested):
            return nested

    raise FileNotFoundError(
        f"Could not locate 'data_cleaned_adapted_full' under '{dir}'."
    )


def _init_empty_sensor_storage(
    session_ids: List[int],
) -> Dict[str, Dict[int, List[pd.DataFrame]]]:
    return {
        sensor: {sid: [] for sid in session_ids}
        for sensor in REAL_LIFE_HAR_SENSOR_SPECS.keys()
    }


def _load_sensor_data_by_session(
    root_path: str,
    session_bounds: pd.DataFrame,
    chunksize: int = 750_000,
) -> Tuple[Dict[str, Dict[int, List[pd.DataFrame]]], Dict[int, str]]:
    sensor_values_by_session = _init_empty_sensor_storage(
        [int(sid) for sid in session_bounds["session_id"].tolist()]
    )
    session_activity_names: Dict[int, str] = {}

    relevant_cols = [
        "session_id",
        "username",
        "activity_id",
        "init_timestamp",
        "end_timestamp",
    ]
    session_ref = session_bounds[relevant_cols].copy()
    session_ref = session_ref.astype(
        {
            "session_id": "int32",
            "username": "int32",
            "activity_id": "int32",
            "init_timestamp": "float64",
            "end_timestamp": "float64",
        }
    )

    for sensor, spec in REAL_LIFE_HAR_SENSOR_SPECS.items():
        file_name = str(spec["file"])
        raw_cols = list(spec["raw_cols"])
        std_cols = list(spec["std_cols"])
        sensor_path = os.path.join(root_path, file_name)

        usecols = ["username", "timestamp", "activity_id", "activity"] + raw_cols

        for chunk in pd.read_csv(
            sensor_path,
            usecols=usecols,
            chunksize=chunksize,
            dtype={"username": "int32", "activity_id": "int32"},
            low_memory=False,
        ):
            chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
            for col in raw_cols:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            merged = chunk.merge(
                session_ref,
                on=["username", "activity_id"],
                how="inner",
            )
            if merged.empty:
                continue

            valid_mask = (merged["timestamp"] >= merged["init_timestamp"]) & (
                merged["timestamp"] <= merged["end_timestamp"]
            )
            merged = merged[valid_mask]
            if merged.empty:
                continue

            merged = merged.rename(columns=dict(zip(raw_cols, std_cols)))
            keep_cols = ["session_id", "timestamp", "activity"] + std_cols
            merged = merged[keep_cols]
            merged = merged.dropna(subset=["timestamp"])

            for session_id, session_chunk in merged.groupby("session_id"):
                sid = int(session_id)
                if sid not in session_activity_names:
                    activity_values = session_chunk["activity"].dropna()
                    if len(activity_values) > 0:
                        session_activity_names[sid] = str(activity_values.iloc[0])

                values = session_chunk.drop(columns=["activity"]).copy()
                values["timestamp"] = pd.to_datetime(
                    values["timestamp"], unit="s", errors="coerce"
                )
                values = values.dropna(subset=["timestamp"])
                if values.empty:
                    continue

                values = values.sort_values("timestamp")
                values = values[["timestamp"] + std_cols]
                sensor_values_by_session[sensor][sid].append(values)

    return sensor_values_by_session, session_activity_names


def _merge_session_modalities(
    session_info: pd.Series,
    sensor_values_by_session: Dict[str, Dict[int, List[pd.DataFrame]]],
    sampling_freq: int,
) -> pd.DataFrame:
    session_id = int(session_info["session_id"])
    start_dt = pd.to_datetime(float(session_info["init_timestamp"]), unit="s")
    end_dt = pd.to_datetime(float(session_info["end_timestamp"]), unit="s")

    if end_dt < start_dt:
        raise ValueError(f"Invalid session bounds for session_id={session_id}.")

    freq_ms = max(int(1e3 / sampling_freq), 1)
    timeline = pd.date_range(start=start_dt, end=end_dt, freq=f"{freq_ms}ms")
    if len(timeline) == 0:
        timeline = pd.DatetimeIndex([start_dt])

    session_df = pd.DataFrame({"timestamp": timeline})

    for sensor, spec in REAL_LIFE_HAR_SENSOR_SPECS.items():
        std_cols = list(spec["std_cols"])
        parts = sensor_values_by_session[sensor][session_id]
        if not parts:
            for col in std_cols:
                session_df[col] = 0.0
            continue

        sensor_df = pd.concat(parts, ignore_index=True)
        sensor_df = sensor_df.sort_values("timestamp")
        sensor_df = sensor_df.drop_duplicates(subset=["timestamp"], keep="first")

        session_df = pd.merge_asof(
            session_df,
            sensor_df,
            on="timestamp",
            direction="nearest",
        )

    values = session_df[REAL_LIFE_HAR_SENSOR_CHANNELS].apply(
        pd.to_numeric, errors="coerce"
    )
    values = values.interpolate(method="linear", limit_direction="both").ffill().bfill()
    values = values.fillna(0.0).astype("float32").round(6)

    session_df = pd.concat([session_df[["timestamp"]], values], axis=1)
    session_df = session_df.astype(
        {
            **{col: "float32" for col in REAL_LIFE_HAR_SENSOR_CHANNELS},
            "timestamp": "datetime64[ms]",
        }
    )

    return session_df


def parse_real_life_har(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    root_path = _find_real_life_har_root(dir)
    valid_sessions_path = os.path.join(root_path, "validSessions_20.csv")
    valid_sessions = pd.read_csv(valid_sessions_path)

    required_cols = {"username", "activity_id", "init_timestamp", "end_timestamp"}
    if not required_cols.issubset(valid_sessions.columns):
        raise ValueError(
            f"Missing required columns in '{valid_sessions_path}'. "
            f"Expected at least: {sorted(required_cols)}."
        )

    valid_sessions = valid_sessions[list(required_cols)].copy()
    valid_sessions = valid_sessions.astype(
        {
            "username": "int32",
            "activity_id": "int32",
            "init_timestamp": "float64",
            "end_timestamp": "float64",
        }
    )

    valid_sessions = valid_sessions.sort_values(
        by=["username", "init_timestamp", "end_timestamp"]
    ).reset_index(drop=True)
    valid_sessions["session_id"] = range(len(valid_sessions))

    sensor_values_by_session, session_activity_names = _load_sensor_data_by_session(
        root_path, valid_sessions
    )

    sessions: Dict[int, pd.DataFrame] = {}
    session_rows: List[Dict[str, int]] = []

    activity_name_to_id = {
        name: idx for idx, name in enumerate(REAL_LIFE_HAR_ACTIVITY_NAMES)
    }

    for row in valid_sessions.itertuples(index=False):
        session_id = int(row.session_id)  # type: ignore
        subject_raw = int(row.username)  # type: ignore

        activity_name = session_activity_names.get(session_id)
        if activity_name not in activity_name_to_id:
            continue

        merged_session = _merge_session_modalities(
            pd.Series(
                {
                    "session_id": session_id,
                    "init_timestamp": float(row.init_timestamp),  # type: ignore
                    "end_timestamp": float(row.end_timestamp),  # type: ignore
                }
            ),
            sensor_values_by_session,
            sampling_freq=20,
        )
        if merged_session.empty:
            continue

        sessions[session_id] = merged_session
        session_rows.append(
            {
                "session_id": session_id,
                "subject_id": subject_raw,
                "activity_id": activity_name_to_id[activity_name],
            }
        )

    if not sessions:
        raise ValueError(
            "No sessions were parsed from RealLifeHAR. "
            "Please verify extracted files under 'data_cleaned_adapted_full'."
        )

    session_metadata = pd.DataFrame(session_rows)
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )
    session_metadata["subject_id"] = pd.factorize(
        session_metadata["subject_id"], sort=True
    )[0].astype("int32")
    session_metadata = session_metadata[["session_id", "subject_id", "activity_id"]]

    activity_metadata = pd.DataFrame(
        {
            "activity_id": list(activity_name_to_id.values()),
            "activity_name": list(activity_name_to_id.keys()),
        }
    ).astype({"activity_id": "int32", "activity_name": "string"})

    return activity_metadata, session_metadata, sessions


cfg_real_life_har = WHARConfig(
    # Info + common
    dataset_id="real_life_har",
    download_url="https://lbd.udc.es/research/real-life-HAR-dataset/data_cleaned_adapted_full.zip",
    sampling_freq=20,
    num_of_subjects=17,
    num_of_activities=4,
    num_of_channels=len(REAL_LIFE_HAR_SENSOR_CHANNELS),
    datasets_dir="./datasets",
    parallelize=True,
    # Parsing
    parse=parse_real_life_har,
    activity_id_col="activity_id",
    # Preprocessing
    activity_names=REAL_LIFE_HAR_ACTIVITY_NAMES,
    sensor_channels=REAL_LIFE_HAR_SENSOR_CHANNELS,
    window_time=2.56,
    window_overlap=0.5,
)

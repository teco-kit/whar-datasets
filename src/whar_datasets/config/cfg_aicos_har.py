"""Configuration and parser for the AICOS-HAR dataset."""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

logger = logging.getLogger(__name__)

AICOS_HAR_RAW_ACTIVITY_TO_NAME: dict[str, str] = {
    "Downstairs": "Downstairs",
    "ElevatorDown": "Elevator Down",
    "ElevatorUp": "Elevator Up",
    "LyingDown": "Lying Down",
    "RampDown": "Ramp Down",
    "RampUp": "Ramp Up",
    "Running": "Running",
    "Sitting": "Sitting",
    "Standing": "Standing",
    "Upstairs": "Upstairs",
    "Walking": "Walking",
}

AICOS_HAR_ACTIVITY_NAMES: list[str] = list(AICOS_HAR_RAW_ACTIVITY_TO_NAME.values())

AICOS_HAR_CHANNELS: list[str] = [
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
    "magnetometer_x",
    "magnetometer_y",
    "magnetometer_z",
    "barometer",
]
# README units: acceleration m/s^2, angular velocity rad/s, magnetic field
# microtesla, and barometric pressure mbar.

AICOS_HAR_SENSOR_FILES: dict[str, tuple[str, ...]] = {
    "Accelerometer.txt": (
        "accelerometer_x",
        "accelerometer_y",
        "accelerometer_z",
    ),
    "Gyroscope.txt": ("gyroscope_x", "gyroscope_y", "gyroscope_z"),
    "Magnetometer.txt": ("magnetometer_x", "magnetometer_y", "magnetometer_z"),
    "Barometer.txt": ("barometer",),
}

# The README specifies nanoseconds. Accelerometer is the reference timeline.
# Inertial streams share that timebase; the slower barometer is matched within
# 150 ms and unmatched reference samples are omitted. Joins are always limited
# to the same subject/activity/trial/device folder.
AICOS_HAR_INERTIAL_TOLERANCE_NS: int = 25_000_000
AICOS_HAR_BAROMETER_TOLERANCE_NS: int = 150_000_000
AICOS_HAR_MAX_SESSION_GAP_SECONDS: float = 2.0


def _find_aicos_har_root(data_dir: Path) -> Path:
    candidates = (data_dir / "AICOS-HAR", data_dir)
    for candidate in candidates:
        if (candidate / "metadata.csv").is_file() and any(
            child.is_dir() and re.fullmatch(r"S\d+", child.name)
            for child in candidate.iterdir()
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not locate extracted AICOS-HAR participant folders under '{data_dir}'. "
        "Expected AICOS-HAR/metadata.csv and participant folders such as S1."
    )


def _natural_activity_order(activity_dir: Path) -> tuple[int, int, str]:
    match = re.fullmatch(r"(.+)_([0-9]+)", activity_dir.name)
    if match is None:
        raise ValueError(
            f"Unrecognized AICOS-HAR activity/trial directory '{activity_dir.name}'; "
            "expected <ActivityName>_<TrialNumber>."
        )
    return (0, int(match.group(2)), match.group(1))


def _normalize_metadata_acquisition_id(value: str) -> str:
    """Normalize two documented stale names in the archive metadata.csv."""
    parts = value.split("/")
    if len(parts) != 3:
        raise ValueError(f"Malformed AcquisitionID in AICOS-HAR metadata: {value!r}.")
    subject, activity_trial, device_position = parts
    activity_match = re.fullmatch(r"(.+)_([0-9]+)", activity_trial)
    if activity_match is None:
        raise ValueError(f"Malformed activity trial in AcquisitionID: {value!r}.")
    activity, trial = activity_match.groups()
    activity = {"LiftDown": "ElevatorDown", "LiftUp": "ElevatorUp"}.get(
        activity, activity
    )
    if device_position.startswith("AppleiPhone_"):
        device_position = (
            "AppleiPhoneSE2_" + device_position.removeprefix("AppleiPhone_")
        )
    return f"{subject}/{activity}_{trial}/{device_position}"


def _read_sensor_file(
    path: Path, channel_names: tuple[str, ...]
) -> tuple[pd.DataFrame, int]:
    column_names = ["timestamp_ns", *channel_names]
    dtype = {0: "int64", **{idx: "float64" for idx in range(1, len(column_names))}}
    try:
        frame = pd.read_csv(path, header=None, names=column_names, dtype=dtype)
    except (pd.errors.ParserError, ValueError, OSError) as exc:
        raise ValueError(f"Could not read AICOS-HAR sensor file '{path}': {exc}") from exc

    if frame.empty:
        raise ValueError(f"AICOS-HAR sensor file is empty: '{path}'.")
    if frame.isna().any().any():
        raise ValueError(f"AICOS-HAR sensor file contains missing values: '{path}'.")
    values = frame[list(channel_names)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"AICOS-HAR sensor file contains non-finite values: '{path}'.")

    timestamps = frame["timestamp_ns"].to_numpy(dtype=np.int64)
    steps = np.diff(timestamps)
    if (steps < 0).any():
        raise ValueError(
            f"AICOS-HAR timestamps reset or run backwards in source row order in '{path}'. "
            "The reset must be reconciled across sensor files before parsing this acquisition."
        )
    duplicate_count = int((steps == 0).sum())
    if duplicate_count:
        # Keep the first sample at each duplicate timestamp, preserving source order.
        frame = frame.loc[~frame["timestamp_ns"].duplicated(keep="first")].copy()
    return frame, duplicate_count


def _align_sensor_to_accelerometer(
    reference: pd.DataFrame,
    sensor: pd.DataFrame,
    sensor_name: str,
    tolerance_ns: int,
) -> tuple[pd.DataFrame, int, np.ndarray]:
    sensor_time_col = f"{sensor_name.removesuffix('.txt').lower()}_timestamp_ns"
    right = sensor.rename(columns={"timestamp_ns": sensor_time_col})
    aligned = pd.merge_asof(
        reference,
        right,
        left_on="timestamp_ns",
        right_on=sensor_time_col,
        direction="nearest",
        tolerance=tolerance_ns,
        allow_exact_matches=True,
    )
    offsets = (
        aligned[sensor_time_col].to_numpy(dtype=np.float64)
        - aligned["timestamp_ns"].to_numpy(dtype=np.float64)
    )
    matched = np.isfinite(offsets)
    return aligned, int((~matched).sum()), offsets[matched]


def _split_at_large_gaps(
    frame: pd.DataFrame, max_gap_ns: int | None
) -> list[pd.DataFrame]:
    if max_gap_ns is None or len(frame) < 2:
        return [frame]
    steps = np.diff(frame["timestamp_ns"].to_numpy(dtype=np.int64))
    split_points = np.flatnonzero(steps > max_gap_ns) + 1
    boundaries = [0, *split_points.tolist(), len(frame)]
    return [frame.iloc[start:end].copy() for start, end in zip(boundaries, boundaries[1:])]


def parse_aicos_har(
    dir: str, activity_id_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    """Parse complete four-sensor AICOS-HAR device acquisitions.

    The source defines sessions by participant/activity trial/device-position.
    Accelerometer timestamps are the reference samples. Gyroscope, magnetometer,
    and barometer values are joined with bounded nearest-time matching within
    that one acquisition only. Acquisitions lacking any one of the four sensor
    streams are excluded because the common session format requires one fixed
    channel matrix and fabricating an absent sensor would be misleading.
    """
    del activity_id_col  # AICOS-HAR has one directory-derived fine-grained label scheme.
    data_root = _find_aicos_har_root(Path(dir))

    participant_dirs: list[tuple[int, Path]] = []
    for path in data_root.iterdir():
        if path.is_dir() and not path.name.startswith("._"):
            match = re.fullmatch(r"S([0-9]+)", path.name)
            if match is None:
                raise ValueError(f"Unexpected AICOS-HAR top-level directory '{path.name}'.")
            participant_dirs.append((int(match.group(1)), path))
    participant_dirs.sort(key=lambda item: item[0])
    observed_raw_subjects = {subject for subject, _ in participant_dirs}
    expected_raw_subjects = set(range(1, 107))
    if observed_raw_subjects != expected_raw_subjects:
        raise ValueError(
            "AICOS-HAR participant folders differ from the documented S1-S106 range: "
            f"missing={sorted(expected_raw_subjects - observed_raw_subjects)}, "
            f"unexpected={sorted(observed_raw_subjects - expected_raw_subjects)}."
        )

    acquisition_dirs: list[tuple[int, str, int, str, Path]] = []
    seen_raw_activity_names: set[str] = set()
    for raw_subject, subject_dir in participant_dirs:
        activity_dirs = sorted(
            (
                path
                for path in subject_dir.iterdir()
                if path.is_dir() and not path.name.startswith("._")
            ),
            key=lambda path: _natural_activity_order(path),
        )
        for activity_dir in activity_dirs:
            trial_match = re.fullmatch(r"(.+)_([0-9]+)", activity_dir.name)
            if trial_match is None:
                raise ValueError(
                    f"Unrecognized AICOS-HAR directory '{activity_dir}'; expected "
                    "<ActivityName>_<TrialNumber>."
                )
            raw_activity, trial_text = trial_match.groups()
            if raw_activity not in AICOS_HAR_RAW_ACTIVITY_TO_NAME:
                raise ValueError(
                    f"Unknown AICOS-HAR activity directory '{raw_activity}' in '{activity_dir}'."
                )
            seen_raw_activity_names.add(raw_activity)
            trial = int(trial_text)
            device_dirs = sorted(
                (
                    path
                    for path in activity_dir.iterdir()
                    if path.is_dir() and not path.name.startswith("._")
                ),
                key=lambda path: path.name,
            )
            for device_dir in device_dirs:
                acquisition_dirs.append(
                    (raw_subject, raw_activity, trial, device_dir.name, device_dir)
                )

    if seen_raw_activity_names != set(AICOS_HAR_RAW_ACTIVITY_TO_NAME):
        raise ValueError(
            "AICOS-HAR activity directories differ from the configured label list: "
            f"missing={sorted(set(AICOS_HAR_RAW_ACTIVITY_TO_NAME) - seen_raw_activity_names)}, "
            f"unexpected={sorted(seen_raw_activity_names - set(AICOS_HAR_RAW_ACTIVITY_TO_NAME))}."
        )

    # Validate folder-derived provenance against the accompanying acquisition
    # table. Older metadata spelling aliases LiftDown/LiftUp and AppleiPhoneSE2.
    metadata_path = data_root / "metadata.csv"
    metadata = pd.read_csv(metadata_path, dtype={"AcquisitionID": "string"})
    if "AcquisitionID" not in metadata.columns:
        raise ValueError(f"'{metadata_path}' lacks the required AcquisitionID column.")
    metadata_ids = {
        _normalize_metadata_acquisition_id(str(value))
        for value in metadata["AcquisitionID"].dropna()
    }
    metadata_by_id = {
        _normalize_metadata_acquisition_id(str(row["AcquisitionID"])): row
        for row in metadata.to_dict(orient="records")
        if pd.notna(row["AcquisitionID"])
    }
    raw_ids = {
        f"S{subject}/{activity}_{trial}/{device}"
        for subject, activity, trial, device, _ in acquisition_dirs
    }
    if len(metadata) != 5231 or len(raw_ids) != 5231:
        raise ValueError(
            "AICOS-HAR source inventory differs from the published 5,231 acquisitions: "
            f"metadata rows={len(metadata)}, raw folders={len(raw_ids)}."
        )
    if metadata_ids != raw_ids:
        raise ValueError(
            "AICOS-HAR metadata.csv AcquisitionID entries do not match source folders "
            "after the documented Lift/Elevator and AppleiPhoneSE2 spelling fixes: "
            f"metadata-only={sorted(metadata_ids - raw_ids)[:5]}, "
            f"folders-only={sorted(raw_ids - metadata_ids)[:5]}."
        )

    complete_acquisitions = [
        acquisition
        for acquisition in acquisition_dirs
        if all(
            (acquisition[4] / sensor_file).is_file()
            for sensor_file in AICOS_HAR_SENSOR_FILES
        )
    ]
    incomplete_count = len(acquisition_dirs) - len(complete_acquisitions)
    retained_raw_subjects = sorted({item[0] for item in complete_acquisitions})
    subject_id_map = {
        raw_subject: idx for idx, raw_subject in enumerate(retained_raw_subjects)
    }
    activity_id_map = {
        name: idx
        for idx, name in enumerate(
            canonicalize_activity_name_list(AICOS_HAR_ACTIVITY_NAMES)
        )
    }

    session_rows: list[dict[str, int | str]] = []
    sessions: dict[int, pd.DataFrame] = {}
    session_id = 0
    unmatched_by_sensor: dict[str, int] = {
        "Gyroscope.txt": 0,
        "Magnetometer.txt": 0,
        "Barometer.txt": 0,
    }
    reference_samples_by_sensor: dict[str, int] = {
        "Gyroscope.txt": 0,
        "Magnetometer.txt": 0,
        "Barometer.txt": 0,
    }
    unmatched_total = 0
    duplicate_count = 0
    max_gap_ns = int(round(AICOS_HAR_MAX_SESSION_GAP_SECONDS * 1e9))
    empty_alignment_acquisitions = 0
    aligned_acquisitions = 0
    alignment_offset_stats: dict[str, list[tuple[float, float, float]]] = {
        "Gyroscope.txt": [],
        "Magnetometer.txt": [],
        "Barometer.txt": [],
    }

    for (
        raw_subject,
        raw_activity,
        trial,
        device,
        acquisition_dir,
    ) in complete_acquisitions:
        source_acquisition_id = f"S{raw_subject}/{raw_activity}_{trial}/{device}"
        accel, duplicates = _read_sensor_file(
            acquisition_dir / "Accelerometer.txt",
            AICOS_HAR_SENSOR_FILES["Accelerometer.txt"],
        )
        duplicate_count += duplicates
        if not accel["timestamp_ns"].is_monotonic_increasing:
            raise ValueError(
                f"Accelerometer timestamps are not monotonic in '{acquisition_dir}'."
            )

        aligned = accel
        drop_masks: list[pd.Series] = []
        for sensor_file, tolerance in (
            ("Gyroscope.txt", AICOS_HAR_INERTIAL_TOLERANCE_NS),
            ("Magnetometer.txt", AICOS_HAR_INERTIAL_TOLERANCE_NS),
            ("Barometer.txt", AICOS_HAR_BAROMETER_TOLERANCE_NS),
        ):
            sensor, duplicates = _read_sensor_file(
                acquisition_dir / sensor_file, AICOS_HAR_SENSOR_FILES[sensor_file]
            )
            duplicate_count += duplicates
            if not sensor["timestamp_ns"].is_monotonic_increasing:
                raise ValueError(
                    f"{sensor_file} timestamps are not monotonic in '{acquisition_dir}'."
                )
            aligned, unmatched_count, offsets = _align_sensor_to_accelerometer(
                aligned, sensor, sensor_file, tolerance
            )
            reference_samples_by_sensor[sensor_file] += len(aligned)
            unmatched_by_sensor[sensor_file] += unmatched_count
            unmatched_total += unmatched_count
            if len(offsets):
                absolute_offsets_ms = np.abs(offsets) / 1e6
                alignment_offset_stats[sensor_file].append(
                    tuple(
                        float(value)
                        for value in np.quantile(absolute_offsets_ms, [0.5, 0.9, 0.99])
                    )
                )
            matched_column = f"{sensor_file.removesuffix('.txt').lower()}_timestamp_ns"
            drop_masks.append(aligned[matched_column].notna())

        keep = np.logical_and.reduce([mask.to_numpy() for mask in drop_masks])
        aligned = aligned.loc[keep].copy()
        if aligned.empty:
            empty_alignment_acquisitions += 1
            logger.warning(
                "No in-tolerance multimodal samples for '%s'; skipping it.",
                acquisition_dir,
            )
            continue
        aligned_acquisitions += 1

        aligned["timestamp"] = pd.to_datetime(
            aligned["timestamp_ns"], unit="ns", origin="unix", errors="raise"
        )
        channel_order = AICOS_HAR_CHANNELS
        aligned[channel_order] = aligned[channel_order].astype("float32")
        aligned = aligned[["timestamp", *channel_order]]

        aligned["timestamp_ns"] = accel.loc[keep, "timestamp_ns"].to_numpy()
        for segment in _split_at_large_gaps(aligned, max_gap_ns):
            if segment.empty:
                continue
            session_rows.append(
                {
                    "session_id": session_id,
                    "subject_id": subject_id_map[raw_subject],
                    "activity_id": activity_id_map[
                        AICOS_HAR_RAW_ACTIVITY_TO_NAME[raw_activity]
                    ],
                    "source_subject_id": raw_subject,
                    "source_acquisition_id": source_acquisition_id,
                    "device_model": device.rsplit("_", maxsplit=1)[0],
                    "body_position": device.rsplit("_", maxsplit=1)[-1],
                    "position_fullname": str(
                        metadata_by_id[source_acquisition_id]["Position_FullName"]
                    ),
                }
            )
            sessions[session_id] = segment[
                ["timestamp", *channel_order]
            ].reset_index(drop=True)
            session_id += 1

    activity_names = canonicalize_activity_name_list(AICOS_HAR_ACTIVITY_NAMES)
    activity_df = pd.DataFrame(
        {"activity_id": range(len(activity_names)), "activity_name": activity_names}
    ).astype({"activity_id": "int32", "activity_name": "string"})
    session_df = pd.DataFrame(
        session_rows,
        columns=[
            "session_id",
            "subject_id",
            "activity_id",
            "source_subject_id",
            "source_acquisition_id",
            "device_model",
            "body_position",
            "position_fullname",
        ],
    ).astype(
        {
            "session_id": "int32",
            "subject_id": "int32",
            "activity_id": "int32",
            "source_subject_id": "int32",
            "source_acquisition_id": "string",
            "device_model": "string",
            "body_position": "string",
            "position_fullname": "string",
        }
    )

    observed_activities = set(session_df["activity_id"].unique().tolist())
    expected_activities = set(range(len(activity_names)))
    if observed_activities != expected_activities:
        missing_names = [
            activity_names[index]
            for index in sorted(expected_activities - observed_activities)
        ]
        raise ValueError(
            "After excluding incomplete acquisitions, AICOS-HAR activities have no "
            f"usable sessions: {missing_names}."
        )

    logger.warning(
        "AICOS-HAR found %d complete four-sensor acquisitions and excluded %d/%d "
        "acquisitions missing at least one sensor stream; %d of %d subjects remain. "
        "At configured join tolerances, %d reference samples were unmatched "
        "(gyro=%d, magnetometer=%d, barometer=%d). Duplicate timestamp "
        "rows removed: %d.",
        len(complete_acquisitions),
        incomplete_count,
        len(acquisition_dirs),
        len(retained_raw_subjects),
        len(participant_dirs),
        unmatched_total,
        unmatched_by_sensor["Gyroscope.txt"],
        unmatched_by_sensor["Magnetometer.txt"],
        unmatched_by_sensor["Barometer.txt"],
        duplicate_count,
    )
    logger.warning(
        "AICOS-HAR retained %d of %d complete acquisitions after alignment; "
        "%d had no reference samples matching all sensor streams.",
        aligned_acquisitions,
        len(complete_acquisitions),
        empty_alignment_acquisitions,
    )
    for sensor_file in unmatched_by_sensor:
        denominator = reference_samples_by_sensor[sensor_file]
        match_rate = 100.0 * (1.0 - unmatched_by_sensor[sensor_file] / denominator) if denominator else 0.0
        acquisition_offsets = np.asarray(alignment_offset_stats[sensor_file], dtype=np.float64)
        offset_quantiles = (
            np.quantile(acquisition_offsets[:, 0], [0.5, 0.9, 0.99])
            if len(acquisition_offsets)
            else np.zeros(3, dtype=np.float64)
        )
        logger.info(
            "AICOS-HAR %s match rate=%.5f%% (%d unmatched/%d samples) within %d ms; "
            "acquisition-median absolute offset q50/q90/q99=%.3f/%.3f/%.3f ms.",
            sensor_file,
            match_rate,
            unmatched_by_sensor[sensor_file],
            denominator,
            (AICOS_HAR_BAROMETER_TOLERANCE_NS if sensor_file == "Barometer.txt" else AICOS_HAR_INERTIAL_TOLERANCE_NS) // 1_000_000,
            *offset_quantiles,
        )

    return activity_df, session_df, sessions


cfg_aicos_har = WHARConfig(
    dataset_id="aicos_har",
    dataset_url="https://zenodo.org/records/19452049",
    download_url="https://zenodo.org/records/19452049/files/AICOS-HAR.zip?download=1",
    # Source rates vary by device; the library infers each session's rate and
    # resamples to 50 Hz for fixed-size windows.
    sampling_freq=None,
    source_rate_mode="per_session",
    resampling_freq=50.0,
    # The fixed 10-channel parser requires all four modalities. Twelve source
    # participants have no acquisition containing the full set of sensor files.
    num_of_subjects=94,
    num_of_activities=len(AICOS_HAR_ACTIVITY_NAMES),
    num_of_channels=len(AICOS_HAR_CHANNELS),
    parse=parse_aicos_har,
    activity_id_col="activity_id",
    available_activities=canonicalize_activity_name_list(AICOS_HAR_ACTIVITY_NAMES),
    selected_activities=canonicalize_activity_name_list(AICOS_HAR_ACTIVITY_NAMES),
    available_channels=AICOS_HAR_CHANNELS,
    selected_channels=AICOS_HAR_CHANNELS,
    max_session_gap_seconds=AICOS_HAR_MAX_SESSION_GAP_SECONDS,
)

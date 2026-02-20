import os
import re
from typing import Dict, List, Tuple, Any

import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.config.timestamps import to_datetime64_ms

CAPTURE24_SAMPLING_FREQ = 100
CAPTURE24_GAP_THRESHOLD_MS = 2000.0

CAPTURE24_ACTIVITY_NAMES = [
    "sleep",
    "sit_stand",
    "vehicle",
    "walking",
    "bicycling",
    "mixed",
]

CAPTURE24_SENSOR_CHANNELS = ["acc_wrist_x", "acc_wrist_y", "acc_wrist_z"]

CAPTURE24_ACTIVITY_TO_ID = {
    name: idx for idx, name in enumerate(CAPTURE24_ACTIVITY_NAMES)
}


def _normalize_token(token: Any) -> str:
    if token is None or (isinstance(token, float) and pd.isna(token)):
        normalized = ""
    else:
        normalized = str(token).strip().lower()
    normalized = normalized.replace("/", "_")
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _canonicalize_activity(label: str) -> str:
    token = _normalize_token(label)

    if token == "" or token in {"none", "unknown", "unclassified"}:
        return "mixed"
    if "sleep" in token or "bed" in token:
        return "sleep"
    if "sit" in token or "stand" in token or "sedentary" in token:
        return "sit_stand"
    if (
        "vehicle" in token
        or "car" in token
        or "bus" in token
        or "train" in token
        or "transport" in token
        or "driv" in token
    ):
        return "vehicle"
    if "bicycl" in token or "cycl" in token or "bike" in token:
        return "bicycling"
    if "walk" in token:
        return "walking"

    return "mixed"


def _find_label_dictionary_file(dir: str) -> str | None:
    candidates: List[str] = []
    for root, _dirs, files in os.walk(dir):
        for file_name in files:
            lowered = file_name.lower()
            if not lowered.endswith(".csv"):
                continue
            if "annotation" in lowered and "dictionary" in lowered:
                candidates.append(os.path.join(root, file_name))
            elif "label" in lowered and "dictionary" in lowered:
                candidates.append(os.path.join(root, file_name))

    if not candidates:
        return None

    return sorted(candidates)[0]


def _find_subject_files(dir: str, dictionary_file: str | None) -> List[str]:
    pattern = re.compile(r"^p\d{3}\.csv(\.gz)?$", re.IGNORECASE)
    preferred_by_subject: Dict[str, str] = {}

    for root, _dirs, files in os.walk(dir):
        for file_name in files:
            if pattern.match(file_name) is None:
                continue
            full_path = os.path.join(root, file_name)
            if dictionary_file is not None and full_path == dictionary_file:
                continue
            subject_token = _extract_subject_token(file_name)
            current = preferred_by_subject.get(subject_token)
            if current is None:
                preferred_by_subject[subject_token] = full_path
                continue
            # Prefer uncompressed CSV over CSV.GZ when both are present.
            if current.lower().endswith(".gz") and not full_path.lower().endswith(".gz"):
                preferred_by_subject[subject_token] = full_path

    return sorted(preferred_by_subject.values())


def _resolve_dictionary_columns(
    dictionary_df: pd.DataFrame, activity_id_col: str
) -> Tuple[str, str]:
    col_norm: Dict[str, str] = {col: _normalize_token(col) for col in dictionary_df}
    normalized_to_original = {norm: col for col, norm in col_norm.items()}

    annotation_candidates = [
        "annotation",
        "annotation_name",
        "label",
        "activity",
    ]
    annotation_col = ""
    for candidate in annotation_candidates:
        if candidate in normalized_to_original:
            annotation_col = normalized_to_original[candidate]
            break

    if annotation_col == "":
        raise ValueError(
            "Could not identify annotation column in CAPTURE-24 label dictionary."
        )

    requested = _normalize_token(activity_id_col)
    if requested in normalized_to_original:
        return annotation_col, normalized_to_original[requested]

    for col, normalized in col_norm.items():
        if "walmsley2020" in normalized:
            return annotation_col, col

    for col, normalized in col_norm.items():
        if col == annotation_col:
            continue
        if "label" in normalized:
            return annotation_col, col

    fallback_cols = [col for col in dictionary_df.columns if col != annotation_col]
    if not fallback_cols:
        raise ValueError(
            "Could not identify scheme/label column in CAPTURE-24 label dictionary."
        )

    return annotation_col, fallback_cols[0]


def _build_annotation_map(
    dictionary_df: pd.DataFrame,
    annotation_col: str,
    scheme_col: str,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for _idx, row in dictionary_df[[annotation_col, scheme_col]].dropna().iterrows():
        raw_annotation = str(row[annotation_col])
        raw_target = str(row[scheme_col])
        mapping[_normalize_token(raw_annotation)] = _canonicalize_activity(raw_target)
    return mapping


def _resolve_timestamp_col(df: pd.DataFrame) -> str:
    col_norm: Dict[str, str] = {col: _normalize_token(col) for col in df.columns}
    normalized_to_original = {norm: col for col, norm in col_norm.items()}

    for token in ("time", "timestamp", "datetime", "date_time", "epoch", "unix_time"):
        if token in normalized_to_original:
            return normalized_to_original[token]

    for col, normalized in col_norm.items():
        if "time" in normalized:
            return col

    raise ValueError("Could not identify timestamp column in CAPTURE-24 subject CSV.")


def _resolve_annotation_col(df: pd.DataFrame) -> str:
    col_norm: Dict[str, str] = {col: _normalize_token(col) for col in df.columns}
    normalized_to_original = {norm: col for col, norm in col_norm.items()}

    for token in ("annotation", "label", "activity", "activity_name"):
        if token in normalized_to_original:
            return normalized_to_original[token]

    for col, normalized in col_norm.items():
        if "annotation" in normalized or "label" in normalized:
            return col

    raise ValueError("Could not identify annotation column in CAPTURE-24 subject CSV.")


def _resolve_accel_cols(df: pd.DataFrame) -> List[str]:
    col_norm: Dict[str, str] = {col: _normalize_token(col) for col in df.columns}
    normalized_to_original = {norm: col for col, norm in col_norm.items()}

    explicit_triplets = [
        ("x", "y", "z"),
        ("acc_x", "acc_y", "acc_z"),
        ("accelerometer_x", "accelerometer_y", "accelerometer_z"),
    ]

    for x_col, y_col, z_col in explicit_triplets:
        if (
            x_col in normalized_to_original
            and y_col in normalized_to_original
            and z_col in normalized_to_original
        ):
            return [
                normalized_to_original[x_col],
                normalized_to_original[y_col],
                normalized_to_original[z_col],
            ]

    numeric_candidates: List[str] = []
    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.8:
            numeric_candidates.append(col)

    if len(numeric_candidates) < 3:
        raise ValueError(
            "Could not identify three accelerometer channels in CAPTURE-24 subject CSV."
        )

    return numeric_candidates[:3]


def _to_timestamp(values: pd.Series, fallback_length: int) -> pd.Series:
    as_datetime = pd.to_datetime(values, errors="coerce")
    if as_datetime.notna().mean() >= 0.8:
        return as_datetime.astype("datetime64[ms]")

    as_numeric = pd.to_numeric(values, errors="coerce")
    converted = to_datetime64_ms(as_numeric, default_unit="s")
    if converted.notna().mean() >= 0.8:
        return converted

    return to_datetime64_ms(
        pd.Series(range(fallback_length), dtype="float64")
        * (1.0 / CAPTURE24_SAMPLING_FREQ),
        default_unit="s",
    )


def _extract_subject_token(path: str) -> str:
    file_name = os.path.basename(path)
    match = re.search(r"(p\d{3})", file_name, flags=re.IGNORECASE)
    if match is None:
        return os.path.splitext(file_name)[0]
    return match.group(1).upper()


def parse_capture24(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    dictionary_file = _find_label_dictionary_file(dir)
    annotation_map: Dict[str, str] = {}
    if dictionary_file is not None:
        dictionary_df = pd.read_csv(dictionary_file)
        annotation_col, scheme_col = _resolve_dictionary_columns(
            dictionary_df, activity_id_col
        )
        annotation_map = _build_annotation_map(dictionary_df, annotation_col, scheme_col)

    subject_files = _find_subject_files(dir, dictionary_file)

    if not subject_files:
        raise FileNotFoundError(
            f"Could not locate CAPTURE-24 subject files under '{dir}'."
        )

    subject_tokens = [_extract_subject_token(path) for path in subject_files]
    subject_id_map = {
        token: idx for idx, token in enumerate(sorted(set(subject_tokens)))
    }

    session_rows: List[Dict[str, int]] = []
    sessions: Dict[int, pd.DataFrame] = {}
    session_id = 0

    for file_path in sorted(subject_files):
        raw_df = pd.read_csv(file_path, compression="infer")
        if raw_df.empty:
            continue

        timestamp_col = _resolve_timestamp_col(raw_df)
        annotation_col = _resolve_annotation_col(raw_df)
        accel_cols = _resolve_accel_cols(raw_df)

        df = raw_df[[timestamp_col, annotation_col] + accel_cols].copy()
        df = df.rename(
            columns={
                timestamp_col: "timestamp",
                annotation_col: "annotation",
                accel_cols[0]: CAPTURE24_SENSOR_CHANNELS[0],
                accel_cols[1]: CAPTURE24_SENSOR_CHANNELS[1],
                accel_cols[2]: CAPTURE24_SENSOR_CHANNELS[2],
            }
        )

        df["timestamp"] = _to_timestamp(df["timestamp"], fallback_length=len(df))
        for col in CAPTURE24_SENSOR_CHANNELS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["timestamp"])
        if df.empty:
            continue

        df[CAPTURE24_SENSOR_CHANNELS] = (
            df[CAPTURE24_SENSOR_CHANNELS]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
        df = df.dropna(subset=CAPTURE24_SENSOR_CHANNELS)
        if df.empty:
            continue

        normalized_annotation = df["annotation"].astype(str).map(_normalize_token)
        if annotation_map:
            df["activity_name"] = normalized_annotation.map(annotation_map).fillna("mixed")
        else:
            df["activity_name"] = df["annotation"].astype(str).map(_canonicalize_activity)

        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
        if df.empty:
            continue

        time_diff_ms = (
            df["timestamp"].diff().dt.total_seconds().fillna(0.0) * 1000.0
        )
        is_new_session = (
            (df["activity_name"] != df["activity_name"].shift(1))
            | (time_diff_ms > CAPTURE24_GAP_THRESHOLD_MS)
            | (time_diff_ms <= 0.0)
        )
        local_session_ids = is_new_session.cumsum()

        subject_token = _extract_subject_token(file_path)
        subject_id = subject_id_map[subject_token]

        for _local_id, group in df.groupby(local_session_ids, sort=False):
            if group.empty:
                continue

            activity_name = str(group["activity_name"].iloc[0])
            activity_id = CAPTURE24_ACTIVITY_TO_ID.get(activity_name, CAPTURE24_ACTIVITY_TO_ID["mixed"])

            session_df = group[["timestamp"] + CAPTURE24_SENSOR_CHANNELS].copy()
            session_df[CAPTURE24_SENSOR_CHANNELS] = session_df[
                CAPTURE24_SENSOR_CHANNELS
            ].astype("float32").round(6)
            session_df["timestamp"] = session_df["timestamp"].astype("datetime64[ms]")
            session_df = session_df.astype(
                {
                    "timestamp": "datetime64[ms]",
                    CAPTURE24_SENSOR_CHANNELS[0]: "float32",
                    CAPTURE24_SENSOR_CHANNELS[1]: "float32",
                    CAPTURE24_SENSOR_CHANNELS[2]: "float32",
                }
            )

            sessions[session_id] = session_df
            session_rows.append(
                {
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "activity_id": activity_id,
                }
            )
            session_id += 1

    if not sessions:
        raise ValueError("No valid CAPTURE-24 sessions could be parsed.")

    activity_metadata = pd.DataFrame(
        {
            "activity_id": list(range(len(CAPTURE24_ACTIVITY_NAMES))),
            "activity_name": CAPTURE24_ACTIVITY_NAMES,
        }
    ).astype({"activity_id": "int32", "activity_name": "string"})

    session_metadata = pd.DataFrame(session_rows).astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_capture_24 = WHARConfig(
    dataset_id="capture_24",
    download_url="https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b",
    sampling_freq=CAPTURE24_SAMPLING_FREQ,
    num_of_subjects=151,
    num_of_activities=len(CAPTURE24_ACTIVITY_NAMES),
    num_of_channels=len(CAPTURE24_SENSOR_CHANNELS),
    parse=parse_capture24,
    activity_id_col="label:Walmsley2020",
    activity_names=CAPTURE24_ACTIVITY_NAMES,
    sensor_channels=CAPTURE24_SENSOR_CHANNELS,
    window_time=2.56,
    window_overlap=0.5,
)

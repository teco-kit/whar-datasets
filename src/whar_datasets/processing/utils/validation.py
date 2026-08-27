import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.utils.loading import load_session, load_sessions
from whar_datasets.utils.logging import logger


def validate_common_format(
    cfg: WHARConfig,
    sessions_dir: Path,
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
) -> bool:
    """Validate LOSO metadata and all time-series session payloads."""
    logger.info("Validating common format")
    required_session = {"session_id", "subject_id", "activity_id"}
    required_activity = {"activity_id", "activity_name"}
    if not required_session.issubset(session_df.columns):
        logger.error("Session metadata lacks required columns: %s", required_session - set(session_df))
        return False
    if not required_activity.issubset(activity_df.columns):
        logger.error("Activity metadata lacks required columns: %s", required_activity - set(activity_df))
        return False

    for column in required_session:
        if not pd.api.types.is_integer_dtype(session_df[column]):
            logger.error("'%s' column is not integer type.", column)
            return False
    if (session_df[["session_id", "subject_id", "activity_id"]].min() != 0).any():
        logger.error("Session, subject, and activity identifiers must start at zero.")
        return False
    if session_df["session_id"].duplicated().any():
        logger.error("Each session_id must have exactly one metadata row.")
        return False
    if session_df["subject_id"].nunique() != cfg.num_of_subjects:
        logger.error("Subject count does not match the dataset configuration.")
        return False
    if session_df["activity_id"].nunique() != cfg.num_of_activities:
        logger.error("Activity count does not match the dataset configuration.")
        return False
    if not pd.api.types.is_integer_dtype(activity_df["activity_id"]):
        logger.error("activity_id is not integer type.")
        return False
    if activity_df["activity_name"].isna().any():
        logger.error("One or more activity names are missing.")
        return False
    if activity_df["activity_id"].nunique() != cfg.num_of_activities:
        logger.error("Activity metadata count does not match the configuration.")
        return False

    use_processes = cfg.execution_backend == "process"
    valid = (
        validate_sessions_para(cfg, sessions_dir, session_df)
        if use_processes
        else validate_sessions_seq(cfg, sessions_dir, session_df)
    )
    if valid:
        logger.info("Common format validated.")
    return valid


def _validate_session_frame(
    cfg: WHARConfig, session_id: int, session: pd.DataFrame
) -> bool:
    if "timestamp" not in session:
        logger.error("Session %s has no timestamp column.", session_id)
        return False
    if not pd.api.types.is_datetime64_any_dtype(session["timestamp"]):
        logger.error("timestamp in session %s is not datetime64.", session_id)
        return False
    if not session["timestamp"].is_monotonic_increasing:
        logger.error("Timestamps in session %s are not monotonic.", session_id)
        return False
    if cfg.max_session_gap_seconds is not None:
        gaps = session["timestamp"].diff().dt.total_seconds()
        if (gaps > cfg.max_session_gap_seconds).any():
            logger.error(
                "Session %s contains a %.3fs gap; split it into separate sessions.",
                session_id,
                float(gaps.max()),
            )
            return False
    sensor_columns = session.columns.difference(["timestamp"])
    if len(sensor_columns) != cfg.num_of_channels:
        logger.error(
            "Session %s has %d channels; expected %d.",
            session_id,
            len(sensor_columns),
            cfg.num_of_channels,
        )
        return False
    if any(not pd.api.types.is_float_dtype(session[column]) for column in sensor_columns):
        logger.error("Session %s contains a non-floating sensor channel.", session_id)
        return False
    if session.isna().any().any():
        logger.error("Session %s contains NaN values.", session_id)
        return False
    return True


def validate_sessions_seq(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> bool:
    session_ids = [int(value) for value in session_df["session_id"]]
    sessions = load_sessions(sessions_dir, session_ids=session_ids)
    if set(sessions) != set(session_ids):
        logger.error("Session cache and metadata identifiers differ.")
        return False
    return all(
        _validate_session_frame(cfg, session_id, sessions[session_id])
        for session_id in tqdm(session_ids, desc="Validating sessions")
    )


def _validate_from_disk(args: tuple[WHARConfig, Path, int]) -> bool:
    cfg, sessions_dir, session_id = args
    return _validate_session_frame(cfg, session_id, load_session(sessions_dir, session_id))


def validate_sessions_para(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> bool:
    session_ids = [int(value) for value in session_df["session_id"]]
    requested = cfg.num_workers or (os.cpu_count() or 1)
    workers = max(1, min(requested, len(session_ids)))
    if workers == 1:
        return validate_sessions_seq(cfg, sessions_dir, session_df)
    tasks = [(cfg, sessions_dir, session_id) for session_id in session_ids]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results: List[bool] = list(
            tqdm(
                executor.map(_validate_from_disk, tasks),
                total=len(tasks),
                desc="Validating sessions",
            )
        )
    return len(results) == len(session_ids) and all(results)


def validate_session(cfg: WHARConfig, sessions_dir: Path, session_id: int) -> bool:
    return _validate_session_frame(cfg, session_id, load_session(sessions_dir, session_id))

from pathlib import Path
from typing import List

import numpy as np
import dask.dataframe as dd
import pandas as pd
from dask.base import compute
from dask.delayed import delayed
from dask.diagnostics.progress import ProgressBar
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.utils.loading import load_session
from whar_datasets.utils.logging import logger


def validate_common_format(
    cfg: WHARConfig,
    sessions_dir: Path,
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
) -> bool:
    logger.info("Validating common format")

    # Check session_df
    if not pd.api.types.is_integer_dtype(session_df["session_id"]):
        logger.error("'session_id' column is not integer type.")
        return False
    if not pd.api.types.is_integer_dtype(session_df["subject_id"]):
        logger.error("'subject_id' column is not integer type.")
        return False
    
    is_list_of_ints = (
        session_df["activity_id"]
        .apply(
            lambda x: isinstance(x, list)
            and all(isinstance(i, (int, np.integer)) for i in x)
        )
        .all()
    )
    if not is_list_of_ints:
        logger.error("'activity_id' column is not a list of integers.")
        return False

    if session_df["session_id"].min() != 0:
        logger.error("Minimum session_id is not 0.")
        return False
    if session_df["subject_id"].min() != 0:
        logger.error("Minimum subject_id is not 0.")
        return False

    # checks, if min. one session contrains activity 0
    if session_df["activity_id"].explode().min() != 0:
        logger.error("Minimum activity_id is not 0 (or 0 is missing).")
        return False

    if session_df["subject_id"].nunique() != cfg.num_of_subjects:
        logger.error(
            f"In session_df, num of subject_ids {session_df['subject_id'].nunique()} does not match num of subjects {cfg.num_of_subjects}."
        )
        return False

    if session_df["activity_id"].explode().nunique() != cfg.num_of_activities:
        logger.error(
            f"In session_df, number of activity_ids {session_df['activity_id'].explode().nunique()} does not match number of activities {cfg.num_of_activities} ."
        )
        return False

    # Check activity_df
    if not pd.api.types.is_integer_dtype(activity_df["activity_id"]):
        logger.error("'activity_id' column is not integer type.")
        return False
    if not pd.api.types.is_string_dtype(activity_df["activity_name"]):
        logger.error("'activity_name' column is not string type.")
        return False
    if activity_df["activity_id"].min() != 0:
        logger.error("Minimum activity_id is not 0.")
        return False
    if activity_df["activity_id"].nunique() != cfg.num_of_activities:
        logger.error("Number of activity_ids does not match number of activities.")
        return False

    validated = (
        validate_sessions_para(cfg, sessions_dir, session_df)
        if cfg.parallelize
        else validate_sessions_seq(cfg, sessions_dir, session_df)
    )

    if not validated:
        return False

    logger.info("Common format validated.")
    return True


def validate_sessions_seq(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> bool:
    loop = tqdm(session_df["session_id"])
    loop.set_description("Validating sessions")

    for session_id in loop:
        if not validate_session(cfg, sessions_dir, session_id):
            return False

    return True


def validate_sessions_para(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> bool:
    relevant_ids = set(session_df["session_id"])

    # Read sessions parquet with dask
    ddf = dd.read_parquet(sessions_dir / "sessions.parquet", engine="pyarrow")

    def validate_partition(df: pd.DataFrame) -> List[bool]:
        results: List[bool] = []
        if df.empty:
            return results

        if "session_id" not in df.columns:
            return results

        # Filter for relevant sessions
        mask = df["session_id"].isin(relevant_ids)
        if not mask.any():
            return results

        subset = df[mask]

        for session_id, group in subset.groupby("session_id"):
            # Drop session_id to match load_session behavior
            session = group.drop(columns=["session_id"]).reset_index(drop=True)

            # Validation logic
            is_valid = True

            if not pd.api.types.is_datetime64_dtype(session["timestamp"]):
                logger.error(session["timestamp"].dtype)
                logger.error(
                    f"'timestamp' column in {session_id} is not datetime64 type."
                )
                is_valid = False

            elif len(session.columns.difference(["timestamp"])) != cfg.num_of_channels:
                logger.error(session.columns.difference(["timestamp"]))
                logger.error(
                    f"Number of columns {len(session.columns.difference(['timestamp']))} in {session_id} does not match number of channel {cfg.num_of_channels}."
                )
                is_valid = False

            else:
                for col in session.columns.difference(["timestamp"]):
                    if not pd.api.types.is_float_dtype(session[col]):
                        logger.error(
                            f"Column '{col}' in {session_id} is not float type."
                        )
                        is_valid = False
                        break

                if is_valid and session.isna().any().any():
                    logger.error(f"Session file {session_id} contains NaN values.")
                    is_valid = False

            results.append(is_valid)

        return results

    logger.info("Validating sessions (parallelized)")

    # Create delayed tasks
    delayed_partitions = ddf.to_delayed()

    @delayed
    def validate_delayed(partition):
        return validate_partition(partition)

    tasks = [validate_delayed(part) for part in delayed_partitions]

    # execute tasks in parallel
    pbar = ProgressBar()
    pbar.register()
    results_list = list(compute(*tasks, scheduler="processes"))
    pbar.unregister()

    # Flatten results
    all_results = []
    for partition_results in results_list:
        all_results.extend(partition_results)

    return all(all_results)


def validate_session(cfg: WHARConfig, sessions_dir: Path, session_id: int) -> bool:
    session = load_session(sessions_dir, session_id)

    if not pd.api.types.is_datetime64_dtype(session["timestamp"]):
        logger.error(session["timestamp"].dtype)
        logger.error(f"'timestamp' column in {session_id} is not datetime64 type.")
        return False

    if len(session.columns.difference(["timestamp"])) != cfg.num_of_channels:
        logger.error(session.columns.difference(["timestamp"]))
        logger.error(
            f"Number of columns {len(session.columns.difference(['timestamp']))} in {session_id} does not match number of channel {cfg.num_of_channels}."
        )
        return False

    for col in session.columns.difference(["timestamp"]):
        if not pd.api.types.is_float_dtype(session[col]):
            logger.error(f"Column '{col}' in {session_id} is not float type.")
            return False

    if session.isna().any().any():
        logger.error(f"Session file {session_id} contains NaN values.")
        return False

    return True

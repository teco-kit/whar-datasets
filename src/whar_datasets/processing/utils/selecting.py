from typing import List

import pandas as pd

from whar_datasets.utils.logging import logger


def select_activities(
    session_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    activity_names: List[str],
) -> pd.DataFrame:
    logger.info("Selecting activities")

    target_ids = activity_df[activity_df["activity_name"].isin(activity_names)]["activity_id"].tolist()

    mask = session_df["activity_id"].explode().isin(target_ids).groupby(level=0).any()

    sessions_containing_ids = session_df[mask]

    return sessions_containing_ids


def select_channels(session_df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    # if channels is empty, return df
    return session_df[channels + ["timestamp"]] if len(channels) != 0 else session_df

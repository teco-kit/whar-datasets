from typing import List

import pandas as pd

from whar_datasets.utils.logging import logger


def select_activities(
    session_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    selected_activities: List[str],
) -> pd.DataFrame:
    logger.info("Selecting activities")

    # get activity ids corresponding to selected activity names
    activity_ids = activity_df[
        activity_df["activity_name"].isin(selected_activities)
    ]["activity_id"]

    # get session ids corresponding to activiy ids
    session_ids = session_df[session_df["activity_id"].isin(activity_ids)]

    return session_ids


def select_channels(session_df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    # if channels is empty, return df
    return session_df[channels + ["timestamp"]] if len(channels) != 0 else session_df

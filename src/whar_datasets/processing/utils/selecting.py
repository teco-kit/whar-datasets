from typing import Dict, List, Tuple

import pandas as pd

from whar_datasets.utils.logging import logger


def select_activities(
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
    selected_activities: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter activities and remap activity ids to contiguous integers."""
    logger.info("Selecting activities")

    selected_activity_df = activity_df.copy()
    if selected_activities:
        selected_activity_df = selected_activity_df[
            selected_activity_df["activity_name"].isin(selected_activities)
        ].copy()

    selected_activity_df = selected_activity_df.drop_duplicates(
        subset=["activity_id"], keep="first"
    )

    old_ids = (
        selected_activity_df["activity_id"]
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    if len(old_ids) == 0:
        raise ValueError(
            "No activities selected. Check cfg.selected_activities against activity metadata."
        )

    # Ensure labels are contiguous after activity filtering, e.g. [1..12] -> [0..11].
    id_map: Dict[int, int] = {old_id: new_id for new_id, old_id in enumerate(old_ids)}

    selected_activity_df["activity_id"] = (
        selected_activity_df["activity_id"]
        .astype(int)
        .map(id_map)
        .astype("int32")
    )
    selected_activity_df = selected_activity_df.sort_values("activity_id").reset_index(
        drop=True
    )

    selected_session_df = session_df[
        session_df["activity_id"].astype(int).isin(old_ids)
    ].copy()
    selected_session_df["activity_id"] = (
        selected_session_df["activity_id"]
        .astype(int)
        .map(id_map)
        .astype("int32")
    )
    selected_session_df = selected_session_df.reset_index(drop=True)

    return selected_activity_df, selected_session_df


def select_channels(session_df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    """Return selected sensor channels plus timestamp."""
    # if channels is empty, return df
    return session_df[channels + ["timestamp"]] if len(channels) != 0 else session_df

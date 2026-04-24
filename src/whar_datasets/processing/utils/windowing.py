from typing import Dict, Tuple

import pandas as pd
import shortuuid


def generate_windowing(
    session_id: int,
    session_df: pd.DataFrame,
    window_time: float,
    overlap: float,
    sampling_freq: float,
) -> Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]:
    """Generate fixed-length sliding windows for one session."""
    # create containers for windows and index
    window_df_dict: Dict[str, list] = {
        "session_id": [],
        "window_id": [],
    }
    windows: Dict[str, pd.DataFrame] = {}

    # specify window and stride as timedeltas
    window_timedelta = pd.Timedelta(seconds=window_time)
    stride_timedelta = pd.Timedelta(seconds=window_time * (1 - overlap))

    # specifiy times in session
    start_time = session_df["timestamp"].min()
    end_time = session_df["timestamp"].max()
    current_start_time = start_time

    # generate windows from session
    while current_start_time + window_timedelta <= end_time:
        current_end_time = current_start_time + window_timedelta

        # get mask corresponding to window
        mask = (session_df["timestamp"] >= current_start_time) & (
            session_df["timestamp"] < current_end_time
        )

        # get window based on mask and keep_cols and reset index
        window_df = session_df[mask]
        window_df = window_df.drop(columns=["timestamp"]).reset_index(drop=True)

        # create unique window id
        window_id = shortuuid.uuid(
            name=f"{session_id}_{current_start_time.isoformat()}"
        )

        # add window to windows
        windows[window_id] = window_df

        # add window info to window index
        window_df_dict["session_id"].append(session_id)
        window_df_dict["window_id"].append(window_id)

        # step to next window
        current_start_time += stride_timedelta

    if windows == {}:
        return None, None

    # compute window size from sampling freq and window time
    window_size = int(window_time * sampling_freq)

    # trim to window size for batching
    windows = {
        window_id: window_df[:window_size] for window_id, window_df in windows.items()
    }

    # create window index
    window_df = pd.DataFrame(window_df_dict)
    window_df.astype({"window_id": "string", "session_id": "int"})

    return window_df, windows

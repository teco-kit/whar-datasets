from typing import Dict, Tuple

import numpy as np
import pandas as pd


def generate_windowing(
    session_id: int,
    session_df: pd.DataFrame,
    window_time: float,
    overlap: float,
    sampling_freq: float,
) -> Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]:
    """Generate fixed-length sliding windows for one session."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    window_size = int(round(window_time * sampling_freq))
    stride = max(int(round(window_size * (1 - overlap))), 1)
    if window_size <= 0:
        raise ValueError("window_time and sampling_freq must define a non-empty window.")
    if len(session_df) < window_size:
        return None, None

    starts = np.arange(0, len(session_df) - window_size + 1, stride, dtype=np.int64)
    sensor_df = session_df.drop(columns=["timestamp"])
    timestamps = session_df["timestamp"].reset_index(drop=True)
    windows: Dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    duration = pd.to_timedelta(window_size / sampling_freq, unit="s")
    for ordinal, start in enumerate(starts.tolist()):
        end = start + window_size
        window_id = f"{session_id}:{ordinal}"
        windows[window_id] = sensor_df.iloc[start:end].reset_index(drop=True)
        start_time = timestamps.iloc[start]
        rows.append(
            {
                "session_id": session_id,
                "window_id": window_id,
                "start_index": start,
                "end_index": end,
                "window_start": start_time,
                "window_end": start_time + duration,
            }
        )

    window_df = pd.DataFrame(rows).astype(
        {
            "session_id": "int64",
            "window_id": "string",
            "start_index": "int64",
            "end_index": "int64",
        }
    )

    return window_df, windows

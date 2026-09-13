import numpy as np
import pandas as pd

from whar_datasets.utils.logging import logger


def generate_windowing(
    session_id: int,
    session_df: pd.DataFrame,
    window_time: float,
    overlap: float,
    sampling_freq: float,
    max_gap_seconds: float | None = None,
) -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame] | None]:
    """Generate fixed-length sliding windows for one session."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    window_size = round(window_time * sampling_freq)
    stride = max(round(window_size * (1 - overlap)), 1)
    if window_size <= 0:
        raise ValueError(
            "window_time and sampling_freq must define a non-empty window."
        )
    if len(session_df) < window_size:
        return None, None

    starts = np.arange(0, len(session_df) - window_size + 1, stride, dtype=np.int64)
    sensor_df = session_df.drop(columns=["timestamp"])
    timestamps = session_df["timestamp"].reset_index(drop=True)
    invalid_intervals = session_df.attrs.get("invalid_intervals", [])
    windows: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    duration = pd.to_timedelta(window_size / sampling_freq, unit="s")
    rejected_gap_windows = 0
    for ordinal, start in enumerate(starts.tolist()):
        end = start + window_size
        start_time = timestamps.iloc[start]
        end_time = start_time + duration
        overlaps_invalid_interval = any(
            left < end_time and right > start_time
            for left, right in invalid_intervals
        )
        window = session_df.iloc[start:end]
        has_missing_sensor_value = window.drop(columns=["timestamp"]).isna().any().any()
        if max_gap_seconds is not None:
            timestamp_gaps = (
                window["timestamp"].diff().dt.total_seconds().dropna()
            )
            has_large_timestamp_gap = (
                not timestamp_gaps.empty
                and float(timestamp_gaps.max()) > max_gap_seconds
            )
        else:
            has_large_timestamp_gap = False
        if (
            overlaps_invalid_interval
            or has_missing_sensor_value
            or has_large_timestamp_gap
        ):
            if overlaps_invalid_interval or has_large_timestamp_gap:
                rejected_gap_windows += 1
            continue

        window_id = f"{session_id}:{ordinal}"
        windows[window_id] = sensor_df.iloc[start:end].reset_index(drop=True)
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

    if rejected_gap_windows:
        logger.info(
            "Session %s: rejected %d candidate windows due to unsafe "
            "timestamp gaps.",
            session_id,
            rejected_gap_windows,
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

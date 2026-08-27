import pandas as pd


def get_effective_sampling_freq(sampling_freq: float, resampling_freq: float | None) -> float:
    """Return the configured output frequency used by windows and transforms."""
    return float(resampling_freq if resampling_freq is not None else sampling_freq)


def resample(
    session_df: pd.DataFrame,
    resampling_freq: float,
    max_gap_seconds: float | None = None,
) -> pd.DataFrame:
    """Resample one session to a fixed frequency using interpolation."""
    if resampling_freq <= 0:
        raise ValueError("resampling_freq must be greater than zero.")

    session_df = session_df.sort_values("timestamp").copy()
    # Parquet can preserve millisecond-resolution timestamps; promote to ns so
    # rates such as 32, 60, 64, or 98 Hz remain representable by pandas.
    session_df["timestamp"] = session_df["timestamp"].astype("datetime64[ns]")
    gaps = session_df["timestamp"].diff().dt.total_seconds()
    if max_gap_seconds is not None and (gaps > max_gap_seconds).any():
        largest_gap = float(gaps.max())
        raise ValueError(
            "A session contains a timestamp gap of "
            f"{largest_gap:.3f}s, exceeding max_session_gap_seconds="
            f"{max_gap_seconds}. Split the recording into separate sessions or set "
            "max_session_gap_seconds=None to allow interpolation."
        )

    time_delta = pd.to_timedelta(1.0 / resampling_freq, unit="s")

    # Set timestamp as index
    session_df.set_index("timestamp", inplace=True)

    # Remove duplicates in index
    session_df = session_df[~session_df.index.duplicated()]

    # Resample to new frequency
    resampled_df = session_df.resample(time_delta).mean().interpolate()

    # Reset index and add timestamp back
    resampled_df.reset_index(inplace=True, drop=False)

    return resampled_df

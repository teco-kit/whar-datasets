import pandas as pd


def resample(session_df: pd.DataFrame, resampling_freq: float) -> pd.DataFrame:
    """Resample one session to a fixed frequency using interpolation."""
    # convert resampling freq to time delta in ms
    time_delta_ns = int(1e3 / resampling_freq)

    # Set timestamp as index
    session_df.set_index("timestamp", inplace=True)

    # Remove duplicates in index
    session_df = session_df[~session_df.index.duplicated()]

    # Resample to new frequency
    resampled_df = session_df.resample(f"{time_delta_ns}ms").mean().interpolate()

    # Reset index and add timestamp back
    resampled_df.reset_index(inplace=True, drop=False)

    return resampled_df

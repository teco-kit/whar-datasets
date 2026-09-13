from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


def get_effective_sampling_freq(
    sampling_freq: float, resampling_freq: float | None
) -> float:
    """Return the configured output frequency used by windows and transforms."""
    return float(resampling_freq if resampling_freq is not None else sampling_freq)


def _infer_sampling_freq(timestamps: pd.Series) -> float:
    """Infer a robust source rate from strictly increasing timestamps."""
    if len(timestamps) < 2:
        raise ValueError(
            "source_freq is required when resampling a session with fewer than "
            "two samples."
        )

    differences = np.diff(timestamps.astype("int64").to_numpy()) / 1e9
    median_step = float(np.median(differences))
    if not np.isfinite(median_step) or median_step <= 0:
        raise ValueError("Could not infer a positive source sampling interval.")
    return 1.0 / median_step


def _regularize_session(
    session_df: pd.DataFrame,
    source_freq: float,
    max_interpolation_gap_seconds: float,
) -> pd.DataFrame:
    """Map timestamp-jittered samples onto a regular source-rate grid."""
    timestamps = session_df["timestamp"]
    source_period = pd.to_timedelta(1.0 / source_freq, unit="s")
    regular_index = pd.date_range(
        start=timestamps.iloc[0],
        end=timestamps.iloc[-1],
        freq=source_period,
    )

    gaps = timestamps.diff().dt.total_seconds().dropna()
    invalid_intervals = [
        (timestamps.iloc[index - 1], timestamps.iloc[index])
        for index, gap in enumerate(gaps.tolist(), start=1)
        if float(gap) > max_interpolation_gap_seconds
    ]

    indexed = session_df.set_index("timestamp")
    combined_index = indexed.index.union(regular_index).sort_values()
    combined = indexed.reindex(combined_index)
    combined = combined.interpolate(method="time", limit_area="inside")
    regular = combined.reindex(regular_index).reset_index(names="timestamp")

    sensor_columns = [column for column in regular.columns if column != "timestamp"]
    for left, right in invalid_intervals:
        missing = regular["timestamp"].gt(left) & regular["timestamp"].lt(right)
        regular.loc[missing, sensor_columns] = np.nan
    regular.attrs["invalid_intervals"] = invalid_intervals
    return regular


def resample(
    session_df: pd.DataFrame,
    resampling_freq: float,
    max_gap_seconds: float | None = None,
    *,
    source_freq: float | None = None,
    max_interpolation_gap_seconds: float | None = None,
) -> pd.DataFrame:
    """Resample one regular sensor session with bounded interpolation.

    ``source_freq`` is the verified nominal input rate. If omitted, it is
    inferred from the median timestamp step. Timestamp jitter is first mapped
    to a regular source-rate grid, then SciPy's polyphase filter performs the
    actual rate conversion. The output grid starts at the first source sample
    and never extends beyond the last source sample.
    """
    if resampling_freq <= 0:
        raise ValueError("resampling_freq must be greater than zero.")
    if source_freq is not None and source_freq <= 0:
        raise ValueError("source_freq must be greater than zero.")
    if max_gap_seconds is not None and max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be greater than zero.")
    if (
        max_interpolation_gap_seconds is not None
        and max_interpolation_gap_seconds <= 0
    ):
        raise ValueError("max_interpolation_gap_seconds must be greater than zero.")
    if "timestamp" not in session_df.columns:
        raise ValueError("Session must contain a timestamp column.")

    result = session_df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    if result.empty:
        return result
    if result["timestamp"].isna().any():
        raise ValueError("Session timestamps must not be null.")
    if not result["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Session timestamps must be chronological; resolve source ordering "
            "in the dataset parser."
        )
    if result["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps require an explicit parser-level policy."
        )
    # Parquet can preserve millisecond-resolution timestamps; promote to ns so
    # rates such as 32, 60, 64, or 98 Hz remain representable by pandas.
    result["timestamp"] = result["timestamp"].astype("datetime64[ns]")
    timestamps = result["timestamp"]
    gaps = timestamps.diff().dt.total_seconds().dropna()
    if max_gap_seconds is not None and not gaps.empty:
        largest_gap = float(gaps.max())
        if largest_gap > max_gap_seconds:
            raise ValueError(
                "A session contains a timestamp gap of "
                f"{largest_gap:.6f}s, exceeding max_session_gap_seconds="
                f"{max_gap_seconds}. Split the recording into separate sessions."
            )

    actual_source_freq = float(source_freq) if source_freq is not None else _infer_sampling_freq(timestamps)
    observed_source_freq = (
        _infer_sampling_freq(timestamps) if len(result) >= 2 else None
    )
    if observed_source_freq is not None and source_freq is not None and not np.isclose(
        observed_source_freq,
        actual_source_freq,
        rtol=0.05,
        atol=0.05,
    ):
        raise ValueError(
            "Timestamp-derived source frequency "
            f"({observed_source_freq:.6f}Hz) disagrees with configured source "
            f"frequency ({actual_source_freq:.6f}Hz)."
        )

    interpolation_limit = max_interpolation_gap_seconds
    if interpolation_limit is None:
        # Allow a small amount of timestamp jitter and short dropouts by
        # default. Larger gaps are retained as invalid intervals and cause
        # overlapping windows to be rejected downstream.
        interpolation_limit = 3.0 / actual_source_freq

    regular = _regularize_session(
        result,
        actual_source_freq,
        interpolation_limit,
    )
    channel_columns = [column for column in regular.columns if column != "timestamp"]
    if not channel_columns:
        raise ValueError("Session must contain at least one sensor channel.")

    ratio = (
        Fraction(str(resampling_freq)) / Fraction(str(actual_source_freq))
    ).limit_denominator(10_000)
    values = regular[channel_columns].to_numpy(dtype=np.float64)
    finite_or_missing = np.isfinite(values) | np.isnan(values)
    if not finite_or_missing.all():
        raise ValueError("Sensor channels must contain only finite values or NaN gaps.")

    if ratio.numerator == ratio.denominator:
        resampled_values = values
    else:
        resampled_values = resample_poly(
            values,
            up=ratio.numerator,
            down=ratio.denominator,
            axis=0,
            padtype="line",
        )

    start = regular["timestamp"].iloc[0]
    end = regular["timestamp"].iloc[-1]
    span_seconds = (end.value - start.value) / 1e9
    output_count = int(np.floor(span_seconds * resampling_freq + 1e-9)) + 1
    if len(resampled_values) < output_count:
        raise ValueError(
            "Polyphase resampling returned fewer samples than the session grid "
            "requires."
        )
    resampled_values = resampled_values[:output_count]

    target_offsets = pd.to_timedelta(
        np.arange(output_count, dtype=np.float64) / resampling_freq,
        unit="s",
    )
    output = pd.DataFrame(resampled_values, columns=channel_columns)
    output.insert(0, "timestamp", start + target_offsets)
    output[channel_columns] = output[channel_columns].astype(np.float32)
    output.attrs["invalid_intervals"] = regular.attrs.get("invalid_intervals", [])
    return output

from __future__ import annotations

from typing import Literal

import pandas as pd

TimeUnit = Literal["s", "ms", "us", "ns"]


def _select_time_unit(values: pd.Series, default_unit: TimeUnit) -> TimeUnit:
    non_na = values.dropna()
    if non_na.empty:
        return default_unit

    abs_median = float(non_na.abs().median())

    if default_unit == "s":
        if abs_median >= 1e17:
            return "ns"
        if abs_median >= 1e14:
            return "us"
        if abs_median >= 1e11:
            return "ms"

    if default_unit == "ms":
        if abs_median >= 1e16:
            return "ns"
        if abs_median >= 1e13:
            return "us"

    if default_unit == "us" and abs_median >= 1e16:
        return "ns"

    return default_unit


def to_datetime64_ms(
    values: pd.Series,
    default_unit: TimeUnit = "ms",
) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce").astype("datetime64[ms]")

    if pd.api.types.is_timedelta64_dtype(values):
        origin = pd.Timestamp("1970-01-01")
        return (origin + values).astype("datetime64[ms]")

    numeric = pd.to_numeric(values, errors="coerce")
    inferred_unit = _select_time_unit(numeric, default_unit)
    return pd.to_datetime(numeric, unit=inferred_unit, errors="coerce").astype(
        "datetime64[ms]"
    )

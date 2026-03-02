from __future__ import annotations

import pandas as pd

from whar_datasets.config.timestamps import to_datetime64_ms


def test_to_datetime64_ms_supports_seconds_input() -> None:
    ts = pd.Series([0.0, 0.5, 1.0], dtype="float64")
    out = to_datetime64_ms(ts, default_unit="s")

    assert str(out.dtype) == "datetime64[ms]"
    deltas_ms = out.diff().dropna().dt.total_seconds().mul(1e3).tolist()
    assert deltas_ms == [500.0, 500.0]


def test_to_datetime64_ms_supports_milliseconds_even_with_default_seconds() -> None:
    ts = pd.Series([1_700_000_000_000, 1_700_000_000_010], dtype="int64")
    out = to_datetime64_ms(ts, default_unit="s")

    assert str(out.dtype) == "datetime64[ms]"
    deltas_ms = out.diff().dropna().dt.total_seconds().mul(1e3).tolist()
    assert deltas_ms == [10.0]

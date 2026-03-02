from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
import pytest

from whar_datasets.config.config import WHARConfig
from whar_datasets.config.getter import WHARDatasetID, har_dataset_dict
from whar_datasets.processing.utils.sessions import process_session

CFG_ITEMS: List[Tuple[WHARDatasetID, WHARConfig]] = sorted(
    har_dataset_dict.items(),
    key=lambda item: item[0].value,
)

DATASETS_ROOT = Path("notebooks/datasets")
MAX_ALLOWED_GAP_MULTIPLIER = 3.0
WINDOW_SAMPLE_COUNT = 32


def _drop_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    index_cols = [col for col in df.columns if col.startswith("Unnamed:")]
    return df.drop(columns=index_cols) if index_cols else df


def _dataset_cache_paths(dataset_id: str) -> Tuple[Path, Path, Path, Path]:
    dataset_dir = DATASETS_ROOT / dataset_id
    metadata_dir = dataset_dir / "metadata"
    sessions_dir = dataset_dir / "sessions"
    windows_dir = dataset_dir / "windows"
    raw_dir = dataset_dir / "data"
    return metadata_dir, sessions_dir, windows_dir, raw_dir


def _require_cached_common_format(
    dataset_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    metadata_dir, sessions_dir, _, _ = _dataset_cache_paths(dataset_id)
    session_df_path = metadata_dir / "session_df.csv"
    activity_df_path = metadata_dir / "activity_df.csv"
    sessions_path = sessions_dir / "sessions.parquet"

    if (
        not session_df_path.exists()
        or not activity_df_path.exists()
        or not sessions_path.exists()
    ):
        pytest.skip(
            f"Missing cached common format files for '{dataset_id}' under '{DATASETS_ROOT}'."
        )

    session_df = _drop_index_cols(pd.read_csv(session_df_path))
    activity_df = _drop_index_cols(pd.read_csv(activity_df_path))

    return activity_df, session_df, sessions_path


def _expected_step_ms(cfg: WHARConfig) -> float:
    return 1e3 / float(cfg.sampling_freq)


def _assert_non_time_series_modalities_are_excluded(channels: Iterable[str]) -> None:
    forbidden_tokens = ("audio", "mic", "video", "image", "img", "camera", "cam")

    for channel in channels:
        lowered = channel.lower()
        assert not any(token in lowered for token in forbidden_tokens), (
            f"Non-time-series modality leaked into channel schema: '{channel}'."
        )


def _assert_metadata_contract(
    cfg: WHARConfig,
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
) -> None:
    assert {"activity_id", "activity_name"}.issubset(activity_df.columns)
    assert {"session_id", "subject_id", "activity_id"}.issubset(session_df.columns)

    assert not activity_df[["activity_id", "activity_name"]].isna().any().any()
    assert (
        not session_df[["session_id", "subject_id", "activity_id"]].isna().any().any()
    )

    assert session_df["session_id"].is_unique
    assert activity_df["activity_id"].is_unique

    def coerce_int(series: pd.Series, name: str) -> pd.Series:
        if pd.api.types.is_integer_dtype(series):
            return series.astype("int64")
        normalized = series.astype(str).str.strip().str.strip("[]")
        converted = pd.to_numeric(normalized, errors="coerce")
        assert not converted.isna().any(), (
            f"Column '{name}' contains non-integer values."
        )
        assert (converted % 1 == 0).all(), (
            f"Column '{name}' contains non-integral values."
        )
        return converted.astype("int64")

    _ = coerce_int(session_df["session_id"], "session_id")
    subject_ids = coerce_int(session_df["subject_id"], "subject_id")
    session_activity_ids = coerce_int(session_df["activity_id"], "activity_id")
    activity_ids = coerce_int(activity_df["activity_id"], "activity_id")

    assert subject_ids.nunique() == int(cfg.num_of_subjects)
    assert session_activity_ids.nunique() == int(cfg.num_of_activities)
    assert activity_ids.nunique() == int(cfg.num_of_activities)

    assert subject_ids.min() == 0
    assert session_activity_ids.min() == 0
    assert activity_ids.min() == 0

    assert set(session_activity_ids).issubset(set(activity_ids))


def _assert_parquet_sessions_integrity(
    cfg: WHARConfig,
    session_df: pd.DataFrame,
    sessions_path: Path,
) -> None:
    parquet_schema = pq.read_schema(sessions_path)
    column_names = parquet_schema.names

    assert "session_id" in column_names
    assert "timestamp" in column_names

    sensor_cols = [
        col for col in column_names if col not in {"session_id", "timestamp"}
    ]
    assert len(sensor_cols) == int(cfg.num_of_channels)
    _assert_non_time_series_modalities_are_excluded(sensor_cols)

    expected_step_ms = _expected_step_ms(cfg)
    max_allowed_gap_ms = expected_step_ms * MAX_ALLOWED_GAP_MULTIPLIER

    ts_df = pq.read_table(
        sessions_path, columns=["session_id", "timestamp"]
    ).to_pandas()
    assert not ts_df.empty
    if not pd.api.types.is_datetime64_dtype(ts_df["timestamp"]):
        raise AssertionError("Timestamp column in sessions parquet is not datetime64.")

    ts_df["session_id"] = pd.to_numeric(ts_df["session_id"], errors="raise").astype(
        "int64"
    )
    ts_df = ts_df.sort_values(by=["session_id", "timestamp"]).reset_index(drop=True)

    observed_session_ids: set[int] = set(
        int(x) for x in ts_df["session_id"].unique().tolist()
    )
    expected_session_ids = set(int(x) for x in session_df["session_id"].tolist())
    assert observed_session_ids == expected_session_ids

    diffs_ms = (
        ts_df.groupby("session_id")["timestamp"].diff().dropna().dt.total_seconds()
        * 1e3
    )
    if not diffs_ms.empty and (diffs_ms <= 0).any():
        raise AssertionError(
            "At least one session has non-increasing timestamps in cached parse output."
        )

    max_gap_by_session = (
        ts_df.groupby("session_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .mul(1e3)
        .groupby(ts_df["session_id"])
        .max()
        .dropna()
    )

    bad_sessions = sorted(
        int(sid)
        for sid, max_gap in max_gap_by_session.items()
        if float(max_gap) > max_allowed_gap_ms
    )
    assert not bad_sessions, (
        f"Detected unusually large timestamp gaps in sessions {bad_sessions[:8]} "
        f"(threshold {max_allowed_gap_ms:.2f}ms). "
        "This violates the session continuity requirement and may cause interpolation artifacts."
    )


def _assert_windowing_integrity(
    cfg: WHARConfig, dataset_id: str, session_df: pd.DataFrame
) -> None:
    _, sessions_dir, windows_dir, _ = _dataset_cache_paths(dataset_id)
    metadata_dir = DATASETS_ROOT / dataset_id / "metadata"
    window_df_path = metadata_dir / "window_df.csv"
    windows_path = windows_dir / "windows.parquet"

    if not window_df_path.exists() or not windows_path.exists():
        pytest.skip(f"Window artifacts not present for '{dataset_id}'.")

    window_df = _drop_index_cols(pd.read_csv(window_df_path))
    assert {"session_id", "window_id"}.issubset(window_df.columns)
    assert not window_df[["session_id", "window_id"]].isna().any().any()
    assert window_df["window_id"].is_unique

    assert set(window_df["session_id"]).issubset(set(session_df["session_id"]))

    window_size = max(int(float(cfg.window_time) * float(cfg.sampling_freq)), 1)
    sampled_window_df = window_df.sample(
        n=min(WINDOW_SAMPLE_COUNT, len(window_df)),
        random_state=0,
    )

    # Validate sampled windows by reconstructing with the same processing path.
    sampled_session_ids = sorted(
        int(x) for x in sampled_window_df["session_id"].unique()
    )
    generated_windows: Dict[str, pd.DataFrame] = {}

    for sid in sampled_session_ids:
        generated_window_df, generated_window_map = process_session(
            cfg, sessions_dir, sid
        )
        assert generated_window_df is not None
        assert generated_window_map is not None
        generated_windows.update(generated_window_map)

    for window_id in sampled_window_df["window_id"]:
        wid = str(window_id)
        assert wid in generated_windows
        window = generated_windows[wid]
        assert len(window) > 0
        assert len(window) <= window_size
        assert list(window.columns) == list(cfg.sensor_channels)
        assert not window.isna().any().any()
        assert all(pd.api.types.is_float_dtype(window[col]) for col in window.columns)


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_cached_dataset_common_format_meets_requirements(
    dataset_id: WHARDatasetID, cfg: WHARConfig
) -> None:
    activity_df, session_df, sessions_path = _require_cached_common_format(
        dataset_id.value
    )

    _assert_metadata_contract(cfg, activity_df, session_df)
    _assert_non_time_series_modalities_are_excluded(cfg.sensor_channels)
    _assert_parquet_sessions_integrity(cfg, session_df, sessions_path)


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_cached_windowing_does_not_break_labels_or_channels(
    dataset_id: WHARDatasetID,
    cfg: WHARConfig,
) -> None:
    _, session_df, _ = _require_cached_common_format(dataset_id.value)
    _assert_windowing_integrity(cfg, dataset_id.value, session_df)


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_parser_output_requirements_when_raw_data_is_available(
    dataset_id: WHARDatasetID,
    cfg: WHARConfig,
) -> None:
    if os.getenv("WHAR_RUN_PARSE_E2E", "0") != "1":
        pytest.skip("Set WHAR_RUN_PARSE_E2E=1 to run full parser regression checks.")

    _, _, _, raw_dir = _dataset_cache_paths(dataset_id.value)
    if not raw_dir.exists():
        pytest.skip(f"No raw directory available for '{dataset_id.value}'.")

    try:
        activity_df, session_df, sessions = cfg.parse(str(raw_dir), cfg.activity_id_col)
    except (FileNotFoundError, NotADirectoryError) as exc:
        pytest.skip(f"Raw dataset layout for '{dataset_id.value}' is incomplete: {exc}")

    _assert_metadata_contract(cfg, activity_df, session_df)
    _assert_non_time_series_modalities_are_excluded(cfg.sensor_channels)

    assert len(sessions) == len(session_df)
    assert set(int(sid) for sid in sessions.keys()) == set(
        int(sid) for sid in session_df["session_id"]
    )

    expected_step_ms = _expected_step_ms(cfg)
    max_allowed_gap_ms = expected_step_ms * MAX_ALLOWED_GAP_MULTIPLIER

    for sid, session in sessions.items():
        assert "timestamp" in session.columns
        assert pd.api.types.is_datetime64_dtype(session["timestamp"])
        assert not session.isna().any().any()

        sensor_cols = [col for col in session.columns if col != "timestamp"]
        assert len(sensor_cols) == int(cfg.num_of_channels)
        _assert_non_time_series_modalities_are_excluded(sensor_cols)

        ts = session["timestamp"].reset_index(drop=True)
        diffs_ms = ts.diff().dropna().dt.total_seconds() * 1e3
        if not diffs_ms.empty:
            assert not (diffs_ms <= 0).any(), (
                f"Session {sid} has non-increasing timestamps."
            )
            assert float(diffs_ms.max()) <= max_allowed_gap_ms, (
                f"Session {sid} has an unusually large gap ({float(diffs_ms.max()):.2f}ms). "
                "Sessions should be split at discontinuities to avoid invalid interpolation."
            )

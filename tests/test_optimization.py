from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from whar_datasets.config.config import WHARConfig
from whar_datasets.config.getter import WHARDatasetID, get_dataset_cfg
from whar_datasets.processing.utils.caching import (
    cache_common_format,
    cache_samples,
    cache_windows,
)
from whar_datasets.processing.utils.preparation import (
    prepare_windows_para,
    prepare_windows_seq,
)
from whar_datasets.processing.utils.resampling import resample
from whar_datasets.processing.utils.windowing import generate_windowing
from whar_datasets.splitting.splitter_loso import LOSOSplitter
from whar_datasets.utils.loading import (
    load_activity_df,
    load_sample,
    load_samples,
    load_window,
    open_sample_store,
    open_window_store,
)


def _unused_parser(
    data_dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del data_dir, activity_id_col
    raise NotImplementedError


def _config(**updates: object) -> WHARConfig:
    values: dict[str, object] = {
        "dataset_id": "test",
        "dataset_url": "",
        "download_url": "unused",
        "sampling_freq": 10,
        "num_of_subjects": 2,
        "num_of_activities": 1,
        "num_of_channels": 1,
        "available_activities": ["walk"],
        "available_channels": ["x"],
        "parse": _unused_parser,
        "selected_activities": None,
        "selected_channels": None,
    }
    values.update(updates)
    return WHARConfig(**values)  # type: ignore[arg-type]


def test_windowing_uses_sample_offsets_and_keeps_last_window() -> None:
    session = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="100ms"),
            "x": np.arange(100, dtype=np.float32),
        }
    )
    metadata, windows = generate_windowing(3, session, 2.0, 0.5, 10)
    assert metadata is not None and windows is not None
    assert len(metadata) == 9
    assert metadata.iloc[-1]["end_index"] == 100
    assert all(len(window) == 20 for window in windows.values())


def test_resampling_supports_non_integer_millisecond_rates() -> None:
    session = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=61, freq="16666667ns"),
            "x": np.arange(61, dtype=np.float32),
        }
    )
    result = resample(session, 60.0)
    delta_ns = result["timestamp"].diff().dropna().median().value
    assert abs(delta_ns - round(1e9 / 60)) <= 1


def test_strict_train_validation_split_purges_overlaps() -> None:
    cfg = _config(strict_train_val_separation=True, val_percentage=0.2)
    session_df = pd.DataFrame(
        {"session_id": [0, 1], "subject_id": [0, 1], "activity_id": [0, 0]}
    )
    rows = []
    for session_id in range(2):
        for ordinal in range(20):
            rows.append(
                {
                    "session_id": session_id,
                    "window_id": f"{session_id}:{ordinal}",
                    "start_index": ordinal * 5,
                    "end_index": ordinal * 5 + 10,
                }
            )
    window_df = pd.DataFrame(rows)
    split = LOSOSplitter(cfg, subject_ids=[0]).get_splits(session_df, window_df)[0]
    assert len(split.val_indices) == 4
    for train_index in split.train_indices:
        train = window_df.loc[train_index]
        for val_index in split.val_indices:
            val = window_df.loc[val_index]
            if train["session_id"] == val["session_id"]:
                assert train["end_index"] <= val["start_index"] or val["end_index"] <= train["start_index"]


def test_strict_split_allocates_distributed_validation_per_session() -> None:
    cfg = _config(
        strict_train_val_separation=True,
        val_percentage=0.2,
        seed=17,
    )
    rows = []
    for session_id in range(2):
        for ordinal in range(30):
            rows.append(
                {
                    "session_id": session_id,
                    "window_id": f"{session_id}:{ordinal}",
                    "start_index": ordinal * 5,
                    "end_index": ordinal * 5 + 10,
                }
            )
    window_df = pd.DataFrame(rows)

    splitter = LOSOSplitter(cfg)
    train, validation = splitter._get_train_val_indices(
        list(window_df.index), window_df
    )

    for session_id in range(2):
        session_indices = set(window_df.index[window_df["session_id"] == session_id])
        session_train = sorted(session_indices.intersection(train))
        session_validation = sorted(session_indices.intersection(validation))
        assert len(session_validation) == 6
        assert session_train

        runs = 1 + sum(
            right > left + 1
            for left, right in zip(session_validation, session_validation[1:])
        )
        assert runs >= 2

        for train_index in session_train:
            train_row = window_df.loc[train_index]
            for val_index in session_validation:
                val_row = window_df.loc[val_index]
                assert (
                    train_row["end_index"] <= val_row["start_index"]
                    or val_row["end_index"] <= train_row["start_index"]
                )

    repeated = LOSOSplitter(cfg)._get_train_val_indices(
        list(window_df.index), window_df
    )
    assert repeated == (train, validation)


def test_strict_split_keeps_unsplittable_session_in_training() -> None:
    cfg = _config(strict_train_val_separation=True, val_percentage=0.2)
    window_df = pd.DataFrame(
        {
            "session_id": [7, 7],
            "window_id": ["7:0", "7:1"],
            "start_index": [0, 5],
            "end_index": [10, 15],
        }
    )

    train, validation = LOSOSplitter(cfg)._get_train_val_indices(
        list(window_df.index), window_df
    )
    assert train == [0, 1]
    assert validation == []


def test_strict_split_allocates_and_summarizes_unsplittable_sessions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _config(
        strict_train_val_separation=True,
        val_percentage=0.2,
        seed=9,
    )
    rows = []
    for session_id in range(20):
        window_count = 1 if session_id < 10 else 2
        for ordinal in range(window_count):
            rows.append(
                {
                    "session_id": session_id,
                    "window_id": f"{session_id}:{ordinal}",
                    "start_index": ordinal * 5,
                    "end_index": ordinal * 5 + 10,
                }
            )
    window_df = pd.DataFrame(rows)

    caplog.set_level("WARNING", logger="whar-datasets")
    train, validation = LOSOSplitter(cfg)._get_train_val_indices(
        list(window_df.index), window_df
    )

    assert len(train) == 24
    assert len(validation) == 6
    for _, session_windows in window_df.groupby("session_id"):
        indices = set(int(index) for index in session_windows.index)
        assert indices.issubset(train) or indices.issubset(validation)

    summary = caplog.text
    assert "assigned 20 unsplittable sessions as whole units" in summary
    assert "1-window sessions: 10 total, 2 validation, 8 training" in summary
    assert "2-window sessions: 10 total, 2 validation, 8 training" in summary


def test_legacy_train_validation_split_keeps_every_candidate() -> None:
    cfg = _config(strict_train_val_separation=False, val_percentage=0.2)
    splitter = LOSOSplitter(cfg)
    window_df = pd.DataFrame(
        {"session_id": [0] * 10, "window_id": [f"0:{i}" for i in range(10)]}
    )
    train, validation = splitter._get_train_val_indices(list(window_df.index), window_df)
    assert len(train) + len(validation) == len(window_df)
    assert len(validation) == 2


def test_array_caches_round_trip_without_pickle(tmp_path: Path) -> None:
    window_df = pd.DataFrame(
        {"session_id": [0, 0], "window_id": ["0:0", "0:1"]}
    )
    windows = {
        "0:0": pd.DataFrame(np.arange(6, dtype=np.float32).reshape(3, 2), columns=["x", "y"]),
        "0:1": pd.DataFrame(np.arange(6, 12, dtype=np.float32).reshape(3, 2), columns=["x", "y"]),
    }
    windows_dir = tmp_path / "windows"
    cache_windows(windows_dir, window_df, windows)
    np.testing.assert_array_equal(load_window(windows_dir, "0:1").to_numpy(), windows["0:1"].to_numpy())

    samples_dir = tmp_path / "samples"
    samples = {window_id: [frame.to_numpy()] for window_id, frame in windows.items()}
    cache_samples(samples_dir, window_df, samples)
    assert not (samples_dir / "samples.pkl").exists()
    np.testing.assert_array_equal(load_sample(samples_dir, "0:0")[0], samples["0:0"][0])
    np.testing.assert_array_equal(open_sample_store(samples_dir).get("0:1")[0], samples["0:1"][0])
    assert set(load_samples(samples_dir)) == {"0:0", "0:1"}


def test_legacy_cache_formats_are_rejected(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "activity_df.csv").write_text(
        "activity_id,activity_name\n0,walk\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="Rerun preprocessing"):
        load_activity_df(metadata_dir)

    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()
    (windows_dir / "windows.parquet").touch()
    with pytest.raises(FileNotFoundError, match="Rerun preprocessing"):
        open_window_store(windows_dir)

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "samples.pkl").touch()
    with pytest.raises(FileNotFoundError, match="Rerun preprocessing"):
        open_sample_store(samples_dir)


def test_get_dataset_config_returns_independent_objects() -> None:
    first = get_dataset_cfg(WHARDatasetID.WISDM, "first")
    second = get_dataset_cfg(WHARDatasetID.WISDM, "second")
    assert first is not second
    assert first.datasets_dir == "first"
    assert second.datasets_dir == "second"


def test_process_backend_matches_single_core_preparation(tmp_path: Path) -> None:
    cfg = _config(
        normalization=None,
        execution_backend="process",
        num_workers=2,
    )
    window_df = pd.DataFrame(
        {"session_id": [0] * 8, "window_id": [f"0:{i}" for i in range(8)]}
    )
    windows = {
        f"0:{i}": pd.DataFrame(
            np.full((10, 1), i, dtype=np.float32), columns=["x"]
        )
        for i in range(8)
    }
    windows_dir = tmp_path / "parallel_windows"
    cache_windows(windows_dir, window_df, windows)
    store = open_window_store(windows_dir)
    sequential = prepare_windows_seq(cfg, None, window_df, windows_dir, store)
    parallel = prepare_windows_para(cfg, None, window_df, windows_dir, store)
    assert sequential.keys() == parallel.keys()
    for window_id in sequential:
        np.testing.assert_array_equal(sequential[window_id][0], parallel[window_id][0])


def test_common_cache_writes_one_readable_session_row_group(tmp_path: Path) -> None:
    activity_df = pd.DataFrame({"activity_id": [0], "activity_name": ["walk"]})
    session_df = pd.DataFrame(
        {"session_id": [0, 1], "subject_id": [0, 1], "activity_id": [0, 0]},
        dtype=np.int32,
    )
    sessions = {
        session_id: pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=20, freq="100ms"),
                "x": np.arange(20, dtype=np.float32),
            }
        )
        for session_id in range(2)
    }
    cache_common_format(
        tmp_path / "metadata", tmp_path / "sessions", activity_df, session_df, sessions
    )
    loaded = pd.read_parquet(tmp_path / "sessions" / "sessions.parquet")
    assert loaded["session_id"].nunique() == 2
    assert loaded["x"].dtype == np.float32

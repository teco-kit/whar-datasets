import inspect
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pytest

from whar_datasets.config.getter import WHARDatasetID, get_dataset_cfg, har_dataset_dict
from whar_datasets.processing.utils.caching import cache_common_format
from whar_datasets.processing.utils.sessions import process_session
from whar_datasets.processing.utils.validation import validate_common_format

CFG_ITEMS: List[Tuple[WHARDatasetID, object]] = sorted(
    har_dataset_dict.items(),
    key=lambda item: item[0].value,
)


def _make_activity_names(cfg) -> List[str]:
    names = list(cfg.selected_activities or [])

    if len(names) < cfg.num_of_activities:
        names.extend(
            f"activity_{idx}" for idx in range(len(names), cfg.num_of_activities)
        )

    return names[: cfg.num_of_activities]


def _make_all_channel_names(cfg) -> List[str]:
    channel_names = list(cfg.selected_channels or [])
    extra_needed = cfg.num_of_channels - len(channel_names)

    for idx in range(max(extra_needed, 0)):
        channel_names.append(f"extra_channel_{idx}")

    return channel_names


def _make_common_format_payload(
    cfg,
    session_length: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    activity_names = _make_activity_names(cfg)
    channels = _make_all_channel_names(cfg)

    num_sessions = max(cfg.num_of_subjects, cfg.num_of_activities, 1)

    activity_df = pd.DataFrame(
        {
            "activity_id": list(range(cfg.num_of_activities)),
            "activity_name": activity_names,
        }
    ).astype({"activity_id": "int32", "activity_name": "string"})

    session_rows = []
    sessions: Dict[int, pd.DataFrame] = {}

    base_freq_ms = max(int(1e3 / cfg.sampling_freq), 1)

    for session_id in range(num_sessions):
        subject_id = session_id % cfg.num_of_subjects
        activity_id = session_id % cfg.num_of_activities

        session_rows.append(
            {
                "session_id": session_id,
                "subject_id": subject_id,
                "activity_id": activity_id,
            }
        )

        ts = pd.date_range(
            "2020-01-01",
            periods=session_length,
            freq=f"{base_freq_ms}ms",
        )
        data = {"timestamp": ts}
        for col_idx, col_name in enumerate(channels):
            data[col_name] = [
                float(col_idx + row_idx) for row_idx in range(session_length)
            ]

        sessions[session_id] = pd.DataFrame(data).astype(
            {
                **{col: "float32" for col in channels},
                "timestamp": "datetime64[ms]",
            }
        )

    session_df = pd.DataFrame(session_rows).astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_df, session_df, sessions


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_dataset_cfg_basic_semantics(dataset_id: WHARDatasetID, cfg) -> None:
    assert cfg.dataset_id == dataset_id.value
    if isinstance(cfg.download_url, str):
        download_urls = [cfg.download_url]
    else:
        download_urls = list(cfg.download_url)
    assert len(download_urls) > 0
    assert all(url.startswith(("http://", "https://")) for url in download_urls)
    assert cfg.sampling_freq > 0
    assert cfg.num_of_subjects > 0
    assert cfg.num_of_activities > 0
    assert cfg.num_of_channels > 0
    assert cfg.window_time > 0
    assert 0 <= cfg.window_overlap < 1
    assert isinstance(cfg.activity_id_col, str) and cfg.activity_id_col

    assert len(cfg.available_channels) > 0
    assert len(cfg.available_channels) == len(set(cfg.available_channels))
    assert all(isinstance(ch, str) and ch for ch in cfg.available_channels)

    assert len(cfg.available_activities) > 0
    assert len(cfg.available_activities) == len(set(cfg.available_activities))
    assert all(isinstance(name, str) and name for name in cfg.available_activities)

    assert cfg.selected_activities is not None
    assert cfg.selected_channels is not None
    assert len(cfg.selected_channels) > 0
    assert len(cfg.selected_activities) > 0

    # Some datasets expose only a selected subset for downstream usage.
    assert cfg.num_of_channels >= len(cfg.available_channels)
    assert cfg.num_of_activities >= len(cfg.available_activities)
    assert set(cfg.selected_channels).issubset(set(cfg.available_channels))
    assert set(cfg.selected_activities).issubset(set(cfg.available_activities))


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_dataset_cfg_parse_function_contract(dataset_id: WHARDatasetID, cfg) -> None:
    assert callable(cfg.parse)
    sig = inspect.signature(cfg.parse)
    assert list(sig.parameters.keys()) == ["dir", "activity_id_col"]

    assert cfg.parse.__name__.startswith("parse_")
    assert dataset_id.value.split("_")[0] in cfg.parse.__name__


def test_whar_dataset_enum_and_registry_are_in_sync() -> None:
    implemented_ids = {enum_member for enum_member in WHARDatasetID}
    assert set(har_dataset_dict.keys()) == implemented_ids


def test_w_har_defaults_exclude_unknown_activity() -> None:
    cfg = har_dataset_dict[WHARDatasetID.W_HAR]
    assert "Unknown" in cfg.available_activities
    assert "Unknown" not in (cfg.selected_activities or [])


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_get_dataset_cfg_returns_expected_config_object(
    dataset_id: WHARDatasetID, cfg
) -> None:
    custom_dir = f"/tmp/{dataset_id.value}_dataset_cache"
    resolved_cfg = get_dataset_cfg(dataset_id, datasets_dir=custom_dir)

    assert resolved_cfg.dataset_id == dataset_id.value
    assert resolved_cfg.parse is cfg.parse
    assert resolved_cfg.datasets_dir == custom_dir


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_common_format_validation_contract_can_be_satisfied_for_all_datasets(
    dataset_id: WHARDatasetID,
    cfg,
    tmp_path: Path,
) -> None:
    activity_df, session_df, sessions = _make_common_format_payload(
        cfg, session_length=8
    )

    metadata_dir = tmp_path / dataset_id.value / "metadata"
    sessions_dir = tmp_path / dataset_id.value / "sessions"
    cache_common_format(metadata_dir, sessions_dir, activity_df, session_df, sessions)

    assert validate_common_format(cfg, sessions_dir, activity_df, session_df)


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_process_session_windowing_semantics_hold_for_all_datasets(
    dataset_id: WHARDatasetID,
    cfg,
    tmp_path: Path,
) -> None:
    channels = _make_all_channel_names(cfg)
    window_size = max(int(cfg.window_time * cfg.sampling_freq), 1)
    session_length = max(window_size * 3, 12)

    activity_df = pd.DataFrame(
        {"activity_id": [0], "activity_name": [_make_activity_names(cfg)[0]]}
    ).astype({"activity_id": "int32", "activity_name": "string"})
    session_df = pd.DataFrame(
        {"session_id": [0], "subject_id": [0], "activity_id": [0]}
    ).astype({"session_id": "int32", "subject_id": "int32", "activity_id": "int32"})

    base_freq_ms = max(int(1e3 / cfg.sampling_freq), 1)
    ts = pd.date_range("2020-01-01", periods=session_length, freq=f"{base_freq_ms}ms")
    session_data = {"timestamp": ts}
    for col_idx, col_name in enumerate(channels):
        session_data[col_name] = [
            float(col_idx + row_idx) for row_idx in range(session_length)
        ]
    sessions = {
        0: pd.DataFrame(session_data).astype(
            {
                **{col: "float32" for col in channels},
                "timestamp": "datetime64[ms]",
            }
        )
    }

    metadata_dir = tmp_path / dataset_id.value / "metadata_window"
    sessions_dir = tmp_path / dataset_id.value / "sessions_window"
    cache_common_format(metadata_dir, sessions_dir, activity_df, session_df, sessions)

    window_df, windows = process_session(cfg, sessions_dir, 0)

    assert window_df is not None
    assert windows is not None
    assert len(window_df) > 0
    assert len(windows) == len(window_df)

    first_window_id = window_df["window_id"].iloc[0]
    first_window = windows[first_window_id]

    assert list(first_window.columns) == (cfg.selected_channels or [])
    assert not first_window.isna().any().any()
    assert all(
        pd.api.types.is_float_dtype(first_window[col]) for col in first_window.columns
    )
    assert len(first_window) <= window_size

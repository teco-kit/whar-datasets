import inspect
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from whar_datasets.config.getter import WHARDatasetID, get_dataset_cfg, har_dataset_dict
from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WINDOW_TIME_MEDIUM
from whar_datasets.config.cfg_daphnet import parse_daphnet
from whar_datasets.config.cfg_uci_har import get_df_from_files_uci_har
from whar_datasets.config.cfg_wisdm_19_phone import (
    WISDM_19_MAX_GAP_NS,
    parse_wisdm_19_phone,
)
from whar_datasets.config.cfg_wisdm_19_watch import parse_wisdm_19_watch
from whar_datasets.config.cfg_w_har import parse_w_har
from whar_datasets.processing.utils.caching import cache_common_format
from whar_datasets.processing.utils.selecting import select_activities
from whar_datasets.processing.utils.sessions import process_session
from whar_datasets.processing.utils.validation import validate_common_format
from whar_datasets.processing.steps.parsing_step import _align_activity_ids_to_config

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
            data[col_name] = np.asarray(
                [float(col_idx + row_idx) for row_idx in range(session_length)],
                dtype=np.float32,
            )

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


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_activity_config_names_are_canonicalized_and_stable(
    dataset_id: WHARDatasetID, cfg
) -> None:
    del dataset_id
    available = list(cfg.available_activities)
    selected = list(cfg.selected_activities or [])

    assert canonicalize_activity_name_list(available) == available
    assert canonicalize_activity_name_list(selected) == selected
    assert len(available) == len(set(available))
    assert len(selected) == len(set(selected))
    assert set(selected).issubset(available)


def test_activity_name_canonicalization_handles_config_label_variants() -> None:
    raw_names = [
        "falling forward using hands",
        "sittingSofa",
        "not_labeled",
        "walking-downstairs",
    ]

    canonical = canonicalize_activity_name_list(raw_names)

    assert canonical == [
        "Falling Forward Using Hands",
        "Sitting Sofa",
        "Not Labeled",
        "Walking Downstairs",
    ]
    assert canonicalize_activity_name_list(canonical) == canonical


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_all_dataset_configs_use_global_window_time(
    dataset_id: WHARDatasetID, cfg
) -> None:
    del dataset_id
    assert cfg.window_time == WINDOW_TIME_MEDIUM == 2.0


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_all_dataset_configs_use_global_processing_defaults(
    dataset_id: WHARDatasetID, cfg
) -> None:
    del dataset_id
    assert cfg.window_overlap == 0.5
    assert cfg.execution_backend == "sequential"
    assert cfg.datasets_dir == "./datasets/"


def test_w_har_defaults_exclude_unknown_activity() -> None:
    cfg = har_dataset_dict[WHARDatasetID.W_HAR]
    assert "Unknown" in cfg.available_activities
    assert "Unknown" not in (cfg.selected_activities or [])


@pytest.mark.parametrize(("dataset_id", "cfg"), CFG_ITEMS)
def test_selected_activities_exclude_null_classes(
    dataset_id: WHARDatasetID, cfg
) -> None:
    del dataset_id
    null_names = {
        "null",
        "unknown",
        "undefined",
        "none",
        "other",
        "notlabeled",
        "notlabelled",
        "unlabeled",
        "unlabelled",
        "noactivity",
        "background",
        "na",
        "nan",
    }

    def normalize(label: str) -> str:
        return "".join(char for char in label.lower() if char.isalnum())

    assert not any(
        normalize(activity) in null_names
        for activity in (cfg.selected_activities or [])
    )


def test_activity_ids_are_aligned_by_names_not_parser_order() -> None:
    cfg = har_dataset_dict[WHARDatasetID.UCI_HAR]
    activity_df = pd.DataFrame(
        {
            "activity_id": [0, 1],
            "activity_name": ["STANDING", "WALKING"],
        }
    )
    session_df = pd.DataFrame(
        {
            "session_id": [0, 1],
            "subject_id": [0, 0],
            "activity_id": [0, 1],
        }
    )

    aligned_activity, aligned_sessions = _align_activity_ids_to_config(
        cfg, activity_df, session_df
    )

    assert aligned_activity["activity_id"].tolist() == [0, 4]
    assert aligned_activity["activity_name"].tolist() == ["Walking", "Standing"]
    assert aligned_sessions["activity_id"].tolist() == [4, 0]


def test_uci_har_deoverlap_keeps_one_nonoverlapping_half(tmp_path: Path) -> None:
    files = [
        "total_acc_x_train.txt",
        "total_acc_y_train.txt",
        "total_acc_z_train.txt",
    ]
    for file_name, offset in zip(files, [0, 1000, 2000]):
        values = " ".join(str(offset + value) for value in range(128))
        (tmp_path / file_name).write_text(values + "\n")

    subjects_path = tmp_path / "subject_train.txt"
    labels_path = tmp_path / "y_train.txt"
    subjects_path.write_text("1\n")
    labels_path.write_text("1\n")

    parsed = get_df_from_files_uci_har(
        files=files,
        files_dir=str(tmp_path),
        subj_path=str(subjects_path),
        labels_path=str(labels_path),
        slice_end=-10,
    )

    assert len(parsed) == 64
    assert parsed["total_acc_x"].tolist() == list(range(64))


def test_wisdm19_gap_threshold_uses_raw_nanoseconds() -> None:
    assert WISDM_19_MAX_GAP_NS == 1_000_000_000


def test_daphnet_parser_ignores_macos_sidecar_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "dataset_fog_release" / "dataset"
    raw_dir.mkdir(parents=True)
    rows = "\n".join(
        f"{index * 16} 1 2 3 4 5 6 7 8 9 {activity}"
        for index, activity in enumerate([0, 0, 1, 1])
    )
    (raw_dir / "S01R01.txt").write_text(rows + "\n")
    (raw_dir / "._S01R01.txt").write_bytes(b"Mac OS X sidecar metadata")

    activity_df, session_df, sessions = parse_daphnet(str(tmp_path), "activity_id")

    assert len(activity_df) == 2
    assert len(session_df) == len(sessions) == 2


def test_w_har_parser_maps_undefined_to_excluded_unknown_class(tmp_path: Path) -> None:
    motion_header = "Time (s),User,Scenerio,Trial,Ax,Ay,Az,GyroX,GyroY,GyroZ"
    motion_rows = [
        "0.0,1,1,1,1,2,3,4,5,6,undefined",
        "0.004,1,1,1,1,2,3,4,5,6,walk",
    ]
    stretch_header = "Time (s),User,Scenerio,Trial,Stretch Value"
    stretch_rows = [
        "0.0,1,1,1,10,undefined",
        "0.01,1,1,1,11,walk",
    ]
    (tmp_path / "motion_data_22_users.csv").write_text(
        "\n".join([motion_header, *motion_rows]) + "\n"
    )
    (tmp_path / "stretch_data_22_users.csv").write_text(
        "\n".join([stretch_header, *stretch_rows]) + "\n"
    )

    activity_df, session_df, sessions = parse_w_har(str(tmp_path), "activity_id")

    assert "Unknown" in set(activity_df["activity_name"])
    assert len(session_df) == len(sessions) == 2


@pytest.mark.parametrize(
    ("device", "parser", "channels"),
    [
        (
            "phone",
            parse_wisdm_19_phone,
            [
                "accel_phone_x",
                "accel_phone_y",
                "accel_phone_z",
                "gyro_phone_x",
                "gyro_phone_y",
                "gyro_phone_z",
            ],
        ),
        (
            "watch",
            parse_wisdm_19_watch,
            [
                "accel_watch_x",
                "accel_watch_y",
                "accel_watch_z",
                "gyro_watch_x",
                "gyro_watch_y",
                "gyro_watch_z",
            ],
        ),
    ],
)
def test_wisdm19_alignment_is_bounded_and_preserves_one_session(
    tmp_path: Path,
    device: str,
    parser: Callable[[str, str], Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]],
    channels: List[str],
) -> None:
    raw_root = tmp_path / "wisdm-dataset" / "wisdm-dataset" / "raw" / device
    timestamps = [1_700_000_000_000_000_000 + idx * 50_000_000 for idx in range(4)]
    gyro_timestamps = [timestamp + 10_000_000 for timestamp in timestamps]

    for sensor, sensor_timestamps, offset in [
        ("accel", timestamps, 0.0),
        ("gyro", gyro_timestamps, 10.0),
    ]:
        sensor_path = raw_root / sensor / f"data_1600_{sensor}_{device}.txt"
        sensor_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, timestamp in enumerate(sensor_timestamps):
            values = [offset + idx, offset + idx + 1, offset + idx + 2]
            rows.append(f"1600,A,{timestamp},{values[0]},{values[1]},{values[2]};")
        sensor_path.write_text("\n".join(rows) + "\n")

    activity_df, session_df, sessions = parser(str(tmp_path), "activity_id")

    assert len(activity_df) == 18
    assert len(session_df) == len(sessions) == 1
    session = sessions[int(session_df.iloc[0]["session_id"])]
    assert list(session.columns) == ["timestamp", *channels]
    assert len(session) == 4
    assert session["timestamp"].diff().dropna().dt.total_seconds().tolist() == [
        0.05,
        0.05,
        0.05,
    ]


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
        session_data[col_name] = np.asarray(
            [float(col_idx + row_idx) for row_idx in range(session_length)],
            dtype=np.float32,
        )
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


def test_select_activities_remaps_activity_ids_to_contiguous_range() -> None:
    activity_df = pd.DataFrame(
        {
            "activity_id": [0, 1, 4, 7],
            "activity_name": ["other", "lying", "walking", "running"],
        }
    ).astype({"activity_id": "int32", "activity_name": "string"})

    session_df = pd.DataFrame(
        {
            "session_id": [10, 11, 12, 13],
            "subject_id": [0, 0, 1, 1],
            "activity_id": [0, 1, 4, 7],
        }
    ).astype({"session_id": "int32", "subject_id": "int32", "activity_id": "int32"})

    selected_activity_df, selected_session_df = select_activities(
        activity_df=activity_df,
        session_df=session_df,
        selected_activities=["lying", "walking", "running"],
    )

    assert selected_activity_df["activity_id"].tolist() == [0, 1, 2]
    assert selected_activity_df["activity_name"].tolist() == [
        "lying",
        "walking",
        "running",
    ]
    assert selected_session_df["activity_id"].tolist() == [0, 1, 2]


def test_select_activities_removes_null_class_from_loaded_sessions() -> None:
    activity_df = pd.DataFrame(
        {
            "activity_id": [0, 1],
            "activity_name": ["Unknown", "walking"],
        }
    ).astype({"activity_id": "int32", "activity_name": "string"})
    session_df = pd.DataFrame(
        {
            "session_id": [10, 11],
            "subject_id": [0, 0],
            "activity_id": [0, 1],
        }
    ).astype({"session_id": "int32", "subject_id": "int32", "activity_id": "int32"})

    selected_activity_df, selected_session_df = select_activities(
        activity_df=activity_df,
        session_df=session_df,
        selected_activities=["walking"],
    )

    assert selected_activity_df["activity_name"].tolist() == ["walking"]
    assert selected_session_df["session_id"].tolist() == [11]

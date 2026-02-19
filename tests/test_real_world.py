import zipfile
from pathlib import Path

import pandas as pd
import pytest

from whar_datasets.config.cfg_real_world import (
    REAL_WORLD_ACTIVITY_NAMES,
    REAL_WORLD_SENSOR_CHANNELS,
    cfg_real_world,
    parse_real_world,
)
from whar_datasets.config.getter import WHARDatasetID, get_dataset_cfg, har_dataset_dict
from whar_datasets.processing.utils.caching import cache_common_format
from whar_datasets.processing.utils.validation import validate_common_format


def _csv_text(header: str, rows: list[str]) -> str:
    return header + "\n" + "\n".join(rows) + "\n"


def _write_zip(zip_path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in files.items():
            zf.writestr(name, text)
        zf.writestr("readMe", "test")


def _build_minimal_real_world_subject(base_dir: Path, subject_num: int) -> Path:
    data_dir = base_dir / f"proband{subject_num}" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    _write_zip(
        data_dir / "acc_walking_csv.zip",
        {
            "acc_walking_chest.csv": _csv_text(
                "id,attr_time,attr_x,attr_y,attr_z",
                [
                    "1,1000,1.0,2.0,3.0",
                    "2,1020,1.1,2.1,3.1",
                    "3,1040,1.2,2.2,3.2",
                ],
            ),
            "acc_walking_forearm.csv": _csv_text(
                "id,attr_time,attr_x,attr_y,attr_z",
                [
                    "1,1000,4.0,5.0,6.0",
                    "2,1020,4.1,5.1,6.1",
                    "3,1040,4.2,5.2,6.2",
                ],
            ),
        },
    )

    gyr_dir = data_dir / "gyr_walking_csv"
    gyr_dir.mkdir(parents=True, exist_ok=True)
    (gyr_dir / "Gyroscope_walking_chest.csv").write_text(
        _csv_text(
            "id,attr_time,attr_x,attr_y,attr_z",
            [
                "1,1000,0.1,0.2,0.3",
                "2,1020,0.11,0.21,0.31",
                "3,1040,0.12,0.22,0.32",
            ],
        ),
        encoding="utf-8",
    )

    _write_zip(
        data_dir / "mag_walking_csv.zip",
        {
            "MagneticField_walking_chest.csv": _csv_text(
                "id,attr_time,attr_x,attr_y,attr_z",
                [
                    "1,1000,7.0,8.0,9.0",
                    "2,1020,7.1,8.1,9.1",
                    "3,1040,7.2,8.2,9.2",
                ],
            )
        },
    )

    lig_dir = data_dir / "lig_walking_csv"
    lig_dir.mkdir(parents=True, exist_ok=True)
    (lig_dir / "Light_walking_chest.csv").write_text(
        _csv_text(
            "id,attr_time,attr_light",
            [
                "1,1000,10.0",
                "2,1020,11.0",
                "3,1040,12.0",
            ],
        ),
        encoding="utf-8",
    )

    _write_zip(
        data_dir / "gps_walking_csv.zip",
        {
            "GPS_walking_chest.csv": _csv_text(
                "id,attr_time,attr_lat,attr_lng",
                [
                    "1,1000,49.0,8.0",
                    "2,1040,49.1,8.1",
                ],
            )
        },
    )

    # Should be ignored completely.
    _write_zip(
        data_dir / "mic_walking_csv.zip",
        {
            "Microphone_walking_chest.csv": _csv_text(
                "id,attr_time,attr_db",
                [
                    "1,1000,44.0",
                    "2,1020,43.0",
                    "3,1040,42.0",
                ],
            )
        },
    )

    return data_dir


def _build_full_subject_activity_fixture(base_dir: Path) -> None:
    for subject_num in range(1, 16):
        data_dir = base_dir / f"proband{subject_num}" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        for activity in REAL_WORLD_ACTIVITY_NAMES:
            # Minimal valid RealWorld activity source: one accelerometer csv with a few rows.
            # Parser should still emit full channel schema consistently.
            _write_zip(
                data_dir / f"acc_{activity}_csv.zip",
                {
                    f"acc_{activity}_chest.csv": _csv_text(
                        "id,attr_time,attr_x,attr_y,attr_z",
                        [
                            "1,1000,1.0,2.0,3.0",
                            "2,1020,1.1,2.1,3.1",
                            "3,1040,1.2,2.2,3.2",
                        ],
                    )
                },
            )


def test_real_world_cfg_has_expected_non_sound_channels() -> None:
    assert cfg_real_world.dataset_id == "real_world"
    assert cfg_real_world.num_of_subjects == 15
    assert cfg_real_world.num_of_activities == 8
    assert cfg_real_world.num_of_channels == len(REAL_WORLD_SENSOR_CHANNELS)
    assert all(not channel.startswith("mic_") for channel in REAL_WORLD_SENSOR_CHANNELS)

    # Spot-check key channels across all included sensor groups.
    expected = {
        "acc_chest_x",
        "gyr_forearm_z",
        "mag_waist_y",
        "lig_chest",
        "gps_head_lat",
        "gps_waist_lng",
    }
    assert expected.issubset(set(REAL_WORLD_SENSOR_CHANNELS))


def test_parse_real_world_parses_non_sound_sensors_and_ignores_mic(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "realworld2016_dataset"
    _build_minimal_real_world_subject(raw_root, subject_num=6)

    activity_df, session_df, sessions = parse_real_world(str(tmp_path), "activity_id")

    assert set(activity_df["activity_name"]) == set(REAL_WORLD_ACTIVITY_NAMES)
    assert len(session_df) == 1
    assert session_df["subject_id"].tolist() == [0]

    walking_id = int(
        activity_df.loc[activity_df["activity_name"] == "walking", "activity_id"].iloc[
            0
        ]
    )
    assert int(session_df.at[0, "activity_id"]) == walking_id

    session_id = int(session_df.at[0, "session_id"])
    session = sessions[session_id]

    assert list(session.columns) == ["timestamp"] + REAL_WORLD_SENSOR_CHANNELS
    assert not session.isna().any().any()
    assert pd.api.types.is_datetime64_dtype(session["timestamp"])
    assert all(
        pd.api.types.is_float_dtype(session[col])
        for col in session.columns
        if col != "timestamp"
    )

    # Missing channels are normalized to 0.0 for schema consistency.
    assert (session["acc_head_x"] == 0.0).all()
    assert (session["gyr_head_x"] == 0.0).all()

    # Provided channels contain real signal.
    assert session["acc_chest_x"].abs().sum() > 0
    assert session["lig_chest"].abs().sum() > 0
    assert session["gps_chest_lat"].abs().sum() > 0

    # Sound must not be represented.
    assert all(not column.startswith("mic_") for column in session.columns)

    # Semantic mapping check for known raw values (not just schema presence).
    first_row = session.iloc[0]
    assert float(first_row["acc_chest_x"]) == pytest.approx(1.0)
    assert float(first_row["acc_forearm_x"]) == pytest.approx(4.0)
    assert float(first_row["gyr_chest_x"]) == pytest.approx(0.1)
    assert float(first_row["mag_chest_x"]) == pytest.approx(7.0)
    assert float(first_row["lig_chest"]) == pytest.approx(10.0)
    assert float(first_row["gps_chest_lat"]) == pytest.approx(49.0)
    assert float(first_row["gps_chest_lng"]) == pytest.approx(8.0)


def test_parse_real_world_output_is_semantically_compatible_with_common_format_validation(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "realworld2016_dataset"
    _build_full_subject_activity_fixture(raw_root)

    activity_df, session_df, sessions = parse_real_world(str(tmp_path), "activity_id")

    metadata_dir = tmp_path / "metadata"
    sessions_dir = tmp_path / "sessions"
    cache_common_format(metadata_dir, sessions_dir, activity_df, session_df, sessions)

    assert validate_common_format(cfg_real_world, sessions_dir, activity_df, session_df)
    assert session_df["subject_id"].nunique() == cfg_real_world.num_of_subjects
    assert session_df["activity_id"].nunique() == cfg_real_world.num_of_activities
    assert len(sessions) == cfg_real_world.num_of_subjects * cfg_real_world.num_of_activities


def test_real_world_is_glued_into_dataset_getter() -> None:
    assert WHARDatasetID.REAL_WORLD in har_dataset_dict

    cfg = get_dataset_cfg(WHARDatasetID.REAL_WORLD, datasets_dir="/tmp/realworld-tests")
    assert cfg.dataset_id == "real_world"
    assert cfg.parse.__name__ == "parse_real_world"
    assert cfg.datasets_dir == "/tmp/realworld-tests"

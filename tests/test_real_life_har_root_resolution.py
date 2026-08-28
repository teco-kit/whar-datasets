from pathlib import Path

from whar_datasets.config.cfg_real_life_har import (
    REAL_LIFE_HAR_REQUIRED_FILES,
    _find_real_life_har_root,
)


def _touch_required_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for file_name in REAL_LIFE_HAR_REQUIRED_FILES:
        (root / file_name).write_text("header\n", encoding="utf-8")


def test_find_root_when_files_are_in_expected_subdirectory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    real_life_root = data_dir / "data_cleaned_adapted_full"
    _touch_required_files(real_life_root)

    assert _find_real_life_har_root(str(data_dir)) == str(real_life_root)


def test_find_root_when_files_are_extracted_flat_into_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _touch_required_files(data_dir)

    assert _find_real_life_har_root(str(data_dir)) == str(data_dir)

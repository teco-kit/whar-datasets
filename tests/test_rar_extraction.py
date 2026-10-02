from pathlib import Path
from unittest.mock import patch

import pytest

from whar_datasets.processing.utils.extracting import extract


def test_missing_unrar_raises_and_preserves_archive(tmp_path: Path) -> None:
    archive = tmp_path / "sad.rar"
    archive.write_bytes(b"RAR fixture placeholder")
    destination = tmp_path / "extracted"

    with patch("whar_datasets.processing.utils.extracting.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="'unrar' command is required"):
            extract(archive, destination)

    assert archive.exists()
    assert not destination.exists()

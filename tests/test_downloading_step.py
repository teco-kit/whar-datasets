"""Focused regressions for streaming archive downloads."""

import io
import tempfile
import unittest
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from whar_datasets.processing.steps.downloading_step import DownloadingStep


class _Response:
    def __init__(
        self, body: bytes, content_type: str = "application/zip", status_code: int = 200
    ) -> None:
        self.body = body
        self.ok = status_code == 200
        self.status_code = status_code
        self.url = "https://example.org/sample.zip"
        self.headers = {
            "content-type": content_type,
            "content-length": str(len(body)),
        }

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        yield self.body


class _MetadataResponse(_Response):
    def __init__(self, metadata: dict[str, object]) -> None:
        super().__init__(b"{}", "application/json")
        self.metadata = metadata

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.metadata


class DownloadingStepTests(unittest.TestCase):
    def test_partial_file_does_not_satisfy_download_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sample.zip.part").write_bytes(b"partial")
            cfg = SimpleNamespace(dataset_id="sample", download_url="unused")
            step = DownloadingStep(cfg, root, root, root)  # type: ignore[arg-type]
            self.assertFalse(step.output_exists())

    def test_streamed_zip_is_committed_without_partial_file(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("record.csv", "subject,activity\n1,walk\n")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = SimpleNamespace(dataset_id="sample", download_url="unused")
            step = DownloadingStep(cfg, root, root, root)  # type: ignore[arg-type]
            target = root / "sample.zip"
            with patch(
                "whar_datasets.processing.steps.downloading_step.requests.get",
                return_value=_Response(buffer.getvalue()),
            ):
                step._download_single_url("https://example.org/sample.zip", target)

            self.assertEqual(target.read_bytes(), buffer.getvalue())
            self.assertFalse((root / "sample.zip.part").exists())

    def test_html_response_retries_with_origin_session(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("record.csv", "data")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = SimpleNamespace(dataset_id="sample", download_url="unused")
            step = DownloadingStep(cfg, root, root, root)  # type: ignore[arg-type]
            target = root / "sample.zip"
            session = MagicMock()
            session.__enter__.return_value = session
            session.get.side_effect = [
                _Response(b"home", "text/html"),
                _Response(buffer.getvalue()),
            ]
            with (
                patch(
                    "whar_datasets.processing.steps.downloading_step.requests.get",
                    return_value=_Response(b"<html>blocked</html>", "text/html", 403),
                ),
                patch(
                    "whar_datasets.processing.steps.downloading_step.requests.Session",
                    return_value=session,
                ),
            ):
                step._download_single_url("https://example.org/sample.zip", target)

            self.assertEqual(target.read_bytes(), buffer.getvalue())
            self.assertEqual(session.get.call_count, 2)
            self.assertEqual(session.get.call_args.kwargs["headers"]["Referer"], "https://example.org/")

    def test_zenodo_api_resolves_file_from_record_metadata(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("record.csv", "data")

        url = "https://zenodo.org/records/123/files/sample.zip?download=1"
        content_url = "https://zenodo.org/api/records/123/files/sample.zip/content"
        metadata = {
            "files": {
                "entries": {
                    "sample.zip": {"links": {"content": content_url}}
                }
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = SimpleNamespace(dataset_id="sample", download_url=url)
            step = DownloadingStep(cfg, root, root, root)  # type: ignore[arg-type]
            target = root / "sample.zip"
            with patch(
                "whar_datasets.processing.steps.downloading_step.requests.get",
                side_effect=[
                    _Response(b"<html>blocked</html>", "text/html", 403),
                    _MetadataResponse(metadata),
                    _Response(buffer.getvalue()),
                ],
            ) as get:
                step._download_single_url(url, target)

            self.assertEqual(target.read_bytes(), buffer.getvalue())
            self.assertEqual(get.call_args_list[1].args[0], "https://zenodo.org/api/records/123")
            self.assertEqual(get.call_args_list[2].args[0], content_url)

    def test_zenodo_content_403_reports_cluster_cache_path(self) -> None:
        url = "https://zenodo.org/records/123/files/sample.zip?download=1"
        content_url = "https://zenodo.org/api/records/123/files/sample.zip/content"
        metadata = {
            "files": {
                "entries": {
                    "sample.zip": {"links": {"content": content_url}}
                }
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = SimpleNamespace(dataset_id="sample", download_url=url)
            step = DownloadingStep(cfg, root, root, root)  # type: ignore[arg-type]
            target = root / "sample.zip"
            with patch(
                "whar_datasets.processing.steps.downloading_step.requests.get",
                side_effect=[
                    _Response(b"<html>blocked</html>", "text/html", 403),
                    _MetadataResponse(metadata),
                    _Response(b"<html>blocked</html>", "text/html", 403),
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "shared cache"):
                    step._download_single_url(url, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()

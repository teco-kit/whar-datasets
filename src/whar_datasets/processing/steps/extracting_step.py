import tarfile
import zipfile
from pathlib import Path
from typing import Any, List, Set, TypeAlias

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.steps.abstract_step import AbstractStep
from whar_datasets.processing.utils.extracting import extract
from whar_datasets.utils.logging import logger

base_type: TypeAlias = Any
result_type: TypeAlias = None


class ExtractingStep(AbstractStep):
    def __init__(
        self,
        cfg: WHARConfig,
        data_dir: Path,
        dependent_on: List[AbstractStep],
    ):
        super().__init__(cfg, data_dir, dependent_on)

        self.data_dir = data_dir

        self.hash_name: str = "extracting_hash"
        self.relevant_cfg_keys: Set[str] = {"dataset_id", "download_url"}

    def get_base(self) -> base_type:
        return None

    def check_initial_format(self, base: base_type) -> bool:
        return self.data_dir.exists()

    def compute_results(self, base: base_type) -> result_type:
        archive_paths = self._find_archives(self.data_dir)

        if len(archive_paths) == 0:
            logger.info(f"No archives found to extract for {self.cfg.dataset_id}")
            return None

        logger.info(
            f"Extracting {len(archive_paths)} archive(s) for {self.cfg.dataset_id}"
        )
        for archive_path in archive_paths:
            extract(archive_path, self.data_dir)

    def save_results(self, results: result_type) -> None:
        return None

    def load_results(self) -> result_type:
        return None

    @staticmethod
    def _find_archives(root_dir: Path) -> List[Path]:
        archive_paths: List[Path] = []

        for file_path in root_dir.rglob("*"):
            if not file_path.is_file():
                continue

            if (
                tarfile.is_tarfile(file_path)
                or zipfile.is_zipfile(file_path)
                or file_path.suffix.lower() in {".rar", ".gz"}
            ):
                archive_paths.append(file_path)

        return archive_paths

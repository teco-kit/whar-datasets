from pathlib import Path
from typing import List, Set, TypeAlias

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.steps.abstract_step import AbstractStep
from whar_datasets.processing.utils.extracting import extract, find_archives
from whar_datasets.utils.logging import logger

InputT: TypeAlias = None
OutputT: TypeAlias = None


class ExtractingStep(AbstractStep[InputT, OutputT]):
    """Extract downloaded archives under the dataset `data` directory.

    Input/output are `None` because this step mutates files in-place.
    """

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

    def load_input(self) -> InputT:
        return None

    def validate_input(self, step_input: InputT) -> bool:
        return self.data_dir.exists()

    def build_output(self, step_input: InputT) -> OutputT:
        archive_paths = find_archives(self.data_dir)

        if len(archive_paths) == 0:
            logger.info(f"No archives found to extract for {self.cfg.dataset_id}")
            return None

        logger.info(
            f"Extracting {len(archive_paths)} archive(s) for {self.cfg.dataset_id}"
        )

        for archive_path in archive_paths:
            extract(archive_path, self.data_dir)

    def save_output(self, step_output: OutputT) -> None:
        return None

    def load_output(self) -> OutputT:
        return None

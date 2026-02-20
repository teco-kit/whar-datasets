from pathlib import Path
from typing import Any, Set, TypeAlias

import requests

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.steps.abstract_step import AbstractStep
from whar_datasets.utils.logging import logger

base_type: TypeAlias = Any
result_type: TypeAlias = None


class DownloadingStep(AbstractStep):
    def __init__(
        self,
        cfg: WHARConfig,
        datasets_dir: Path,
        dataset_dir: Path,
        data_dir: Path,
    ):
        super().__init__(cfg, data_dir)

        self.datasets_dir = datasets_dir
        self.dataset_dir = dataset_dir
        self.data_dir = data_dir

        self.hash_name: str = "download_hash"
        self.relevant_cfg_keys: Set[str] = {"dataset_id", "download_url"}

    def get_base(self) -> base_type:
        return None

    def check_initial_format(self, base: base_type) -> bool:
        return True

    def compute_results(self, base: base_type) -> result_type:
        # Use filename to define file path
        filename = self.cfg.download_url.split("/")[-1]
        file_path = self.data_dir / filename

        # download file from url
        logger.info(f"Downloading {self.cfg.dataset_id}")
        response = requests.get(self.cfg.download_url)
        with open(file_path, "wb") as f:
            f.write(response.content)

    def save_results(self, results: result_type) -> None:
        return None

    def load_results(self) -> result_type:
        return None

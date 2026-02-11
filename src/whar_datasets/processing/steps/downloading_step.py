from pathlib import Path
from typing import Any, Set, TypeAlias

import requests

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.steps.processing_step import ProcessingStep
from whar_datasets.processing.utils.extracting import extract
from whar_datasets.utils.logging import logger

base_type: TypeAlias = Any
result_type: TypeAlias = None


class DownloadingStep(ProcessingStep):
    def __init__(
        self,
        cfg: WHARConfig,
        datasets_dir: Path,
        dataset_dir: Path,
        raw_dir: Path,
    ):
        super().__init__(cfg, raw_dir)

        self.datasets_dir = datasets_dir
        self.dataset_dir = dataset_dir
        self.download_dir = raw_dir

        self.hash_name: str = "download_hash"
        self.relevant_cfg_keys: Set[str] = {
            "dataset_id",
            "download_url",
            "datasets_dir",
        }

    def get_base(self) -> base_type:
        return None

    def check_initial_format(self, base: base_type) -> bool:
        return True

    def compute_results(self, base: base_type) -> result_type:
        raw_urls = self.cfg.download_url

        urls = [raw_urls] if isinstance(raw_urls, str) else raw_urls

        for url in urls:
            filename = url.split("/")[-1]
            file_path = self.download_dir / filename

            logger.info(f"Downloading {filename} (Dataset: {self.cfg.dataset_id})")

            response = requests.get(url)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Extracting {filename}")
            extract(file_path, self.download_dir)

    def save_results(self, results: result_type) -> None:
        return None

    def load_results(self) -> result_type:
        return None

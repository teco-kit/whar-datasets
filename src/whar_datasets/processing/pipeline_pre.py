from pathlib import Path
from typing import List, Tuple

import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.pipeline import ProcessingPipeline
from whar_datasets.processing.steps.downloading_step import DownloadingStep
from whar_datasets.processing.steps.extracting_step import ExtractingStep
from whar_datasets.processing.steps.parsing_step import ParsingStep
from whar_datasets.processing.steps.windowing_step import WindowingStep


class PreProcessingPipeline(ProcessingPipeline):
    def __init__(self, cfg: WHARConfig):
        # define directories
        self.datasets_dir = Path(cfg.datasets_dir)
        self.dataset_dir = self.datasets_dir / cfg.dataset_id
        self.data_dir = self.dataset_dir / "data"
        self.metadata_dir = self.dataset_dir / "metadata"
        self.sessions_dir = self.dataset_dir / "sessions"
        self.windows_dir = self.dataset_dir / "windows"

        # create directories
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.windows_dir.mkdir(parents=True, exist_ok=True)

        # Create gitignore file if it doesn't exist
        gitignore_path = self.datasets_dir / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, "w") as f:
                f.write("*")

        # define steos
        self.downloading_step = DownloadingStep(
            cfg=cfg,
            datasets_dir=self.datasets_dir,
            dataset_dir=self.dataset_dir,
            data_dir=self.data_dir,
        )

        self.extracting_step = ExtractingStep(
            cfg=cfg,
            data_dir=self.data_dir,
            dependent_on=[self.downloading_step],
        )

        self.parsing_step = ParsingStep(
            cfg=cfg,
            data_dir=self.data_dir,
            metadata_dir=self.metadata_dir,
            sessions_dir=self.sessions_dir,
            dependent_on=[self.extracting_step],
        )

        self.windowing_step = WindowingStep(
            cfg=cfg,
            metadata_dir=self.metadata_dir,
            sessions_dir=self.sessions_dir,
            windows_dir=self.windows_dir,
            dependent_on=[self.parsing_step],
        )

        super().__init__(
            steps=[
                self.downloading_step,
                self.extracting_step,
                self.parsing_step,
                self.windowing_step,
            ]
        )

    def run(
        self, force_recompute: bool | List[bool] | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        super().run(force_recompute)

        activity_df, session_df, window_df, _ = self.windowing_step.load_results()

        return activity_df, session_df, window_df

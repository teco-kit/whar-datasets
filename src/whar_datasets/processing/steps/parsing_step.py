from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, TypeAlias

import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.steps.abstract_step import AbstractStep
from whar_datasets.processing.utils.caching import cache_common_format
from whar_datasets.utils.loading import load_activity_df, load_session_df, load_sessions
from whar_datasets.utils.logging import logger

base_type: TypeAlias = Any
result_type: TypeAlias = Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[int, pd.DataFrame],
]


class ParsingStep(AbstractStep):
    def __init__(
        self,
        cfg: WHARConfig,
        data_dir: Path,
        metadata_dir: Path,
        sessions_dir: Path,
        dependent_on: List[AbstractStep],
    ):
        super().__init__(cfg, sessions_dir, dependent_on)

        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.sessions_dir = sessions_dir

        self.hash_name: str = "parsing_hash"
        self.relevant_cfg_keys: Set[str] = {"dataset_id", "activity_id_col"}

    def get_base(self) -> base_type:
        return None

    def check_initial_format(self, base: base_type) -> bool:
        logger.info("Checking extracted data")

        if not self.data_dir.exists():
            logger.warning(f"Data directory not found at '{self.data_dir}'.")
            return False

        logger.info("Data directory exists")
        return True

    def compute_results(self, base: base_type) -> result_type:
        logger.info("Parsing to common format")

        activity_df, session_df, sessions = self.cfg.parse(
            str(self.data_dir), self.cfg.activity_id_col
        )

        return activity_df, session_df, sessions

    def save_results(self, results: result_type) -> None:
        activity_df, session_df, sessions = results

        logger.info("Saving common format")

        cache_common_format(
            self.metadata_dir, self.sessions_dir, activity_df, session_df, sessions
        )

    def load_results(self) -> result_type:
        logger.info("Loading common format")

        session_df = load_session_df(self.metadata_dir)
        activity_df = load_activity_df(self.metadata_dir)
        sessions = load_sessions(self.sessions_dir, session_df)

        return activity_df, session_df, sessions

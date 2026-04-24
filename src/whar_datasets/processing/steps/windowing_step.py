from pathlib import Path
from typing import Dict, List, Set, Tuple, TypeAlias

import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.pipeline import AbstractStep
from whar_datasets.processing.utils.caching import cache_window_df, cache_windows
from whar_datasets.processing.utils.selecting import select_activities
from whar_datasets.processing.utils.sessions import (
    process_sessions_para,
    process_sessions_seq,
)
from whar_datasets.processing.utils.validation import validate_common_format
from whar_datasets.utils.loading import (
    load_activity_df,
    load_session_df,
    load_window_df,
)
from whar_datasets.utils.logging import logger

InputT: TypeAlias = Tuple[pd.DataFrame, pd.DataFrame]
OutputT: TypeAlias = Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, pd.DataFrame],
]


class WindowingStep(AbstractStep[InputT, OutputT]):
    """Create fixed-length windows from session-level recordings."""

    def __init__(
        self,
        cfg: WHARConfig,
        metadata_dir: Path,
        sessions_dir: Path,
        windows_dir: Path,
        dependent_on: List[AbstractStep],
    ):
        super().__init__(cfg, windows_dir, dependent_on)

        self.metadata_dir = metadata_dir
        self.sessions_dir = sessions_dir
        self.windows_dir = windows_dir

        self.hash_name: str = "windowing_hash"
        self.relevant_cfg_keys: Set[str] = {
            "sampling_freq",
            "selected_activities",
            "selected_channels",
            "window_time",
            "window_overlap",
            "resampling_freq",
        }

    def load_input(self) -> InputT:
        activity_df = load_activity_df(self.metadata_dir)
        session_df = load_session_df(self.metadata_dir)
        return activity_df, session_df

    def validate_input(self, step_input: InputT) -> bool:
        activity_df, session_df = step_input
        return validate_common_format(
            self.cfg, self.sessions_dir, activity_df, session_df
        )

    def build_output(self, step_input: InputT) -> OutputT:
        activity_df, session_df = step_input

        logger.info("Compute windowing")

        # Select and remap activity labels to contiguous ids for downstream training.
        activity_df, session_df = select_activities(
            activity_df,
            session_df,
            self.cfg.selected_activities or [],
        )

        # generate windowing
        process_sessions = (
            process_sessions_para if self.cfg.parallelize else process_sessions_seq
        )

        window_df, windows = process_sessions(self.cfg, self.sessions_dir, session_df)

        return activity_df, session_df, window_df, windows

    def save_output(self, step_output: OutputT) -> None:
        logger.info("Saving windowing")

        _, _, window_df, windows = step_output
        cache_window_df(self.metadata_dir, window_df)
        cache_windows(self.windows_dir, window_df, windows)

    def load_output(self) -> OutputT:
        logger.info("Loading windowing")

        activity_df = load_activity_df(self.metadata_dir)
        session_df = load_session_df(self.metadata_dir)
        window_df = load_window_df(self.metadata_dir)
        activity_df, session_df = select_activities(
            activity_df,
            session_df,
            self.cfg.selected_activities or [],
        )

        df = activity_df["activity_id"]
        logger.info(f"activity_ids from {df.min()} to {df.max()}")

        df = session_df["subject_id"]
        logger.info(f"subject_ids from {df.min()} to {df.max()}")

        return activity_df, session_df, window_df, {}

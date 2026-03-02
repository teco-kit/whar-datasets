from pathlib import Path
from typing import Dict, List, Set, TypeAlias

import numpy as np
import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.steps.abstract_step import AbstractStep
from whar_datasets.processing.utils.caching import cache_samples
from whar_datasets.processing.utils.normalization import get_norm_params
from whar_datasets.processing.utils.preparation import (
    prepare_windows_para,
    prepare_windows_seq,
)
from whar_datasets.utils.loading import load_samples, load_windows
from whar_datasets.utils.logging import logger

base_type: TypeAlias = Dict[str, pd.DataFrame]
result_type: TypeAlias = Dict[str, List[np.ndarray]]


class SamplingStep(AbstractStep):
    def __init__(
        self,
        cfg: WHARConfig,
        metadata_dir: Path,
        samples_dir: Path,
        windows_dir: Path,
        window_df: pd.DataFrame,
        indices: List[int],
        dependent_on: List[AbstractStep],
    ):
        super().__init__(cfg, samples_dir, dependent_on)

        self.metadata_dir = metadata_dir
        self.samples_dir = samples_dir
        self.windows_dir = windows_dir
        self.window_df = window_df
        self.indices = indices

        self.hash_name: str = "sampling_hash"
        self.relevant_cfg_keys: Set[str] = {
            "given_fold",
            "fold_groups",
            "val_percentage",
            "normalization",
            "transform",
        }
        self.relevant_values = [str(i) for i in self.indices]

    def get_base(self) -> base_type:
        windows = load_windows(self.window_df, self.windows_dir)
        return windows

    def check_initial_format(self, base: base_type) -> bool:
        return True

    def compute_results(self, base: base_type) -> result_type:
        windows = base

        logger.info("Computing samples")

        norm_params = get_norm_params(self.cfg, self.indices, self.window_df, windows)

        prepare_windows = (
            prepare_windows_para if self.cfg.parallelize else prepare_windows_seq
        )

        samples = prepare_windows(
            self.cfg, norm_params, self.window_df, self.windows_dir
        )

        return samples

    def save_results(self, results: result_type) -> None:
        samples = results
        logger.info("Saving samples")
        cache_samples(self.samples_dir, self.window_df, samples)

    def load_results(self) -> result_type:
        logger.info("Loading samples")
        return load_samples(self.window_df, self.samples_dir)

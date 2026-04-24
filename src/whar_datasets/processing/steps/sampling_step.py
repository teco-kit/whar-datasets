import hashlib
import json
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

InputT: TypeAlias = Dict[str, pd.DataFrame]
OutputT: TypeAlias = Dict[str, List[np.ndarray]]


class SamplingStep(AbstractStep[InputT, OutputT]):
    """Materialize model-ready samples from cached window artifacts."""

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
        self.samples_root_dir = samples_dir
        self.metadata_dir = metadata_dir
        self.split_hash = self._compute_split_hash(indices)
        self.samples_dir = (
            self.samples_root_dir / self.split_hash
            if cfg.cache_each_split
            else self.samples_root_dir
        )
        self.windows_dir = windows_dir
        self.window_df = window_df
        self.indices = indices

        super().__init__(cfg, self.samples_dir, dependent_on)

        self.hash_name: str = "sampling_hash"
        self.relevant_cfg_keys: Set[str] = {
            "given_fold",
            "fold_groups",
            "val_percentage",
            "normalization",
            "transform",
            "cache_each_split",
        }
        self.relevant_values = [str(i) for i in self.indices]

    def load_input(self) -> InputT:
        windows = load_windows(self.windows_dir)
        return windows

    def validate_input(self, step_input: InputT) -> bool:
        return True

    def build_output(self, step_input: InputT) -> OutputT:
        windows = step_input

        logger.info("Computing samples")

        norm_params = get_norm_params(self.cfg, self.indices, self.window_df, windows)

        prepare_windows = (
            prepare_windows_para if self.cfg.parallelize else prepare_windows_seq
        )

        samples = prepare_windows(
            self.cfg, norm_params, self.window_df, self.windows_dir
        )

        return samples

    def save_output(self, step_output: OutputT) -> None:
        logger.info("Saving samples")

        samples = step_output
        cache_samples(self.samples_dir, self.window_df, samples)

    def load_output(self) -> OutputT:
        logger.info("Loading samples")

        return load_samples(self.samples_dir)

    def _compute_split_hash(self, indices: List[int]) -> str:
        # Hash normalized train indices so identical splits map to one cache directory.
        payload = json.dumps(sorted(indices), separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split


class Splitter(ABC):
    """Base interface for split generation strategies."""

    def __init__(self, cfg: WHARConfig):
        self.val_percentage = cfg.val_percentage
        self.rng = np.random.RandomState(cfg.seed)

    @abstractmethod
    def get_splits(
        self, session_df: pd.DataFrame, window_df: pd.DataFrame
    ) -> List[Split]:
        """Return train/validation/test splits for the provided metadata."""
        pass

    def _get_train_val_indices(self, indices: List[int]) -> Tuple[List[int], List[int]]:
        """Split candidate indices into train/validation subsets."""
        n_train = len(indices)
        n_val = int(n_train * self.val_percentage)

        shuffled_indices: List[int] = self.rng.permutation(indices).tolist()

        val_indices = shuffled_indices[:n_val]
        train_indices = shuffled_indices[n_val:]

        return train_indices, val_indices

    def _check_indices_overlap(
        self, train_indices: List[int], val_indices: List[int], test_indices: List[int]
    ) -> bool:
        """Return ``True`` when any split pair shares at least one index."""
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)

        if train_set.intersection(val_set):
            return True
        if train_set.intersection(test_set):
            return True
        if val_set.intersection(test_set):
            return True

        return False

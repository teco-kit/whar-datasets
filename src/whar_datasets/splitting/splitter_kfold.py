from typing import List

import numpy as np
import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split
from whar_datasets.splitting.splitter import Splitter


class KFoldSplitter(Splitter):
    """Random K-fold split over window indices."""

    def __init__(self, cfg: WHARConfig):
        super().__init__(cfg)

        if cfg.num_folds is None:
            raise ValueError("num_folds must be configured for K-fold splitting.")

        self.n_folds = cfg.num_folds

    def get_splits(
        self, session_df: pd.DataFrame, window_df: pd.DataFrame
    ) -> List[Split]:
        """Create ``n_folds`` train/val/test splits over shuffled window indices."""
        indices = list(window_df.index)
        self.rng.shuffle(indices)

        folds = np.array_split(indices, self.n_folds)

        self._reset_split_diagnostics()
        splits: List[Split] = []
        for fold_idx in range(self.n_folds):
            test_indices = folds[fold_idx].tolist()
            train_val_indices = [
                idx for i, fold in enumerate(folds) if i != fold_idx for idx in fold
            ]

            train_indices, val_indices = self._get_train_val_indices(
                train_val_indices, window_df, emit_diagnostics=False
            )

            split = Split(
                identifier=f"fold_{fold_idx}",
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )

            if self._check_indices_overlap(
                split.train_indices, split.val_indices, split.test_indices
            ):
                raise RuntimeError("Overlap detected in split indices.")

            splits.append(split)

        self._log_split_diagnostics()
        return splits

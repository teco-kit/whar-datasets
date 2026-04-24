from typing import List

import numpy as np
import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split
from whar_datasets.splitting.splitter import Splitter


class KFoldSplitter(Splitter):
    def __init__(self, cfg: WHARConfig):
        super().__init__(cfg)

        assert cfg.num_folds is not None

        self.n_folds = cfg.num_folds

    def get_splits(
        self, session_df: pd.DataFrame, window_df: pd.DataFrame
    ) -> List[Split]:
        indices = list(window_df.index)
        self.rng.shuffle(indices)

        folds = np.array_split(indices, self.n_folds)

        splits: List[Split] = []
        for fold_idx in range(self.n_folds):
            test_indices = folds[fold_idx].tolist()
            train_val_indices = [
                idx for i, fold in enumerate(folds) if i != fold_idx for idx in fold
            ]

            train_indices, val_indices = self._get_train_val_indices(train_val_indices)

            split = Split(
                identifier=f"fold_{fold_idx}",
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )

            assert not self._check_indices_overlap(
                split.train_indices, split.val_indices, split.test_indices
            ), "Overlap detected in indices!"

            splits.append(split)

        return splits

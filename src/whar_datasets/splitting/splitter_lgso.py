from typing import List

import numpy as np
import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split
from whar_datasets.splitting.splitter import Splitter


class LGSOSplitter(Splitter):
    def __init__(self, cfg: WHARConfig, subject_ids: List[int] | None = None):
        super().__init__(cfg)

        assert cfg.num_subject_groups is not None

        self.num_subject_groups = cfg.num_subject_groups
        self.subject_ids = subject_ids

    def get_splits(
        self,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
    ) -> List[Split]:
        # 1. Identify unique subjects
        unique_subjects = self.subject_ids or session_df["subject_id"].unique().tolist()
        unique_subjects = np.array(unique_subjects)
        self.rng.shuffle(unique_subjects)

        # 2. Split subjects into N groups
        groups = np.array_split(unique_subjects, self.num_subject_groups)

        splits: List[Split] = []

        for i, test_subjects in enumerate(groups):
            # 3. Identify sessions belonging to the current group of subjects
            test_sessions = session_df[session_df["subject_id"].isin(test_subjects)][
                "session_id"
            ].tolist()

            # 4. Filter window indices
            test_indices = window_df[
                window_df["session_id"].isin(test_sessions)
            ].index.tolist()

            train_val_indices = window_df[
                ~window_df["session_id"].isin(test_sessions)
            ].index.tolist()

            # 5. Handle internal train/val split logic
            train_indices, val_indices = self._get_train_val_indices(train_val_indices)

            split = Split(
                identifier=f"group_{i}",
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )

            # Safety check
            assert not self._check_indices_overlap(
                split.train_indices, split.val_indices, split.test_indices
            ), f"Overlap detected in group {i} indices!"

            splits.append(split)

        return splits

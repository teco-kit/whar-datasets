from typing import List

import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split
from whar_datasets.splitting.splitter import Splitter


class LKSOSplitter(Splitter):
    """K-fold splitter that operates on subject groups.

    Subjects are partitioned into ``k`` folds. For each fold, all windows from
    the subjects in that fold are used as test data, and windows from all
    remaining subjects are used for train/validation.

    The grouping is deterministic when the subject IDs stay the same.
    """

    def __init__(self, cfg: WHARConfig, subject_ids: List[int] | None = None):
        super().__init__(cfg)

        assert cfg.num_folds is not None

        self.n_folds = cfg.num_folds
        self.subject_ids = subject_ids

    def get_splits(
        self,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
    ) -> List[Split]:
        # 1. Identify unique subjects and sort them deterministically
        unique_subjects = self.subject_ids or session_df["subject_id"].unique().tolist()
        unique_subjects = sorted(unique_subjects)

        # 2. Determine effective number of folds (cannot exceed #subjects)
        n_subjects = len(unique_subjects)
        n_folds = min(self.n_folds, n_subjects)

        # 3. Assign each subject to a fold in round-robin fashion
        # Example: 8 subjects, k=5 -> assignments (1->0, 2->1, 3->2, 4->3, 5->4, 6->0, 7->1, 8->2)
        subject_to_fold = {
            subj_id: idx % n_folds for idx, subj_id in enumerate(unique_subjects)
        }

        splits: List[Split] = []

        for fold_idx in range(n_folds):
            # subjects assigned to this fold
            test_subjects = [
                subj_id
                for subj_id, f_idx in subject_to_fold.items()
                if f_idx == fold_idx
            ]

            # 4. Identify sessions belonging to the current group of subjects
            test_sessions = session_df[session_df["subject_id"].isin(test_subjects)][
                "session_id"
            ].tolist()

            # 5. Filter window indices
            test_indices = window_df[
                window_df["session_id"].isin(test_sessions)
            ].index.tolist()

            train_val_indices = window_df[
                ~window_df["session_id"].isin(test_sessions)
            ].index.tolist()

            # 6. Internal train/val split
            train_indices, val_indices = self._get_train_val_indices(train_val_indices)

            split = Split(
                identifier=f"group_kfold_{fold_idx}",
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )

            # Safety check: ensure no overlaps between index sets
            assert not self._check_indices_overlap(
                split.train_indices, split.val_indices, split.test_indices
            ), f"Overlap detected in group_kfold_{fold_idx} indices!"

            splits.append(split)

        return splits

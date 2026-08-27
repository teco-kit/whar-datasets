from typing import List

import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split
from whar_datasets.splitting.splitter import Splitter


class LKSOSplitter(Splitter):
    """Seeded K-fold splitter that operates on subject groups.

    Subjects are partitioned into ``k`` folds. For each fold, all windows from
    the subjects in that fold are used as test data, and windows from all
    remaining subjects are used for train/validation.

    Subjects are shuffled reproducibly by default. Set ``shuffle_subject=False``
    for stable sorted round-robin grouping independent of ``cfg.seed``.
    """

    def __init__(self, cfg: WHARConfig, subject_ids: List[int] | None = None):
        super().__init__(cfg)

        if cfg.num_folds is None:
            raise ValueError("num_folds must be configured for LKSO.")

        self.n_folds = cfg.num_folds
        self.shuffle_subject = cfg.shuffle_subject
        self.subject_ids = subject_ids

    def get_splits(
        self,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
    ) -> List[Split]:
        # 1. Start from a canonical order, then optionally apply seeded shuffling.
        unique_subjects = self.subject_ids or session_df["subject_id"].unique().tolist()
        unique_subjects = sorted(unique_subjects)
        if self.shuffle_subject:
            self.rng.shuffle(unique_subjects)

        # 2. Determine effective number of folds (cannot exceed #subjects)
        n_subjects = len(unique_subjects)
        n_folds = min(self.n_folds, n_subjects)

        # 3. Assign each subject to a balanced fold in round-robin fashion.
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
            train_indices, val_indices = self._get_train_val_indices(
                train_val_indices, window_df
            )

            split = Split(
                identifier=f"group_kfold_{fold_idx}",
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )

            # Safety check: ensure no overlaps between index sets
            if self._check_indices_overlap(
                split.train_indices, split.val_indices, split.test_indices
            ):
                raise RuntimeError(
                    f"Overlap detected in group_kfold_{fold_idx} indices."
                )

            splits.append(split)

        return splits

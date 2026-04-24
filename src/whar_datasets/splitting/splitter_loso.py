from typing import List

import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split
from whar_datasets.splitting.splitter import Splitter


class LOSOSplitter(Splitter):
    """Leave-one-subject-out splitting strategy."""

    def __init__(self, cfg: WHARConfig, subject_ids: List[int] | None = None):
        super().__init__(cfg)

        self.subject_ids = subject_ids

    def get_splits(
        self,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
    ) -> List[Split]:
        """Create one split per held-out subject."""
        subject_ids = self.subject_ids or session_df["subject_id"].unique().tolist()

        splits: List[Split] = []

        for s in subject_ids:
            test_sessions = session_df[session_df["subject_id"] == s][
                "session_id"
            ].tolist()

            test_indices = window_df[
                window_df["session_id"].isin(test_sessions)
            ].index.tolist()

            train_val_indices = window_df[
                ~window_df["session_id"].isin(test_sessions)
            ].index.tolist()

            train_indices, val_indices = self._get_train_val_indices(train_val_indices)

            split = Split(
                identifier=f"subject_{s}",
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )

            assert not self._check_indices_overlap(
                split.train_indices, split.val_indices, split.test_indices
            ), "Overlap detected in indices!"

            splits.append(split)

        return splits

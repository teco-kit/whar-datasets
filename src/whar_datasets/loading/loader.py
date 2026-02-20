from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from whar_datasets.loading.weighting import compute_class_weights
from whar_datasets.utils.loading import load_sample


class Loader:
    """Dataset loader with precomputed metadata for fast sampling/filtering."""

    def __init__(
        self,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
        samples_dir: Path,
        samples_dict: Dict[str, List[np.ndarray]] | None = None,
    ) -> None:
        self.session_df = session_df
        self.window_df = window_df
        self.samples_dir = samples_dir
        self.samples_dict = samples_dict

        # Canonical window index labels from window_df.
        self._window_indices_np = self.window_df.index.to_numpy()
        self._window_indices = self._window_indices_np.tolist()

        # session_id -> (subject_id, activity_id) mapping used to build window-level arrays.
        session_meta = self.session_df.drop_duplicates("session_id").set_index(
            "session_id"
        )
        # Per-window arrays where position i corresponds to row i in window_df.
        self._session_id_by_pos = self.window_df["session_id"].to_numpy(dtype=np.int64)
        self._window_id_by_pos = self.window_df["window_id"].to_numpy()

        subject_series = session_meta["subject_id"].reindex(self._session_id_by_pos)
        activity_series = session_meta["activity_id"].reindex(self._session_id_by_pos)
        assert not subject_series.isna().any(), (
            "Missing subject_id for one or more session_ids."
        )
        assert not activity_series.isna().any(), (
            "Missing activity_id for one or more session_ids."
        )

        self._subject_by_pos = subject_series.to_numpy(dtype=np.int64)
        self._activity_by_pos = activity_series.to_numpy(dtype=np.int64)

        # Fast filter lookups when caller requests global filtering (indices=None).
        self._indices_by_subject = {
            int(subject): self._window_indices_np[self._subject_by_pos == subject]
            for subject in np.unique(self._subject_by_pos)
        }
        self._indices_by_activity = {
            int(activity): self._window_indices_np[self._activity_by_pos == activity]
            for activity in np.unique(self._activity_by_pos)
        }
        self._indices_by_subject_activity = {
            (int(subject), int(activity)): self._window_indices_np[
                (self._subject_by_pos == subject) & (self._activity_by_pos == activity)
            ]
            for subject, activity in np.unique(
                np.column_stack((self._subject_by_pos, self._activity_by_pos)), axis=0
            )
        }

        self._sample_loader: Callable[[str], List[np.ndarray]]
        if self.samples_dict is None:
            # Cache repeated disk reads for frequently sampled windows.
            self._sample_loader = lru_cache(maxsize=4096)(self._load_sample_from_disk)
        else:
            self._sample_loader = self._load_sample_from_dict

    def __len__(self) -> int:
        return len(self.window_df)

    def sample_items(
        self,
        batch_size: int,
        indices: List[int] | None = None,
        activity_id: int | None = None,
        subject_id: int | None = None,
        seed: int | None = None,
    ) -> Tuple[List[int], List[int], List[List[np.ndarray]]]:
        """Sample a batch with replacement, optionally filtered by subject/activity."""
        inds = self.filter_indices(indices, subject_id, activity_id)
        assert len(inds) > 0, "No samples found for the given filters."

        rng = np.random.default_rng(seed)
        sampled_inds = rng.choice(np.asarray(inds), size=batch_size, replace=True)
        sampled_inds_list = sampled_inds.tolist()

        sampled_pos = self.window_df.index.get_indexer(pd.Index(sampled_inds))
        assert np.all(sampled_pos >= 0), (
            "One or more sampled indices are missing in window_df."
        )

        activity_labels = self._activity_by_pos[sampled_pos].astype(int).tolist()
        subject_labels = self._subject_by_pos[sampled_pos].astype(int).tolist()
        samples = [self.get_sample(int(idx)) for idx in sampled_inds_list]

        return activity_labels, subject_labels, samples

    def get_item(self, index: int) -> Tuple[int, int, List[np.ndarray]]:
        """Return (activity_label, subject_label, sample) for a window index label."""
        pos = self._get_pos(index)

        activity_label = int(self._activity_by_pos[pos])
        subject_label = int(self._subject_by_pos[pos])
        sample = self.get_sample(index)

        return activity_label, subject_label, sample

    def get_activity_label(self, index: int) -> int:
        """Return the activity label for a window index label."""
        pos = self._get_pos(index)
        return int(self._activity_by_pos[pos])

    def get_subject_label(self, index: int) -> int:
        """Return the subject label for a window index label."""
        pos = self._get_pos(index)
        return int(self._subject_by_pos[pos])

    def get_sample(self, index: int) -> List[np.ndarray]:
        """Return sample data for a window index label."""
        pos = self._get_pos(index)
        window_id = self._window_id_by_pos[pos]
        assert isinstance(window_id, str)
        sample = self._sample_loader(window_id)

        return sample

    def _load_sample_from_disk(self, window_id: str) -> List[np.ndarray]:
        """Load sample by window_id from disk."""
        return load_sample(self.samples_dir, window_id)

    def _load_sample_from_dict(self, window_id: str) -> List[np.ndarray]:
        """Load sample by window_id from in-memory dictionary."""
        assert self.samples_dict is not None
        return self.samples_dict[window_id]

    def _get_pos(self, index: int) -> int:
        """Map external window index label to its positional row offset."""
        pos = self.window_df.index.get_loc(index)
        assert isinstance(pos, (int, np.integer)), "Expected a unique window index."
        return int(pos)

    def filter_indices(
        self,
        indices: List[int] | None = None,
        subject_id: int | None = None,
        activity_id: int | None = None,
    ) -> List[int]:
        """Filter indices by subject/activity using cached or vectorized paths."""
        if indices is None:
            if subject_id is not None and activity_id is not None:
                return self._indices_by_subject_activity.get(
                    (subject_id, activity_id),
                    np.array([], dtype=self._window_indices_np.dtype),
                ).tolist()
            if subject_id is not None:
                return self._indices_by_subject.get(
                    subject_id, np.array([], dtype=self._window_indices_np.dtype)
                ).tolist()
            if activity_id is not None:
                return self._indices_by_activity.get(
                    activity_id, np.array([], dtype=self._window_indices_np.dtype)
                ).tolist()
            return self._window_indices.copy()

        if not indices:
            return []

        inds_np = np.asarray(indices)
        pos = self.window_df.index.get_indexer(pd.Index(inds_np))
        assert np.all(pos >= 0), (
            "One or more provided indices are missing in window_df."
        )

        mask = np.ones(len(pos), dtype=bool)
        if subject_id is not None:
            mask &= self._subject_by_pos[pos] == subject_id
        if activity_id is not None:
            mask &= self._activity_by_pos[pos] == activity_id

        return inds_np[mask].tolist()

    def plot_indices_statistics(self, indices: List[int] | None = None) -> None:
        indices = indices or self._window_indices.copy()

        subset = self.window_df.loc[indices]
        merged = subset.merge(
            self.session_df[["session_id", "subject_id", "activity_id"]],
            on="session_id",
            how="left",
        )
        counts = (
            merged.groupby(["subject_id", "activity_id"])
            .size()
            .reset_index(name="num_samples")
        )

        # pivot for easier plotting (subjects on x, activities as groups)
        pivot_table = counts.pivot(
            index="subject_id", columns="activity_id", values="num_samples"
        ).fillna(0)

        # plot
        pivot_table.plot(kind="bar", stacked=False, figsize=(12, 4))

        plt.title("number of samples per subject and activity")
        plt.xlabel("subject_id")
        plt.ylabel("number of samples")
        plt.legend(title="activity_id")
        plt.tight_layout()
        plt.show()

    def get_class_weights(self, indices: List[int] | None = None) -> dict:
        indices = indices or self._window_indices.copy()

        return compute_class_weights(
            self.session_df,
            self.window_df.loc[indices],
        )

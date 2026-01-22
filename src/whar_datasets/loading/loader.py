from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from whar_datasets.loading.weighting import compute_class_weights
from whar_datasets.utils.loading import load_sample
import ast


class Loader:
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

    def __len__(self) -> int:
        return len(self.window_df)

    def sample_items(
        self,
        batch_size: int,
        indices: List[int] | None = None,
        activity_id: int | None = None,
        subject_id: int | None = None,
        seed: int | None = None,
    ) -> Tuple[List[List[int]], List[int], List[List[np.ndarray]]]:
        inds = indices or list(range(len(self)))

        inds = self.filter_indices(inds, subject_id, activity_id)
        assert len(inds) > 0, "No samples found for the given filters."

        if seed is not None:
            np.random.seed(seed)

        inds = np.random.choice(inds, size=batch_size, replace=True).tolist()

        items = [self.get_item(idx) for idx in inds]

        # TODO: zip angucken
        activity_labels, subject_labels, samples = zip(*items)

        return list(activity_labels), list(subject_labels), list(samples)

    def get_item(self, index: int) -> Tuple[int, int, List[np.ndarray]]:
        activity_label = self.get_activity_label(index)
        subject_label = self.get_subject_label(index)
        sample = self.get_sample(index)

        return activity_label, subject_label, sample

    def get_activity_label(self, index: int) -> int:
        session_id = int(self.window_df.at[index, "session_id"])
        assert isinstance(session_id, int)

        activity_label = self.session_df.loc[
            self.session_df["session_id"] == session_id, "activity_id"
        ].item()
        assert isinstance(activity_label, list)

        return activity_label

    def get_subject_label(self, index: int) -> int:
        session_id = int(self.window_df.at[index, "session_id"])
        assert isinstance(session_id, int)

        subject_label = self.session_df.loc[
            self.session_df["session_id"] == session_id, "subject_id"
        ].item()
        assert isinstance(subject_label, int)

        return subject_label

    def get_sample(self, index: int) -> List[np.ndarray]:
        window_id = self.window_df.at[index, "window_id"]
        assert isinstance(window_id, str)

        sample = (
            self.samples_dict[window_id]
            if self.samples_dict is not None
            else load_sample(self.samples_dir, window_id)
        )

        return sample

    def filter_indices(
        self,
        indices: List[int] | None = None,
        subject_id: int | None = None,
        activity_id: int | None = None,
    ):
        indices = indices or list(range(len(self)))

        if subject_id is not None:
            subset = self.window_df.iloc[indices].copy()
            subset["orig_index"] = subset.index

            # Merge with session_df to get subject_id info
            merged = subset.merge(
                self.session_df[["session_id", "subject_id"]],
                on="session_id",
                how="left",
            )

            # Filter by subject_id
            filtered = merged[merged["subject_id"] == subject_id]
            indices = filtered["orig_index"].to_list()

        if activity_id is not None:
            subset = self.window_df.iloc[indices].copy()
            subset["orig_index"] = subset.index

            # Merge with session_df to get activity_id info
            merged = subset.merge(
                self.session_df[["session_id", "activity_id"]],
                on="session_id",
                how="left",
            )

            # filter by activity_list containing activity_id
            exploded = merged.explode("activity_id")
            filtered = exploded[exploded["activity_id"] == activity_id]
            indices = filtered["orig_index"].unique().tolist()

        return indices

    def plot_indices_statistics(self, indices: List[int] | None = None) -> None:
        indices = indices or list(range(len(self)))

        subset = self.window_df.iloc[indices]
        merged = subset.merge(
            self.session_df[["session_id", "subject_id", "activity_id"]],
            on="session_id",
            how="left",
        )

        counts = (
            merged.explode("activity_id")
            .groupby(["subject_id", "activity_id"])
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
        indices = indices or list(range(len(self)))

        return compute_class_weights(
            self.session_df,
            self.window_df.iloc[indices],
        )

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split


class Splitter(ABC):
    def __init__(self, cfg: WHARConfig):
        self.val_percentage = cfg.val_percentage
        self.rng = np.random.RandomState(cfg.seed)

    @abstractmethod
    def get_splits(
        self, session_df: pd.DataFrame, window_df: pd.DataFrame
    ) -> List[Split]:
        pass

    def _get_train_val_indices(self, indices: List[int]) -> Tuple[List[int], List[int]]:
        n_train = len(indices)
        n_val = int(n_train * self.val_percentage)

        shuffled_indices: List[int] = self.rng.permutation(indices).tolist()

        val_indices = shuffled_indices[:n_val]
        train_indices = shuffled_indices[n_val:]

        return train_indices, val_indices

    def _check_indices_overlap(
        self, train_indices: List[int], val_indices: List[int], test_indices: List[int]
    ) -> bool:
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


# import math
# import random
# from typing import List, Tuple

# import pandas as pd
# from whar_datasets.config.config import WHARConfig
# from whar_datasets.core.utils.logging import logger


# def get_split_train_test(
#     cfg: WHARConfig,
#     session_df: pd.DataFrame,
#     window_df: pd.DataFrame,
#     split_group_index: int,
# ) -> Tuple[List[int], List[int]]:
#     groups = cfg.split_groups
#     assert groups is not None
#     assert split_group_index < len(groups)

#     # get subject_ids for both groups
#     test_subj_ids = groups[split_group_index]
#     train_subj_ids = [
#         subj_id
#         for i, group in enumerate(groups)
#         if i != split_group_index
#         for subj_id in group
#     ]

#     # get window indices for all groups
#     train_indices = get_window_indices(
#         session_df, window_df, train_subj_ids
#     )
#     test_indices = get_window_indices(session_df, window_df, test_subj_ids)

#     logger.info(f"train: {len(train_indices)} | test: {len(test_indices)}")

#     return train_indices, test_indices


# def split_indices(
#     cfg: WHARConfig, indices: List[int], percentages: Tuple[float, ...]
# ) -> Tuple[List[int], ...]:
#     assert math.isclose(sum(percentages), 1.0)

#     # shuffle so subjects are not in order
#     random.seed(cfg.seed)
#     shuffled_indices = indices.copy()
#     random.shuffle(shuffled_indices)

#     # calculate split sizes
#     nums = [int(p * len(indices)) for p in percentages]

#     # adjust last split to take remaining items (to handle rounding)
#     nums[-1] = len(indices) - sum(nums[:-1])

#     # create splits
#     split = []
#     start = 0
#     for n in nums:
#         split.append(shuffled_indices[start : start + n])
#         start += n

#     return tuple(split)


# def get_split_train_val_test(
#     cfg: WHARConfig,
#     session_df: pd.DataFrame,
#     window_df: pd.DataFrame,
#     split_group_index: int | None = None,
# ) -> Tuple[List[int], List[int], List[int]]:
#     # if no split group is specified, use given split
#     if split_group_index is None:
#         assert cfg.given_split is not None

#         # get subject_ids for each group
#         train_subj_ids, test_subj_ids = cfg.given_split

#         # get window indices for each group
#         train_indices = get_window_indices(
#             session_df, window_df, train_subj_ids
#         )

#         test_indices = get_window_indices(
#             session_df, window_df, test_subj_ids
#         )

#         # split train into train and val
#         train_indices, val_indices = split_indices(
#             cfg, train_indices, (1 - cfg.val_percentage, cfg.val_percentage)
#         )

#     # if split group is specified, use subject cross validation
#     else:
#         groups = cfg.split_groups
#         assert groups is not None
#         assert split_group_index < len(groups)

#         # get subject_ids for both groups
#         test_subj_ids = groups[split_group_index]
#         train_subj_ids = [
#             subj_id
#             for i, group in enumerate(groups)
#             if i != split_group_index
#             for subj_id in group
#         ]

#         # get window indices for all groups
#         train_indices = get_window_indices(
#             session_df, window_df, train_subj_ids
#         )

#         test_indices = get_window_indices(
#             session_df, window_df, test_subj_ids
#         )

#         # split train into train and val
#         train_indices, val_indices = split_indices(
#             cfg, train_indices, (1 - cfg.val_percentage, cfg.val_percentage)
#         )

#     logger.info(
#         f"train: {len(train_indices)} | val: {len(val_indices)} | test: {len(test_indices)}"
#     )

#     return train_indices, val_indices, test_indices


# def get_window_indices(
#     session_df: pd.DataFrame,
#     window_df: pd.DataFrame,
#     subject_ids: List[int],
# ) -> List[int]:
#     # get session_ids of sessions with subjects in group
#     sessions = session_df[session_df["subject_id"].isin(subject_ids)]
#     session_ids = sessions["session_id"].unique()

#     # get window indices of sessions with subjects in group
#     windows = window_df[window_df["session_id"].isin(session_ids)]
#     window_indices = windows.index.to_list()

#     return window_indices

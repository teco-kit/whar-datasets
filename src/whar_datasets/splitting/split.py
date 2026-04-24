from dataclasses import dataclass
from typing import List


@dataclass
class Split:
    """Index split for one evaluation fold.

    All indices refer to rows in ``window_df``.
    """

    identifier: str
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]

from dataclasses import dataclass


@dataclass
class Split:
    """Index split for one evaluation fold.

    All indices refer to rows in ``window_df``.
    """

    identifier: str
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]

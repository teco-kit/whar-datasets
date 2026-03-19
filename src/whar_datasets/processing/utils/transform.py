from typing import Callable, List

import numpy as np

from whar_datasets.config.config import TransformType, WHARConfig
from whar_datasets.processing.transforms.dwt import signal_to_dwt_grid


def get_transform(cfg: WHARConfig) -> Callable[[np.ndarray], List[np.ndarray]]:
    transform: Callable[[np.ndarray], List[np.ndarray]]
    match cfg.transform:
        case TransformType.DWT:

            def transform_dwt(x: np.ndarray):
                grid, lengths = signal_to_dwt_grid(x)
                return [grid, np.array(lengths)]

            transform = transform_dwt
        case TransformType.STFT:
            transform = lambda x: []  # noqa: E731
        case _:
            transform = lambda x: []  # noqa: E731
    return transform

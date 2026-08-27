from typing import Callable, List

import numpy as np

from whar_datasets.config.config import TransformType, WHARConfig
from whar_datasets.processing.utils.resampling import get_effective_sampling_freq


def get_transform(cfg: WHARConfig) -> Callable[[np.ndarray], List[np.ndarray]]:
    """Build an optional feature-transform callable from config."""
    transform: Callable[[np.ndarray], List[np.ndarray]]
    match cfg.transform:
        case TransformType.DWT:
            from whar_datasets.processing.transforms.dwt import signal_to_dwt_grid

            def transform_dwt(x: np.ndarray):
                grid, lengths = signal_to_dwt_grid(x)
                return [grid, np.array(lengths)]

            transform = transform_dwt
        case TransformType.STFT:
            from whar_datasets.processing.transforms.stft import signal_to_stft
            sampling_freq = get_effective_sampling_freq(
                cfg.sampling_freq, getattr(cfg, "resampling_freq", None)
            )

            def transform_stft(x: np.ndarray):
                stft_mag, stft_phase, stft_info = signal_to_stft(
                    x, sampling_freq=sampling_freq
                )
                return [stft_mag, stft_phase, stft_info]

            transform = transform_stft
        case _:
            transform = lambda x: []  # noqa: E731
    return transform

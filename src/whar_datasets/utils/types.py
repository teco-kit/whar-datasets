from enum import Enum
from typing import Callable, Dict, Tuple, TypeAlias

import pandas as pd

Parse: TypeAlias = Callable[
    [str, str], Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]
]
# Signature for dataset parser callbacks:
# (data_dir, activity_id_column_name) -> (activity_df, session_df, sessions_by_id)


class NormType(str, Enum):
    """Supported normalization modes for post-processing."""

    STD_GLOBALLY = "std_globally"
    MIN_MAX_GLOBALLY = "min_max_globally"
    ROBUST_SCALE_GLOBALLY = "robust_scale_globally"
    STD_PER_SAMPLE = "std_per_sample"
    MIN_MAX_PER_SAMPLE = "min_max_per_sample"
    ROBUST_SCALE_PER_SAMPLE = "robust_scale_per_sample"


class TransformType(Enum):
    """Optional feature transforms applied to normalized windows."""

    DWT = "dwt"
    STFT = "stft"

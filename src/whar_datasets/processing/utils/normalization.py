from functools import partial
from typing import Callable, Dict, Hashable, List, Tuple, TypeAlias

import numpy as np
import pandas as pd

from whar_datasets.config.config import NormType, WHARConfig
from whar_datasets.utils.loading import load_window
from whar_datasets.utils.logging import logger

NormParams: TypeAlias = Tuple[
    Dict[Hashable, float], Dict[Hashable, float]
]  # Tuple[Dict[str, float], Dict[str, float]]


def _safe_denominator(values: pd.Series, eps: float = 1e-12) -> pd.Series:
    safe = pd.to_numeric(values, errors="coerce").astype("float64")
    safe = safe.replace([np.inf, -np.inf], np.nan)
    safe = safe.mask(safe.abs() <= eps)
    return safe.fillna(1.0)


def _sanitize_normalized(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def get_normalize(
    cfg: WHARConfig, norm_params: NormParams | None
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build a normalization callable from the configured normalization mode."""
    match cfg.normalization:
        case NormType.MIN_MAX_PER_SAMPLE:
            normalize = partial(min_max, norm_params=None)
        case NormType.STD_PER_SAMPLE:
            normalize = partial(standardize, norm_params=None)
        case NormType.ROBUST_SCALE_PER_SAMPLE:
            normalize = partial(robust_scale, norm_params=None)
        case NormType.MIN_MAX_GLOBALLY:
            normalize = partial(min_max, norm_params=norm_params)
        case NormType.STD_GLOBALLY:
            normalize = partial(standardize, norm_params=norm_params)
        case NormType.ROBUST_SCALE_GLOBALLY:
            normalize = partial(robust_scale, norm_params=norm_params)
        case _:
            normalize = partial(load_window)
    return normalize


def get_norm_params(
    cfg: WHARConfig,
    indices: List[int],
    window_df: pd.DataFrame,
    windows: Dict[str, pd.DataFrame],
) -> NormParams | None:
    """Compute global normalization statistics for the provided train indices."""
    logger.info("Getting normalization parameters")

    # return None if per sample normalization
    if (
        cfg.normalization == NormType.MIN_MAX_PER_SAMPLE
        or cfg.normalization == NormType.STD_PER_SAMPLE
        or cfg.normalization == NormType.ROBUST_SCALE_PER_SAMPLE
    ):
        return None

    if len(indices) == 0:
        raise ValueError(
            "Cannot compute global normalization parameters from an empty "
            "training split. This usually happens when LOSO is used with a "
            "single-subject dataset (e.g. SKODA). Use KFold splitting or a "
            "per-sample normalization mode."
        )

    # concat to single df
    windows_df = pd.concat(
        [windows[str(window_df.at[index, "window_id"])] for index in indices],
        ignore_index=True,
    )

    # get normalization params
    match cfg.normalization:
        case NormType.MIN_MAX_GLOBALLY:
            return get_min_max_params(windows_df)
        case NormType.STD_GLOBALLY:
            return get_standardize_params(windows_df)
        case NormType.ROBUST_SCALE_GLOBALLY:
            return get_robust_scale_params(windows_df)
        case _:
            return None


def get_min_max_params(df: pd.DataFrame, exclude_columns: List[str] = []) -> NormParams:
    """Compute min/max statistics used by min-max normalization."""
    cols = df.columns.difference(exclude_columns)

    # Compute min and max for each column
    min_values = df[cols].min()
    max_values = df[cols].max()

    # round to 6 decimal places
    min_values = min_values.round(6)
    max_values = max_values.round(6)

    return (min_values.to_dict(), max_values.to_dict())


def get_standardize_params(
    df: pd.DataFrame, exclude_columns: List[str] = []
) -> NormParams:
    """Compute mean/std statistics used by standardization."""
    cols = df.columns.difference(exclude_columns)

    # Compute mean and standard deviation for each column
    mean_values = df[cols].mean()
    std_values = df[cols].std()

    # round to 6 decimal places
    mean_values = mean_values.round(6)
    std_values = std_values.round(6)

    return (mean_values.to_dict(), std_values.to_dict())


def get_robust_scale_params(
    df: pd.DataFrame, exclude_columns: List[str] = []
) -> NormParams:
    """Compute median/IQR statistics used by robust scaling."""
    cols = df.columns.difference(exclude_columns)

    # Compute median and IQR (q3 - q1) for each column
    median_values = df[cols].median()
    iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)

    # round to 6 decimal places
    median_values = median_values.round(6)
    iqr = iqr.round(6)

    return (median_values.to_dict(), iqr.to_dict())


def min_max(
    df: pd.DataFrame, norm_params: NormParams | None, exclude_columns: List[str] = []
) -> pd.DataFrame:
    """Apply min-max normalization to numeric columns."""
    norm_params = (
        get_min_max_params(df, exclude_columns) if norm_params is None else norm_params
    )

    min_values = pd.Series(norm_params[0])
    max_values = pd.Series(norm_params[1])

    # Apply min-max normalization
    denom = _safe_denominator(max_values - min_values)
    df_normalized = (df - min_values) / denom

    return _sanitize_normalized(df_normalized)


def standardize(
    df: pd.DataFrame, norm_params: NormParams | None, exclude_columns: List[str] = []
) -> pd.DataFrame:
    """Apply z-score standardization to numeric columns."""
    norm_params = (
        get_standardize_params(df, exclude_columns)
        if norm_params is None
        else norm_params
    )

    mean_values = pd.Series(norm_params[0])
    std_values = pd.Series(norm_params[1])

    # Apply standardization
    denom = _safe_denominator(std_values)
    df_normalized = (df - mean_values) / denom

    return _sanitize_normalized(df_normalized)


def robust_scale(
    df: pd.DataFrame, norm_params: NormParams | None, exclude_columns: List[str] = []
) -> pd.DataFrame:
    """Apply robust scaling (median/IQR) to numeric columns."""
    norm_params = (
        get_robust_scale_params(df, exclude_columns)
        if norm_params is None
        else norm_params
    )

    median_values = pd.Series(norm_params[0])
    iqr = pd.Series(norm_params[1])

    # Apply robust scaling
    denom = _safe_denominator(iqr)
    df_normalized = (df - median_values) / denom

    return _sanitize_normalized(df_normalized)

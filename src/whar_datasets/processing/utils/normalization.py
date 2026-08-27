from functools import partial
from typing import Callable, Dict, Hashable, List, Tuple, TypeAlias

import numpy as np
import pandas as pd

from whar_datasets.config.config import NormType, WHARConfig
from whar_datasets.utils.loading import WindowStore
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
    normalize: Callable[[pd.DataFrame], pd.DataFrame]
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
            normalize = lambda frame: frame  # noqa: E731
    return normalize


def get_norm_params(
    cfg: WHARConfig,
    indices: List[int],
    window_df: pd.DataFrame,
    windows: Dict[str, pd.DataFrame] | WindowStore,
) -> NormParams | None:
    """Compute global normalization statistics for the provided train indices."""
    logger.info("Getting normalization parameters")

    if cfg.normalization is None:
        return None

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

    window_ids = [str(window_df.at[index, "window_id"]) for index in indices]
    if isinstance(windows, WindowStore):
        return _get_array_norm_params(cfg, window_ids, windows)

    windows_df = pd.concat([windows[window_id] for window_id in window_ids], ignore_index=True)

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


def _get_array_norm_params(
    cfg: WHARConfig, window_ids: List[str], store: WindowStore
) -> NormParams | None:
    """Compute global statistics from bounded memory-mapped batches."""
    positions = np.asarray([store.row_by_id[window_id] for window_id in window_ids])
    if cfg.normalization == NormType.ROBUST_SCALE_GLOBALLY:
        values = np.asarray(store.data[positions]).reshape(-1, store.data.shape[-1])
        median = np.median(values, axis=0)
        iqr = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
        return dict(zip(store.columns, median)), dict(zip(store.columns, iqr))

    count = 0
    mean = np.zeros(store.data.shape[-1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    minimum = np.full_like(mean, np.inf)
    maximum = np.full_like(mean, -np.inf)
    for start in range(0, len(positions), 1024):
        batch = np.asarray(store.data[positions[start : start + 1024]], dtype=np.float64)
        batch = batch.reshape(-1, batch.shape[-1])
        minimum = np.minimum(minimum, np.nanmin(batch, axis=0))
        maximum = np.maximum(maximum, np.nanmax(batch, axis=0))
        batch_count = len(batch)
        batch_mean = np.nanmean(batch, axis=0)
        batch_m2 = np.nansum((batch - batch_mean) ** 2, axis=0)
        delta = batch_mean - mean
        total = count + batch_count
        mean += delta * batch_count / total
        m2 += batch_m2 + delta**2 * count * batch_count / total
        count = total

    if cfg.normalization == NormType.MIN_MAX_GLOBALLY:
        return dict(zip(store.columns, minimum)), dict(zip(store.columns, maximum))
    if cfg.normalization == NormType.STD_GLOBALLY:
        std = np.sqrt(m2 / max(count - 1, 1))
        return dict(zip(store.columns, mean)), dict(zip(store.columns, std))
    return None


def normalize_array(
    cfg: WHARConfig,
    values: np.ndarray,
    columns: List[str],
    norm_params: NormParams | None,
) -> np.ndarray:
    """Normalize a window directly as NumPy and return compact float32 data."""
    array = np.asarray(values, dtype=np.float32)
    mode = cfg.normalization
    if mode is None:
        return array

    if mode in {
        NormType.MIN_MAX_PER_SAMPLE,
        NormType.STD_PER_SAMPLE,
        NormType.ROBUST_SCALE_PER_SAMPLE,
    }:
        if mode == NormType.MIN_MAX_PER_SAMPLE:
            center = np.nanmin(array, axis=0)
            scale = np.nanmax(array, axis=0) - center
        elif mode == NormType.STD_PER_SAMPLE:
            center = np.nanmean(array, axis=0)
            scale = np.nanstd(array, axis=0, ddof=1)
        else:
            center = np.nanmedian(array, axis=0)
            scale = np.nanquantile(array, 0.75, axis=0) - np.nanquantile(
                array, 0.25, axis=0
            )
    else:
        if norm_params is None:
            raise ValueError("Global normalization requires fitted parameters.")
        center = np.asarray([norm_params[0][column] for column in columns], dtype=np.float32)
        second = np.asarray([norm_params[1][column] for column in columns], dtype=np.float32)
        if mode == NormType.MIN_MAX_GLOBALLY:
            scale = second - center
        else:
            scale = second

    scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-12), scale, 1.0)
    normalized = (array - center) / scale
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


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

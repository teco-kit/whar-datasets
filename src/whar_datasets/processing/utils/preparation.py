from pathlib import Path
from typing import Dict, List, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.base import compute
from dask.delayed import delayed
from dask.diagnostics.progress import ProgressBar
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.utils.normalization import NormParams, get_normalize
from whar_datasets.processing.utils.transform import get_transform
from whar_datasets.utils.loading import load_window
from whar_datasets.utils.logging import logger


def prepare_windows_seq(
    cfg: WHARConfig,
    norm_params: NormParams | None,
    window_df: pd.DataFrame,
    windows_dir: Path,
) -> Dict[str, List[np.ndarray]]:
    logger.info("Normalizing and transforming windows")

    normalize = get_normalize(cfg, norm_params)
    transform = get_transform(cfg)

    def prepare(window_id: str) -> Tuple[str, List[np.ndarray]]:
        window = load_window(windows_dir, window_id)
        normalized = normalize(window).values
        transformed = transform(normalized)
        return window_id, [normalized, *transformed]

    loop = tqdm(window_df["window_id"])
    loop.set_description("Normalizing and transforming windows")

    prepared: Dict[str, List[np.ndarray]] = {
        window_id: values for window_id, values in map(prepare, loop)
    }

    return prepared


def prepare_windows_para(
    cfg: WHARConfig,
    norm_params: NormParams | None,
    window_df: pd.DataFrame,
    windows_dir: Path,
) -> Dict[str, List[np.ndarray]]:
    logger.info("Normalizing and transforming windows (parallelized)")

    normalize = get_normalize(cfg, norm_params)
    transform = get_transform(cfg)

    relevant_ids = set(window_df["window_id"])

    # Read parquet with dask to handle partitions efficiently
    ddf = dd.read_parquet(windows_dir / "windows.parquet", engine="pyarrow")

    def process_partition(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        if "window_id" not in df.columns:
            return pd.DataFrame()

        mask = df["window_id"].isin(relevant_ids)
        if not mask.any():
            return pd.DataFrame()

        return df.loc[mask].copy()

    # Create delayed tasks for each partition
    delayed_partitions = ddf.to_delayed()

    @delayed
    def process_delayed(partition):
        return process_partition(partition)

    tasks = [process_delayed(part) for part in delayed_partitions]

    # execute tasks in parallel
    pbar = ProgressBar()
    pbar.register()
    partition_subsets = list(compute(*tasks, scheduler="processes"))
    pbar.unregister()

    non_empty_subsets = [df for df in partition_subsets if not df.empty]
    if not non_empty_subsets:
        return {}

    # Important: group globally (not per-partition), otherwise one window can be
    # split across partitions and produce truncated arrays.
    full_subset = pd.concat(non_empty_subsets, axis=0, ignore_index=True)

    prepared: Dict[str, List[np.ndarray]] = {}
    for window_id, group in full_subset.groupby("window_id", sort=False):
        window_data = group.drop(columns=["window_id"]).reset_index(drop=True)
        normalized = normalize(window_data).values
        transformed = transform(normalized)
        prepared[str(window_id)] = [normalized, *transformed]

    return prepared

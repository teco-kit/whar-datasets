import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.utils.normalization import NormParams, normalize_array
from whar_datasets.processing.utils.transform import get_transform
from whar_datasets.utils.loading import WindowStore, open_window_store
from whar_datasets.utils.logging import logger

WindowSource = Dict[str, pd.DataFrame] | WindowStore


def _prepare_one(
    cfg: WHARConfig,
    norm_params: NormParams | None,
    source: WindowSource,
    window_id: str,
) -> Tuple[str, List[np.ndarray]]:
    if isinstance(source, WindowStore):
        values = source.get_array(window_id)
        columns = source.columns
    else:
        frame = source[window_id]
        values = frame.to_numpy(copy=False)
        columns = [str(column) for column in frame.columns]
    normalized = normalize_array(cfg, values, columns, norm_params)
    transformed = get_transform(cfg)(normalized)
    return window_id, [normalized, *transformed]


def prepare_windows_seq(
    cfg: WHARConfig,
    norm_params: NormParams | None,
    window_df: pd.DataFrame,
    windows_dir: Path,
    windows: WindowSource | None = None,
) -> Dict[str, List[np.ndarray]]:
    """Normalize and transform windows from one scan or memory map."""
    logger.info("Normalizing and transforming windows")
    source = windows or open_window_store(windows_dir)
    ids = [str(value) for value in window_df["window_id"]]
    return {
        window_id: values
        for window_id, values in (
            _prepare_one(cfg, norm_params, source, window_id)
            for window_id in tqdm(ids, desc="Preparing windows")
        )
    }


def _prepare_chunk(
    args: tuple[WHARConfig, NormParams | None, Path, List[str]]
) -> Dict[str, List[np.ndarray]]:
    cfg, norm_params, windows_dir, window_ids = args
    source = open_window_store(windows_dir)
    return {
        window_id: _prepare_one(cfg, norm_params, source, window_id)[1]
        for window_id in window_ids
    }


def prepare_windows_para(
    cfg: WHARConfig,
    norm_params: NormParams | None,
    window_df: pd.DataFrame,
    windows_dir: Path,
    windows: WindowSource | None = None,
) -> Dict[str, List[np.ndarray]]:
    """Normalize/transform bounded window chunks in actual worker processes."""
    ids = [str(value) for value in window_df["window_id"]]
    requested = cfg.num_workers or (os.cpu_count() or 1)
    workers = max(1, min(requested, len(ids)))
    if workers == 1:
        return prepare_windows_seq(cfg, norm_params, window_df, windows_dir, windows)

    chunk_size = max(1, int(np.ceil(len(ids) / (workers * 4))))
    chunks = [ids[start : start + chunk_size] for start in range(0, len(ids), chunk_size)]
    tasks = [(cfg, norm_params, windows_dir, chunk) for chunk in chunks]
    logger.info("Preparing windows with %d worker processes", workers)
    prepared: Dict[str, List[np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in tqdm(
            executor.map(_prepare_chunk, tasks),
            total=len(tasks),
            desc="Preparing window chunks",
        ):
            prepared.update(result)
    return prepared

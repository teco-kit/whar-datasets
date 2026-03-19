from pathlib import Path
from typing import Dict, List, Tuple

import dask.dataframe as dd
import pandas as pd
from dask.base import compute
from dask.delayed import delayed
from dask.diagnostics.progress import ProgressBar
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.utils.resampling import resample
from whar_datasets.processing.utils.selecting import select_channels
from whar_datasets.processing.utils.windowing import generate_windowing
from whar_datasets.utils.loading import load_session
from whar_datasets.utils.logging import logger


def process_sessions_seq(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # loop over sessions
    loop = tqdm([int(x) for x in session_df["session_id"].unique()])
    loop.set_description("Processing sessions")

    pairs = [process_session(cfg, sessions_dir, session_id) for session_id in loop]
    window_dfs, window_dicts = zip(*pairs)

    # compute global window metadata and windows
    window_df = pd.concat([w for w in window_dfs if w is not None])
    window_df.reset_index(drop=True, inplace=True)
    windows = {k: v for d in window_dicts if d is not None for k, v in d.items()}

    # assert uniqueness of window ids
    assert window_df["window_id"].nunique() == len(window_df)

    return window_df, windows


def process_sessions_para(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    relevant_ids = set(session_df["session_id"])

    # Read sessions parquet with dask
    ddf = dd.read_parquet(sessions_dir / "sessions.parquet", engine="pyarrow")

    def process_partition(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        if "session_id" not in df.columns:
            return pd.DataFrame()

        mask = df["session_id"].isin(relevant_ids)
        if not mask.any():
            return pd.DataFrame()

        return df.loc[mask].copy()

    logger.info("Processing sessions (parallelized)")

    # Create delayed tasks
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
        return pd.DataFrame(columns=["window_id"]), {}

    # Important: group globally (not per-partition), otherwise one session can be
    # split across partitions and produce incomplete windows.
    full_subset = pd.concat(non_empty_subsets, axis=0, ignore_index=True)

    all_pairs: List[Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]] = []
    for session_id, group in full_subset.groupby("session_id", sort=False):
        session_data = group.drop(columns=["session_id"]).reset_index(drop=True)
        session_data = select_channels(session_data, cfg.selected_channels or [])
        session_data = resample(session_data, cfg.sampling_freq)

        window_df_local, windows_local = generate_windowing(
            int(session_id),  # type: ignore[arg-type]
            session_data,
            cfg.window_time,
            cfg.window_overlap,
            cfg.sampling_freq,
        )
        if window_df_local is not None and windows_local is not None:
            all_pairs.append((window_df_local, windows_local))

    if not all_pairs:
        return pd.DataFrame(columns=["window_id"]), {}

    window_dfs, window_dicts = zip(*all_pairs)

    # compute global window metadata and windows
    window_df = pd.concat([w for w in window_dfs if w is not None])
    window_df.reset_index(drop=True, inplace=True)
    windows = {k: v for d in window_dicts if d is not None for k, v in d.items()}

    # assert uniqueness of window ids
    assert window_df["window_id"].nunique() == len(window_df)

    return window_df, windows


def process_session(
    cfg: WHARConfig, sessions_dir: Path, session_id: int
) -> Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]:
    # laod and process session
    session = load_session(sessions_dir, session_id)
    session = select_channels(session, cfg.selected_channels or [])
    session = resample(session, cfg.sampling_freq)

    # generate windowing
    window_df, windows = generate_windowing(
        session_id,
        session,
        cfg.window_time,
        cfg.window_overlap,
        cfg.sampling_freq,
    )

    if window_df is None or windows is None:
        return None, None

    return window_df, windows

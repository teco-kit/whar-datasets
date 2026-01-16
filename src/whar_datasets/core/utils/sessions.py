from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from dask.delayed import delayed
from dask.base import compute
from tqdm import tqdm
from dask.diagnostics.progress import ProgressBar

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.loading import load_session
from whar_datasets.core.utils.resampling import resample
from whar_datasets.core.utils.selecting import select_channels
from whar_datasets.core.utils.windowing import generate_windowing


def process_sessions_seq(
    cfg: WHARConfig, sessions_dir: Path, session_metadata: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # loop over sessions

    session_metadata = pd.read_parquet(
        r"C:\Users\hohma\Teco\whar-datasets\notebooks\datasets\hhar\metadata\session_metadata.parquet"
    )

    loop = tqdm([int(x) for x in session_metadata["session_id"].unique()])
    loop.set_description("Processing sessions")

    pairs = [process_session(cfg, sessions_dir, session_id) for session_id in loop]
    window_metadatas, window_dicts = zip(*pairs)

    # compute global window metadata and windows
    window_metadata = pd.concat([w for w in window_metadatas if w is not None])
    window_metadata.reset_index(drop=True, inplace=True)
    windows = {k: v for d in window_dicts if d is not None for k, v in d.items()}

    # assert uniqueness of window ids
    assert window_metadata["window_id"].nunique() == len(window_metadata)

    return window_metadata, windows


def process_sessions_para(
    cfg: WHARConfig, sessions_dir: Path, session_metadata: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    @delayed
    def process_session_delayed(
        session_id: int,
    ) -> Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]:
        return process_session(cfg, sessions_dir, session_id)

    # define processing tasks
    tasks = [
        process_session_delayed(session_id)
        for session_id in [int(x) for x in session_metadata["session_id"]]
    ]

    # execute tasks in parallel
    pbar = ProgressBar()
    pbar.register()
    pairs = list(compute(*tasks, scheduler="processes"))
    pbar.unregister()

    window_metadatas, window_dicts = zip(*pairs)

    # compute global window metadata and windows
    window_metadata = pd.concat([w for w in window_metadatas if w is not None])
    window_metadata.reset_index(drop=True, inplace=True)
    windows = {k: v for d in window_dicts if d is not None for k, v in d.items()}

    # assert uniqueness of window ids
    assert window_metadata["window_id"].nunique() == len(window_metadata)

    return window_metadata, windows


def process_session(
    cfg: WHARConfig, sessions_dir: Path, session_id: int
) -> Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]:
    # laod and process session
    session = load_session(sessions_dir, session_id)
    session = select_channels(session, cfg.sensor_channels)
    session = resample(session, cfg.sampling_freq)

    # generate windowing
    window_metadata, windows = generate_windowing(
        session_id,
        session,
        cfg.window_time,
        cfg.window_overlap,
        cfg.sampling_freq,
    )

    if window_metadata is None or windows is None:
        return None, None

    return window_metadata, windows

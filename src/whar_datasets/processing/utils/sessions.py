import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.utils.resampling import (
    get_effective_sampling_freq,
    resample,
)
from whar_datasets.processing.utils.selecting import select_channels
from whar_datasets.processing.utils.windowing import generate_windowing
from whar_datasets.utils.loading import load_session, load_sessions
from whar_datasets.utils.logging import logger

SessionResult = Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]


def _effective_worker_count(cfg: WHARConfig, task_count: int) -> int:
    requested = cfg.num_workers or (os.cpu_count() or 1)
    return max(1, min(requested, task_count))


def _process_session_data(
    cfg: WHARConfig, session_id: int, session: pd.DataFrame
) -> SessionResult:
    session = select_channels(session, cfg.selected_channels or [])
    frequency = get_effective_sampling_freq(cfg.sampling_freq, cfg.resampling_freq)
    session = resample(session, frequency, cfg.max_session_gap_seconds)
    return generate_windowing(
        session_id,
        session,
        cfg.window_time,
        cfg.window_overlap,
        frequency,
    )


def _process_session_from_disk(args: tuple[WHARConfig, Path, int]) -> SessionResult:
    cfg, sessions_dir, session_id = args
    return _process_session_data(cfg, session_id, load_session(sessions_dir, session_id))


def _combine_results(results: List[SessionResult]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    valid = [(frame, data) for frame, data in results if frame is not None and data]
    if not valid:
        return pd.DataFrame(columns=["session_id", "window_id"]), {}
    window_df = pd.concat([frame for frame, _ in valid], ignore_index=True)
    windows = {key: value for _, data in valid for key, value in data.items()}
    if window_df["window_id"].nunique() != len(window_df):
        raise ValueError("Window identifiers are not unique.")
    return window_df, windows


def process_sessions_seq(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Generate windows after scanning the sessions cache only once."""
    session_ids = [int(value) for value in session_df["session_id"].unique()]
    sessions = load_sessions(sessions_dir, session_ids=session_ids)
    results = [
        _process_session_data(cfg, session_id, sessions[session_id])
        for session_id in tqdm(session_ids, desc="Processing sessions")
    ]
    return _combine_results(results)


def process_sessions_para(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Process complete sessions concurrently using bounded local workers."""
    session_ids = [int(value) for value in session_df["session_id"].unique()]
    workers = _effective_worker_count(cfg, len(session_ids))
    if workers == 1:
        return process_sessions_seq(cfg, sessions_dir, session_df)

    logger.info("Processing sessions with %d worker processes", workers)
    tasks = [(cfg, sessions_dir, session_id) for session_id in session_ids]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(_process_session_from_disk, tasks),
                total=len(tasks),
                desc="Processing sessions",
            )
        )
    return _combine_results(results)


def process_session(
    cfg: WHARConfig, sessions_dir: Path, session_id: int
) -> SessionResult:
    """Generate windows from one cached session."""
    return _process_session_data(cfg, session_id, load_session(sessions_dir, session_id))

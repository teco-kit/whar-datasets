import shutil
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd


def cache_samples(
    samples_dir: Path,
    window_df: pd.DataFrame,
    samples: Dict[str, List[np.ndarray]],
) -> None:
    # delete windowing directory if it exists
    if samples_dir.exists():
        shutil.rmtree(samples_dir)

    # create samples directory if it does not exist
    samples_dir.mkdir(parents=True, exist_ok=True)

    # save samples as a single file
    samples_path = samples_dir / "samples.npy"
    np.save(samples_path, samples)  # type: ignore


def cache_windows(
    windows_dir: Path, window_df: pd.DataFrame, windows: Dict[str, pd.DataFrame]
) -> None:
    # delete windowing directory if it exists
    if windows_dir.exists():
        shutil.rmtree(windows_dir)

    # create windowing directory if it does not exist
    windows_dir.mkdir(parents=True, exist_ok=True)

    # Combine all windows into one DataFrame
    window_list = []
    for window_id, df in windows.items():
        df = df.copy()
        df["window_id"] = window_id
        window_list.append(df)

    if window_list:
        combined_windows = pd.concat(window_list)
        # Sort by window_id to optimize read filtering
        combined_windows = combined_windows.sort_values("window_id")

        combined_windows.to_parquet(
            windows_dir / "windows.parquet", index=False, engine="pyarrow"
        )


def cache_window_df(metadata_dir: Path, window_df: pd.DataFrame) -> None:
    # create directories if do not exist
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # define window index path
    window_df_path = metadata_dir / "window_df.csv"

    # save window index
    window_df.to_csv(window_df_path, index=True)


def cache_common_format(
    metadata_dir: Path,
    sessions_dir: Path,
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
    sessions: Dict[int, pd.DataFrame],
) -> None:
    # delete sessions directory if it exists
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)

    # create directories if do not exist
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # define paths
    activity_df_path = metadata_dir / "activity_df.csv"
    session_df_path = metadata_dir / "session_df.csv"

    # save activity and session index
    activity_df.to_csv(activity_df_path, index=True)
    session_df.to_csv(session_df_path, index=True)

    # Combine all sessions into one DataFrame
    session_list = []
    for session_id, df in sessions.items():
        df = df.copy()
        df["session_id"] = session_id
        session_list.append(df)

    if session_list:
        combined_sessions = pd.concat(session_list)
        # Sort by session_id to optimize read filtering
        combined_sessions = combined_sessions.sort_values("session_id")

        combined_sessions.to_parquet(
            sessions_dir / "sessions.parquet", index=False, engine="pyarrow"
        )

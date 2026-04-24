import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_window_df(cache_dir: Path) -> pd.DataFrame:
    """Load cached window metadata."""
    window_df_path = cache_dir / "window_df.csv"
    return pd.read_csv(window_df_path)


def load_session_df(cache_dir: Path) -> pd.DataFrame:
    """Load cached session metadata."""
    session_df_path = cache_dir / "session_df.csv"
    return pd.read_csv(session_df_path)


def load_activity_df(cache_dir: Path) -> pd.DataFrame:
    """Load cached activity metadata."""
    activity_df_path = cache_dir / "activity_df.csv"
    return pd.read_csv(activity_df_path)


def load_samples(samples_dir: Path) -> Dict[str, List[np.ndarray]]:
    """Load all cached samples as ``window_id -> feature list`` mapping."""
    pickle_path = samples_dir / "samples.pkl"
    if pickle_path.exists():
        with pickle_path.open("rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected dict in {pickle_path}, got {type(loaded)}")
        return loaded

    # Backward compatibility for older caches.
    legacy_path = samples_dir / "samples.npy"
    if legacy_path.exists():
        loaded = np.load(legacy_path, allow_pickle=True).item()
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected dict in {legacy_path}, got {type(loaded)}")
        return loaded

    raise FileNotFoundError(f"No samples found at {pickle_path} or {legacy_path}")


def load_windows(windows_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all cached windows from parquet into a dictionary."""
    window_path = windows_dir / "windows.parquet"

    # Load all windows at once
    df = pd.read_parquet(window_path, engine="pyarrow")

    # Group by window_id and create dictionary
    # This is much faster than reading from disk for every window
    windows = {
        str(k): v.drop(columns=["window_id"]).reset_index(drop=True)
        for k, v in df.groupby("window_id")
    }

    return windows


def load_sessions(sessions_dir: Path) -> Dict[int, pd.DataFrame]:
    """Load all cached sessions from parquet into a dictionary."""
    session_path = sessions_dir / "sessions.parquet"

    # Load all sessions at once
    df = pd.read_parquet(session_path, engine="pyarrow")

    # Group by session_id and create dictionary
    sessions = {
        int(k): v.drop(columns=["session_id"]).reset_index(drop=True)  # type: ignore
        for k, v in df.groupby("session_id")
    }

    return sessions


def load_sample(samples_dir: Path, window_id: str) -> List[np.ndarray]:
    """Load one cached sample by ``window_id``."""
    # Try loading from single file first
    pickle_path = samples_dir / "samples.pkl"
    if pickle_path.exists():
        # Warning: This loads the entire dataset to get one sample.
        # Use in_memory=True for better performance.
        with pickle_path.open("rb") as f:
            all_samples = pickle.load(f)
        return all_samples[window_id]

    legacy_path = samples_dir / "samples.npy"
    if legacy_path.exists():
        # Warning: This loads the entire dataset to get one sample.
        # Use in_memory=True for better performance.
        all_samples = np.load(legacy_path, allow_pickle=True).item()
        return all_samples[window_id]

    # Fallback to individual file
    sample_path = samples_dir / f"sample_{window_id}.npy"
    return np.load(sample_path, allow_pickle=True).tolist()


def load_window(windows_dir: Path, window_id: str) -> pd.DataFrame:
    """Load one window by id from the window parquet cache."""
    window_path = windows_dir / "windows.parquet"
    # Use filters to efficiently read only the specific window
    df = pd.read_parquet(
        window_path, filters=[("window_id", "==", window_id)], engine="pyarrow"
    )
    # Drop the ID column to match previous behavior (returning only data)
    if "window_id" in df.columns:
        df = df.drop(columns=["window_id"])
    return df


def load_session(sessions_dir: Path, session_id: int) -> pd.DataFrame:
    """Load one session by id from the session parquet cache."""
    session_path = sessions_dir / "sessions.parquet"
    # Use filters to efficiently read only the specific session
    df = pd.read_parquet(
        session_path, filters=[("session_id", "==", session_id)], engine="pyarrow"
    )
    # Drop the ID column to match previous behavior
    if "session_id" in df.columns:
        df = df.drop(columns=["session_id"])
    return df

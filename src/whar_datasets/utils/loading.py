import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from whar_datasets.config.config import CACHE_SCHEMA_VERSION


def _require_current_manifest(cache_dir: Path, artifact: str) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Schema-v{CACHE_SCHEMA_VERSION} {artifact} cache not found under "
            f"'{cache_dir}'. Rerun preprocessing."
        )
    manifest: dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {artifact} cache schema; rerun preprocessing."
        )
    return manifest


def _load_metadata(cache_dir: Path, stem: str) -> pd.DataFrame:
    parquet_path = cache_dir / f"{stem}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Schema-v{CACHE_SCHEMA_VERSION} metadata cache not found at "
            f"'{parquet_path}'. Rerun preprocessing."
        )
    return pd.read_parquet(parquet_path, engine="pyarrow")


def load_window_df(cache_dir: Path) -> pd.DataFrame:
    return _load_metadata(cache_dir, "window_df")


def load_session_df(cache_dir: Path) -> pd.DataFrame:
    return _load_metadata(cache_dir, "session_df")


def load_activity_df(cache_dir: Path) -> pd.DataFrame:
    return _load_metadata(cache_dir, "activity_df")


class WindowStore:
    """Memory-mapped fixed-shape window cache."""

    def __init__(self, windows_dir: Path):
        self.windows_dir = windows_dir
        _require_current_manifest(windows_dir, "window")
        self.data = np.load(windows_dir / "data.npy", mmap_mode="c", allow_pickle=False)
        self.ids = np.load(windows_dir / "window_ids.npy", allow_pickle=False)
        self.columns = json.loads((windows_dir / "columns.json").read_text(encoding="utf-8"))
        self.row_by_id = {str(window_id): row for row, window_id in enumerate(self.ids)}

    def get_array(self, window_id: str, copy: bool = False) -> np.ndarray:
        value = self.data[self.row_by_id[str(window_id)]]
        return np.array(value, copy=True) if copy else value

    def get_frame(self, window_id: str) -> pd.DataFrame:
        return pd.DataFrame(self.get_array(window_id), columns=self.columns)


def open_window_store(windows_dir: Path) -> WindowStore:
    return WindowStore(windows_dir)


class ArraySampleStore:
    """Memory-mapped feature tensors indexed by stable window identifier."""

    def __init__(self, samples_dir: Path):
        manifest = _require_current_manifest(samples_dir, "sample")
        self.ids = np.load(samples_dir / "window_ids.npy", allow_pickle=False)
        self.row_by_id = {str(window_id): row for row, window_id in enumerate(self.ids)}
        self.features = [
            np.load(samples_dir / feature["path"], mmap_mode="c", allow_pickle=False)
            for feature in manifest["features"]
        ]

    def get(self, window_id: str, copy: bool = False) -> List[np.ndarray]:
        row = self.row_by_id[str(window_id)]
        if copy:
            return [np.array(feature[row], copy=True) for feature in self.features]
        return [feature[row] for feature in self.features]


def open_sample_store(samples_dir: Path) -> ArraySampleStore:
    return ArraySampleStore(samples_dir)


def load_samples(samples_dir: Path) -> Dict[str, List[np.ndarray]]:
    store = open_sample_store(samples_dir)
    return {
        str(window_id): store.get(str(window_id), copy=True)
        for window_id in store.ids
    }


@lru_cache(maxsize=16)
def _cached_sample_store(path: str, cache_stamp: int) -> ArraySampleStore:
    del cache_stamp
    return open_sample_store(Path(path))


def load_sample(samples_dir: Path, window_id: str) -> List[np.ndarray]:
    manifest = samples_dir / "manifest.json"
    stamp = manifest.stat().st_mtime_ns if manifest.exists() else 0
    return _cached_sample_store(str(samples_dir.resolve()), stamp).get(window_id)


def load_windows(
    windows_dir: Path, window_ids: List[str] | None = None
) -> Dict[str, pd.DataFrame]:
    store = open_window_store(windows_dir)
    ids = window_ids or [str(value) for value in store.ids]
    return {window_id: store.get_frame(window_id) for window_id in ids}


def load_sessions(
    sessions_dir: Path, session_ids: List[int] | None = None
) -> Dict[int, pd.DataFrame]:
    _require_current_manifest(sessions_dir, "session")
    filters = [("session_id", "in", session_ids)] if session_ids else None
    frame = pd.read_parquet(
        sessions_dir / "sessions.parquet", filters=filters, engine="pyarrow"
    )
    return {
        int(str(key)): value.drop(columns=["session_id"]).reset_index(drop=True)
        for key, value in frame.groupby("session_id", sort=False)
    }


def load_window(windows_dir: Path, window_id: str) -> pd.DataFrame:
    return open_window_store(windows_dir).get_frame(window_id)


def load_session(sessions_dir: Path, session_id: int) -> pd.DataFrame:
    _require_current_manifest(sessions_dir, "session")
    frame = pd.read_parquet(
        sessions_dir / "sessions.parquet",
        filters=[("session_id", "==", session_id)],
        engine="pyarrow",
    )
    return frame.drop(columns=["session_id"], errors="ignore")

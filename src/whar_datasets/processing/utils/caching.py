import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from whar_datasets.config.config import CACHE_SCHEMA_VERSION


def _replace_directory(target: Path, build: Callable[[Path], None]) -> None:
    """Build a cache beside its target and atomically swap it into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    try:
        build(temporary)
        if target.exists():
            target.rename(backup)
        temporary.rename(target)
        if backup.exists():
            shutil.rmtree(backup)
    except BaseException:
        if not target.exists() and backup.exists():
            backup.rename(target)
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        df.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_samples(
    samples_dir: Path,
    window_df: pd.DataFrame,
    samples: Dict[str, List[np.ndarray]],
) -> None:
    """Persist fixed-shape features as memory-mappable arrays."""
    ordered_ids = [str(value) for value in window_df["window_id"]]
    if set(ordered_ids) != set(samples):
        raise ValueError("Sample identifiers do not match window metadata.")

    def build(directory: Path) -> None:
        if not ordered_ids:
            _write_json(
                directory / "manifest.json",
                {"schema_version": CACHE_SCHEMA_VERSION, "count": 0, "features": []},
            )
            np.save(directory / "window_ids.npy", np.asarray([], dtype="U1"))
            return

        feature_count = len(samples[ordered_ids[0]])
        if any(len(samples[window_id]) != feature_count for window_id in ordered_ids):
            raise ValueError("All samples must contain the same feature tensors.")

        np.save(directory / "window_ids.npy", np.asarray(ordered_ids, dtype=str))
        features: list[dict[str, object]] = []
        for feature_index in range(feature_count):
            first = np.asarray(samples[ordered_ids[0]][feature_index])
            shape = first.shape
            if any(
                np.asarray(samples[window_id][feature_index]).shape != shape
                for window_id in ordered_ids
            ):
                raise ValueError(
                    f"Feature {feature_index} has variable shapes and cannot be array-cached."
                )
            dtype = np.dtype(np.float32 if np.issubdtype(first.dtype, np.floating) else first.dtype)
            filename = f"feature_{feature_index}.npy"
            target = np.lib.format.open_memmap(
                directory / filename,
                mode="w+",
                dtype=dtype,
                shape=(len(ordered_ids), *shape),
            )
            for row, window_id in enumerate(ordered_ids):
                target[row] = np.asarray(samples[window_id][feature_index], dtype=dtype)
            target.flush()
            del target
            features.append(
                {"path": filename, "dtype": dtype.str, "shape": list(shape)}
            )

        _write_json(
            directory / "manifest.json",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "count": len(ordered_ids),
                "features": features,
            },
        )

    _replace_directory(samples_dir, build)


def cache_windows(
    windows_dir: Path, window_df: pd.DataFrame, windows: Dict[str, pd.DataFrame]
) -> None:
    """Persist windows as one dense float32 tensor with constant-time row access."""
    ordered_ids = [str(value) for value in window_df["window_id"]]

    def build(directory: Path) -> None:
        if not ordered_ids:
            raise ValueError("Cannot cache an empty window collection.")
        first = windows[ordered_ids[0]]
        shape = first.shape
        columns = [str(column) for column in first.columns]
        if any(windows[window_id].shape != shape for window_id in ordered_ids):
            raise ValueError("All resampled windows must have the same shape.")
        data = np.lib.format.open_memmap(
            directory / "data.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(ordered_ids), *shape),
        )
        for row, window_id in enumerate(ordered_ids):
            data[row] = windows[window_id].to_numpy(dtype=np.float32, copy=False)
        data.flush()
        del data
        np.save(directory / "window_ids.npy", np.asarray(ordered_ids, dtype=str))
        _write_json(directory / "columns.json", columns)
        _write_json(
            directory / "manifest.json",
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "count": len(ordered_ids),
                "shape": list(shape),
                "dtype": np.dtype(np.float32).str,
            },
        )

    _replace_directory(windows_dir, build)


def cache_window_df(metadata_dir: Path, window_df: pd.DataFrame) -> None:
    """Persist typed window metadata."""
    _atomic_parquet(window_df.reset_index(drop=True), metadata_dir / "window_df.parquet")


def cache_common_format(
    metadata_dir: Path,
    sessions_dir: Path,
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
    sessions: Dict[int, pd.DataFrame],
) -> None:
    """Persist metadata and one Parquet row group per session without concatenation."""
    _atomic_parquet(activity_df.reset_index(drop=True), metadata_dir / "activity_df.parquet")
    _atomic_parquet(session_df.reset_index(drop=True), metadata_dir / "session_df.parquet")

    def build(directory: Path) -> None:
        writer: pq.ParquetWriter | None = None
        try:
            for session_id in sorted(sessions):
                frame = sessions[session_id].copy()
                frame["timestamp"] = pd.to_datetime(frame["timestamp"])
                sensor_columns = frame.columns.difference(["timestamp"])
                frame[sensor_columns] = frame[sensor_columns].astype(np.float32)
                table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata()
                table = table.append_column(
                    "session_id",
                    pa.array(np.full(len(frame), session_id, dtype=np.int64)),
                )
                if writer is None:
                    writer = pq.ParquetWriter(
                        directory / "sessions.parquet",
                        table.schema,
                        compression="zstd",
                    )
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table, row_group_size=len(frame))
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError("Cannot cache a dataset without sessions.")
        _write_json(
            directory / "manifest.json",
            {"schema_version": CACHE_SCHEMA_VERSION, "session_count": len(sessions)},
        )

    _replace_directory(sessions_dir, build)

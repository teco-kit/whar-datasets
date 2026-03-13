from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from whar_datasets.config.getter import WHARDatasetID, har_dataset_dict

REQUIRED_ACTIVITY_COLUMNS = {"activity_id"}
REQUIRED_SESSION_COLUMNS = {"session_id", "subject_id", "activity_id"}
REQUIRED_WINDOW_COLUMNS = {"session_id"}


@dataclass(frozen=True)
class DatasetAnalysisResult:
    dataset_id: str
    num_samples: int
    window_time_s: float
    window_overlap: float
    estimated_time_s: float
    estimated_time_h: float
    num_subjects: int
    num_classes_total: int
    num_classes_observed: int
    class_coverage_mean: float
    class_coverage_std: float
    global_normalized_entropy: float
    intra_subject_entropy_mean: float
    intra_subject_entropy_std: float
    inter_subject_kl_mean: float
    inter_subject_kl_std: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def drop_unnamed_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    index_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    return df.drop(columns=index_cols) if index_cols else df


def load_cached_metadata(
    dataset_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_dir = dataset_dir / "metadata"
    activity_df = drop_unnamed_index_cols(pd.read_csv(metadata_dir / "activity_df.csv"))
    session_df = drop_unnamed_index_cols(pd.read_csv(metadata_dir / "session_df.csv"))
    window_df = drop_unnamed_index_cols(pd.read_csv(metadata_dir / "window_df.csv"))
    return activity_df, session_df, window_df


def validate_required_metadata(
    dataset_id: str,
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
) -> None:
    missing_activity = REQUIRED_ACTIVITY_COLUMNS.difference(activity_df.columns)
    if missing_activity:
        raise ValueError(
            f"[{dataset_id}] Missing required activity metadata columns: "
            f"{sorted(missing_activity)}"
        )

    missing_session = REQUIRED_SESSION_COLUMNS.difference(session_df.columns)
    if missing_session:
        raise ValueError(
            f"[{dataset_id}] Missing required session metadata columns: "
            f"{sorted(missing_session)}"
        )

    missing_window = REQUIRED_WINDOW_COLUMNS.difference(window_df.columns)
    if missing_window:
        raise ValueError(
            f"[{dataset_id}] Missing required window metadata columns: "
            f"{sorted(missing_window)}"
        )

    if session_df["subject_id"].isna().any():
        raise ValueError(f"[{dataset_id}] subject_id contains missing values.")
    if session_df["activity_id"].isna().any():
        raise ValueError(f"[{dataset_id}] activity_id contains missing values.")

    session_ids = set(
        coerce_integer_series(session_df["session_id"], "session_id", dataset_id)
        .astype(int)
        .tolist()
    )
    window_session_ids = set(
        coerce_integer_series(window_df["session_id"], "session_id", dataset_id)
        .astype(int)
        .tolist()
    )

    unknown_sessions = window_session_ids.difference(session_ids)
    if unknown_sessions:
        preview = sorted(unknown_sessions)[:10]
        raise ValueError(
            f"[{dataset_id}] window_df references unknown session_id values: {preview}"
        )


def coerce_integer_series(series: pd.Series, name: str, dataset_id: str) -> pd.Series:
    if pd.api.types.is_integer_dtype(series):
        return series.astype("int64")

    normalized = series.astype(str).str.strip().str.strip("[]")
    converted = pd.to_numeric(normalized, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"[{dataset_id}] Column '{name}' contains non-integer values.")
    if (converted % 1 != 0).any():
        raise ValueError(
            f"[{dataset_id}] Column '{name}' contains non-integral values."
        )

    return converted.astype("int64")


def _probabilities_from_counts(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0:
        return np.array([], dtype=np.float64)
    return counts.astype(np.float64) / total


def normalized_entropy(counts: np.ndarray, num_classes_total: int) -> float:
    if counts.size == 0:
        return 0.0
    probs = _probabilities_from_counts(counts)
    if probs.size == 0:
        return 0.0
    entropy = -float(np.sum(probs * np.log2(probs + np.finfo(float).eps)))
    if num_classes_total <= 1:
        return 1.0
    return float(entropy / np.log2(float(num_classes_total)))


def kl_divergence(subject_probs: np.ndarray, global_probs: np.ndarray) -> float:
    mask = subject_probs > 0.0
    if not np.any(mask):
        return 0.0
    if np.any(global_probs[mask] <= 0.0):
        return float("inf")
    ratio = subject_probs[mask] / global_probs[mask]
    return float(np.sum(subject_probs[mask] * np.log2(ratio)))


def _compute_subject_activity_counts(
    dataset_id: str, session_df: pd.DataFrame, window_df: pd.DataFrame
) -> pd.DataFrame:
    session_clean = session_df.copy()
    window_clean = window_df.copy()
    session_clean["session_id"] = coerce_integer_series(
        session_clean["session_id"], "session_id", dataset_id
    )
    session_clean["subject_id"] = coerce_integer_series(
        session_clean["subject_id"], "subject_id", dataset_id
    )
    session_clean["activity_id"] = coerce_integer_series(
        session_clean["activity_id"], "activity_id", dataset_id
    )
    window_clean["session_id"] = coerce_integer_series(
        window_clean["session_id"], "session_id", dataset_id
    )

    window_labels = (
        window_clean[["session_id"]]
        .merge(
            session_clean[["session_id", "subject_id", "activity_id"]],
            on="session_id",
            how="left",
            validate="many_to_one",
        )
        .dropna(subset=["subject_id", "activity_id"])
    )

    window_labels = window_labels.astype(
        {"subject_id": "int64", "activity_id": "int64"}
    )
    counts = (
        window_labels.groupby(["subject_id", "activity_id"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    return counts


def analyze_dataset_metadata(
    dataset_id: str,
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
    window_df: pd.DataFrame,
) -> DatasetAnalysisResult:
    validate_required_metadata(dataset_id, activity_df, session_df, window_df)

    activity_ids = coerce_integer_series(
        activity_df["activity_id"], "activity_id", dataset_id
    )
    num_classes_total = int(activity_ids.nunique())
    num_samples = int(len(window_df))
    subject_activity_counts = _compute_subject_activity_counts(
        dataset_id, session_df, window_df
    )

    if subject_activity_counts.empty:
        raise ValueError(
            f"[{dataset_id}] No valid subject/activity assignments could be derived from window_df."
        )

    num_subjects = int(subject_activity_counts["subject_id"].nunique())
    num_classes_observed = int(subject_activity_counts["activity_id"].nunique())

    per_subject_table = subject_activity_counts.pivot_table(
        index="subject_id",
        columns="activity_id",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    # Coverage_i = recorded classes by subject i / total classes.
    coverage_by_subject = (per_subject_table > 0).sum(axis=1).astype(np.float64)
    coverage_by_subject = coverage_by_subject / max(num_classes_total, 1)

    class_counts_global = per_subject_table.sum(axis=0).to_numpy(dtype=np.float64)
    global_normalized_entropy = normalized_entropy(
        class_counts_global, num_classes_total
    )
    global_probs = _probabilities_from_counts(class_counts_global)

    subject_entropies: List[float] = []
    subject_kls: List[float] = []
    for _, row in per_subject_table.iterrows():
        counts = row.to_numpy(dtype=np.float64)
        subject_entropies.append(normalized_entropy(counts, num_classes_total))
        subject_probs = _probabilities_from_counts(counts)
        if subject_probs.size == 0 or global_probs.size == 0:
            subject_kls.append(0.0)
        else:
            subject_kls.append(kl_divergence(subject_probs, global_probs))

    cfg = None
    for ds_enum, ds_cfg in har_dataset_dict.items():
        if ds_enum.value == dataset_id:
            cfg = ds_cfg
            break
    if cfg is None:
        raise ValueError(
            f"[{dataset_id}] Could not resolve dataset config to compute duration."
        )
    window_time_s = float(cfg.window_time)
    window_overlap = float(cfg.window_overlap)
    step_time_s = window_time_s * max(1.0 - window_overlap, 0.0)
    if num_samples <= 0:
        estimated_time_s = 0.0
    elif num_samples == 1:
        estimated_time_s = window_time_s
    else:
        estimated_time_s = window_time_s + (num_samples - 1) * step_time_s

    return DatasetAnalysisResult(
        dataset_id=dataset_id,
        num_samples=num_samples,
        window_time_s=window_time_s,
        window_overlap=window_overlap,
        estimated_time_s=float(estimated_time_s),
        estimated_time_h=float(estimated_time_s / 3600.0),
        num_subjects=num_subjects,
        num_classes_total=num_classes_total,
        num_classes_observed=num_classes_observed,
        class_coverage_mean=float(
            np.mean(coverage_by_subject.to_numpy(dtype=np.float64))
        ),
        class_coverage_std=float(
            np.std(coverage_by_subject.to_numpy(dtype=np.float64))
        ),
        global_normalized_entropy=global_normalized_entropy,
        intra_subject_entropy_mean=float(np.mean(np.array(subject_entropies))),
        intra_subject_entropy_std=float(np.std(np.array(subject_entropies))),
        inter_subject_kl_mean=float(np.mean(np.array(subject_kls))),
        inter_subject_kl_std=float(np.std(np.array(subject_kls))),
    )


def default_dataset_ids() -> List[str]:
    return sorted(ds.value for ds in WHARDatasetID)


def available_cached_dataset_ids(datasets_root: Path) -> List[str]:
    if not datasets_root.exists():
        return []

    known = set(default_dataset_ids())
    result: List[str] = []
    for child in datasets_root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name not in known:
            continue
        metadata_dir = child / "metadata"
        required_files = [
            metadata_dir / "activity_df.csv",
            metadata_dir / "session_df.csv",
            metadata_dir / "window_df.csv",
        ]
        if all(path.exists() for path in required_files):
            result.append(child.name)

    return sorted(result)


def analyze_cached_datasets(
    datasets_root: Path,
    dataset_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    chosen_ids = (
        list(dataset_ids)
        if dataset_ids is not None
        else available_cached_dataset_ids(datasets_root)
    )
    if not chosen_ids:
        raise ValueError(
            f"No cached datasets with metadata found under '{datasets_root}'."
        )

    results: List[Dict[str, object]] = []
    for dataset_id in chosen_ids:
        dataset_dir = datasets_root / dataset_id
        activity_df, session_df, window_df = load_cached_metadata(dataset_dir)
        row = analyze_dataset_metadata(dataset_id, activity_df, session_df, window_df)
        results.append(row.to_dict())

    return pd.DataFrame(results).sort_values("dataset_id").reset_index(drop=True)

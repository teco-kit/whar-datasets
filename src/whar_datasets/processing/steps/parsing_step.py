import inspect
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple, TypeAlias

import pandas as pd

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.steps.abstract_step import AbstractStep
from whar_datasets.processing.utils.caching import cache_common_format
from whar_datasets.utils.loading import load_activity_df, load_session_df, load_sessions
from whar_datasets.utils.logging import logger

InputT: TypeAlias = None
OutputT: TypeAlias = Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[int, pd.DataFrame],
]


def _align_activity_ids_to_config(
    cfg: WHARConfig,
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align parser IDs by activity names, never by parser/file order.

    Parsers are allowed to discover activities in dataset-specific order, but
    the common format must use the deterministic IDs declared by the config.
    Mapping by names also keeps session metadata synchronized with the activity
    metadata table.
    """
    required_activity_columns = {"activity_id", "activity_name"}
    required_session_columns = {"activity_id"}
    if not required_activity_columns.issubset(activity_df.columns):
        raise ValueError(
            "Parser activity metadata must contain 'activity_id' and "
            "'activity_name'."
        )
    if not required_session_columns.issubset(session_df.columns):
        raise ValueError("Parser session metadata must contain 'activity_id'.")

    config_names = canonicalize_activity_name_list(cfg.available_activities)
    config_name_to_id = {name: idx for idx, name in enumerate(config_names)}
    if len(config_name_to_id) != len(config_names):
        raise ValueError("Configured activity names are not unique after normalization.")

    parsed = activity_df.copy()
    parsed_names = canonicalize_activity_name_list(parsed["activity_name"].tolist())
    unknown_names = sorted(set(parsed_names).difference(config_name_to_id))
    if unknown_names:
        raise ValueError(
            "Parser emitted activities not covered by cfg.available_activities: "
            + ", ".join(unknown_names)
        )

    parsed_id_to_config_id: dict[int, int] = {}
    for raw_id, config_name in zip(parsed["activity_id"], parsed_names):
        raw_id_int = int(raw_id)
        config_id = config_name_to_id[config_name]
        previous = parsed_id_to_config_id.setdefault(raw_id_int, config_id)
        if previous != config_id:
            raise ValueError(
                f"Parser activity_id {raw_id_int} maps to multiple activity names."
            )

    parsed["activity_id"] = [config_name_to_id[name] for name in parsed_names]
    parsed["activity_name"] = [config_names[idx] for idx in parsed["activity_id"]]
    if parsed["activity_id"].duplicated().any():
        raise ValueError("Parser activity metadata contains duplicate activity IDs.")
    parsed = parsed.sort_values("activity_id").reset_index(drop=True)

    sessions = session_df.copy()
    session_ids = pd.to_numeric(sessions["activity_id"], errors="coerce")
    if session_ids.isna().any():
        raise ValueError("Parser session metadata contains non-numeric activity IDs.")
    unmapped_session_ids = sorted(
        set(session_ids.astype(int)).difference(parsed_id_to_config_id)
    )
    if unmapped_session_ids:
        raise ValueError(
            "Parser session metadata references activity IDs absent from activity "
            "metadata: "
            + ", ".join(str(value) for value in unmapped_session_ids)
        )
    sessions["activity_id"] = session_ids.astype(int).map(parsed_id_to_config_id)
    if sessions["activity_id"].isna().any():
        raise ValueError("Could not map all parser session activity IDs to config IDs.")
    sessions["activity_id"] = sessions["activity_id"].astype("int32")

    return parsed, sessions


class ParsingStep(AbstractStep[InputT, OutputT]):
    """Parse extracted raw files into the common WHAR dataset format."""

    def __init__(
        self,
        cfg: WHARConfig,
        data_dir: Path,
        metadata_dir: Path,
        sessions_dir: Path,
        dependent_on: List[AbstractStep],
    ):
        super().__init__(cfg, sessions_dir, dependent_on)

        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.sessions_dir = sessions_dir

        self.hash_name: str = "parsing_hash"
        self.relevant_cfg_keys: Set[str] = {
            "dataset_id",
            "activity_id_col",
            "available_activities",
            "available_channels",
            "num_of_subjects",
            "num_of_activities",
            "num_of_channels",
        }
        try:
            self.relevant_values = [inspect.getsource(cfg.parse)]
        except (OSError, TypeError):
            self.relevant_values = [cfg.parse.__module__, cfg.parse.__qualname__]

    def load_input(self) -> InputT:
        return None

    def validate_input(self, step_input: InputT) -> bool:
        logger.info("Checking extracted data")

        if not self.data_dir.exists():
            logger.warning(f"Data directory not found at '{self.data_dir}'.")
            return False

        logger.info("Data directory exists")
        return True

    def build_output(self, step_input: InputT) -> OutputT:
        logger.info("Parsing to common format")

        with _ignore_sidecar_files():
            activity_df, session_df, sessions = self.cfg.parse(
                str(self.data_dir), self.cfg.activity_id_col
            )

        activity_df, session_df = _align_activity_ids_to_config(
            self.cfg, activity_df, session_df
        )

        return activity_df, session_df, sessions

    def save_output(self, step_output: OutputT) -> None:
        activity_df, session_df, sessions = step_output

        logger.info("Saving common format")

        cache_common_format(
            self.metadata_dir, self.sessions_dir, activity_df, session_df, sessions
        )

    def load_output(self) -> OutputT:
        logger.info("Loading common format")

        session_df = load_session_df(self.metadata_dir)
        activity_df = load_activity_df(self.metadata_dir)
        sessions = load_sessions(self.sessions_dir)

        return activity_df, session_df, sessions

    def output_exists(self) -> bool:
        metadata = all(
            (self.metadata_dir / f"{stem}.parquet").exists()
            for stem in ("activity_df", "session_df")
        )
        return (
            metadata
            and (self.sessions_dir / "sessions.parquet").exists()
            and (self.sessions_dir / "manifest.json").exists()
        )


def _is_sidecar_entry(entry_name: str) -> bool:
    lowered = entry_name.lower()
    return (
        entry_name.startswith("._")
        or entry_name == ".DS_Store"
        or entry_name == "__MACOSX"
        or "hash" in lowered
    )


def _is_sidecar_path(path: Path) -> bool:
    return any(_is_sidecar_entry(part) for part in path.parts)


@contextmanager
def _ignore_sidecar_files() -> Iterator[None]:
    original_listdir = os.listdir
    original_walk = os.walk
    original_path_glob = Path.glob
    original_path_rglob = Path.rglob

    def _filtered_listdir(path: str | os.PathLike[str] = ".") -> List[str]:
        return [name for name in original_listdir(path) if not _is_sidecar_entry(name)]

    def _filtered_walk(
        top: str | os.PathLike[str],
        topdown: bool = True,
        onerror=None,
        followlinks: bool = False,
    ):
        for root, dirs, files in original_walk(top, topdown, onerror, followlinks):
            dirs[:] = [name for name in dirs if not _is_sidecar_entry(name)]
            files = [name for name in files if not _is_sidecar_entry(name)]
            yield root, dirs, files

    def _filtered_glob(self: Path, pattern: str):
        for path in original_path_glob(self, pattern):
            if not _is_sidecar_path(path):
                yield path

    def _filtered_rglob(self: Path, pattern: str):
        for path in original_path_rglob(self, pattern):
            if not _is_sidecar_path(path):
                yield path

    os.listdir = _filtered_listdir  # type: ignore
    os.walk = _filtered_walk  # type: ignore
    Path.glob = _filtered_glob  # type: ignore
    Path.rglob = _filtered_rglob  # type: ignore
    try:
        yield
    finally:
        os.listdir = original_listdir  # type: ignore[method-assign]
        os.walk = original_walk  # type: ignore[method-assign]
        Path.glob = original_path_glob  # type: ignore[method-assign]
        Path.rglob = original_path_rglob  # type: ignore[method-assign]

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple, TypeAlias

import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.steps.abstract_step import AbstractStep
from whar_datasets.processing.utils.caching import cache_common_format
from whar_datasets.utils.loading import load_activity_df, load_session_df, load_sessions
from whar_datasets.utils.logging import logger

base_type: TypeAlias = Any
result_type: TypeAlias = Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[int, pd.DataFrame],
]


class ParsingStep(AbstractStep):
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
        }

    def get_base(self) -> base_type:
        return None

    def check_initial_format(self, base: base_type) -> bool:
        logger.info("Checking extracted data")

        if not self.data_dir.exists():
            logger.warning(f"Data directory not found at '{self.data_dir}'.")
            return False

        logger.info("Data directory exists")
        return True

    def compute_results(self, base: base_type) -> result_type:
        logger.info("Parsing to common format")

        with _ignore_sidecar_files():
            activity_df, session_df, sessions = self.cfg.parse(
                str(self.data_dir), self.cfg.activity_id_col
            )

        # Dataset-specific label adaptation: align parsed labels to the configured
        # labels of the current dataset via activity_id when cardinalities match.
        if (
            "activity_id" in activity_df.columns
            and "activity_name" in activity_df.columns
            and len(self.cfg.available_activities) == len(activity_df)
        ):
            by_id = activity_df.sort_values("activity_id").reset_index(drop=True)
            by_id["activity_name"] = self.cfg.available_activities
            activity_df = by_id.astype({"activity_name": "string"})

        return activity_df, session_df, sessions

    def save_results(self, results: result_type) -> None:
        activity_df, session_df, sessions = results

        logger.info("Saving common format")

        cache_common_format(
            self.metadata_dir, self.sessions_dir, activity_df, session_df, sessions
        )

    def load_results(self) -> result_type:
        logger.info("Loading common format")

        session_df = load_session_df(self.metadata_dir)
        activity_df = load_activity_df(self.metadata_dir)
        sessions = load_sessions(self.sessions_dir)

        return activity_df, session_df, sessions


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
        os.listdir = original_listdir
        os.walk = original_walk
        Path.glob = original_path_glob
        Path.rglob = original_path_rglob

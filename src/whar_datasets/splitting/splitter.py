from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split
from whar_datasets.utils.logging import logger


class Splitter(ABC):
    """Base interface for split generation strategies."""

    def __init__(self, cfg: WHARConfig):
        self.val_percentage = cfg.val_percentage
        self.strict_train_val_separation = cfg.strict_train_val_separation
        self.rng = np.random.RandomState(cfg.seed)
        self.dataset_id = cfg.dataset_id
        self._reset_split_diagnostics()

    @abstractmethod
    def get_splits(
        self, session_df: pd.DataFrame, window_df: pd.DataFrame
    ) -> List[Split]:
        """Return train/validation/test splits for the provided metadata."""
        pass

    def _get_train_val_indices(
        self,
        indices: List[int],
        window_df: pd.DataFrame,
        *,
        emit_diagnostics: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """Split candidate indices into train/validation subsets."""
        if self.strict_train_val_separation:
            result = self._get_purged_train_val_indices(indices, window_df)
            if emit_diagnostics:
                self._log_split_diagnostics()
            return result

        n_train = len(indices)
        n_val = int(n_train * self.val_percentage)

        shuffled_indices: List[int] = self.rng.permutation(indices).tolist()

        val_indices = shuffled_indices[:n_val]
        train_indices = shuffled_indices[n_val:]

        return train_indices, val_indices

    def _get_purged_train_val_indices(
        self, indices: List[int], window_df: pd.DataFrame
    ) -> Tuple[List[int], List[int]]:
        """Split eligible sessions into validation blocks and purge overlaps."""
        if not indices:
            return [], []

        subset = window_df.loc[indices].copy()
        required_columns = {"session_id", "start_index", "end_index"}
        missing_columns = required_columns.difference(subset.columns)
        if missing_columns:
            raise ValueError(
                "Strict train/validation separation requires schema-v2 window "
                f"metadata columns {sorted(missing_columns)}. Rerun preprocessing."
            )

        if self.val_percentage == 0:
            return list(indices), []

        training: set[int] = set()
        validation: set[int] = set()
        unsplittable: list[tuple[int, pd.DataFrame]] = []
        for session_id, group in subset.groupby("session_id", sort=False):
            ordered = group.sort_values("start_index")
            session_id_int = int(str(session_id))
            session_split = self._split_session_strict(ordered)
            if session_split is None:
                unsplittable.append((session_id_int, ordered))
                continue
            session_validation, session_training = session_split
            validation.update(session_validation)
            training.update(session_training)

        fallback_validation, fallback_training = self._assign_unsplittable_sessions(
            unsplittable
        )
        validation.update(fallback_validation)
        training.update(fallback_training)

        return sorted(training), sorted(validation)

    def _reset_split_diagnostics(self) -> None:
        """Clear diagnostics collected while generating one set of splits."""
        self._unsplittable_session_count = 0
        self._unsplittable_by_window_count: dict[int, list[int]] = {}

    def _log_split_diagnostics(self) -> None:
        """Emit one compact summary of indivisible-session assignments."""
        if self._unsplittable_session_count == 0:
            return

        summaries = []
        for window_count in sorted(self._unsplittable_by_window_count):
            total, validation, training = self._unsplittable_by_window_count[
                window_count
            ]
            summaries.append(
                f"{window_count}-window sessions: {total} total, "
                f"{validation} validation, {training} training"
            )

        logger.warning(
            "Strict separation assigned %d unsplittable sessions as whole units: "
            "%s (dataset=%s).",
            self._unsplittable_session_count,
            "; ".join(summaries),
            self.dataset_id,
        )
        self._reset_split_diagnostics()

    def _split_session_strict(
        self, group: pd.DataFrame
    ) -> Tuple[set[int], set[int]] | None:
        """Create a feasible, distributed strict split for one session."""
        count = len(group)
        validation_count = max(
            1,
            min(int(np.floor(count * self.val_percentage + 0.5)), count - 1),
        )
        if count < 2:
            return None

        group_indices = np.asarray(group.index, dtype=np.int64)
        starts = group["start_index"].to_numpy(dtype=np.int64)
        ends = group["end_index"].to_numpy(dtype=np.int64)
        preferred_blocks = min(3, validation_count)
        minimum_blocks = 2 if validation_count > 1 else 1
        for block_count in range(preferred_blocks, minimum_blocks - 1, -1):
            if count - validation_count < block_count - 1:
                continue
            for positions in self._distributed_block_candidates(
                count, validation_count, block_count
            ):
                validation_positions = np.asarray(positions, dtype=np.int64)
                validation_mask = np.zeros(count, dtype=bool)
                validation_mask[validation_positions] = True

                # All candidates belong to this one session. Vectorizing the
                # interval test avoids repeated DataFrame indexing for every
                # candidate layout and is important for datasets with many
                # short sessions.
                validation_starts = starts[validation_positions]
                validation_ends = ends[validation_positions]
                overlaps = (
                    (starts[:, np.newaxis] < validation_ends[np.newaxis, :])
                    & (ends[:, np.newaxis] > validation_starts[np.newaxis, :])
                ).any(axis=1)
                training_positions = np.flatnonzero(~validation_mask & ~overlaps)
                if len(training_positions) > 0:
                    session_validation = {
                        int(group_indices[position]) for position in positions
                    }
                    session_training = {
                        int(group_indices[position]) for position in training_positions
                    }
                    return session_validation, session_training

        return None

    def _distributed_block_candidates(
        self, length: int, validation_count: int, block_count: int
    ) -> List[List[int]]:
        """Generate reproducible block layouts spread over a session timeline."""
        block_sizes = np.full(block_count, validation_count // block_count, dtype=int)
        block_sizes[: validation_count % block_count] += 1

        minimum_gaps = np.asarray(
            [0, *([1] * (block_count - 1)), 0], dtype=int
        )
        remaining = length - validation_count - int(minimum_gaps.sum())
        if remaining < 0:
            return []

        gap_layouts: list[np.ndarray] = []
        balanced = np.full(block_count + 1, remaining // (block_count + 1), dtype=int)
        remainder = remaining % (block_count + 1)
        if remainder:
            extra_positions = self.rng.permutation(block_count + 1)[:remainder]
            balanced[extra_positions] += 1
        gap_layouts.append(minimum_gaps + balanced)

        # Alternative seeded layouts avoid rejecting a session merely because one
        # balanced placement happens to purge all possible training windows.
        for _ in range(31):
            extras = self.rng.multinomial(
                remaining, np.full(block_count + 1, 1.0 / (block_count + 1))
            )
            gap_layouts.append(minimum_gaps + extras)

        layouts: List[List[int]] = []
        for gaps in gap_layouts:
            positions: list[int] = []
            cursor = int(gaps[0])
            for block_index, block_size in enumerate(block_sizes):
                positions.extend(range(cursor, cursor + int(block_size)))
                cursor += int(block_size) + int(gaps[block_index + 1])
            layouts.append(positions)
        return layouts

    def _assign_unsplittable_sessions(
        self, sessions: list[tuple[int, pd.DataFrame]]
    ) -> Tuple[set[int], set[int]]:
        """Assign indivisible sessions wholly, stratified by their window count."""
        if not sessions:
            return set(), set()

        by_window_count: dict[int, list[tuple[int, pd.DataFrame]]] = {}
        for session in sessions:
            by_window_count.setdefault(len(session[1]), []).append(session)

        validation: set[int] = set()
        training: set[int] = set()
        self._unsplittable_session_count += len(sessions)
        for window_count in sorted(by_window_count):
            group = by_window_count[window_count]
            order = self.rng.permutation(len(group)).tolist()
            validation_session_count = int(
                np.floor(len(group) * self.val_percentage + 0.5)
            )
            validation_positions = set(order[:validation_session_count])
            for position, (_, session_df) in enumerate(group):
                target = validation if position in validation_positions else training
                target.update(int(index) for index in session_df.index)

            training_session_count = len(group) - validation_session_count
            summary = self._unsplittable_by_window_count.setdefault(
                window_count, [0, 0, 0]
            )
            summary[0] += len(group)
            summary[1] += validation_session_count
            summary[2] += training_session_count
        return validation, training

    def _purge_overlaps(
        self,
        training: set[int],
        validation: set[int],
        window_df: pd.DataFrame,
    ) -> set[int]:
        """Remove training intervals intersecting any validation interval."""
        if not validation:
            return training

        keep = training.copy()
        # ``training`` is produced from one grouped session by
        # ``_split_session_strict``. Restricting this lookup to the candidate
        # rows avoids repeatedly materializing the complete window table for
        # every session in large datasets.
        train_group = window_df.loc[list(training)]
        for session_id, val_group in window_df.loc[list(validation)].groupby(
            "session_id"
        ):
            session_train_group = train_group[
                train_group["session_id"] == session_id
            ]
            merged_intervals: list[tuple[int, int]] = []
            intervals = val_group[["start_index", "end_index"]].sort_values(
                "start_index"
            )
            for start, end in intervals.itertuples(index=False, name=None):
                start_int, end_int = int(start), int(end)
                if merged_intervals and start_int <= merged_intervals[-1][1]:
                    previous_start, previous_end = merged_intervals[-1]
                    merged_intervals[-1] = (
                        previous_start,
                        max(previous_end, end_int),
                    )
                else:
                    merged_intervals.append((start_int, end_int))

            overlaps_any = pd.Series(False, index=session_train_group.index)
            for start, end in merged_intervals:
                overlaps_any |= (session_train_group["start_index"] < end) & (
                    session_train_group["end_index"] > start
                )
            keep.difference_update(
                int(index) for index in session_train_group.index[overlaps_any]
            )
        return keep

    def _check_indices_overlap(
        self, train_indices: List[int], val_indices: List[int], test_indices: List[int]
    ) -> bool:
        """Return ``True`` when any split pair shares at least one index."""
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)

        if train_set.intersection(val_set):
            return True
        if train_set.intersection(test_set):
            return True
        if val_set.intersection(test_set):
            return True

        return False

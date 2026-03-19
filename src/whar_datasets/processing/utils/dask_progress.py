from typing import Any, Set

from dask.callbacks import Callback
from tqdm import tqdm


class SharedTqdmDaskCallback(Callback):
    """Update an existing tqdm instance for each finished dask task."""

    def __init__(self, pbar: tqdm):
        self._pbar = pbar
        self._seen: Set[Any] = set()

    def _start(self, dsk):
        self._seen = set()

    def _posttask(self, key, result, dsk, state, worker_id):
        if key in self._seen:
            return
        self._seen.add(key)
        self._pbar.update(1)

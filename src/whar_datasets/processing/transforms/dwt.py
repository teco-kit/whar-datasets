from typing import List, Tuple

import numpy as np
import pywt
from matplotlib import pyplot as plt


def repeat_each_cell(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Repeat each entry so the returned array reaches ``target_len``."""
    original_len = len(arr)
    repeat_count = target_len // original_len
    remainder = target_len % original_len

    repeated = np.repeat(arr, repeat_count)
    if remainder > 0:
        repeated = np.concatenate([repeated, arr[:remainder]])
    return repeated


def blockwise_average(arr: np.ndarray, original_len: int) -> np.ndarray:
    """Collapse a repeated array into its original size by averaging blocks."""
    block_size = len(arr) // original_len
    return np.mean(
        arr[: original_len * block_size].reshape(original_len, block_size), axis=1
    )


def signal_to_dwt_grid(
    signal: np.ndarray,
    wavelet: str = "db4",
    mode: str = "periodization",
    level: int | None = None,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Convert ``(time, channels)`` signals into padded multi-level DWT grids."""
    # (time_steps, sensor_channels)

    level = level or int(
        pywt.dwt_max_level(signal.shape[0], pywt.Wavelet("db4").dec_len)  # type: ignore
    )

    signal = signal.T  # (T, C) → (C, T)
    C, T = signal.shape

    all_grids = []
    all_lengths = []

    for c in range(C):
        coeffs = pywt.wavedec(signal[c], wavelet, level=level, mode=mode)
        lengths = [len(c) for c in coeffs]
        max_len = max(lengths)

        # padded = [np.pad(c, (0, max_len - len(c))) for c in coeffs]
        repeated = [
            repeat_each_cell(arr, max_len) if len(arr) < max_len else arr
            for arr in coeffs
        ]

        grid = np.stack(repeated, axis=0)  # (levels+1, max_len)
        all_grids.append(grid)
        all_lengths.append(lengths)

    grid = np.stack(all_grids, axis=0)
    # (sensor_channels, levels+1, max_len)

    return grid, all_lengths


def dtw_grid_to_signal(
    grid: np.ndarray,
    all_lengths: List[List[int]],
    wavelet: str = "db4",
    mode: str = "periodization",
) -> np.ndarray:
    """Approximate inverse transform from DWT grid representation to signal."""
    # (sensor_channels, levels+1, max_len)

    C = grid.shape[0]
    signals = []

    for c in range(C):
        # lengths = all_lengths[c]
        # coeffs = [grid[c, i, : lengths[i]] for i in range(len(lengths))]
        # signal = pywt.waverec(coeffs, wavelet, mode=mode)
        # signals.append(signal)

        lengths = all_lengths[c]
        max_len = grid.shape[-1]

        coeffs = []
        for i, orig_len in enumerate(lengths):
            repeated_arr = grid[c, i]
            recovered = blockwise_average(repeated_arr, orig_len)
            coeffs.append(recovered)

        signal = pywt.waverec(coeffs, wavelet, mode=mode)
        signals.append(signal)

    max_len = max(len(s) for s in signals)
    signals = [np.pad(s, (0, max_len - len(s))) for s in signals]

    signals_np = np.stack(signals, axis=0)  # (C, T)
    signals_np = signals_np.T  # → back to (T, C)

    return signals_np


def plot_dwt_grid(grid, channel_names=None, cmap="viridis"):
    """
    Plot DWT grid (C, levels+1, max_len) as heatmaps.
    Each channel gets its own subplot.
    """
    C, L, T = grid.shape
    fig, axes = plt.subplots(C, 1, figsize=(8, 1 * C), sharex=True)

    if C == 1:
        axes = [axes]  # make it iterable for 1 channel

    for i in range(C):
        ax = axes[i]  # type: ignore
        im = ax.imshow(grid[i], aspect="auto", origin="lower", cmap=cmap)
        ax.set_ylabel("DWT Level")
        title = f"Channel {i}" if channel_names is None else channel_names[i]
        ax.set_title(title)
        fig.colorbar(im, ax=ax, orientation="vertical")

    plt.xlabel("Time (DWT padded axis)")
    plt.tight_layout()
    plt.show()

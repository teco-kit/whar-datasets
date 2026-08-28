from types import SimpleNamespace

import numpy as np

from whar_datasets.processing.transforms.stft import signal_to_stft
from whar_datasets.processing.utils.transform import get_transform
from whar_datasets.utils.types import TransformType


def test_signal_to_stft_shape_contract() -> None:
    x = np.random.randn(128, 3).astype(np.float32)

    stft_mag, stft_phase, stft_info = signal_to_stft(x, sampling_freq=50.0)

    assert stft_mag.ndim == 3
    assert stft_mag.shape[0] == 3
    assert stft_mag.shape[1] > 0
    assert stft_mag.shape[2] > 0
    assert np.all(stft_mag >= 0)
    assert np.isfinite(stft_mag).all()
    assert stft_phase.shape == stft_mag.shape
    assert np.isfinite(stft_phase).all()
    assert stft_info.shape == (3,)
    assert stft_info.dtype.kind in {"i", "u"}
    assert np.all(stft_info > 0)


def test_get_transform_stft_returns_reconstruction_info() -> None:
    x = np.random.randn(96, 2).astype(np.float32)
    cfg = SimpleNamespace(transform=TransformType.STFT, sampling_freq=100)

    transform = get_transform(cfg)  # type: ignore[arg-type]
    out = transform(x)

    assert len(out) == 3
    stft_mag, stft_phase, stft_info = out

    assert stft_mag.ndim == 3
    assert stft_mag.shape[0] == 2
    assert stft_phase.shape == stft_mag.shape
    assert stft_info.shape == (3,)


def test_get_transform_dwt_still_returns_grid_and_metadata() -> None:
    x = np.random.randn(96, 2).astype(np.float32)
    cfg = SimpleNamespace(transform=TransformType.DWT, sampling_freq=100)

    transform = get_transform(cfg)  # type: ignore[arg-type]
    out = transform(x)

    assert len(out) == 2
    dwt_grid, dwt_lengths = out

    assert dwt_grid.ndim == 3
    assert dwt_grid.shape[0] == 2
    assert dwt_lengths.shape[0] == 2

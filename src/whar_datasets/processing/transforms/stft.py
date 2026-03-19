import numpy as np
from scipy.signal import stft


def signal_to_stft(
    signal: np.ndarray,
    sampling_freq: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # (time_steps, sensor_channels)
    if signal.ndim != 2:
        raise ValueError("Expected signal shape (time_steps, sensor_channels).")

    signal_t = signal.T  # (T, C) -> (C, T)
    channels, time_steps = signal_t.shape

    segment_len = nperseg if nperseg is not None else min(64, time_steps)
    segment_len = max(1, int(segment_len))
    overlap = noverlap if noverlap is not None else segment_len // 2
    overlap = max(0, int(overlap))
    if overlap >= segment_len:
        raise ValueError("noverlap must be strictly smaller than nperseg.")

    magnitudes: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    for channel_idx in range(channels):
        _, _, stft_values = stft(
            signal_t[channel_idx],
            fs=sampling_freq,
            window=window,
            nperseg=segment_len,
            noverlap=overlap,
            boundary="zeros",
            padded=True,
        )
        magnitudes.append(np.abs(stft_values).astype(np.float32))
        phases.append(np.angle(stft_values).astype(np.float32))

    # (sensor_channels, freq_bins, time_bins)
    magnitude = np.stack(magnitudes, axis=0)
    phase = np.stack(phases, axis=0)
    reconstruction_info = np.array(
        [time_steps, segment_len, overlap], dtype=np.int32
    )
    return magnitude, phase, reconstruction_info

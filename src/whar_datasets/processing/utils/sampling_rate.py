from collections.abc import Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.utils.logging import logger

_SOURCE_RATE_RTOL = 0.05
_SOURCE_RATE_ATOL = 0.05
_MIN_SAMPLES_FOR_RATE_ESTIMATE = 11


def audit_session_sampling_rates(
    cfg: WHARConfig,
    sessions: Mapping[int, pd.DataFrame],
) -> None:
    """Report timestamp-derived rates and reject fixed-rate config mismatches.

    Rates are estimated from the median positive timestamp interval per
    session, which is robust to occasional dropped samples. Input timestamps
    must already have been converted by the parser to datetime values.
    """
    session_rates: dict[int, float] = {}
    insufficient_sessions: list[int] = []

    for session_id, session in tqdm(
        sessions.items(), desc="Auditing session sampling rates"
    ):
        if "timestamp" not in session:
            raise ValueError(
                f"Session {session_id} has no timestamp column; cannot audit its "
                "sampling rate."
            )
        if not pd.api.types.is_datetime64_any_dtype(session["timestamp"]):
            raise ValueError(
                f"Session {session_id} timestamps are not datetime typed; resolve "
                "their raw units and convert them in the parser before auditing "
                "the sampling rate."
            )
        timestamps = session["timestamp"]
        if timestamps.isna().any():
            raise ValueError(
                f"Session {session_id} contains null timestamps; cannot audit its "
                "sampling rate."
            )
        if len(timestamps) < _MIN_SAMPLES_FOR_RATE_ESTIMATE:
            insufficient_sessions.append(int(session_id))
            continue

        # Use pandas' duration conversion so datetime64[ms], datetime64[us],
        # and datetime64[ns] caches all produce seconds correctly.
        steps_seconds = (
            timestamps.diff().dt.total_seconds().dropna().to_numpy(dtype=np.float64)
        )
        if (steps_seconds <= 0).any():
            raise ValueError(
                f"Session {session_id} has non-increasing timestamps; sampling "
                "rate cannot be audited until timestamp ordering/duplicates are "
                "resolved."
            )

        if cfg.max_session_gap_seconds is not None:
            large_gaps = steps_seconds > cfg.max_session_gap_seconds
            if large_gaps.any():
                largest_gap = float(steps_seconds.max())
                raise ValueError(
                    f"Session {session_id} contains a {largest_gap:.6f}s timestamp "
                    "gap while auditing its sampling rate; split the session at "
                    "that gap before windowing."
                )

        median_step = float(np.median(steps_seconds))
        session_rates[int(session_id)] = 1.0 / median_step

    if not session_rates:
        logger.warning(
            "Could not estimate sampling rate for any of the %d parsed sessions; "
            "all sessions have fewer than %d samples.",
            len(sessions),
            _MIN_SAMPLES_FOR_RATE_ESTIMATE,
        )
        return

    rates = np.asarray(list(session_rates.values()), dtype=np.float64)
    q05, median_rate, q95 = np.quantile(rates, [0.05, 0.50, 0.95])
    logger.info(
        "Timestamp-derived sampling rates across %d sessions: "
        "5th percentile=%.4f Hz, median=%.4f Hz, 95th percentile=%.4f Hz.",
        len(session_rates),
        q05,
        median_rate,
        q95,
    )
    if insufficient_sessions:
        logger.warning(
            "Could not reliably estimate a sampling rate for %d short sessions "
            "with fewer than %d samples (first session IDs: %s).",
            len(insufficient_sessions),
            _MIN_SAMPLES_FOR_RATE_ESTIMATE,
            insufficient_sessions[:10],
        )

    if cfg.sampling_freq is None:
        return

    mismatches = {
        session_id: rate
        for session_id, rate in session_rates.items()
        if not np.isclose(
            rate,
            cfg.sampling_freq,
            rtol=_SOURCE_RATE_RTOL,
            atol=_SOURCE_RATE_ATOL,
        )
    }
    if mismatches:
        examples = ", ".join(
            f"{session_id}: {rate:.4f} Hz"
            for session_id, rate in list(mismatches.items())[:10]
        )
        raise ValueError(
            f"Dataset '{cfg.dataset_id}' is configured for a fixed source rate "
            f"of {cfg.sampling_freq:g} Hz, but {len(mismatches)} of "
            f"{len(session_rates)} parsed sessions disagree by more than "
            f"5% (median across sessions {median_rate:.4f} Hz). Example "
            f"session rates: {examples}. Verify the raw timestamp units and "
            "dataset metadata; correct sampling_freq or set source_rate_mode="
            "'per_session' if the source genuinely varies."
        )

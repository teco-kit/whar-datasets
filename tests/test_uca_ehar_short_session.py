"""Short transition fragments must not determine UCA-EHAR's source rate."""

import unittest

import numpy as np
import pandas as pd

from whar_datasets.config.cfg_uca_ehar import cfg_uca_ehar
from whar_datasets.processing.utils.sessions import _process_session_data


class UcaEharShortSessionTests(unittest.TestCase):
    def test_two_sample_fragment_cannot_produce_window(self) -> None:
        cfg = cfg_uca_ehar.model_copy(update={"resampling_freq": 50})
        session = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([300_937, 300_941], unit="ms"),
                **{
                    channel: np.ones(2, dtype=np.float32)
                    for channel in cfg.available_channels
                },
            }
        )

        metadata, windows = _process_session_data(cfg, 215, session)

        self.assertIsNone(metadata)
        self.assertIsNone(windows)

    def test_regular_25_hz_session_resamples_to_50_hz(self) -> None:
        cfg = cfg_uca_ehar.model_copy(update={"resampling_freq": 50})
        samples = 100
        session = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=samples, freq="40ms"
                ),
                **{
                    channel: np.arange(samples, dtype=np.float32)
                    for channel in cfg.available_channels
                },
            }
        )

        metadata, windows = _process_session_data(cfg, 0, session)

        self.assertIsNotNone(metadata)
        self.assertIsNotNone(windows)
        self.assertEqual(len(next(iter(windows.values()))), 100)

    def test_long_genuinely_mismatched_session_still_raises(self) -> None:
        cfg = cfg_uca_ehar.model_copy(update={"resampling_freq": 50})
        samples = 750
        session = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2020-01-01", periods=samples, freq="4ms"
                ),
                **{
                    channel: np.arange(samples, dtype=np.float32)
                    for channel in cfg.available_channels
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "disagrees with configured"):
            _process_session_data(cfg, 0, session)


if __name__ == "__main__":
    unittest.main()

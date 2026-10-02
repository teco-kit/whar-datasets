"""Per-session source rates still produce fixed-duration output windows."""

import unittest

import numpy as np
import pandas as pd
from pydantic import ValidationError

from whar_datasets.config.cfg_hhar import cfg_hhar
from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.utils.sessions import _process_session_data


class VariableSourceRateTests(unittest.TestCase):
    def test_target_rate_is_required(self) -> None:
        values = {**cfg_hhar.__dict__, "resampling_freq": None}
        with self.assertRaisesRegex(ValidationError, "requires resampling_freq"):
            WHARConfig.model_validate(values)

    def test_different_source_rates_produce_the_same_window_size(self) -> None:
        self.assertIsNone(cfg_hhar.sampling_freq)
        self.assertEqual(cfg_hhar.source_rate_mode, "per_session")
        self.assertEqual(cfg_hhar.resampling_freq, 50)
        for source_rate in (100, 200):
            with self.subTest(source_rate=source_rate):
                count = source_rate * 3
                session = pd.DataFrame(
                    {
                        "timestamp": pd.date_range(
                            "2020-01-01",
                            periods=count,
                            freq=pd.to_timedelta(1 / source_rate, unit="s"),
                        ),
                        **{
                            channel: np.arange(count, dtype=np.float32)
                            for channel in cfg_hhar.available_channels
                        },
                    }
                )
                metadata, windows = _process_session_data(cfg_hhar, 0, session)
                self.assertIsNotNone(metadata)
                self.assertIsNotNone(windows)
                self.assertEqual(len(next(iter(windows.values()))), 100)
                self.assertEqual(
                    (
                        metadata.window_end.iloc[0]
                        - metadata.window_start.iloc[0]
                    ).total_seconds(),
                    2.0,
                )


if __name__ == "__main__":
    unittest.main()

"""GOTOV uses a millisecond-rounded 600/7 Hz merged sensor timeline."""

import unittest

import numpy as np
import pandas as pd

from whar_datasets.config.cfg_gotov import GOTOV_SOURCE_FREQ, cfg_gotov
from whar_datasets.processing.utils.resampling import resample


class GotovSourceRateTests(unittest.TestCase):
    def test_rounded_millisecond_timeline_resamples_to_50_hz(self) -> None:
        self.assertAlmostEqual(cfg_gotov.sampling_freq, 600 / 7)
        raw_ms = 1_456_483_921_000 + np.rint(
            np.arange(300, dtype=np.float64) * 1_000 / GOTOV_SOURCE_FREQ
        ).astype(np.int64)
        source = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(raw_ms, unit="ms"),
                "ankle_x": np.arange(300, dtype=np.float32),
            }
        )

        output = resample(source, 50, source_freq=cfg_gotov.sampling_freq)

        self.assertEqual(output.columns.tolist(), ["timestamp", "ankle_x"])
        self.assertTrue(output.timestamp.is_monotonic_increasing)
        self.assertAlmostEqual(
            (output.timestamp.iloc[1] - output.timestamp.iloc[0]).total_seconds(),
            0.02,
        )
        self.assertGreater(len(output), 170)


if __name__ == "__main__":
    unittest.main()

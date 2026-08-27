from typing import Dict, Tuple

import pandas as pd

from whar_datasets.config.config import WHARConfig


def parse_dummy(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del dir, activity_id_col
    raise NotImplementedError("The dummy configuration has no dataset parser.")


cfg_dummy = WHARConfig(
    # Info + common
    dataset_id="dummy",
    dataset_url="https://example.com/dummy",
    download_url="https://example.com/dummy.tar.gz",
    sampling_freq=20,
    num_of_subjects=36,
    num_of_activities=6,
    num_of_channels=3,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_dummy,
    # Preprocessing (selections + sliding window)
    available_activities=[],
    selected_activities=[],
    available_channels=[],
    selected_channels=[],
    window_time=2,
    window_overlap=0.5,
    # Training (split info)
)

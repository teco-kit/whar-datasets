from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from whar_datasets.config.activity_name_utils import canonicalize_activity_name_list
from whar_datasets.config.config import WHARConfig

HARMES_SENSOR_CHANNELS: list[str] = [
    "acc_x_left",
    "acc_y_left",
    "acc_z_left",
    "gyro_x_left",
    "gyro_y_left",
    "gyro_z_left",
    "acc_x_right",
    "acc_y_right",
    "acc_z_right",
    "gyro_x_right",
    "gyro_y_right",
    "gyro_z_right",
]  # For IMU_L AND IMU_R!

HARMES_ACTIVITY_NAMES: list[str] = [
    "Floor cleaning",
    "Making tea",
    "Vacuum Cleaning",
    "Window cleaning",
    "Brushing teeth",
    "Washing dishes",
    "Cutting vegetables",
    "Putting away the dishes",
    "Cleaning out the dishwasher",
    "Watering plant",
    "Cleaning table",
    "Washing hands",
    "Drinking",
    "Cream hands",
    "Disinfecting hands"
]


def parse_harmes(
    dir: str, activity_id_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    """Parse the preprocessed HARMES IMU-only release (one pickle per participant).

    Each pickle holds a list of recordings with keys ``rec_id``, ``IMU_L``,
    ``IMU_R`` and ``Labels``. ``rec_id`` is ``SSRR`` (subject, recording) or
    ``SSRRPP`` when a recording was split into parts recorded on different
    days; each part has its own labels and its own ``ts_sync`` clock that
    restarts at ~0, so parts are parsed independently.

    Audited against the raw files (all 20 participants, 69 scripted recordings,
    2861 start/end label pairs -> 2818 sessions). Known source properties:

    - Recording 04 of every participant is free-form daily life with
      open-vocabulary labels (e.g. "Ironing laundry", "Cutting wood") that are
      not among the 15 scripted activities, so it is skipped.
    - In the first part of 9 split recordings (050101, 090301, 130101, 130201,
      140101, 150101, 180201, 190201, 200101) the IMU stream ends before the
      last label pairs. These 43 label pairs have no sensor data and are
      ignored; the remaining data is correct.
    - Recording 0101 has one overlapping pair: "Vacuum Cleaning" ends exactly
      120 s after its start (likely an auto-filled end), after the following
      "Brushing teeth" has started. Labels are applied in source order and a
      later label overwrites an earlier one, so vacuuming is cut at the start
      of brushing teeth (6001 -> 4571 samples).
    """
    del activity_id_col

    data_root = Path(dir)
    file_paths = sorted(data_root.glob("participant*.pkl"))
    if not file_paths:
        raise FileNotFoundError(f"No HARMES pkl files found under '{dir}'.")

    activity_map = {
        activity_name: idx for idx, activity_name in enumerate(HARMES_ACTIVITY_NAMES)
    }
    step_ms = int(1e3 / 50)

    sessions: dict[int, pd.DataFrame] = {}
    session_rows: list[dict[str, int]] = []
    next_session_id = 0

    loop = tqdm(file_paths, desc="Parsing HARMES (IMU-ONLY)")
    for file_path in loop:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        for recording in data:
            if recording["rec_id"][2:4] == "04":
                # Recording 04 is free-form with labels outside the 15 scripted
                # activities; skip it entirely.
                continue
            left = recording["IMU_L"]
            right = recording["IMU_R"]
            # IMU_L and IMU_R share the same 50 Hz row index; their ts_sync values
            # differ by at most one sample (19 ms) without drift, so the two
            # sensors are aligned by row and trimmed to the shorter stream
            # (drops at most 138 trailing samples).
            min_len = min(left.shape[0], right.shape[0])
            left = left.iloc[:min_len].copy()
            right = right.iloc[:min_len].copy()
            left["label"] = pd.NA
            desc: str | None = None
            start_ts = None
            # Labels are start/end pairs in ms on the IMU_L ts_sync clock;
            # "comment" rows (notes such as "voice -20") are ignored. Pairs are
            # applied in source order, so on overlap the later label wins (only
            # occurs once, in 0101). Pairs beyond the end of the IMU stream match
            # no rows and are thereby ignored (43 pairs, see docstring).
            for i, label_col in recording["Labels"].iterrows():
                start = label_col["Type"] == "start"
                if start:
                    start_ts = label_col["ts_sync"]
                    desc = label_col["Description"]
                elif label_col["Type"] == "end" and desc == label_col["Description"]:
                    mask = (
                        (left["ts_sync"] >= start_ts) &
                        (left["ts_sync"] <= label_col["ts_sync"])
                    )
                    left.loc[mask, "label"] = desc
            frame = left.drop(columns=["ts_sync"]).rename(
                columns=lambda c: f"{c}_left" if c in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"] else c
                ).join(
                    right.drop(columns=["ts_sync"]).rename(
                        columns=lambda c: f"{c}_right" if c in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"] else c
                    )
                )


            frame["label"] = frame["label"].astype("string").str.strip()
            frame["label"] = frame["label"].replace(
                {"null": pd.NA, "None": pd.NA, "nan": pd.NA, "": pd.NA}  # type: ignore
            )
            frame["sbj_id"] = int(recording["rec_id"][:2]) - 1


            frame["activity_id"] = frame["label"].map(activity_map).astype("Int64")
            frame["subject_id"] = frame["sbj_id"].astype("int32")


            activity = frame["activity_id"]
            prev_activity = activity.shift()

            activity_breaks = (
                activity.ne(prev_activity).fillna(False)
                | (activity.isna() != prev_activity.isna())
            )

            session_breaks = (
                frame["subject_id"].ne(frame["subject_id"].shift()).fillna(True)
                | activity_breaks
            )

            session_breaks.iloc[0] = True
            local_session_ids = session_breaks.astype("int64").cumsum() - 1

            for _, chunk in frame.groupby(local_session_ids, sort=False):
                if chunk.empty or pd.isna(chunk["activity_id"].iloc[0]):
                    continue

                # Raw ts_sync steps are <= 39 ms (no recording gaps inside a
                # label), so a regular 50 Hz time axis is synthesized.
                timestamps = pd.to_datetime(
                    np.arange(len(chunk), dtype="int64") * step_ms, unit="ms"
                )
                session = pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        **{col: chunk[col].to_numpy() for col in HARMES_SENSOR_CHANNELS},
                    }
                )
                session = session.astype(
                    {
                        "timestamp": "datetime64[ms]",
                        **{col: "float32" for col in HARMES_SENSOR_CHANNELS},
                    }
                )
                session[HARMES_SENSOR_CHANNELS] = session[HARMES_SENSOR_CHANNELS].round(6)
                sessions[next_session_id] = session.reset_index(drop=True)

                session_rows.append(
                    {
                        "session_id": next_session_id,
                        "subject_id": int(chunk["subject_id"].iloc[0]),
                        "activity_id": int(chunk["activity_id"].iloc[0]),
                    }
                )
                next_session_id += 1

    if not sessions:
        raise ValueError(
            "No HARMES sessions were created. Activity labels or subject identifiers are missing."
        )

    activity_df = pd.DataFrame(
        {
            "activity_id": list(range(len(HARMES_ACTIVITY_NAMES))),
            "activity_name": HARMES_ACTIVITY_NAMES,
        }
    ).astype({"activity_id": "int32", "activity_name": "string"})

    session_df = pd.DataFrame(session_rows).astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    if session_df["subject_id"].nunique() == 0:
        raise ValueError("HARMES subject identifiers are missing after parsing.")
    if session_df["activity_id"].nunique() == 0:
        raise ValueError("HARMES activity labels are missing after parsing.")

    return activity_df, session_df, sessions


SELECTED_ACTIVITIES = HARMES_ACTIVITY_NAMES

cfg_harmes = WHARConfig(
    # Info + common
    dataset_id="harmes",
    dataset_url="https://doi.org/10.5281/zenodo.19425718",
    download_url="https://zenodo.org/records/23416820/files/HARMES-preprocessed_IMU_ONLY.zip",
    sampling_freq=50,
    num_of_subjects=20,
    num_of_activities=15,
    num_of_channels=12,
    # Parsing
    parse=parse_harmes,
    activity_id_col="activity_id",
    # Preprocessing (selections + sliding window)
    available_activities=canonicalize_activity_name_list(HARMES_ACTIVITY_NAMES),
    selected_activities=canonicalize_activity_name_list(SELECTED_ACTIVITIES),
    available_channels=HARMES_SENSOR_CHANNELS,
    selected_channels=HARMES_SENSOR_CHANNELS,
    # Training (split info)
)

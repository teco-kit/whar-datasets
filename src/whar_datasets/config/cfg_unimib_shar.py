import os
from typing import Dict, List, Tuple

import pandas as pd
import scipy
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig


def find_unimib_full_data_mat(dir: str) -> str:
    direct_path = os.path.join(dir, "UniMiB-SHAR", "data", "full_data.mat")
    if os.path.isfile(direct_path):
        return direct_path

    fallback_path = os.path.join(dir, "full_data.mat")
    if os.path.isfile(fallback_path):
        return fallback_path

    for root, _, files in os.walk(dir):
        if "full_data.mat" in files:
            return os.path.join(root, "full_data.mat")

    raise FileNotFoundError(f"Could not locate 'full_data.mat' under '{dir}'.")


def parse_unimib(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    del activity_id_col

    file_path = find_unimib_full_data_mat(dir)
    full_data = scipy.io.loadmat(file_path, simplify_cells=True)["full_data"]

    session_rows: List[dict] = []
    sessions: Dict[int, pd.DataFrame] = {}
    next_session_id = 0

    for subject_id, row in enumerate(full_data):
        activity_struct = row[0]
        for activity_name in activity_struct._fieldnames:
            trials = getattr(activity_struct, activity_name)
            for trial_data in trials:
                trial_df = pd.DataFrame(
                    trial_data.T,
                    columns=[
                        "acc_x",
                        "acc_y",
                        "acc_z",
                        "timestamp_sys",
                        "timestamp_sec",
                        "magnitude",
                    ],
                ).dropna()

                if trial_df.empty:
                    continue

                trial_df = trial_df.sort_values("timestamp_sec").reset_index(drop=True)
                if trial_df.empty:
                    continue

                session_df = trial_df[["acc_x", "acc_y", "acc_z"]].copy()
                sampling_interval_ms = 1 / 50 * 1e3
                session_df["timestamp"] = pd.to_datetime(
                    session_df.index * sampling_interval_ms, unit="ms"
                )
                session_df = session_df[["timestamp", "acc_x", "acc_y", "acc_z"]]
                session_df[["acc_x", "acc_y", "acc_z"]] = session_df[
                    ["acc_x", "acc_y", "acc_z"]
                ].round(6)
                session_df = session_df.astype(
                    {
                        "timestamp": "datetime64[ms]",
                        "acc_x": "float32",
                        "acc_y": "float32",
                        "acc_z": "float32",
                    }
                )

                sessions[next_session_id] = session_df
                session_rows.append(
                    {
                        "session_id": next_session_id,
                        "subject_id": subject_id,
                        "activity_name": activity_name,
                    }
                )
                next_session_id += 1

    if not sessions:
        raise ValueError("No UniMiB-SHAR sessions could be parsed from full_data.mat.")

    session_metadata = pd.DataFrame(session_rows)
    session_metadata["activity_id"] = pd.factorize(
        session_metadata["activity_name"], sort=False
    )[0]
    session_metadata = session_metadata.drop(columns=["activity_name"])

    activity_metadata = (
        pd.DataFrame(session_rows)[["activity_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    activity_metadata["activity_id"] = pd.factorize(
        activity_metadata["activity_name"], sort=False
    )[0]
    activity_metadata = activity_metadata[["activity_id", "activity_name"]]

    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    loop = tqdm(sorted(sessions.keys()))
    loop.set_description("Creating sessions")

    for session_id in loop:
        sessions[session_id] = sessions[session_id].reset_index(drop=True)

    return activity_metadata, session_metadata, sessions


cfg_unimib = WHARConfig(
    # Info + common
    dataset_id="unimib_shar",
    download_url=r"https://uca49515d91834b9e696812321f7.dl.dropboxusercontent.com/cd/0/get/C7Su1jAcRlICf-4t_U5oSJuHvhm2gar11GVbPhu3OTb3gufsm3YlMjJJ9bLluxkyxCWGrEWSgCAMCqUlJxfv3WbSDoVhgZUGkWyNelvt-wScE8WNn5RsEh7D0B60RBRt-JAiNkOVDNUG5D8rJsygjP-i4jE7BLOvGd6juVN2putmYdq0kSyvI2URhmJIQamICnU/file?_download_id=791856115059715474932647043585648413589735310769281530879669014&_log_download_success=1&_notify_domain=www.dropbox.com&dl=1",
    sampling_freq=50,
    num_of_subjects=30,
    num_of_activities=17,
    num_of_channels=3,
    datasets_dir="./datasets",
    parse=parse_unimib,
    activity_names=[
        "StandingUpFS",
        "StandingUpFL",
        "Walking",
        "Running",
        "GoingUpS",
        "Jumping",
        "GoingDownS",
        "LyingDownFS",
        "SittingDown",
        "FallingForw",
        "FallingRight",
        "FallingBack",
        "HittingObstacle",
        "FallingWithPS",
        "FallingBackSC",
        "Syncope",
        "FallingLeft",
    ],
    sensor_channels=["acc_x", "acc_y", "acc_z"],
    window_time=3.02,
    window_overlap=0.5,
)

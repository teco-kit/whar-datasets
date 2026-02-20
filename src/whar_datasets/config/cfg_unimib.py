import os
import scipy
from typing import Dict, Tuple
import pandas as pd
from tqdm import tqdm
from whar_datasets.config.config import WHARConfig


def parse_unimib(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:

    file_path = os.path.join(dir, "UniMiB-SHAR", "data", "full_data.mat")

    mat = scipy.io.loadmat(file_path, simplify_cells=True)
    full_data = mat["full_data"]

    all_dfs = []

    for subject_idx, row in enumerate(full_data):
        for fieldname in row[0]._fieldnames:
            trials = getattr(row[0], fieldname)
            for _, trial_data in enumerate(trials):
                # x,y,z sind Zeilen, statt spalten
                df_trial = pd.DataFrame(trial_data.T)

                df_trial.columns = [
                    "acc_x",
                    "acc_y",
                    "acc_z",
                    "timestamp_sys",
                    "timestamp_sec",
                    "magnitude",
                ]

                df_trial["subject_id"] = subject_idx
                df_trial["activity_name"] = fieldname
                all_dfs.append(df_trial)

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna().reset_index(drop=True)

    df["timestamp"] = pd.to_datetime(df["timestamp_sec"], unit="s")

    df["activity_id"] = pd.factorize(df["activity_name"])[0]

    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )
    df["session_id"] = changes.cumsum() - 1

    activity_metadata = (
        df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"], keep="first")
        .reset_index(drop=True)
    )

    session_metadata = (
        df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates(subset=["session_id"], keep="first")
        .reset_index(drop=True)
    )

    session_metadata["activity_id"] = session_metadata["activity_id"].apply(
        lambda x: [int(x)]
    )

    sessions: Dict[int, pd.DataFrame] = {}
    loop = tqdm(session_metadata["session_id"].unique())
    loop.set_description("Creating sessions")

    for session_id in loop:
        session_df = df[df["session_id"] == session_id].copy()

        session_df = session_df.drop(
            columns=[
                "session_id",
                "subject_id",
                "activity_id",
                "activity_name",
                "timestamp_sys",
                "timestamp_sec",
                "magnitude",
            ]
        ).reset_index(drop=True)

        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        dtypes["timestamp"] = "datetime64[ms]"

        session_df = session_df.astype(dtypes)

        sessions[session_id] = session_df

    return activity_metadata, session_metadata, sessions


cfg_unimib = WHARConfig(
    # Info + common
    dataset_id="unimib_shar",
    download_url=r"https://www.dropbox.com/scl/fi/g5ig8nw9qqd253dz8woax/UniMiB-SHAR.zip",
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

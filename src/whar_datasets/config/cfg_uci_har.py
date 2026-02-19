import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig


def get_df_from_files_uci_har(
    files: List[str],
    files_dir: str,
    subj_path: str,
    labels_path: str,
    slice_end: int,
) -> pd.DataFrame:
    # hyperparameters since data is already segmented into 128 readings per window with 50% overlap
    SEG_SIZE = 128
    DISPLACEMENT = 64

    # effective segment size
    eff_seg_size = SEG_SIZE - DISPLACEMENT

    # create dict to store df for each channel, use file name without "train" as key
    dict = {
        file[:slice_end]: pd.read_csv(
            os.path.join(files_dir, file), header=None, sep="\\s+"
        ).iloc[:, :eff_seg_size]
        for file in files
    }  # (num_segs, seg_size)

    for value in dict.values():
        assert value.shape[1] == eff_seg_size

    cols = list(dict.keys())

    # creates dfs from each dict, we ensure order of our specified cols through dict
    df = pd.DataFrame(
        np.stack(
            [dict[col].values.reshape(-1) for col in cols],
            axis=1,
        ),
        columns=cols,
    )  # (seg_size * num_segs, num_activities)

    # load subjects and labels as dfs
    subjects_df = pd.read_csv(subj_path, header=None, names=["subjects"])
    labels_df = pd.read_csv(labels_path, header=None, names=["labels"])
    # (num_segs, 1)

    # assert number of segs are the same
    assert df.shape[0] / eff_seg_size == subjects_df.shape[0] == labels_df.shape[0]
    NUM_SEGS = labels_df.shape[0]

    subject_ids = []
    activity_ids = []

    for i in range(NUM_SEGS):
        # get subject and label
        subject_id = subjects_df.loc[i, "subjects"]
        activity_id = labels_df.loc[i, "labels"]

        # repeat for seg_size
        subject_ids.extend(eff_seg_size * [subject_id])
        activity_ids.extend(eff_seg_size * [activity_id])

    df["subject_id"] = subject_ids
    df["activity_id"] = activity_ids
    # (seg_size * num_segs, num_activities + 2)

    return df


def parse_uci_har(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    dir = os.path.join(dir, "UCI HAR Dataset/UCI HAR Dataset/")

    # directories of raw data
    train_path = os.path.join(dir, "train/Inertial Signals/")
    test_path = os.path.join(dir, "test/Inertial Signals/")

    # file paths
    train_subj_path = os.path.join(dir, "train/subject_train.txt")
    test_subj_path = os.path.join(dir, "test/subject_test.txt")
    train_labels_path = os.path.join(dir, "train/y_train.txt")
    test_labels_path = os.path.join(dir, "test/y_test.txt")
    labels_map_path = os.path.join(dir, "activity_labels.txt")

    # get all files in train and test dirs
    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)

    # get train and test dfs
    train_df = get_df_from_files_uci_har(
        files=train_files,
        files_dir=train_path,
        subj_path=train_subj_path,
        labels_path=train_labels_path,
        slice_end=-10,
    )  # (seg_size * num_segs, num_activities + 3)

    test_df = get_df_from_files_uci_har(
        files=test_files,
        files_dir=test_path,
        subj_path=test_subj_path,
        labels_path=test_labels_path,
        slice_end=-9,
    )  # (seg_size * num_segs, num_activities + 3)

    # concat for varying splits and set index
    df = pd.concat([train_df, test_df])
    df = df.reset_index(drop=True)
    # (seg_size * (num_segs_train + num_segs_test), num_activities + 3)

    # add col with activity names
    ldf = pd.read_csv(labels_map_path, sep="\\s+", header=None, names=["id", "label"])
    df["activity_name"] = df["activity_id"].map(dict(zip(ldf.id, ldf.label)))

    # convert activity_id to categorical starting from 0
    df["activity_id"] = pd.factorize(df["activity_id"])[0]

    # identify where activity or subject changes
    changes = (df["activity_id"] != df["activity_id"].shift(1)) | (
        df["subject_id"] != df["subject_id"].shift(1)
    )

    # assign a unique session to each continuous segment
    df["session_id"] = changes.cumsum()

    # add timestamp column per session
    sampling_interval = 1 / 50 * 1e3  # 50 Hz → 0.02 seconds -> to ms
    df["timestamp"] = (
        df.groupby("session_id", group_keys=False).cumcount() * sampling_interval
    )

    # factorize
    df["activity_id"] = df["activity_id"].factorize()[0]
    df["subject_id"] = df["subject_id"].factorize()[0]
    df["session_id"] = df["session_id"].factorize()[0]

    # create activity index
    activity_metadata = (
        df[["activity_id", "activity_name"]]
        .drop_duplicates(subset=["activity_id"], keep="first")
        .reset_index(drop=True)
    )

    # create session_metadata
    session_metadata = (
        df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates(subset=["session_id"], keep="first")
        .reset_index(drop=True)
    )

    # create sessions
    sessions: Dict[int, pd.DataFrame] = {}

    # loop over sessions
    loop = tqdm(session_metadata["session_id"].unique())
    loop.set_description("Creating sessions")

    for session_id in loop:
        # get session df
        session_df = df[df["session_id"] == session_id]

        # drop nan rows
        session_df = session_df.dropna()

        # drop metadata cols
        session_df = session_df.drop(
            columns=[
                "session_id",
                "subject_id",
                "activity_id",
                "activity_name",
            ]
        ).reset_index(drop=True)

        # set types
        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"], unit="ms")
        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        dtypes["timestamp"] = "datetime64[ms]"
        float_cols = [col for col in session_df.columns if col != "timestamp"]
        session_df[float_cols] = session_df[float_cols].round(6)
        session_df = session_df.astype(dtypes)

        # add to sessions
        sessions[session_id] = session_df

    # set metadata types
    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_uci_har = WHARConfig(
    # Info + common
    dataset_id="uci_har",
    download_url="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    sampling_freq=50,
    num_of_subjects=30,
    num_of_activities=6,
    num_of_channels=9,
    datasets_dir="./datasets",
    # Parsing
    parse=parse_uci_har,
    # Preprocessing (selections + sliding window)
    activity_names=[
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ],
    sensor_channels=[
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
    ],
    window_time=2.56,
    window_overlap=0.5,
)

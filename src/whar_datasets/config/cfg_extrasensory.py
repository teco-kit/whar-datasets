from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from whar_datasets.config.config import WHARConfig


def add_session_id(df: pd.DataFrame, gap: str = "3s") -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values("timestamp").copy()
    gap_td = pd.Timedelta(gap)

    curr_labels = df["activity_id"].astype(str)
    label_change = curr_labels.shift() != curr_labels

    time_gap = df["timestamp"].diff() > gap_td

    new_session = label_change | time_gap

    df["session_id"] = new_session.cumsum().astype(int) - 1
    return df


def parse_extrasensory(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:

    session_metadata = pd.DataFrame(columns=["session_id", "subject_id", "activity_id"])
    sessions: Dict[int, pd.DataFrame] = {}
    activity_metadata = pd.DataFrame(columns=["activity_id", "activity_name"])

    # hier müssten die Pfade angepasst werden
    acc_root_path = Path(
        r"C:\Users\hohma\Teco\whar-datasets\notebooks\datasets\extrasensory\data\ExtraSensory.raw_measurements.raw_acc\raw_acc"
    )
    gyro_root_path = Path(
        r"C:\Users\hohma\ExtraSensory.raw_measurements.proc_gyro\proc_gyro"
    )
    acc_watch_root_path = Path(
        r"C:\Users\hohma\ExtraSensory.raw_measurements.watch_acc\watch_acc"
    )
    compass_watch_root_path = Path(
        r"C:\Users\hohma\ExtraSensory.raw_measurements.watch_compass\watch_compass"
    )
    labels_root_path = Path(r"C:\Users\hohma\ExtraSensory.per_uuid_features_labels")

    gyro_header = ["timestamp", "gyro_x", "gyro_y", "gyro_z"]
    acc_header = ["timestamp", "acc_x", "acc_y", "acc_z"]

    global_label_map = {}
    all_subjects = sorted([p.name for p in acc_root_path.iterdir() if p.is_dir()])
    subject_map = {name: idx for idx, name in enumerate(all_subjects)}
    last_subject_max_session_id = 0

    for subject_folder in acc_root_path.iterdir():
        if not subject_folder.is_dir():
            continue

        subject_id = subject_folder.name
        print(f"Verarbeite Subject: {subject_id}")

        label_file = labels_root_path / f"{subject_id}.features_labels.csv"
        label_lookup = {}

        if label_file.exists():
            try:
                # Die Label sind als "label_name" und dann 0/1 im Header
                header_df = pd.read_csv(label_file, nrows=0)
                label_cols = [
                    c
                    for c in header_df.columns
                    if c.startswith("label:") and "label_source" not in c
                ]

                for col in label_cols:
                    if col not in global_label_map:
                        global_label_map[col] = len(global_label_map)

                cols_to_load = ["timestamp"] + label_cols
                df_labels_raw = pd.read_csv(label_file, usecols=cols_to_load)

                sorted_labels = sorted(global_label_map, key=global_label_map.get)
                for l in sorted_labels:
                    if l not in df_labels_raw.columns:
                        df_labels_raw[l] = 0

                timestamps = df_labels_raw["timestamp"].values
                label_matrix = df_labels_raw[sorted_labels].values
                # [0, 1, 0, 1, 0] -> [1,3]
                active_ids_list = [np.nonzero(row)[0].tolist() for row in label_matrix]

                # um die später an die timestmaps zu hängen
                label_lookup = dict(zip(timestamps, active_ids_list))

            except Exception:
                label_lookup = {}

        acc_dfs_list = []
        for file in subject_folder.rglob("*.dat"):
            try:
                file_name_ts = float(file.name.split(".")[0])

                if file_name_ts not in label_lookup:
                    continue

                current_activity_id = label_lookup[file_name_ts]

                tmp_df = pd.read_csv(
                    file, names=acc_header, header=None, sep=r"\s+", engine="python"
                )
                tmp_df = tmp_df.dropna(how="all")

                tmp_df["timestamp"] = pd.to_numeric(
                    tmp_df["timestamp"], errors="coerce"
                )

                if not tmp_df.empty:
                    first_val = tmp_df["timestamp"].iloc[0]

                    if first_val < 1e9:
                        start_dt = pd.to_datetime(file_name_ts, unit="s")
                        offsets = pd.to_timedelta(tmp_df["timestamp"], unit="ms")
                        tmp_df["timestamp"] = start_dt + offsets
                    else:
                        tmp_df["timestamp"] = pd.to_datetime(
                            tmp_df["timestamp"], unit="s"
                        )

                int_sub_id = subject_map[subject_id]
                tmp_df["subject_id"] = int_sub_id
                tmp_df["activity_id"] = [current_activity_id] * len(tmp_df)

                acc_dfs_list.append(tmp_df)
            except Exception:
                continue

        if acc_dfs_list:
            df_acc = pd.concat(acc_dfs_list, ignore_index=True)
            df_acc = df_acc.sort_values("timestamp").reset_index(drop=True)
        else:
            continue

        current_gyro_folder = gyro_root_path / subject_id
        gyro_dfs_list = []
        if current_gyro_folder.exists():
            for file in current_gyro_folder.rglob("*.dat"):
                try:
                    file_name_ts = float(file.name.split(".")[0])

                    tmp_df = pd.read_csv(
                        file,
                        names=gyro_header,
                        header=None,
                        sep=r"\s+",
                        engine="python",
                    )
                    tmp_df = tmp_df.dropna(how="all")
                    tmp_df["timestamp"] = pd.to_numeric(
                        tmp_df["timestamp"], errors="coerce"
                    )

                    if not tmp_df.empty:
                        first_val = tmp_df["timestamp"].iloc[0]

                        if first_val < 1e9:
                            start_dt = pd.to_datetime(file_name_ts, unit="s")
                            offsets = pd.to_timedelta(tmp_df["timestamp"], unit="ms")
                            tmp_df["timestamp"] = start_dt + offsets
                        else:
                            tmp_df["timestamp"] = pd.to_datetime(
                                tmp_df["timestamp"], unit="s"
                            )

                        gyro_dfs_list.append(tmp_df)
                except Exception:
                    continue

        if gyro_dfs_list:
            df_gyro = pd.concat(gyro_dfs_list, ignore_index=True)
            df_gyro = df_gyro.sort_values("timestamp").reset_index(drop=True)
        else:
            df_gyro = pd.DataFrame(columns=gyro_header)

        watch_dfs = []
        current_watch_folder = acc_watch_root_path / subject_id
        if current_watch_folder.exists():
            for file in current_watch_folder.rglob("*.dat"):
                try:
                    file_ts = float(file.name.split(".")[0])
                    start_dt = pd.to_datetime(file_ts, unit="s")

                    tmp = pd.read_csv(file, header=None, sep=r"\s+", engine="python")
                    tmp = tmp.dropna()
                    if tmp.empty:
                        continue

                    if tmp.shape[1] == 4:
                        tmp.columns = [
                            "offset_ms",
                            "watch_acc_x",
                            "watch_acc_y",
                            "watch_acc_z",
                        ]
                        offsets = pd.to_timedelta(tmp["offset_ms"], unit="ms")
                        tmp["timestamp"] = start_dt + offsets
                        tmp = tmp.drop(columns=["offset_ms"])
                        watch_dfs.append(tmp)
                    elif tmp.shape[1] == 3:
                        tmp.columns = ["watch_acc_x", "watch_acc_y", "watch_acc_z"]
                        deltas = pd.to_timedelta(tmp.index * 0.04, unit="s")
                        tmp["timestamp"] = start_dt + deltas
                        watch_dfs.append(tmp)
                except Exception:
                    continue

        if watch_dfs:
            watch_acc = pd.concat(watch_dfs, ignore_index=True)
            watch_acc = watch_acc.sort_values("timestamp").reset_index(drop=True)
        else:
            watch_acc = pd.DataFrame(
                columns=["timestamp", "watch_acc_x", "watch_acc_y", "watch_acc_z"]
            )

        # compass_dfs = []
        # current_compass_folder = compass_watch_root_path / subject_id
        # if current_compass_folder.exists():
        #     for file in current_compass_folder.rglob("*.dat"):
        #         try:
        #             file_ts = float(file.name.split(".")[0])
        #             start_dt = pd.to_datetime(file_ts, unit="s")

        #             tmp = pd.read_csv(file, header=None, sep=r"\s+", engine="python")
        #             tmp = tmp.dropna()
        #             if tmp.empty:
        #                 continue

        #             if tmp.shape[1] >= 2:
        #                 tmp = tmp.iloc[:, :2]
        #                 tmp.columns = ["offset_ms", "compass_deg"]
        #                 offsets = pd.to_timedelta(tmp["offset_ms"], unit="ms")
        #                 tmp["timestamp"] = start_dt + offsets
        #                 tmp = tmp.drop(columns=["offset_ms"])
        #                 compass_dfs.append(tmp)
        #         except Exception:
        #             continue

        # if compass_dfs:
        #     watch_compass = pd.concat(compass_dfs, ignore_index=True)
        #     watch_compass = watch_compass.sort_values("timestamp").reset_index(
        #         drop=True
        #     )
        # else:
        #     watch_compass = pd.DataFrame(columns=["timestamp", "compass_deg"])

        if not df_acc.empty and not df_gyro.empty:
            df_final = pd.merge_asof(
                df_acc,
                df_gyro,
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("40ms"),
                suffixes=("_acc", "_gyro"),
            )
        else:
            df_final = df_acc

        if not df_final.empty and not watch_acc.empty:
            if "acc_x" in watch_acc.columns:
                watch_acc = watch_acc.rename(
                    columns={
                        "acc_x": "watch_acc_x",
                        "acc_y": "watch_acc_y",
                        "acc_z": "watch_acc_z",
                    }
                )

            df_final = pd.merge_asof(
                df_final,
                watch_acc,
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("40ms"),
            )

        # die sessions dauern ca 20 sec, es wird der letzte wert genommen, weil nur ein wert existiert, wenn deg sich um 1 ändert
        # if watch_compass.empty:
        #     print("empty")
        # if not df_final.empty and not watch_compass.empty:
        #     df_final = pd.merge_asof(
        #         df_final,
        #         watch_compass,
        #         on="timestamp",
        #         direction="backward",
        #         tolerance=pd.Timedelta("25s"),
        #     )

        df_final = df_final.dropna().reset_index(drop=True)

        if df_final.empty:
            continue

        df_final = add_session_id(df_final)
        df_final["session_id"] = df_final["session_id"] + last_subject_max_session_id
        last_subject_max_session_id = df_final["session_id"].max() + 1

        session_metadata_tmp = (
            df_final[["session_id", "subject_id", "activity_id"]]
            .drop_duplicates("session_id")
            .sort_values("session_id")
            .reset_index(drop=True)
        )

        session_metadata = pd.concat(
            [session_metadata, session_metadata_tmp], ignore_index=True
        )
        loop = tqdm(session_metadata_tmp["session_id"].unique())
        loop.set_description("Creating sessions")

        drop_cols = ["activity_id", "subject_id", "session_id"]

        for sid in loop:
            sdf = df_final[df_final["session_id"] == sid].copy()
            sdf = sdf.drop(
                columns=[c for c in drop_cols if c in sdf.columns]
            ).reset_index(drop=True)
            sdf = sdf.round(6)
            if sdf.isna().any().any():
                print(f"\nACHTUNG: NaNs gefunden in Session {sid}!")
                print(sdf.isna().sum())  # Zeigt Anzahl NaNs pro Spalte
            sessions[int(sid)] = sdf

    if global_label_map:
        activity_metadata = pd.DataFrame(
            list(global_label_map.items()), columns=["activity_name", "activity_id"]
        )

        activity_metadata["activity_name"] = activity_metadata[
            "activity_name"
        ].str.replace("label:", "")

    return activity_metadata, session_metadata, sessions


cfg_extrasensory = WHARConfig(
    dataset_id="extrasensory",
    download_url="http://extrasensory.ucsd.edu/",
    sampling_freq=40,
    num_of_subjects=2,
    num_of_activities=51,
    num_of_channels=9,  # eigentlich 10
    datasets_dir="./datasets",
    parse=parse_extrasensory,
    activity_id_col="activity_id",
    activity_names=[
        "LYING_DOWN",
        "SITTING",
        "FIX_walking",
        "FIX_running",
        "BICYCLING",
        "SLEEPING",
        "LAB_WORK",
        "IN_CLASS",
        "IN_A_MEETING",
        "LOC_main_workplace",
        "OR_indoors",
        "OR_outside",
        "IN_A_CAR",
        "ON_A_BUS",
        "DRIVE_-_I_M_THE_DRIVER",
        "DRIVE_-_I_M_A_PASSENGER",
        "LOC_home",
        "FIX_restaurant",
        "PHONE_IN_POCKET",
        "OR_exercise",
        "COOKING",
        "SHOPPING",
        "STROLLING",
        "DRINKING__ALCOHOL_",
        "BATHING_-_SHOWER",
        "CLEANING",
        "DOING_LAUNDRY",
        "WASHING_DISHES",
        "WATCHING_TV",
        "SURFING_THE_INTERNET",
        "AT_A_PARTY",
        "AT_A_BAR",
        "LOC_beach",
        "SINGING",
        "TALKING",
        "COMPUTER_WORK",
        "EATING",
        "TOILET",
        "GROOMING",
        "DRESSING",
        "AT_THE_GYM",
        "STAIRS_-_GOING_UP",
        "STAIRS_-_GOING_DOWN",
        "ELEVATOR",
        "OR_standing",
        "AT_SCHOOL",
        "PHONE_IN_HAND",
        "PHONE_IN_BAG",
        "PHONE_ON_TABLE",
        "WITH_CO-WORKERS",
        "WITH_FRIENDS",
    ],
    sensor_channels=[
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "watch_acc_x",
        "watch_acc_y",
        "watch_acc_z",
        # "compass_deg",
    ],
    window_time=5.0,
    window_overlap=0.5,
    given_split=([0], [1]),
    split_groups=[[0], [1]],
)

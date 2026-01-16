from whar_datasets.core.config import WHARConfig

import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import CubicSpline

KEEP_DEVICES: List[str] = [
    "samsungold_2",
    "samsungold_1",
    "s3_2",
    "nexus4_1",
    "nexus4_2",
    "s3mini_1",
    "s3_1",
]


def add_session_id(df: pd.DataFrame, gap: str = "3s") -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    gap_td = pd.Timedelta(gap)

    new_session = (
        (df["subject_raw"].shift() != df["subject_raw"])
        | (df["activity_name"].shift() != df["activity_name"])
        | (df["timestamp"].diff() > gap_td)
    )
    df["session_id_raw"] = new_session.cumsum().astype(int)
    return df


def session_to_wide(
    seg: pd.DataFrame,
    devices: List[str],
    target_hz: float = 50.0,
    min_points: int = 4,
    require_all_devices: bool = True,
) -> pd.DataFrame:
    seg = seg[seg["device"].isin(devices)].copy()
    if seg.empty:
        return pd.DataFrame()

    present = seg["device"].nunique()
    if require_all_devices and present < len(devices):
        return pd.DataFrame()

    ranges = seg.groupby("device")["timestamp"].agg(["min", "max"])
    start_common = ranges["min"].max()
    end_common = ranges["max"].min()
    if start_common >= end_common:
        return pd.DataFrame()

    seg = seg[
        (seg["timestamp"] >= start_common) & (seg["timestamp"] <= end_common)
    ].copy()

    # 50 Hz
    step_ms = int(round(1000.0 / target_hz))
    common_dt = pd.date_range(start_common, end_common, freq=f"{step_ms}ms")
    common_sec = common_dt.view("int64") / 1e9

    wide = pd.DataFrame({"timestamp": common_dt})
    wide["session_id_raw"] = seg["session_id_raw"].iloc[0]
    wide["subject_raw"] = seg["subject_raw"].iloc[0]
    wide["activity_name"] = seg["activity_name"].iloc[0]

    # pro Device spline
    for dev in devices:
        g = seg[seg["device"] == dev].sort_values("timestamp")
        if g.empty:
            if require_all_devices:
                return pd.DataFrame()
            for col in ["x", "y", "z"]:
                wide[f"{dev}_{col}"] = np.nan
            continue

        # Duplikate bei timestamp: mitteln
        g = g.groupby("timestamp", as_index=False).agg(
            {"x": "mean", "y": "mean", "z": "mean"}
        )

        if len(g) < min_points:
            if require_all_devices:
                return pd.DataFrame()
            for col in ["x", "y", "z"]:
                wide[f"{dev}_{col}"] = np.nan
            continue

        t_src = (g["timestamp"].view("int64") / 1e9).to_numpy()

        for col in ["x", "y", "z"]:
            y = g[col].astype(float).to_numpy()
            cs = CubicSpline(t_src, y, bc_type="natural", extrapolate=False)
            y_new = cs(common_sec)

            y_new = (
                pd.Series(y_new, index=common_dt)
                .interpolate(limit_direction="both")
                .to_numpy()
            )
            wide[f"{dev}_{col}"] = y_new

    return wide


def parse_hhar(
    dir: str,
    activity_id_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    csv_path = os.path.join(
        dir,
        "Activity recognition exp",
        "Activity recognition exp",
        "Phones_accelerometer.csv",
    )

    df = pd.read_csv(csv_path)

    df["timestamp"] = pd.to_datetime(df["Arrival_Time"], unit="ms")

    df = df[df["Device"].isin(KEEP_DEVICES)].copy()
    df = df[df["gt"].notna()].copy()

    df = df[["timestamp", "Device", "x", "y", "z", "User", "gt"]].copy()

    df = df.rename(
        columns={
            "Device": "device",
            "User": "subject_raw",
            "gt": "activity_name",
        }
    )

    df = df.sort_values(["device", "timestamp"]).reset_index(drop=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_session_id(df, gap="3s")

    grouped = df.groupby("session_id_raw", group_keys=False)

    wide_df = grouped.apply(
        lambda seg: session_to_wide(
            seg,
            devices=KEEP_DEVICES,
            target_hz=50.0,
            require_all_devices=True,
        )
    ).reset_index(drop=True)

    wide_df = wide_df.dropna(subset=["timestamp"]).reset_index(drop=True)
    if wide_df.empty:
        raise ValueError("wide_df empty")

    wide_df["activity_id"] = pd.factorize(wide_df["activity_name"])[0].astype("int32")
    wide_df["subject_id"] = pd.factorize(wide_df["subject_raw"])[0].astype("int32")
    wide_df["session_id"] = pd.factorize(wide_df["session_id_raw"])[0].astype("int32")

    activity_metadata = (
        wide_df[["activity_id", "activity_name"]]
        .drop_duplicates("activity_id")
        .sort_values("activity_id")
        .reset_index(drop=True)
        .astype({"activity_id": "int32", "activity_name": "string"})
    )

    session_metadata = (
        wide_df[["session_id", "subject_id", "activity_id"]]
        .drop_duplicates("session_id")
        .sort_values("session_id")
        .reset_index(drop=True)
        .astype({"session_id": "int32", "subject_id": "int32", "activity_id": "int32"})
    )

    sessions: Dict[int, pd.DataFrame] = {}

    loop = tqdm(session_metadata["session_id"].unique())
    loop.set_description("Creating sessions")

    drop_cols = [
        "session_id_raw",
        "subject_raw",
        "activity_name",
        "activity_id",
        "subject_id",
        "session_id",
    ]
    for sid in loop:
        # hat noch timestamp, die x,y,z der devices
        sdf = wide_df[wide_df["session_id"] == sid].copy()

        sdf = sdf.drop(columns=[c for c in drop_cols if c in sdf.columns]).reset_index(
            drop=True
        )

        sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
        for c in sdf.columns:
            if c != "timestamp":
                sdf[c] = sdf[c].astype("float32")

        sdf = sdf.round(6)

        sessions[int(sid)] = sdf

    return activity_metadata, session_metadata, sessions


cfg_hhar = WHARConfig(
    # Info + common
    dataset_id="hhar",
    download_url="https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip",
    sampling_freq=50,
    num_of_subjects=8,
    num_of_activities=6,
    num_of_channels=len(KEEP_DEVICES) * 3,
    datasets_dir="./datasets",
    parse=parse_hhar,
    activity_names=["Biking", "Sitting", "Standing", "Walking", "Stair UpStair down"],
    sensor_channels=[f"{dev}_{ax}" for dev in KEEP_DEVICES for ax in ["x", "y", "z"]],
    window_time=1.28,
    window_overlap=0.0,
    given_split=(list(range(0, 6)), list(range(6, 8))),
    split_groups=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
)

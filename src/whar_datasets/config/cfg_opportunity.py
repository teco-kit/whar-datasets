import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig

SENSOR_COLS = [
    "timestamp",  # 1
    "RKN_upper_acc_x",
    "RKN_upper_acc_y",
    "RKN_upper_acc_z",  # 2–4
    "HIP_acc_x",
    "HIP_acc_y",
    "HIP_acc_z",  # 5–7
    "LUA_upper_acc_x",
    "LUA_upper_acc_y",
    "LUA_upper_acc_z",  # 8–10
    "RUA_lower_acc_x",
    "RUA_lower_acc_y",
    "RUA_lower_acc_z",  # 11–13
    "LH_acc_x",
    "LH_acc_y",
    "LH_acc_z",  # 14–16
    "BACK_acc_x",
    "BACK_acc_y",
    "BACK_acc_z",  # 17–19
    "RKN_lower_acc_x",
    "RKN_lower_acc_y",
    "RKN_lower_acc_z",  # 20–22
    "RWR_acc_x",
    "RWR_acc_y",
    "RWR_acc_z",  # 23–25
    "RUA_upper_acc_x",
    "RUA_upper_acc_y",
    "RUA_upper_acc_z",  # 26–28
    "LUA_lower_acc_x",
    "LUA_lower_acc_y",
    "LUA_lower_acc_z",  # 29–31
    "LWR_acc_x",
    "LWR_acc_y",
    "LWR_acc_z",  # 32–34
    "RH_acc_x",
    "RH_acc_y",
    "RH_acc_z",  # 35–37
    # IMU BACK
    "IMU_BACK_acc_x",
    "IMU_BACK_acc_y",
    "IMU_BACK_acc_z",  # 38–40
    "IMU_BACK_gyro_x",
    "IMU_BACK_gyro_y",
    "IMU_BACK_gyro_z",  # 41–43
    "IMU_BACK_mag_x",
    "IMU_BACK_mag_y",
    "IMU_BACK_mag_z",  # 44–46
    "IMU_BACK_quat_1",
    "IMU_BACK_quat_2",
    "IMU_BACK_quat_3",
    "IMU_BACK_quat_4",  # 47–50
    # IMU RUA
    "IMU_RUA_acc_x",
    "IMU_RUA_acc_y",
    "IMU_RUA_acc_z",  # 51–53
    "IMU_RUA_gyro_x",
    "IMU_RUA_gyro_y",
    "IMU_RUA_gyro_z",  # 54–56
    "IMU_RUA_mag_x",
    "IMU_RUA_mag_y",
    "IMU_RUA_mag_z",  # 57–59
    "IMU_RUA_quat_1",
    "IMU_RUA_quat_2",
    "IMU_RUA_quat_3",
    "IMU_RUA_quat_4",  # 60–63
    # IMU RLA
    "IMU_RLA_acc_x",
    "IMU_RLA_acc_y",
    "IMU_RLA_acc_z",  # 64–66
    "IMU_RLA_gyro_x",
    "IMU_RLA_gyro_y",
    "IMU_RLA_gyro_z",  # 67–69
    "IMU_RLA_mag_x",
    "IMU_RLA_mag_y",
    "IMU_RLA_mag_z",  # 70–72
    "IMU_RLA_quat_1",
    "IMU_RLA_quat_2",
    "IMU_RLA_quat_3",
    "IMU_RLA_quat_4",  # 73–76
    # IMU LUA
    "IMU_LUA_acc_x",
    "IMU_LUA_acc_y",
    "IMU_LUA_acc_z",  # 77–79
    "IMU_LUA_gyro_x",
    "IMU_LUA_gyro_y",
    "IMU_LUA_gyro_z",  # 80–82
    "IMU_LUA_mag_x",
    "IMU_LUA_mag_y",
    "IMU_LUA_mag_z",  # 83–85
    "IMU_LUA_quat_1",
    "IMU_LUA_quat_2",
    "IMU_LUA_quat_3",
    "IMU_LUA_quat_4",  # 86–89
    # IMU LLA
    "IMU_LLA_acc_x",
    "IMU_LLA_acc_y",
    "IMU_LLA_acc_z",  # 90–92
    "IMU_LLA_gyro_x",
    "IMU_LLA_gyro_y",
    "IMU_LLA_gyro_z",  # 93–95
    "IMU_LLA_mag_x",
    "IMU_LLA_mag_y",
    "IMU_LLA_mag_z",  # 96–98
    "IMU_LLA_quat_1",
    "IMU_LLA_quat_2",
    "IMU_LLA_quat_3",
    "IMU_LLA_quat_4",  # 99–102
    # IMU L-SHOE
    "IMU_LSHOE_eu_x",
    "IMU_LSHOE_eu_y",
    "IMU_LSHOE_eu_z",  # 103–105
    "IMU_LSHOE_nav_acc_x",
    "IMU_LSHOE_nav_acc_y",
    "IMU_LSHOE_nav_acc_z",  # 106–108
    "IMU_LSHOE_body_acc_x",
    "IMU_LSHOE_body_acc_y",
    "IMU_LSHOE_body_acc_z",  # 109–111
    "IMU_LSHOE_angvel_body_x",
    "IMU_LSHOE_angvel_body_y",
    "IMU_LSHOE_angvel_body_z",  # 112–114
    "IMU_LSHOE_angvel_nav_x",
    "IMU_LSHOE_angvel_nav_y",
    "IMU_LSHOE_angvel_nav_z",  # 115–117
    "IMU_LSHOE_compass",  # 118
    # IMU R-SHOE
    "IMU_RSHOE_eu_x",
    "IMU_RSHOE_eu_y",
    "IMU_RSHOE_eu_z",  # 119–121
    "IMU_RSHOE_nav_acc_x",
    "IMU_RSHOE_nav_acc_y",
    "IMU_RSHOE_nav_acc_z",  # 122–124
    "IMU_RSHOE_body_acc_x",
    "IMU_RSHOE_body_acc_y",
    "IMU_RSHOE_body_acc_z",  # 125–127
    "IMU_RSHOE_angvel_body_x",
    "IMU_RSHOE_angvel_body_y",
    "IMU_RSHOE_angvel_body_z",  # 128–130
    "IMU_RSHOE_angvel_nav_x",
    "IMU_RSHOE_angvel_nav_y",
    "IMU_RSHOE_angvel_nav_z",  # 131–133
    "IMU_RSHOE_compass",  # 134
]

LABEL_COLS = [
    "Locomotion",
    "HL_Activity",
    "LL_Left_Arm",
    "LL_Left_Arm_Object",
    "LL_Right_Arm",
    "LL_Right_Arm_Object",
    "ML_Both_Arms",
]

MAX_SESSION_GAP_MS = 100.0

# Locomotion
LOCOMOTION_MAP = {0: "Unknown", 1: "Stand", 2: "Walk", 4: "Sit", 5: "Lie"}

# HL_Activity
HL_ACTIVITY_MAP = {
    0: "Unknown",
    101: "Relaxing",
    102: "time",
    103: "morning",
    104: "Cleanup",
    105: "time",
}

# LL_Left_Arm
LL_LEFT_ARM_MAP = {
    0: "Unknown",
    201: "unlock",
    202: "stir",
    203: "lock",
    204: "close",
    205: "reach",
    206: "open",
    207: "sip",
    208: "clean",
    209: "bite",
    210: "cut",
    211: "spread",
    212: "release",
    213: "move",
}

# LL_Left_Arm_Object
LL_LEFT_ARM_OBJECT_MAP = {
    0: "Unknown",
    301: "Bottle",
    302: "Salami",
    303: "Bread",
    304: "Sugar",
    305: "Dishwasher",
    306: "Switch",
    307: "Milk",
    308: "lower",  # Drawer3 (lower)
    309: "Spoon",
    310: "cheese",  # Knife cheese
    311: "middle",  # Drawer2 (middle)
    312: "Table",
    313: "Glass",
    314: "Cheese",
    315: "Chair",
    316: "Door1",
    317: "Door2",
    318: "Plate",
    319: "top",  # Drawer1 (top)
    320: "Fridge",
    321: "Cup",
    322: "salami",  # Knife salami
    323: "Lazychair",
}

# LL_Right_Arm
LL_RIGHT_ARM_MAP = {
    0: "Unknown",
    401: "unlock",
    402: "stir",
    403: "lock",
    404: "close",
    405: "reach",
    406: "open",
    407: "sip",
    408: "clean",
    409: "bite",
    410: "cut",
    411: "spread",
    412: "release",
    413: "move",
}

# LL_Right_Arm_Object
LL_RIGHT_ARM_OBJECT_MAP = {
    0: "Unknown",
    501: "Bottle",
    502: "Salami",
    503: "Bread",
    504: "Sugar",
    505: "Dishwasher",
    506: "Switch",
    507: "Milk",
    508: "lower",  # Drawer3 (lower)
    509: "Spoon",
    510: "cheese",  # Knife cheese
    511: "middle",  # Drawer2 (middle)
    512: "Table",
    513: "Glass",
    514: "Cheese",
    515: "Chair",
    516: "Door1",
    517: "Door2",
    518: "Plate",
    519: "top",  # Drawer1 (top)
    520: "Fridge",
    521: "Cup",
    522: "salami",  # Knife salami
    523: "Lazychair",
}

# ML_Both_Arms
ML_BOTH_ARMS_MAP = {
    0: "Unknown",
    406516: "Open Door 1",
    406517: "Open Door 2",
    404516: "Close Door 1",
    404517: "Close Door 2",
    406520: "Open Fridge",
    404520: "Close Fridge",
    406505: "Open Dishwasher",
    404505: "Close Dishwasher",
    406519: "Open Drawer 1",
    404519: "Close Drawer 1",
    406511: "Open Drawer 2",
    404511: "Close Drawer 2",
    406508: "Open Drawer 3",
    404508: "Close Drawer 3",
    408512: "Clean Table",
    407521: "Drink from Cup",
    405506: "Toggle Switch",
}


def parse_opportunity(
    dir: str, activity_id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    dir = os.path.join(dir, "OpportunityUCIDataset/dataset/")

    files = sorted(file for file in os.listdir(dir) if file.endswith(".dat"))

    sub_dfs = []

    for file in files:
        # get subject id from filename
        subject_id = int(file.split("-")[0][-1])

        # use body sensors and labels
        sub_df = pd.read_table(
            os.path.join(dir, file),
            header=None,
            sep="\\s+",
            usecols=list(range(134)) + list(range(243, 250)),
            names=[*SENSOR_COLS, *LABEL_COLS],
        )

        # fill nans in sensor cols with linear interpolation
        sub_df.loc[:, SENSOR_COLS] = (
            sub_df[SENSOR_COLS]
            .interpolate(method="linear", limit_direction="both", axis=0)  # fill edges
            .bfill()  # extra safety for lingering NaNs
            .ffill()
        )

        # # fills nans in label cols with backward fill
        sub_df.loc[:, LABEL_COLS] = sub_df[LABEL_COLS].bfill()

        # add subject/file provenance before concatenation
        sub_df = sub_df.copy()
        sub_df["subject_id"] = subject_id
        sub_df["source_file"] = file

        # append to list
        sub_dfs.append(sub_df)

    # concat all sub_dfs
    df = pd.concat(sub_dfs, ignore_index=True).copy()

    # specify one label column as activity_id column
    df = df.rename(columns={activity_id_col: "activity_id"})
    df = df.drop(columns=[col for col in LABEL_COLS if col != activity_id_col])

    # change ms timestamp from ms to ns
    df["timestamp"] = pd.to_datetime(df["timestamp"] * 1e6, unit="ns")

    # split sessions on subject/activity change, file boundary, timestamp resets,
    # and unusually large time gaps to avoid interpolation artifacts.
    diff_ms = df["timestamp"].diff().dt.total_seconds().mul(1e3)
    changes = (
        df["activity_id"].ne(df["activity_id"].shift(1))
        | df["subject_id"].ne(df["subject_id"].shift(1))
        | df["source_file"].ne(df["source_file"].shift(1))
        | diff_ms.isna()
        | diff_ms.le(0.0)
        | diff_ms.gt(MAX_SESSION_GAP_MS)
    )
    df["session_id"] = changes.cumsum().sub(1).astype("int64")

    # map activity_id to activity_name
    match activity_id_col:
        case "Locomotion":
            df["activity_name"] = df["activity_id"].map(LOCOMOTION_MAP)
        case "HL_Activity":
            df["activity_name"] = df["activity_id"].map(HL_ACTIVITY_MAP)
        case "LL_Left_Arm":
            df["activity_name"] = df["activity_id"].map(LL_LEFT_ARM_MAP)
        case "LL_Left_Arm_Object":
            df["activity_name"] = df["activity_id"].map(LL_LEFT_ARM_OBJECT_MAP)
        case "LL_Right_Arm":
            df["activity_name"] = df["activity_id"].map(LL_RIGHT_ARM_MAP)
        case "LL_Right_Arm_Object":
            df["activity_name"] = df["activity_id"].map(LL_RIGHT_ARM_OBJECT_MAP)
        case "ML_Both_Arms":
            df["activity_name"] = df["activity_id"].map(ML_BOTH_ARMS_MAP)
        case _:
            raise ValueError(f"Unknown activity_id_col: {activity_id_col}")

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
                "source_file",
            ]
        ).reset_index(drop=True)

        # set types
        session_df["timestamp"] = pd.to_datetime(session_df["timestamp"])
        dtypes = {col: "float32" for col in session_df.columns if col != "timestamp"}
        dtypes["timestamp"] = "datetime64[ms]"
        float_cols = [col for col in session_df.columns if col != "timestamp"]
        session_df[float_cols] = session_df[float_cols].round(6)
        session_df = session_df.astype(dtypes)

        if session_df.empty:
            continue

        # add to sessions
        sessions[session_id] = session_df

    # Keep metadata aligned with retained sessions and compact ids.
    valid_session_ids = sorted(int(sid) for sid in sessions.keys())
    session_metadata = session_metadata[
        session_metadata["session_id"].isin(valid_session_ids)
    ].copy()

    remap = {old_sid: new_sid for new_sid, old_sid in enumerate(valid_session_ids)}
    session_metadata["session_id"] = session_metadata["session_id"].map(remap)
    session_metadata = session_metadata.sort_values("session_id").reset_index(
        drop=True
    )
    sessions = {remap[int(old_sid)]: df for old_sid, df in sessions.items()}

    # set metadata types
    activity_metadata = activity_metadata.astype(
        {"activity_id": "int32", "activity_name": "string"}
    )
    session_metadata = session_metadata.astype(
        {"session_id": "int32", "subject_id": "int32", "activity_id": "int32"}
    )

    return activity_metadata, session_metadata, sessions


cfg_opportunity = WHARConfig(
    # Info + common
    dataset_id="opportunity",
    download_url="https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
    sampling_freq=30,
    num_of_subjects=4,
    num_of_activities=18,
    num_of_channels=133,
    datasets_dir="./datasets",
    parallelize=True,
    # Parsing
    parse=parse_opportunity,
    activity_id_col="ML_Both_Arms",
    # Preprocessing (selections + sliding window)
    activity_names=[
        "Unknown",  # 0
        "Open Door 1",  # 406516
        "Open Door 2",  # 406517
        "Close Door 1",  # 404516
        "Close Door 2",  # 404517
        "Open Fridge",  # 406520
        "Close Fridge",  # 404520
        "Open Dishwasher",  # 406505
        "Close Dishwasher",  # 404505
        "Open Drawer 1",  # 406519
        "Close Drawer 1",  # 404519
        "Open Drawer 2",  # 406511
        "Close Drawer 2",  # 404511
        "Open Drawer 3",  # 406508
        "Close Drawer 3",  # 404508
        "Clean Table",  # 408512
        "Drink from Cup",  # 407521
        "Toggle Switch",  # 405506
    ],
    sensor_channels=SENSOR_COLS[1:],
    window_time=1,
    window_overlap=0.5,
    # Training (split info)
)

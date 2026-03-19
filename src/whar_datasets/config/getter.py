from enum import Enum
from typing import Dict

from whar_datasets.config.cfg_bmhad import cfg_bmhad
from whar_datasets.config.cfg_capture24 import cfg_capture_24
from whar_datasets.config.cfg_daphnet import cfg_daphnet
from whar_datasets.config.cfg_dsads import cfg_dsads
from whar_datasets.config.cfg_falldet import cfg_falldet
from whar_datasets.config.cfg_gotov import cfg_gotov
from whar_datasets.config.cfg_hang_time import cfg_hang_time
from whar_datasets.config.cfg_hapt import cfg_hapt
from whar_datasets.config.cfg_har70 import cfg_har70
from whar_datasets.config.cfg_har_sense import cfg_har_sense
from whar_datasets.config.cfg_harth import cfg_harth
from whar_datasets.config.cfg_hhar import cfg_hhar
from whar_datasets.config.cfg_hugadb import cfg_hugadb
from whar_datasets.config.cfg_ku_har import cfg_ku_har
from whar_datasets.config.cfg_mhealth import cfg_mhealth
from whar_datasets.config.cfg_motion_sense import cfg_motion_sense
from whar_datasets.config.cfg_opportunity import cfg_opportunity
from whar_datasets.config.cfg_pamap2 import cfg_pamap2
from whar_datasets.config.cfg_real_disp import cfg_real_disp
from whar_datasets.config.cfg_real_life_har import cfg_real_life_har
from whar_datasets.config.cfg_real_world import cfg_real_world
from whar_datasets.config.cfg_sad import cfg_sad
from whar_datasets.config.cfg_uca_ehar import cfg_uca_ehar
from whar_datasets.config.cfg_uci_har import cfg_uci_har
from whar_datasets.config.cfg_uma_fall import cfg_uma_fall
from whar_datasets.config.cfg_unimib_shar import cfg_unimib
from whar_datasets.config.cfg_up_fall import cfg_up_fall
from whar_datasets.config.cfg_usc_had import cfg_usc_had
from whar_datasets.config.cfg_utd_mhad import cfg_utd_mhad
from whar_datasets.config.cfg_w_har import cfg_w_har
from whar_datasets.config.cfg_wear import cfg_wear
from whar_datasets.config.cfg_wisdm import cfg_wisdm
from whar_datasets.config.cfg_wisdm_19_phone import cfg_wisdm_19_phone
from whar_datasets.config.cfg_wisdm_19_watch import cfg_wisdm_19_watch
from whar_datasets.config.config import WHARConfig


class WHARDatasetID(Enum):
    UCI_HAR = "uci_har"
    WISDM = "wisdm"
    PAMAP2 = "pamap2"
    MOTION_SENSE = "motion_sense"
    OPPORTUNITY = "opportunity"
    MHEALTH = "mhealth"
    DSADS = "dsads"
    KU_HAR = "ku_har"
    DAPHNET = "daphnet"
    HAR_SENSE = "har_sense"
    HAPT = "hapt"
    W_HAR = "w_har"
    USC_HAD = "usc_had"
    HUGADB = "hugadb"
    WISDM_19_PHONE = "wisdm_19_phone"
    WISDM_19_WATCH = "wisdm_19_watch"
    HANG_TIME = "hang_time"
    HHAR = "hhar"
    CAPTURE_24 = "capture_24"
    REAL_WORLD = "real_world"
    REAL_LIFE_HAR = "real_life_har"
    SAD = "sad"
    UNIMIB_SHAR = "unimib_shar"
    UMA_FALL = "uma_fall"
    REAL_DISP = "real_disp"
    HARTH = "harth"
    FALLDET = "falldet"
    HAR70 = "har70"
    GOTOV = "gotov"
    UTD_MHAD = "utd_mhad"
    UP_FALL = "up_fall"
    BMHAD = "bmhad"
    UCA_EHAR = "uca_ehar"
    WEAR = "wear"


har_dataset_dict: Dict[WHARDatasetID, WHARConfig] = {
    WHARDatasetID.UCI_HAR: (cfg_uci_har),
    WHARDatasetID.WISDM: (cfg_wisdm),
    WHARDatasetID.PAMAP2: (cfg_pamap2),
    WHARDatasetID.MOTION_SENSE: (cfg_motion_sense),
    WHARDatasetID.OPPORTUNITY: (cfg_opportunity),
    WHARDatasetID.MHEALTH: (cfg_mhealth),
    WHARDatasetID.DSADS: (cfg_dsads),
    WHARDatasetID.KU_HAR: (cfg_ku_har),
    WHARDatasetID.DAPHNET: (cfg_daphnet),
    WHARDatasetID.HAR_SENSE: (cfg_har_sense),
    WHARDatasetID.HAPT: (cfg_hapt),
    WHARDatasetID.W_HAR: (cfg_w_har),
    WHARDatasetID.USC_HAD: (cfg_usc_had),
    WHARDatasetID.HUGADB: (cfg_hugadb),
    WHARDatasetID.WISDM_19_PHONE: (cfg_wisdm_19_phone),
    WHARDatasetID.WISDM_19_WATCH: (cfg_wisdm_19_watch),
    WHARDatasetID.HANG_TIME: (cfg_hang_time),
    WHARDatasetID.CAPTURE_24: (cfg_capture_24),
    WHARDatasetID.REAL_WORLD: (cfg_real_world),
    WHARDatasetID.REAL_LIFE_HAR: (cfg_real_life_har),
    WHARDatasetID.SAD: (cfg_sad),
    WHARDatasetID.UNIMIB_SHAR: (cfg_unimib),
    WHARDatasetID.UMA_FALL: (cfg_uma_fall),
    WHARDatasetID.REAL_DISP: (cfg_real_disp),
    WHARDatasetID.HARTH: (cfg_harth),
    WHARDatasetID.FALLDET: (cfg_falldet),
    WHARDatasetID.HAR70: (cfg_har70),
    WHARDatasetID.GOTOV: (cfg_gotov),
    WHARDatasetID.HHAR: (cfg_hhar),
    WHARDatasetID.UTD_MHAD: (cfg_utd_mhad),
    WHARDatasetID.UP_FALL: (cfg_up_fall),
    WHARDatasetID.BMHAD: (cfg_bmhad),
    WHARDatasetID.UCA_EHAR: (cfg_uca_ehar),
    WHARDatasetID.WEAR: (cfg_wear),
}


def get_dataset_cfg(
    dataset_id: WHARDatasetID, datasets_dir: str = "./datasets/"
) -> WHARConfig:
    # load dataset-specific config and parser
    cfg = har_dataset_dict[dataset_id]

    # override datasets dir
    cfg.datasets_dir = datasets_dir

    return cfg


# Dataset IDs listed in README.md (Supported Datasets), excluding CAPTURE-24.
BENCHMARK_DATASET_IDS: list[WHARDatasetID] = [
    # Single-Sensor Datasets
    WHARDatasetID.WISDM,
    WHARDatasetID.UCI_HAR,
    WHARDatasetID.UTD_MHAD,
    WHARDatasetID.HAPT,
    WHARDatasetID.USC_HAD,
    WHARDatasetID.UNIMIB_SHAR,
    WHARDatasetID.MOTION_SENSE,
    WHARDatasetID.REAL_LIFE_HAR,
    WHARDatasetID.WISDM_19_PHONE,
    WHARDatasetID.WISDM_19_WATCH,
    WHARDatasetID.KU_HAR,
    WHARDatasetID.HANG_TIME,
    # Multi-Sensor Datasets
    WHARDatasetID.PAMAP2,
    WHARDatasetID.OPPORTUNITY,
    WHARDatasetID.HHAR,
    WHARDatasetID.MHEALTH,
    WHARDatasetID.DSADS,
    WHARDatasetID.SAD,
    WHARDatasetID.DAPHNET,
    WHARDatasetID.REAL_WORLD,
    WHARDatasetID.UP_FALL,
    WHARDatasetID.UMA_FALL,
    WHARDatasetID.REAL_DISP,
    WHARDatasetID.HUGADB,
    WHARDatasetID.HARTH,
    WHARDatasetID.W_HAR,
    WHARDatasetID.WEAR,
    WHARDatasetID.HAR70,
    WHARDatasetID.UCA_EHAR,
    WHARDatasetID.GOTOV,
]

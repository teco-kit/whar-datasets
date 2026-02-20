from enum import Enum
from typing import Dict

from whar_datasets.config.cfg_capture24 import cfg_capture_24
from whar_datasets.config.cfg_daphnet import cfg_daphnet
from whar_datasets.config.cfg_dsads import cfg_dsads
from whar_datasets.config.cfg_hang_time import cfg_hang_time
from whar_datasets.config.cfg_hapt import cfg_hapt
from whar_datasets.config.cfg_har_sense import cfg_har_sense
from whar_datasets.config.cfg_hugadb import cfg_hugadb
from whar_datasets.config.cfg_ku_har import cfg_ku_har
from whar_datasets.config.cfg_mhealth import cfg_mhealth
from whar_datasets.config.cfg_motion_sense import cfg_motion_sense
from whar_datasets.config.cfg_opportunity import cfg_opportunity
from whar_datasets.config.cfg_pamap2 import cfg_pamap2
from whar_datasets.config.cfg_real_life_har import cfg_real_life_har
from whar_datasets.config.cfg_real_world import cfg_real_world
from whar_datasets.config.cfg_sad import cfg_sad
from whar_datasets.config.cfg_uci_har import cfg_uci_har
from whar_datasets.config.cfg_usc_had import cfg_usc_had
from whar_datasets.config.cfg_w_har import cfg_w_har
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
}


def get_dataset_cfg(
    dataset_id: WHARDatasetID, datasets_dir: str = "./datasets/"
) -> WHARConfig:
    # load dataset-specific config and parser
    cfg = har_dataset_dict[dataset_id]

    # override datasets dir
    cfg.datasets_dir = datasets_dir

    return cfg

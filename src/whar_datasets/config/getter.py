from enum import Enum
from typing import Dict

from whar_datasets.config.cfg_daphnet import cfg_daphnet
from whar_datasets.config.cfg_dsads import cfg_dsads
from whar_datasets.config.cfg_har_sense import cfg_har_sense
from whar_datasets.config.cfg_ku_har import cfg_ku_har
from whar_datasets.config.cfg_mhealth import cfg_mhealth
from whar_datasets.config.cfg_motion_sense import cfg_motion_sense
from whar_datasets.config.cfg_opportunity import cfg_opportunity
from whar_datasets.config.cfg_pamap2 import cfg_pamap2
from whar_datasets.config.cfg_uci_har import cfg_uci_har
from whar_datasets.config.cfg_wisdm import cfg_wisdm
from whar_datasets.config.cfg_extrasensory import cfg_extrasensory
from whar_datasets.config.cfg_unimib import cfg_unimib
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
    EXTRASENSORY = "extrasensory"
    UNIMIB = "unimib"


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
    WHARDatasetID.EXTRASENSORY: (cfg_extrasensory),
    WHARDatasetID.UNIMIB: (cfg_unimib),
}


def get_dataset_cfg(
    dataset_id: WHARDatasetID, datasets_dir: str = "./datasets"
) -> WHARConfig:
    # load dataset-specific config and parser
    cfg = har_dataset_dict[dataset_id]

    # override datasets dir
    cfg.datasets_dir = datasets_dir

    return cfg

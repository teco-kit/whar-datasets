from collections.abc import Iterator, Mapping
from enum import Enum
from importlib import import_module

from whar_datasets.config.config import WHARConfig


class WHARDatasetID(Enum):
    """Identifiers for all built-in dataset configurations."""

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
    SKODA = "skoda"
    ACTRECTUT_GESTURES = "actrectut_gestures"
    ACTRECTUT_WALKING = "actrectut_walking"


_CONFIG_IMPORTS: dict[WHARDatasetID, tuple[str, str]] = {
    dataset_id: (
        f"whar_datasets.config.cfg_{dataset_id.value}",
        f"cfg_{dataset_id.value}",
    )
    for dataset_id in WHARDatasetID
}
_CONFIG_IMPORTS[WHARDatasetID.CAPTURE_24] = (
    "whar_datasets.config.cfg_capture24",
    "cfg_capture_24",
)
_CONFIG_IMPORTS[WHARDatasetID.UNIMIB_SHAR] = (
    "whar_datasets.config.cfg_unimib_shar",
    "cfg_unimib",
)


def _load_config(dataset_id: WHARDatasetID) -> WHARConfig:
    module_name, attribute_name = _CONFIG_IMPORTS[dataset_id]
    return getattr(import_module(module_name), attribute_name)


class _LazyConfigMapping(Mapping[WHARDatasetID, WHARConfig]):
    def __getitem__(self, key: WHARDatasetID) -> WHARConfig:
        return _load_config(key)

    def __iter__(self) -> Iterator[WHARDatasetID]:
        return iter(_CONFIG_IMPORTS)

    def __len__(self) -> int:
        return len(_CONFIG_IMPORTS)


har_dataset_dict: Mapping[WHARDatasetID, WHARConfig] = _LazyConfigMapping()


def get_dataset_cfg(
    dataset_id: WHARDatasetID, datasets_dir: str = "./datasets/"
) -> WHARConfig:
    """Return an independent dataset configuration with an overridden cache path."""
    cfg = har_dataset_dict[dataset_id].model_copy(deep=True)
    cfg.datasets_dir = datasets_dir
    return cfg


BENCHMARK_DATASET_IDS: list[WHARDatasetID] = [
    WHARDatasetID.WISDM,
    WHARDatasetID.UCI_HAR,
    WHARDatasetID.UTD_MHAD,
    WHARDatasetID.HAPT,
    WHARDatasetID.USC_HAD,
    WHARDatasetID.UNIMIB_SHAR,
    WHARDatasetID.MOTION_SENSE,
    WHARDatasetID.REAL_LIFE_HAR,
    WHARDatasetID.WISDM_19_PHONE,
    WHARDatasetID.HANG_TIME,
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
    # WHARDatasetID.W_HAR,
    WHARDatasetID.WEAR,
    WHARDatasetID.HAR70,
    WHARDatasetID.UCA_EHAR,
    WHARDatasetID.GOTOV,
    WHARDatasetID.ACTRECTUT_GESTURES,
    WHARDatasetID.BMHAD,
    WHARDatasetID.WISDM_19_WATCH,
]

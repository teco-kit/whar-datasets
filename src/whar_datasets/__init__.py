# from whar_datasets.adapters.adapter_tf import TFAdapter
from whar_datasets.adapters.adapter_torch import TorchAdapter
from whar_datasets.config.config import (
    WINDOW_TIME_LARGE,
    WINDOW_TIME_MEDIUM,
    WINDOW_TIME_SMALL,
    WHARConfig,
)
from whar_datasets.config.getter import WHARDatasetID, get_dataset_cfg
from whar_datasets.loading.loader import Loader
from whar_datasets.processing.pipeline_post import PostProcessingPipeline
from whar_datasets.processing.pipeline_pre import PreProcessingPipeline
from whar_datasets.splitting.splitter_kfold import KFoldSplitter
from whar_datasets.splitting.splitter_lgso import LGSOSplitter
from whar_datasets.splitting.splitter_loso import LOSOSplitter

__all__ = [
    "Loader",
    "KFoldSplitter",
    "LOSOSplitter",
    "LGSOSplitter",
    "PreProcessingPipeline",
    "PostProcessingPipeline",
    "get_dataset_cfg",
    "WHARDatasetID",
    # "TFAdapter",
    "TorchAdapter",
    "WHARConfig",
    "WINDOW_TIME_SMALL",
    "WINDOW_TIME_MEDIUM",
    "WINDOW_TIME_LARGE",
]

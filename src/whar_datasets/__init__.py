from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "TorchAdapter": ("whar_datasets.adapters.adapter_torch", "TorchAdapter"),
    "WHARConfig": ("whar_datasets.config.config", "WHARConfig"),
    "WINDOW_TIME_SMALL": ("whar_datasets.config.config", "WINDOW_TIME_SMALL"),
    "WINDOW_TIME_MEDIUM": ("whar_datasets.config.config", "WINDOW_TIME_MEDIUM"),
    "WINDOW_TIME_LARGE": ("whar_datasets.config.config", "WINDOW_TIME_LARGE"),
    "BENCHMARK_DATASET_IDS": ("whar_datasets.config.getter", "BENCHMARK_DATASET_IDS"),
    "WHARDatasetID": ("whar_datasets.config.getter", "WHARDatasetID"),
    "get_dataset_cfg": ("whar_datasets.config.getter", "get_dataset_cfg"),
    "Loader": ("whar_datasets.loading.loader", "Loader"),
    "PostProcessingPipeline": (
        "whar_datasets.processing.pipeline_post",
        "PostProcessingPipeline",
    ),
    "PreProcessingPipeline": (
        "whar_datasets.processing.pipeline_pre",
        "PreProcessingPipeline",
    ),
    "Split": ("whar_datasets.splitting.split", "Split"),
    "KFoldSplitter": (
        "whar_datasets.splitting.splitter_kfold",
        "KFoldSplitter",
    ),
    "LGSOSplitter": ("whar_datasets.splitting.splitter_lgso", "LGSOSplitter"),
    "LKSOSplitter": ("whar_datasets.splitting.splitter_lkso", "LKSOSplitter"),
    "LOSOSplitter": ("whar_datasets.splitting.splitter_loso", "LOSOSplitter"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)

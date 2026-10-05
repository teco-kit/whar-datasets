from whar_datasets import (
    BENCHMARK_DATASET_IDS,
    PreProcessingPipeline,
    WHARConfig,
    get_dataset_cfg,
)
from whar_datasets.config.getter import WHARDatasetID

# Resampling frequency (Hz) per supported window length in seconds. Windows
# missing from this lookup use the dataset's configured resampling frequency
# for variable-rate sources, or keep the native rate for fixed-rate sources.
WINDOW_RESAMPLING_FREQ: dict[float, int] = {
    1.0: 50,  # 50 samples per window
    2.0: 25,  # 50 samples per window
    3.0: 25,  # 75 samples per window
}


def resolve_resampling_freq(window: float, cfg: WHARConfig) -> float | None:
    """Choose the output rate for a window and dataset source-rate mode."""
    target = WINDOW_RESAMPLING_FREQ.get(window)

    if cfg.sampling_freq is None:
        # Per-session source rates need one common output rate.
        result = target if target is not None else cfg.resampling_freq
        if result is None:
            raise ValueError("Per-session source rates require a resampling rate.")
        return result

    # For fixed-rate data, resample only when the requested rate is higher.
    if target is None or cfg.sampling_freq >= target:
        return None
    return target


ids = BENCHMARK_DATASET_IDS  # [WHARDatasetID.AICOS_HAR]  #
datasets_dir = "/Volumes/Samsung SSD/datasets"  # "./datasets"  #

for id in ids:
    print(f"Preprocessing dataset: {id.value}")
    cfg = get_dataset_cfg(id, datasets_dir=datasets_dir)
    cfg.execution_backend = "sequential"
    cfg.in_memory = True
    cfg.cache_each_split = False
    cfg.num_folds = 10
    cfg.resampling_freq = resolve_resampling_freq(cfg.window_time, cfg)

    force_recompute = False

    pre_pipeline = PreProcessingPipeline(cfg)
    activity_df, session_df, window_df = pre_pipeline.run(force_recompute)

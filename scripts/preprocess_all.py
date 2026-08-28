# %%
from whar_datasets import BENCHMARK_DATASET_IDS, PreProcessingPipeline, get_dataset_cfg

# %%
for id in BENCHMARK_DATASET_IDS:
    print(f"Preprocessing dataset: {id.value}")
    cfg = get_dataset_cfg(id, datasets_dir="/Volumes/Samsung SSD/datasets")
    cfg.execution_backend = "sequential"
    cfg.in_memory = False
    cfg.cache_each_split = False
    cfg.num_folds = 10

    force_recompute = False

    pre_pipeline = PreProcessingPipeline(cfg)
    activity_df, session_df, window_df = pre_pipeline.run(force_recompute)

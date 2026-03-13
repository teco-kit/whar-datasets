from whar_datasets import (
    WINDOW_TIME_MEDIUM,
    Loader,
    LOSOSplitter,
    PostProcessingPipeline,
    PreProcessingPipeline,
    TorchAdapter,
    WHARDatasetID,
    get_dataset_cfg,
)

# create cfg for WISDM dataset
cfg = get_dataset_cfg(WHARDatasetID.WISDM)
cfg.window_time = WINDOW_TIME_MEDIUM

# create and run pre-processing pipeline
pre_pipeline = PreProcessingPipeline(cfg)
activity_df, session_df, window_df = pre_pipeline.run()

# create LOSO splits
splitter = LOSOSplitter(cfg)
splits = splitter.get_splits(session_df, window_df)
split = splits[0]

# create and run post-processing pipeline for the specific split
post_pipeline = PostProcessingPipeline(
    cfg, pre_pipeline, window_df, split.train_indices
)
samples = post_pipeline.run()

# create dataloaders for the specific split
loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
adapter = TorchAdapter(cfg, loader, split)
dataloaders = adapter.get_dataloaders(batch_size=64)

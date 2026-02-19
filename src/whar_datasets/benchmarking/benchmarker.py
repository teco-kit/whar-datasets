from abc import ABC, abstractmethod

from torch import nn

from whar_datasets.adapters.adapter_torch import TorchAdapter
from whar_datasets.benchmarking.dataclasses import Result
from whar_datasets.benchmarking.trainer import TorchTrainer
from whar_datasets.config.config import WHARConfig
from whar_datasets.loading.loader import Loader
from whar_datasets.processing.pipeline_post import PostProcessingPipeline
from whar_datasets.processing.pipeline_pre import PreProcessingPipeline
from whar_datasets.splitting.split import Split
from whar_datasets.splitting.splitter_loso import LOSOSplitter


class Benchmarker(ABC):
    def __init__(self, cfg: WHARConfig) -> None:
        super().__init__()

        self.cfg = cfg

        # create and run pre-processing pipeline
        self.pre_pipeline = PreProcessingPipeline(cfg)
        self.activity_df, self.session_df, self.window_df = self.pre_pipeline.run()

        # create LOSO splits
        splitter = LOSOSplitter(cfg)
        self.splits = splitter.get_splits(self.session_df, self.window_df)

    def benchmark(self) -> None:
        self.results = []

        for split in self.splits:
            # create and run post-processing pipeline for the specific split
            post_pipeline = PostProcessingPipeline(
                self.cfg, self.pre_pipeline, self.window_df, split.train_indices
            )
            samples = post_pipeline.run()

            # create dataloaders for the specific split
            loader = Loader(
                self.session_df, self.window_df, post_pipeline.samples_dir, samples
            )

            # train and evaluate the model for the specific split
            result = self.train_and_eval(split, loader)
            self.results.append(result)

    @abstractmethod
    def train_and_eval(self, split: Split, loader: Loader) -> Result:
        pass


class TorchBenchmarker(Benchmarker):
    def __init__(self, cfg: WHARConfig, model: nn.Module) -> None:
        super().__init__(cfg)
        self.model = model

    def train_and_eval(self, split: Split, loader: Loader) -> Result:

        adapter = TorchAdapter(self.cfg, loader, split)
        dataloaders = adapter.get_dataloaders(batch_size=self.cfg.batch_size)

        trainer = TorchTrainer(self.cfg, dataloaders, self.model)
        train_metrics, val_metrics = trainer.train()
        test_metrics = trainer.evaluate()

        return Result(
            split=split,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )

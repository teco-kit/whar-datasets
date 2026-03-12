from typing import Dict, List

import numpy as np
import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.pipeline import ProcessingPipeline
from whar_datasets.processing.pipeline_pre import PreProcessingPipeline
from whar_datasets.processing.steps.sampling_step import SamplingStep


class PostProcessingPipeline(ProcessingPipeline):
    def __init__(
        self,
        cfg: WHARConfig,
        pre_processing_pipeline: PreProcessingPipeline,
        window_df: pd.DataFrame,
        indices: List[int],
    ):
        self.cfg = cfg
        self.samples_root_dir = pre_processing_pipeline.dataset_dir / "samples"
        self.metadata_dir = pre_processing_pipeline.metadata_dir
        self.windows_dir = pre_processing_pipeline.windows_dir

        self.sampling_step = SamplingStep(
            cfg=cfg,
            metadata_dir=self.metadata_dir,
            samples_dir=self.samples_root_dir,
            windows_dir=self.windows_dir,
            window_df=window_df,
            indices=indices,
            dependent_on=[pre_processing_pipeline.windowing_step],
        )
        self.samples_dir = self.sampling_step.samples_dir

        super().__init__(steps=[self.sampling_step])

    def run(
        self, force_recompute: bool | List[bool] | None = None
    ) -> Dict[str, List[np.ndarray]] | None:
        super().run(force_recompute)

        samples = self.sampling_step.load_results() if self.cfg.in_memory else None

        return samples

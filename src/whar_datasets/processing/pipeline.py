from typing import Any, List

from whar_datasets.processing.steps.abstract_step import AbstractStep


class ProcessingPipeline:
    def __init__(self, steps: List[AbstractStep]):
        self.steps = steps

    def run(self, force_recompute: bool | List[bool] | None = None) -> Any:
        if force_recompute is None:
            force_recompute = False

        if isinstance(force_recompute, list):
            assert len(self.steps) == len(force_recompute)
            for step, fr in zip(self.steps, force_recompute):
                step.run(fr)
        elif isinstance(force_recompute, bool):
            for step in self.steps:
                step.run(force_recompute)

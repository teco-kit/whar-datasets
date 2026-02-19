from typing import Dict, Tuple

from torch import nn
from torch.utils.data import DataLoader

from whar_datasets.benchmarking.dataclasses import Metrics
from whar_datasets.config.config import WHARConfig


class TorchTrainer:
    def __init__(
        self, cfg: WHARConfig, dataloaders: Dict[str, DataLoader], model: nn.Module
    ) -> None:
        self.cfg = cfg
        self.model = model

        self.train_loader = dataloaders["train"]
        self.val_loader = dataloaders["val"]
        self.test_loader = dataloaders["test"]

    def train(self) -> Tuple[Metrics, Metrics]:
        for epoch in range(self.cfg.num_epochs):
            for batch in self.train_loader:
                # Implement training logic here
                pass

            for batch in self.val_loader:
                # Implement validation logic here
                pass

    def evaluate(self) -> Metrics:
        for batch in self.test_loader:
            # Implement evaluation logic here
            pass

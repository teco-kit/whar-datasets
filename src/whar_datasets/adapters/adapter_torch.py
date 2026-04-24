import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from whar_datasets.config.config import WHARConfig
from whar_datasets.loading.loader import Loader
from whar_datasets.splitting.split import Split


class TorchAdapter(Dataset):
    """PyTorch dataset/dataloader bridge for WHAR samples."""

    def __init__(self, cfg: WHARConfig, loader: Loader, split: Split):
        self.cfg = cfg

        self.loader = loader
        self.split = split

        self._set_seed()

    def _set_seed(self) -> None:
        """Seed random number generators used for sampling and dataloaders."""
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.cfg.seed)

    def __len__(self) -> int:
        return len(self.loader)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Return ``(label, sample)`` tensors for one window index."""
        activity_label, subject_label, sample = self.loader.get_item(index)

        y = torch.tensor(activity_label, dtype=torch.long)
        x = torch.tensor(sample[0], dtype=torch.float32)

        return y, x

    def get_dataloaders(self, batch_size: int) -> Dict[str, DataLoader]:
        """Build train/validation/test dataloaders for the current split."""
        train_set = Subset(self, self.split.train_indices)
        val_set = Subset(self, self.split.val_indices)
        test_set = Subset(self, self.split.test_indices)

        train_loader = DataLoader(train_set, batch_size, True, generator=self.generator)
        val_loader = DataLoader(val_set, batch_size, False, generator=self.generator)
        test_loader = DataLoader(test_set, batch_size, False, generator=self.generator)

        return {"train": train_loader, "val": val_loader, "test": test_loader}

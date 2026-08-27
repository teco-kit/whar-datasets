import random
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset, Subset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch support requires the optional dependency: "
        '`pip install "whar-datasets[torch]"`.'
    ) from exc

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
        array = np.asarray(sample[0], dtype=np.float32)
        x = torch.from_numpy(array)

        return y, x

    def get_dataloaders(self, batch_size: int) -> Dict[str, DataLoader]:
        """Build train/validation/test dataloaders for the current split."""
        train_set = Subset(self, self.split.train_indices)
        val_set = Subset(self, self.split.val_indices)
        test_set = Subset(self, self.split.test_indices)

        persistent = (
            self.cfg.dataloader_persistent_workers
            and self.cfg.dataloader_num_workers > 0
        )

        def create(dataset: Dataset, shuffle: bool) -> DataLoader:
            if self.cfg.dataloader_num_workers > 0:
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    generator=self.generator if shuffle else None,
                    num_workers=self.cfg.dataloader_num_workers,
                    pin_memory=self.cfg.dataloader_pin_memory,
                    persistent_workers=persistent,
                    prefetch_factor=self.cfg.dataloader_prefetch_factor,
                )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=self.generator if shuffle else None,
                num_workers=0,
                pin_memory=self.cfg.dataloader_pin_memory,
            )

        train_loader = create(train_set, True)
        val_loader = create(val_set, False)
        test_loader = create(test_set, False)

        return {"train": train_loader, "val": val_loader, "test": test_loader}

from typing import TYPE_CHECKING, Any, Dict, List
import random

import numpy as np

from whar_datasets.config.config import WHARConfig
from whar_datasets.loading.loader import Loader
from whar_datasets.splitting.split import Split

if TYPE_CHECKING:
    import tensorflow as tf


class TFAdapter:
    def __init__(self, cfg: WHARConfig, loader: Loader, split: Split):
        self.cfg = cfg
        self.loader = loader
        self.split = split
        self.tf = self._import_tensorflow()

        # Detect shapes automatically from the first sample.
        self._input_shape = self._infer_shapes()
        self._set_seed()

    @staticmethod
    def _import_tensorflow() -> Any:
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TensorFlow is not installed. Install the optional dependency with "
                "`pip install \"whar-datasets[tf] @ git+https://github.com/teco-kit/whar-datasets.git\"` "
                "or `pip install tensorflow`."
            ) from exc

        return tf

    def _infer_shapes(self) -> "tf.TensorShape":
        _, _, sample = self.loader.get_item(0)
        # Create a list of shapes for each sensor in the sample.
        return self.tf.TensorShape(sample[0].shape)

    def _set_seed(self) -> None:
        self.tf.random.set_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

    def _generator(self, indices: List[int]) -> Any:
        for idx in indices:
            activity_label, _, sample = self.loader.get_item(idx)

            y = np.array(activity_label, dtype=np.int64)
            # Ensure samples are converted to float32 numpy arrays.
            x = np.array(sample[0], dtype=np.float32)

            yield y, x

    def _create_dataset(self, indices: List[int]) -> Any:
        # Define the explicit signature.
        output_signature = (
            self.tf.TensorSpec(shape=(), dtype=self.tf.int64),  # Label
            self.tf.TensorSpec(shape=self._input_shape, dtype=self.tf.float32),  # Sample
        )

        return self.tf.data.Dataset.from_generator(
            lambda: self._generator(indices), output_signature=output_signature
        )

    def get_datasets(self, batch_size: int) -> Dict[str, Any]:
        train_ds = self._create_dataset(self.split.train_indices)
        val_ds = self._create_dataset(self.split.val_indices)
        test_ds = self._create_dataset(self.split.test_indices)

        # Buffer size for shuffle should ideally be the size of the set.
        train_ds = (
            train_ds.shuffle(len(self.split.train_indices))
            .batch(batch_size)
            .prefetch(self.tf.data.AUTOTUNE)
        )

        val_ds = val_ds.batch(len(self.split.val_indices)).prefetch(self.tf.data.AUTOTUNE)
        test_ds = test_ds.batch(1).prefetch(self.tf.data.AUTOTUNE)

        return {"train": train_ds, "val": val_ds, "test": test_ds}

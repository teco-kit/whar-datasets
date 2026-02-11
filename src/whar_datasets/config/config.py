from typing import List, Optional

from pydantic import BaseModel, field_serializer

from whar_datasets.utils.types import NormType, Parse, TransformType


class WHARConfig(BaseModel):
    # metadata fields
    dataset_id: str
    download_url: str | List[str]
    sampling_freq: int
    num_of_subjects: int
    num_of_activities: int
    num_of_channels: int

    # flow fields
    datasets_dir: str  # directory to cache datasets
    in_memory: bool = True  # whether to load the dataset fully into memory
    parallelize: bool = False  # whether to parallelize preprocessing

    # parsing fields
    parse: Parse  # function to parse raw data files to common format
    activity_id_col: str = "activity_id"  # column to use as activity id

    # preprocessing fields
    activity_names: List[str]  # for filtering activities
    sensor_channels: List[str]  # for filtering sensor channels
    window_time: float  # in seconds
    window_overlap: float  # in [0,1]
    resampling_freq: Optional[int] = None

    # postprocessing fields
    val_percentage: float = 0.2  # portion of training data used for validation
    num_subject_groups: Optional[int] = 10  # used for leave-group-out-splitting
    num_folds: Optional[int] = 10  # used for k-fold-splitting
    normalization: Optional[NormType] = NormType.STD_GLOBALLY
    transform: Optional[TransformType] = None

    # training fields
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    seed: int = 0

    @field_serializer("parse")
    def serialize_func(self, func, _info):
        return func.__name__

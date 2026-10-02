from typing import Literal

from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator

from whar_datasets.utils.types import NormType, Parse, TransformType

WINDOW_TIME_SMALL = 1.0
WINDOW_TIME_MEDIUM = 2.0
WINDOW_TIME_LARGE = 3.0
CACHE_SCHEMA_VERSION = 2


class WHARConfig(BaseModel):
    """Unified configuration model used across parsing, preprocessing, and training."""

    # metadata fields
    dataset_id: str
    dataset_url: str
    download_url: str | list[str]
    # None is used only with source_rate_mode="per_session".
    sampling_freq: float | None = Field(gt=0)
    source_rate_mode: Literal["fixed", "per_session"] = "fixed"
    num_of_subjects: int = Field(gt=0)
    num_of_activities: int = Field(gt=0)
    num_of_channels: int = Field(gt=0)
    available_activities: list[str]
    available_channels: list[str]

    # flow fields
    datasets_dir: str = "./datasets/"  # directory to cache datasets
    in_memory: bool = True  # whether to load the dataset fully into memory
    num_workers: int | None = Field(default=None, ge=1)
    execution_backend: Literal["sequential", "process"] = "sequential"
    cache_each_split: bool = True  # cache samples per split hash

    # parsing fields
    parse: Parse  # function to parse raw data files to common format
    activity_id_col: str = "activity_id"  # column to use as activity id

    # preprocessing fields
    selected_activities: list[str] | None
    selected_channels: list[str] | None
    window_time: float = Field(default=WINDOW_TIME_MEDIUM, gt=0)
    window_overlap: float = Field(default=0.5, ge=0, lt=1)
    resampling_freq: float | None = None
    max_session_gap_seconds: float | None = 2.0

    # postprocessing fields
    val_percentage: float = Field(default=0.2, ge=0, lt=1)
    num_folds: int | None = 10  # used for k-fold-splitting
    shuffle_subject: bool = True  # seed-shuffle subjects before LKSO grouping
    normalization: NormType | None = NormType.STD_GLOBALLY
    transform: TransformType | None = None
    strict_train_val_separation: bool = True

    # training fields
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    seed: int = 0
    dataloader_num_workers: int = Field(default=0, ge=0)
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = Field(default=2, ge=1)

    @field_validator("resampling_freq")
    @classmethod
    def validate_resampling_freq(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("resampling_freq must be greater than zero.")
        return value

    @model_validator(mode="after")
    def validate_variable_source_rate(self) -> "WHARConfig":
        if self.source_rate_mode == "fixed" and self.sampling_freq is None:
            raise ValueError("A fixed source rate requires sampling_freq.")
        if self.source_rate_mode == "per_session" and self.sampling_freq is not None:
            raise ValueError(
                "A per-session source rate requires sampling_freq=None."
            )
        if self.source_rate_mode == "per_session" and self.resampling_freq is None:
            raise ValueError(
                "A per-session source rate requires resampling_freq for "
                "fixed-size time windows."
            )
        return self

    @field_validator("max_session_gap_seconds")
    @classmethod
    def validate_max_session_gap(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("max_session_gap_seconds must be greater than zero.")
        return value

    @field_serializer("parse")
    def serialize_func(self, func, _info):
        """Serialize parser callables by their function name."""
        return func.__name__

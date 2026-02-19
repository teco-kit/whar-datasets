from dataclasses import dataclass
from typing import List

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class Result:
    split: Split
    train_metrics: Metrics
    val_metrics: Metrics
    test_metrics: Metrics


@dataclass
class AggregatedResults:
    cfg: WHARConfig
    results: List[Result]

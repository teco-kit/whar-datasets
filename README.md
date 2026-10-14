# WHAR Datasets

## Overview

This library provides a common workflow for wearable human activity recognition (WHAR) datasets:

- automated downloading from original sources and data extraction
- parsing into a unified, standardized data format
- configurable pre-processing (e.g. resampling, windowing) and post-processing (e.g. normalization)
- dataset splitting for common evaluation protocols such as LOSO, LKSO and K-Fold cross-validation
- built-in caching and multiprocessing for improved efficiency
- seamless integration with PyTorch and TensorFlow

It currently supports 38 datasets. New datasets can be added with a parser and configuration.

## Notice

This library does not host any datasets. To use a dataset, please visit its original website and make sure you understand and agree to the dataset’s terms and conditions.

## Installation

With `uv`:

```bash
uv add "whar-datasets[torch] @ git+https://github.com/teco-kit/whar-datasets.git"
```

With `pip`:

```bash
pip install "whar-datasets[torch] @ git+https://github.com/teco-kit/whar-datasets.git"
```

The quickstart uses the `torch` extra. Replace it with `tf` for TensorFlow, or omit `[torch]` to install only the core package.

## Quickstart

```python
from whar_datasets import (
    Loader,
    LOSOSplitter,
    PostProcessingPipeline,
    PreProcessingPipeline,
    TorchAdapter,
    WHARDatasetID,
    get_dataset_cfg,
)

# create cfg for WISDM dataset
cfg = get_dataset_cfg(WHARDatasetID.WISDM)

# create and run pre-processing pipeline
pre_pipeline = PreProcessingPipeline(cfg)
activity_df, session_df, window_df = pre_pipeline.run()

# create LOSO splits
splitter = LOSOSplitter(cfg)
splits = splitter.get_splits(session_df, window_df)
split = splits[0]  # use the first split

# create and run post-processing pipeline for the specific split
post_pipeline = PostProcessingPipeline(
    cfg, pre_pipeline, window_df, split.train_indices
)
samples = post_pipeline.run()

# create dataloaders for the specific split
loader = Loader(activity_df, session_df, window_df, post_pipeline.samples_dir, samples)
adapter = TorchAdapter(cfg, loader, split)
dataloaders = adapter.get_dataloaders(batch_size=64)
```

## Configuration Reference

The easiest way to start is to load a built-in configuration and adjust the fields you need:

```python
from whar_datasets import WHARDatasetID, get_dataset_cfg

cfg = get_dataset_cfg(WHARDatasetID.WISDM, datasets_dir="./datasets/")
cfg.window_time = 3.0
cfg.window_overlap = 0.5
```

`get_dataset_cfg(...)` returns an independent [`WHARConfig`](src/whar_datasets/config/config.py). Common settings are:

| Field | Meaning | Default |
| --- | --- | --- |
| `datasets_dir` | Root directory used to cache downloads, extracted files, metadata, windows, and samples. | `./datasets/` |
| `in_memory` | Whether post-processing keeps samples in memory or loads them from disk when needed. | `True` |
| `num_workers` | Local worker processes; set `1` for single-core or `None` for automatic sizing. | `None` |
| `execution_backend` | Processing backend: `"sequential"` or `"process"`. | `"sequential"` |
| `cache_each_split` | Caches split-specific samples separately so repeated runs can reuse them. | `True` |
| `selected_activities` | Activities retained before windowing. | Dataset-specific |
| `selected_channels` | Sensor channels retained before windowing. | Dataset-specific |
| `window_time` | Sliding window length in seconds. | `2.0` |
| `window_overlap` | Window overlap ratio. | `0.5` |
| `sampling_freq` | Verified source rate in Hz for fixed-rate datasets; `None` when the rate is inferred per session. | Dataset-specific |
| `source_rate_mode` | `"fixed"` uses `sampling_freq`; `"per_session"` estimates each session from its timestamps and requires `resampling_freq`. | `"fixed"` |
| `resampling_freq` | Optional resampling rate in Hz before windowing. | `None` |
| `max_session_gap_seconds` | Reject sessions containing a larger timestamp gap; use `None` to disable. | `2.0` |
| `val_percentage` | Fraction of training data reserved for validation. | `0.2` |
| `num_folds` | Number of folds used for K-fold or LKSO splitting. | `10` |
| `normalization` | Normalization strategy used in post-processing. | `STD_GLOBALLY` |
| `strict_train_val_separation` | Purge overlapping raw intervals between training and validation. | `True` |
| `seed` | Random seed used for sampling and dataloader shuffling. | `0` |

## Supported Datasets

### Single-Sensor Datasets

| Name | Year | Paper |
| --- | --- | --- |
| [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) | 2010 | *Activity Recognition using Cell Phone Accelerometers* |
| [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | 2013 | *A Public Domain Dataset for Human Activity Recognition using Smartphones* |
| [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) | 2015 | *UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor* |
| [HAPT](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) | 2016 | *Transition-aware human activity recognition using smartphones.* |
| [USC-HAD](https://sipi.usc.edu/had/) | 2012 | *USC-HAD: A Daily Activity Dataset for Ubiquitous Activity Recognition Using Wearable Sensors* |
| [UniMiB-SHAR](http://www.sal.disco.unimib.it/technologies/unimib-shar/) | 2017 | *Unimib shar: a dataset for human activity recognition using acceleration data from smartphones* |
| [MotionSense](https://github.com/mmalekzadeh/motion-sense) | 2019 | *Mobile Sensor Data Anonymization* |
| [RealLifeHAR](https://lbd.udc.es/research/real-life-HAR-dataset/) | 2020 | *A Public Domain Dataset for Real-Life Human Activity Recognition Using Smartphone Sensors* |
| [WISDM-19-PHONE](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) | 2019 | *WISDM: Smartphone and Smartwatch Activity and Biometrics Dataset* |
| [WISDM-19-WATCH](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) | 2019 | *WISDM: Smartphone and Smartwatch Activity and Biometrics Dataset* |
| [KU-HAR](https://data.mendeley.com/datasets/45f952y38r/5) | 2021 | *KU-HAR: An open dataset for heterogeneous human activity recognition* |
| [Hang-Time](https://ahoelzemann.github.io/hangtime_har/) | 2023 | *Hang-time HAR: A benchmark dataset for basketball activity recognition using wrist-worn inertial sensors* |
| [CAPTURE-24](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001) | 2024 | *CAPTURE-24: A large dataset of wrist-worn activity tracker data collected in the wild for human activity recognition* |
| [HARSense](https://ieee-dataport.org/open-access/harsense-statistical-human-activity-recognition-dataset) | 2021 | *Harsense: statistical human activity recognition dataset* |
| [FallDet](https://www.kaggle.com/datasets/harnoor343/fall-detection-accelerometer-data) | - | - |

### Multi-Sensor Datasets

| Name | Year | Paper |
| --- | --- | --- |
| [ActRecTut-Gestures](https://github.com/andreas-bulling/ActRecTut) | 2014 | *A tutorial on human activity recognition using body-worn inertial sensors.* |
| [ActRecTut-Walking](https://github.com/andreas-bulling/ActRecTut) | 2014 | *A tutorial on human activity recognition using body-worn inertial sensors.* |
| [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | 2012 | *Introducing a New Benchmarked Dataset for Activity Monitoring* |
| [OPPORTUNITY](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition) | 2010 | *Collecting complex activity datasets in highly rich networked sensor environments* |
| [HHAR](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) | 2015 | *Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition* |
| [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | 2014 | *mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications* |
| [DSADS](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) | 2010 | *Comparative study on classifying human activities with miniature inertial and magnetic sensors* |
| [SAD](https://www.utwente.nl/en/eemcs/ps/research/dataset/) | 2014 | *Fusion of Smartphone Motion Sensors for Physical Activity Recognition* |
| [BMHAD](https://www.kaggle.com/datasets/dasmehdixtr/berkeley-multimodal-human-action-database) | 2013 | *Berkeley MHAD: A Comprehensive Multimodal Human Action Database* |
| [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) | 2009 | *Ambulatory monitoring of freezing of gait in Parkinson’s disease* |
| [SKODA](http://har-dataset.org/doku.php?id=wiki%3Adataset) | 2008 | *Wearable activity tracking in car manufacturing* |
| [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/) | 2016 | *On-body Localization of Wearable Devices: An Investigation of Position-Aware Activity Recognition* |
| [UP-Fall](https://sites.google.com/up.edu.mx/har-up/) | 2019 | *UP-fall detection dataset: A multimodal approach* |
| [UMAFall](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283) | 2017 | *Umafall: A multisensor dataset for the research on automatic fall detection* |
| [REALDISP](https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset) | 2014 | *Dealing with the Effects of Sensor Displacement in Wearable Activity Recognition* |
| [HuGaDB](https://github.com/romanchereshnev/HuGaDB) | 2018 | *HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks* |
| [HARTH](https://archive.ics.uci.edu/dataset/779/harth) | 2021 | *HARTH: A Human Activity Recognition Dataset for Machine Learning* |
| [w-HAR](https://github.com/gmbhat/human-activity-recognition) | 2020 | *w-HAR: An Activity Recognition Dataset and Framework Using Low-Power Wearable Devices* |
| [WEAR](https://mariusbock.github.io/wear/) | 2024 | *Wear: An outdoor sports dataset for wearable and egocentric activity recognition* |
| [HAR70+](https://archive.ics.uci.edu/dataset/780/har70) | 2021 | *A machine learning classifier for detection of physical activity types and postures during free-living* |
| [UCA-EHAR](https://zenodo.org/records/5659336) | 2022 | *UCA-EHAR: A Dataset for Human Activity Recognition with Embedded AI on Smart Glasses* |
| [GOTOV](https://data.4tu.nl/articles/dataset/GOTOV_Human_Physical_Activity_and_Energy_Expenditure_Dataset_on_Older_Individuals/12716081) | 2022 | *A recurrent neural network architecture to model physical activity energy expenditure in older people* |
| [AICOS-HAR](https://zenodo.org/records/19452049) | 2026 | *AICOS-HAR: A human activity recognition dataset spanning multiple devices and body positions* |

## Adding a Dataset

Each integration lives in `src/whar_datasets/config/cfg_<dataset>.py` and contains a parser plus a [`WHARConfig`](src/whar_datasets/config/config.py) instance. See [`cfg_wisdm.py`](src/whar_datasets/config/cfg_wisdm.py) for a complete example.

The parser returns three objects forming the standardized representation:

| Object | Required content |
| --- | --- |
| `activity_df` | One row per activity: integer `activity_id` and string `activity_name`. |
| `session_df` | One row per session: integer `session_id`, `subject_id`, and `activity_id`. |
| `sessions` | A dictionary keyed by `session_id`; each value contains a datetime `timestamp` followed by floating-point sensor channels in `available_channels` order. |

IDs must be zero-based and deterministic; timestamps must be unique and increasing; values must be finite. A session is one continuous recording of one subject performing one activity. Split on subject, activity, source recording, or a large time gap. Align simultaneous sensors by timestamp, but keep modalities recorded at different times as separate datasets. Exclude audio, images, and video.

A minimal configuration looks like this:

```python
import pandas as pd

from whar_datasets.config.config import WHARConfig

ACTIVITIES = ["Walking", "Running", "Sitting", "Standing"]
CHANNELS = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


def parse_example(
    dir: str, activity_id_col: str
) -> tuple[
    pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]
]: ...  # Return activity_df, session_df, and sessions.


cfg_example = WHARConfig(
    dataset_id="example",
    dataset_url="https://example.org/dataset",
    download_url="https://example.org/dataset.zip",
    sampling_freq=50.0,
    num_of_subjects=10,
    num_of_activities=4,
    num_of_channels=6,
    parse=parse_example,
    available_activities=ACTIVITIES,
    selected_activities=ACTIVITIES,
    available_channels=CHANNELS,
    selected_channels=CHANNELS,
)
```

Then:

1. Add the dataset ID to [`WHARDatasetID`](src/whar_datasets/config/getter.py). With the conventional names `cfg_example.py` and `cfg_example`, the lazy registry discovers it automatically.
2. Verify labels, subjects, sensor units, sampling rate, timestamp units, missing recordings, and channel order against the dataset's official documentation and raw files. Never infer these solely from an existing parser.
3. Add focused parser regressions under [`tests/`](tests), especially for filename/label mapping, session boundaries, timestamp conversion, missing values, and multi-sensor alignment. Raw datasets must not be committed.
4. Run the fast checks:

   ```bash
   uv run --with pytest python -m pytest tests/test_dataset_consistency.py -q
   ```

   If the raw dataset is already cached, also run the artifact and end-to-end checks described in [`AGENTS.md`](AGENTS.md). The cache root is `cfg.datasets_dir`; do not redownload data just to test a parser.

To contribute the integration, fork the repository, create a focused branch, and open a pull request. Include links to the source documentation, describe the raw-to-standard mappings and any exclusions or missing recordings, and list the tests you ran.

## Citation

If you use the WHAR Datasets library in your research, please cite our paper:

```
@inproceedings{burzer2025whar,
  title={WHAR Datasets: An Open Source Library for Wearable Human Activity Recognition},
  author={Burzer, Maximilian and King, Tobias and Riedel, Till and Beigl, Michael and R{\"o}ddiger, Tobias},
  booktitle={Companion of the 2025 ACM International Joint Conference on Pervasive and Ubiquitous Computing},
  pages={1315--1322},
  year={2025}
}
```

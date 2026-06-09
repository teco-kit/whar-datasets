# WHAR Datasets

## Overview

This library offers comprehensive support for widely used WHAR (Wearable Human Activity Recognition) datasets, including:

- automated downloading from original sources and data extraction
- parsing into a unified, standardized data format
- configurable pre-processing (e.g., resampling, windowing) and post-processing (e.g., normalization)
- dataset splitting for common evaluation protocols such as LOSO and K-Fold cross-validation
- built-in caching and multi-processing for improved efficiency
- seamless integration with PyTorch and TensorFlow

The library currently includes out-of-the-box support for 37 datasets (listed below). Additional WHAR datasets can be easily integrated by defining a custom configuration with an associated parser and registering it with the framework.

## Notice

This library does not host any datasets. To use a dataset, please visit its original website and make sure you understand and agree to the dataset’s terms and conditions.

## Installation

### With `uv`:

```bash
uv add "whar-datasets @ git+https://github.com/teco-kit/whar-datasets.git"
```

With optional TensorFlow support:

```bash
uv add "whar-datasets[tf] @ git+https://github.com/teco-kit/whar-datasets.git"  
```

### With `pip`:

```bash
pip install "whar-datasets @ git+https://github.com/teco-kit/whar-datasets.git"
```

With optional TensorFlow support:

```bash
pip install "whar-datasets[tf] @ git+https://github.com/teco-kit/whar-datasets.git"
```

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
split = splits[0]

# create and run post-processing pipeline for the specific split
post_pipeline = PostProcessingPipeline(cfg, pre_pipeline, window_df, split.train_indices)
samples = post_pipeline.run()

# create dataloaders for the specific split
loader = Loader(activity_df, session_df, window_df, post_pipeline.samples_dir, samples)
adapter = TorchAdapter(cfg, loader, split)
dataloaders = adapter.get_dataloaders(batch_size=64)
```

## Supported Datasets

### Single-Sensor Datasets

| Supported | Name | Year | Paper | Citations |
| --- | --- | --- | --- | --- |
| ✅ | [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) | 2010 | *Activity Recognition using Cell Phone Accelerometers* | 3862 |
| ✅ | [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | 2013 | *A Public Domain Dataset for Human Activity Recognition using Smartphones* | 3372 |
| ✅ | [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) | 2015 | *UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor* | 997 |
| ✅ | [HAPT](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) | 2016 | *Transition-aware human activity recognition using smartphones.* | 939 |
| ✅ | [USC-HAD](https://sipi.usc.edu/had/) | 2012 | *USC-HAD: A Daily Activity Dataset for Ubiquitous Activity Recognition Using Wearable Sensors* | 753 |
| ✅ | [UniMiB-SHAR](http://www.sal.disco.unimib.it/technologies/unimib-shar/) | 2017 | *Unimib shar: a dataset for human activity recognition using acceleration data from smartphones* | 712 |
| ✅ | [MotionSense](https://github.com/mmalekzadeh/motion-sense) | 2019 | *Mobile Sensor Data Anonymization* | 345 |
| ✅ | [RealLifeHAR](https://lbd.udc.es/research/real-life-HAR-dataset/) | 2020 | *A Public Domain Dataset for Real-Life Human Activity Recognition Using Smartphone Sensors* | 208 |
| ✅ | [WISDM-19-PHONE](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) | 2019 | *WISDM: Smartphone and Smartwatch Activity and Biometrics Dataset* | 198 |
| ✅ | [WISDM-19-WATCH](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) | 2019 | *WISDM: Smartphone and Smartwatch Activity and Biometrics Dataset* | 198 |
| ✅ | [KU-HAR](https://data.mendeley.com/datasets/45f952y38r/5) | 2021 | *KU-HAR: An open dataset for heterogeneous human activity recognition* | 187 |
| ✅ | [Hang-Time](https://ahoelzemann.github.io/hangtime_har/) | 2023 | *Hang-time HAR: A benchmark dataset for basketball activity recognition using wrist-worn inertial sensors* | 52 |
| ✅ | [CAPTURE-24](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001) | 2024 | *CAPTURE-24: A large dataset of wrist-worn activity tracker data collected in the wild for human activity recognition* | 45 |
| ✅ | [HARSense](https://ieee-dataport.org/open-access/harsense-statistical-human-activity-recognition-dataset) | 2021 | *Harsense: statistical human activity recognition dataset* | 5 |
| ✅ | [FallDet](https://www.kaggle.com/datasets/harnoor343/fall-detection-accelerometer-data) | - | - | - |


### Multi-Sensor Datasets

| Supported | Name | Year | Paper | Citations |
| --- | --- | --- | --- | --- |
| ✅ | [ActRecTut-Gestures](https://github.com/andreas-bulling/ActRecTut) | 2014 | *A tutorial on human activity recognition using body-worn inertial sensors.* | 2086 |
| ✅ | [ActRecTut-Walking](https://github.com/andreas-bulling/ActRecTut) | 2014 | *A tutorial on human activity recognition using body-worn inertial sensors.* | 2086 |
| ✅ | [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | 2012 | *Introducing a New Benchmarked Dataset for Activity Monitoring* | 1758 |
| ✅ | [OPPORTUNITY](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition) | 2010 | *Collecting complex activity datasets in highly rich networked sensor environments* | 1024 |
| ✅ | [HHAR](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) | 2015 | *Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition* | 1019 |
| ✅ | [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | 2014 | *mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications* | 887 |
| ✅ | [DSADS](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) | 2010 | *Comparative study on classifying human activities with miniature inertial and magnetic sensors* | 780 |
| ✅ | [SAD](https://www.utwente.nl/en/eemcs/ps/research/dataset/) | 2014 | *Fusion of Smartphone Motion Sensors for Physical Activity Recognition* | 752 |
| ✅ | [BMHAD](https://www.kaggle.com/datasets/dasmehdixtr/berkeley-multimodal-human-action-database) | 2013 | *Berkeley MHAD: A Comprehensive Multimodal Human Action Database* | 668 |
| ✅ | [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) | 2009 | *Ambulatory monitoring of freezing of gait in Parkinson’s disease* | 652 |
| ✅ | [SKODA](http://har-dataset.org/doku.php?id=wiki%3Adataset) | 2008 | *Wearable activity tracking in car manufacturing* | 504 |
| ✅ | [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/) | 2016 | *On-body Localization of Wearable Devices: An Investigation of Position-Aware Activity Recognition* | 482 |
| ✅ | [UP-Fall](https://sites.google.com/up.edu.mx/har-up/) | 2019 | *UP-fall detection dataset: A multimodal approach* | 462 |
| ✅ | [UMAFall](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283) | 2017 | *Umafall: A multisensor dataset for the research on automatic fall detection* | 243 |
| ✅ | [REALDISP](https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset) | 2014 | *Dealing with the Effects of Sensor Displacement in Wearable Activity Recognition* | 216 |
| ✅ | [HuGaDB](https://github.com/romanchereshnev/HuGaDB) | 2018 | *HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks* | 154 |
| ✅ | [HARTH](https://archive.ics.uci.edu/dataset/779/harth) | 2021 | *HARTH: A Human Activity Recognition Dataset for Machine Learning* | 132 |
| ✅ | [w-HAR](https://github.com/gmbhat/human-activity-recognition) | 2020 | *w-HAR: An Activity Recognition Dataset and Framework Using Low-Power Wearable Devices* | 100 |
| ✅ | [WEAR](https://mariusbock.github.io/wear/) | 2024 | *Wear: An outdoor sports dataset for wearable and egocentric activity recognition* | 66 |
| ✅ | [HAR70+](https://archive.ics.uci.edu/dataset/780/har70) | 2021 | *A machine learning classifier for detection of physical activity types and postures during free-living* | 55 |
| ✅ | [UCA-EHAR](https://zenodo.org/records/5659336) | 2022 | *UCA-EHAR: A Dataset for Human Activity Recognition with Embedded AI on Smart Glasses* | 35 |
| ✅ | [GOTOV](https://data.4tu.nl/articles/dataset/GOTOV_Human_Physical_Activity_and_Energy_Expenditure_Dataset_on_Older_Individuals/12716081) | 2022 | *A recurrent neural network architecture to model physical activity energy expenditure in older people* | 33 |

## Configuration Reference

The easiest way to start is to load a built-in configuration and adjust the fields you need:

```python
from whar_datasets import WHARDatasetID, get_dataset_cfg

cfg = get_dataset_cfg(WHARDatasetID.WISDM, datasets_dir="./datasets/")
cfg.parallelize = True
cfg.window_time = 3.0
cfg.window_overlap = 0.5
...
```

`get_dataset_cfg(...)` returns a dataset-specific [`WHARConfig`](/Users/maxburzer/whar-datasets/src/whar_datasets/config/config.py#L12-L59) with sensible defaults. The most useful fields are:

| Field | Meaning | Default |
| --- | --- | --- |
| `datasets_dir` | Root directory used to cache downloads, extracted files, metadata, windows, and samples. | `./datasets/` |
| `in_memory` | Whether post-processing keeps samples in memory or loads them from disk when needed. | `True` |
| `parallelize` | Enables parallel preprocessing steps. | `False` |
| `cache_each_split` | Caches split-specific samples separately so repeated runs can reuse them. | `True` |
| `selected_activities` | Optional activity filter applied before windowing. | `None` |
| `selected_channels` | Optional channel filter applied before windowing. | `None` |
| `window_time` | Sliding window length in seconds. | `3.0` |
| `window_overlap` | Window overlap ratio. | `0.5` |
| `resampling_freq` | Optional resampling rate in Hz before windowing. | `None` |
| `val_percentage` | Fraction of training data reserved for validation. | `0.2` |
| `num_subject_groups` | Number of groups used for leave-group-out splitting. | `10` |
| `num_folds` | Number of folds used for K-fold splitting. | `10` |
| `normalization` | Normalization strategy used in post-processing. | `STD_GLOBALLY` |
| `transform` | Optional transform applied to windows, such as STFT or DWT. | `None` |
| `batch_size` | Default batch size used by adapters and training helpers. | `64` |
| `learning_rate` | Default learning rate for downstream training. | `1e-4` |
| `num_epochs` | Default number of training epochs. | `100` |
| `seed` | Random seed used for sampling and dataloader shuffling. | `0` |

If you want to benchmark multiple built-in datasets, `BENCHMARK_DATASET_IDS` contains the curated subset used by the library.

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


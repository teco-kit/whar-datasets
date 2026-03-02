# WHAR Datasets

This library offers comprehensive support for widely used WHAR (Wearable Human Activity Recognition) datasets, including:

- automated downloading from original sources and data extraction
- parsing into a unified, standardized data format
- configurable pre-processing (e.g., resampling, windowing) and post-processing (e.g., normalization, transformations)
- dataset splitting for common evaluation protocols such as LOSO and K-Fold cross-validation
- built-in caching and multi-processing for improved performance
- seamless integration with PyTorch and TensorFlow

The library currently includes out-of-the-box support for 33 datasets (listed below). Additional WHAR datasets can be easily integrated by defining a custom configuration with an associated parser and registering it with the framework.

# Notice 

This library does not host any datasets. To use a dataset, please visit its original website and make sure you understand and agree to the dataset’s terms and conditions.

# How to Use 

### Installation

```
pip install "git+https://github.com/teco-kit/whar-datasets.git"
```

### Example with PyTorch

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
loader = Loader(session_df, window_df, post_pipeline.samples_dir, samples)
adapter = TorchAdapter(cfg, loader, split)
dataloaders = adapter.get_dataloaders(batch_size=64)
```

# Supported Datasets

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


### Multi-Sensor Datasets

| Supported | Name | Year | Paper | Citations |
| --- | --- | --- | --- | --- |
| ✅ | [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | 2012 | *Introducing a New Benchmarked Dataset for Activity Monitoring* | 1758 |
| ✅ | [OPPORTUNITY](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition) | 2010 | *Collecting complex activity datasets in highly rich networked sensor environments* | 1024 |
| ✅ | [HHAR](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) | 2015 | *Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition* | 1019 |
| ✅ | [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | 2014 | *mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications* | 887 |
| ✅ | [DSADS](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) | 2010 | *Comparative study on classifying human activities with miniature inertial and magnetic sensors* | 780 |
| ✅ | [SAD](https://www.utwente.nl/en/eemcs/ps/research/dataset/) | 2014 | *Fusion of Smartphone Motion Sensors for Physical Activity Recognition* | 752 |
| ✅ | [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) | 2009 | *Ambulatory monitoring of freezing of gait in Parkinson’s disease* | 652 |
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

# Citation

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

<!-- > Another version of the UCI-HAR Dataset --> 
<!-- | ⬜        | [HAPT](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions)           | 2016 | *Transition-aware human activity recognition using smartphones.*                           | -->
<!-- https://zenodo.org/records/3831958 -->
<!-- https://zenodo.org/records/13987073 -->

<!-- Core Benchmark Datasets
https://github.com/haoranD/Awesome-Human-Activity-Recognition
UCI-HAR (UCI Human Activity Recognition Dataset)

WISDM (Wireless Sensor Data Mining)

PAMAP2 (Physical Activity Monitoring Dataset)

Large-Scale & Real-World Datasets
ExtraSensory Dataset

CAPTURE-24 Dataset

SHL (Sussex-Huawei Locomotion Dataset)

Specialized & Domain-Specific Datasets
Opportunity Dataset (and Opportunity++)

MHEALTH (Mobile Health Dataset)

RealWorld HAR Dataset

KU-HAR Dataset

MotionSense Dataset

UniMiB-SHAR Dataset

Daphnet Freezing of Gait Dataset

HAPT (Human Activities and Postural Transitions Dataset)

HHAR (Heterogeneity Activity Recognition Dataset)

AReM (Activity Recognition System Based on Multisensor Data Fusion)

Emerging & Curated Benchmarks
DAGHAR Benchmark

HARTH (Human Activity Recognition Trondheim Dataset)

WEAR (Wearable and Egocentric Activity Recognition Dataset)

HAR70+ (referenced in trends) -->

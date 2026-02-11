# WHAR Datasets

This library provides data handling support for popular WHAR (Wearable Human Activity Recognition) datasets including

<!-- - metadata descriptions in [DCAT-AP](https://www.dcat-ap.de/) and [Croissant](https://github.com/mlcommons/croissant) -->
- downloading from original source
- parsing into a standardized data format
- configuration-driven preprocessing, splitting, normalization, and more
- integration with pytorch and tensorflow

# How to Install

```
pip install "git+https://github.com/teco-kit/whar-datasets.git"
```

This installs the library into the active environment.

# How To Use With PyTorch

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

Not yet natively supported WHAR datasets can be integrated via a custom configuration (with parser).

### Currently Supported Datasets

| Supported | Name | Year | Paper | Citations |
| :--- | :--- | :--- | :--- | :--- |
| ✅ | [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) | 2010 | *Activity Recognition using Cell Phone Accelerometers* | 3862 |
| ✅ | [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | 2013 | *A Public Domain Dataset for Human Activity Recognition using Smartphones* | 3372 |
| ✅ | [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | 2012 | *Introducing a New Benchmarked Dataset for Activity Monitoring* | 1758 |
| ✅ | [OPPORTUNITY](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition) | 2010 | *Collecting complex activity datasets in highly rich networked sensor environments* | 1024 |
| ⬜ | [HHAR](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) | 2015 | *Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition* | 1019 |
| ⬜ | [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) | 2015 | *UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor* | 997 |
| ✅ | [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | 2014 | *mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications* | 887 |
| ✅ | [DSADS](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities) | 2010 | *Comparative study on classifying human activities with miniature inertial and magnetic sensors* | 780 |
| ✅ | [USC-HAD](https://sipi.usc.edu/had/) | 2012 | *USC-HAD: A Daily Activity Dataset for Ubiquitous Activity Recognition Using Wearable Sensors* | 753 |
| ⬜ | [SAD](https://www.utwente.nl/en/eemcs/ps/research/dataset/) | 2014 | *Fusion of Smartphone Motion Sensors for Physical Activity Recognition* | 752 |
| ⬜ | [UniMiB-SHAR](http://www.sal.disco.unimib.it/technologies/unimib-shar/) | 2017 | *Unimib shar: a dataset for human activity recognition using acceleration data from smartphones* | 712 |
| ✅ | [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) | 2009 | *Ambulatory monitoring of freezing of gait in Parkinson’s disease* | 652 |
| ⬜ | [CHAD](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait) | 2009 | *Complex human activity recognition using smartphone and wrist-worn motion sensors* | 554 |
| ⬜ | [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/) | 2016 | *On-body Localization of Wearable Devices: An Investigation of Position-Aware Activity Recognition* | 482 |
| ⬜ | [ExtraSensory](http://extrasensory.ucsd.edu/) | 2016 | *Recognizing Detailed Human Context In-the-Wild from Smartphones and Smartwatches* | 402 |
| ⬜ | [MobiAct](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/#) | 2016 | *The MobiAct dataset: recognition of activities of daily living using smartphones* | 364 |
| ✅ | [MotionSense](https://github.com/mmalekzadeh/motion-sense) | 2019 | *Mobile Sensor Data Anonymization* | 345 |
| ⬜ | [PARDUSS](https://www.utwente.nl/en/eemcs/ps/research/dataset/) | 2013 | *Towards physical activity recognition using smartphone sensors* | 345 |
| ⬜ | [SWELL-KW](https://www.kaggle.com/datasets/qiriro/swell-heart-rate-variability-hrv) | 2014 | *The SWELL Knowledge Work Dataset for Stress and User Modeling Research* | 339 |
| ⬜ | [SHL](http://www.shl-dataset.org/dataset/) | 2018 | *The University of Sussex-Huawei Locomotion and Transportation Dataset for Multimodal Analytics with Mobile Devices* | 317 |
| ⬜ | DA | 2012 | *Recognizing Human Activities User-independently on Smartphones Based on Accelerometer Data* | 302 |
| ⬜ | [UMAFall](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283) | 2017 | *Umafall: A multisensor dataset for the research on automatic fall detection* | 243 |
| ⬜ | [REALDISP](https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset) | 2014 | *Dealing with the Effects of Sensor Displacement in Wearable Activity Recognition* | 216 |
| ⬜ | [RealLifeHAR](https://lbd.udc.es/research/real-life-HAR-dataset/) | 2020 | *A Public Domain Dataset for Real-Life Human Activity Recognition Using Smartphone Sensors* | 208 |
| ✅ | [WISDM-19](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset) | 2019 | *WISDM: Smartphone and Smartwatch Activity and Biometrics Dataset* | 198 |
| ✅ | [KU-HAR](https://data.mendeley.com/datasets/45f952y38r/5) | 2021 | *KU-HAR: An open dataset for heterogeneous human activity recognition* | 187 |
| ⬜ | [HASC-Challenge](http://hasc.jp/) | 2011 | *Hasc challenge: gathering large scale human activity corpus for the real-world activity understandings* | 157 |
| ✅ | [HuGaDB](https://github.com/romanchereshnev/HuGaDB) | 2018 | *HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks* | 154 |
| ⬜ | [HARTH](https://archive.ics.uci.edu/dataset/779/harth) | 2021 | *HARTH: A Human Activity Recognition Dataset for Machine Learning* | 132 |
| ⬜ | [MobiFall](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/#) | 2014 | *The MobiFall Dataset: Fall Detection and Classification with a Smartphone* | 128 |
| ⬜ | [FallAllD](https://www.kaggle.com/datasets/harnoor343/fall-detection-accelerometer-data) | 2020 | *FallAllD: An Open Dataset of Human Falls and Activities of Daily Living for Classical and Deep Learning Applications* | 115 |
| ✅ | [w-HAR](https://github.com/gmbhat/human-activity-recognition) | 2020 | *w-HAR: An Activity Recognition Dataset and Framework Using Low-Power Wearable Devices* | 98 |
| ⬜ | [HAR70+](https://archive.ics.uci.edu/dataset/780/har70) | 2021 | *A machine learning classifier for detection of physical activity types and postures during free-living* | 55 |
| ⬜ | [TNDA-HAR](https://ieee-dataport.org/open-access/tnda-har-0) | 2022 | *Deep transfer learning with graph neural network for sensor-based human activity recognition* | 48 |
| ⬜ | CAPTURE-24 | 2024 | *CAPTURE-24: A large dataset of wrist-worn activity tracker data collected in the wild for human activity recognition* | 45 |
| ⬜ | PAR | 2021 | *Context-aware support for cardiac health monitoring using federated machine learning* | 12 |
| ⬜ | [iSPL](https://github.com/thunguyenth/HAR_IMU_Stretch) | 2022 | *An Investigation on Deep Learning-Based Activity Recognition Using IMUs and Stretch Sensors* | 11 |
| ⬜ | CHARM | 2021 | *A recommendation specific human activity recognition dataset with mobile device's sensor data* | 5 |
| ✅ | [HARSense](https://ieee-dataport.org/open-access/harsense-statistical-human-activity-recognition-dataset) | 2021 | - | - |
| ⬜ | [AReM](https://archive.ics.uci.edu/dataset/366/activity+recognition+system+based+on+multisensor+data+fusion+arem) | 2016 | - | - |

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
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

# How To Use

```python
from whar_datasets.adapters.torch_adapter import TorchAdapter
from whar_datasets.support.getter import WHARDatasetID, get_dataset_cfg

# initialize dataset config given a dataset id
cfg = get_dataset_cfg(dataset_id=WHARDatasetID.WISDM)

# initialize dataset and preprocess
dataset = TorchAdapter(cfg=cfg)
dataset.preprocess()

# postprocess for a given split and get dataloaders
dataset.postprocess(split_group_index=0)
dataloaders = dataset.get_dataloaders(batch_size=32)
```

Not yet natively supported WHAR datasets can be integrated via a custom configuration (with parser).

### Currently Supported Datasets

| Supported | Name           | Year | What To Cite                                                                                       |
|-----------|----------------|------|---------------------------------------------------------------------------------------------|
| ✅        | [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)        | 2009 | *Ambulatory monitoring of freezing of gait in Parkinson’s disease*                         |
| ✅        | [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php)       | 2010 | *Activity Recognition using Cell Phone Accelerometers*                                      |
| ✅        | [OPPORTUNITY](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition)    | 2010 | *Collecting complex activity datasets in highly rich networked sensor environments*        |
| ✅        | [DSADS](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities)          | 2010 | *Comparative study on classifying human activities with miniature inertial and magnetic sensors* |
| ⬜        | [USC-HAD](https://sipi.usc.edu/had/)        | 2012 | *USC-HAD: A Daily Activity Dataset for Ubiquitous Activity Recognition Using Wearable Sensors* |
| ✅        | [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)         | 2012 | *Introducing a New Benchmarked Dataset for Activity Monitoring*                            |
| ⬜        | DA | 2012 | *Recognizing Human Activities User-independently on Smartphones Based on Accelerometer Data* |
| ✅        | [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)        | 2013 | *A Public Domain Dataset for Human Activity Recognition using Smartphones*                  |
| ⬜        | [PARDUSS](https://www.utwente.nl/en/eemcs/ps/research/dataset/) | 2013 | *Towards physical activity recognition using smartphone sensors*     
| ⬜        | [REALDISP](https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset) | 2014 | *Dealing with the Effects of Sensor Displacement in Wearable Activity Recognition*                        |
| ⬜        | [MobiFall](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/#) | 2014 | *The MobiFall Dataset: Fall Detection and Classification with a Smartphone*                |
| ✅        | [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)        | 2014 | *mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications*      |
| ⬜        | [SWELL-KW](https://www.kaggle.com/datasets/qiriro/swell-heart-rate-variability-hrv)          | 2014 | *The SWELL Knowledge Work Dataset for Stress and User Modeling Research*                   |
| ⬜        | [SAD](https://www.utwente.nl/en/eemcs/ps/research/dataset/) | 2014 | *Fusion of Smartphone Motion Sensors for Physical Activity Recognition*                    |
| ⬜        | [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)       | 2015 | *UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor* |
| ⬜        | [MobiAct](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/#) | 2016 | *The MobiAct dataset: recognition of activities of daily living using smartphones*         |
| ⬜        | [ExtraSensory](http://extrasensory.ucsd.edu/)   | 2016 | *Recognizing Detailed Human Context In-the-Wild from Smartphones and Smartwatches*         |
| ⬜        | [HAPT](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions)           | 2016 | *Transition-aware human activity recognition using smartphones.*                           |
| ⬜        | [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/)      | 2016 | *On-body Localization of Wearable Devices: An Investigation of Position-Aware Activity Recognition*    |
| ⬜        | [UniMiB-SHAR](http://www.sal.disco.unimib.it/technologies/unimib-shar/) | 2017 | *Unimib shar: a dataset for human activity recognition using acceleration data from smartphones* |
| ⬜        | [UMAFall](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283) | 2017 | *UMAFall: Fall Detection Dataset*                                                          |
| ⬜        | [SHL](http://www.shl-dataset.org/dataset/)            | 2018 | *The University of Sussex-Huawei Locomotion and Transportation Dataset for Multimodal Analytics with Mobile Devices* |
| ⬜        | [HuGaDB](https://github.com/romanchereshnev/HuGaDB)         | 2018 | *HuGaDB: Human Gait Database for Activity Recognition from Wearable Inertial Sensor Networks* |
| ⬜        | [WISDM-19](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)       | 2019 | WISDM: Smartphone and Smartwatch Activity and Biometrics Dataset |
| ✅        | [MotionSense](https://github.com/mmalekzadeh/motion-sense)    | 2019 | *Mobile Sensor Data Anonymization*                                                         |
| ⬜        | [RealLifeHAR](https://lbd.udc.es/research/real-life-HAR-dataset/)    | 2020 | *A Public Domain Dataset for Real-Life Human Activity Recognition Using Smartphone Sensors* |
| ✅        | [w-HAR](https://github.com/gmbhat/human-activity-recognition)          | 2020 | *w-HAR: An Activity Recognition Dataset and Framework Using Low-Power Wearable Devices*    |
| ✅        | [KU-HAR](https://data.mendeley.com/datasets/45f952y38r/5)         | 2021 | *KU-HAR: An open dataset for heterogeneous human activity recognition*                     |
| ✅        | [HARSense](https://ieee-dataport.org/open-access/harsense-statistical-human-activity-recognition-dataset)       | 2021 | *HARSense: Statistical Human Activity Recognition Dataset*                                 |
| ⬜        | [iSPL](https://github.com/thunguyenth/HAR_IMU_Stretch) | 2022 | *An Investigation on Deep Learning-Based Activity Recognition Using IMUs and Stretch Sensors* |





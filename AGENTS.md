## Dataset Requirements
- Each dataset must include activity labels and subject identifiers
- These are required for Leave-One-Subject-Out Cross-Validation (LOSOCV)
- Inform me if activity labels or subject identifiers are missing
- Make sure activity and subject information is correctly assigned (e.g. subject01_walking.csv indicates subject and activity)

## Session Definition
- A session is one continuous recording of one subject performing one activity
- Create a new session if the subject changes
- Create a new session if the activity changes
- Create a new session if there is a large timestamp gap in the recording (e.g. recording stops and resumes after several minutes)

## Activity Labels
- Inform me if multiple activity label schemes are available (e.g. coarse: walking, running / fine-grained: walking upstairs, walking downstairs)

## Sensor Modalities
- Include only time-series sensor data (e.g. IMU, Accelerometer, Gyroscope, Magnetometer, Physiological signals)
- Exclude non-time-series data (e.g. Audio, Images, Video)
- Inform me if different sensor modalities were not recorded at the same time (e.g. IMU recorded separately from physiological signals)
- Treat non-simultaneously recorded modalities as separate datasets

## Implementation & Data Handling
- Use Python type hints in all implementations
- You can set cfg.parallelize to true for faster preprocessing
- Dataset caches are stored in notebooks/datasets, do not redownload existing datasets
- Check readme files or metadata if needed (e.g. activity labels stored in metadata file)
- Use filenames if needed when information is not available as data columns (e.g. subject01_walking.csv)
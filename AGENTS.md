# Standardized Dataset Format

## Requirements

- Every dataset must contain:
  - Activity labels
  - Subject identifiers  

  This is necessary to enable Leave-One-Subject-Out Cross-Validation (LOSOCV).  
  If either activity labels or subject identifiers are missing, tell me.

## Session Definition

A session is defined as: A continuous recording of a single subject performing a single activity

A new session must be created when:

- The subject changes, or
- The activity changes, or
- There is an unusually large timestamp gap in the recording (indicating discontinuity)

## Activity Labels

- If a dataset provides multiple possible activity label schemes (e.g., coarse vs. fine-grained labels), then tell me.

## Sensor Modalities

- Include only time-series sensor data, such as:
  - IMU
  - Accelerometer
  - Gyroscope
  - Magnetometer
  - Physiological signals

- Exclude all non-time-series modalities, including:
  - Audio
  - Images
  - Video

- if different sensor modalities where not recorded simultaneously, than the effectively constitute several different datasets, tell me about this

## Implementation Details

- All implementations must use Python type hints.

## Other Details

- you can set cfg.parallelize to true for faster preprocessing
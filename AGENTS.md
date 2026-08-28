# Dataset Parser and Configuration Requirements

## Non-negotiable dataset requirements

- Every dataset must provide activity labels and subject identifiers. Stop the
  audit and inform the user if either cannot be recovered reliably.
- Activity and subject assignments must be verified against raw columns,
  filenames, directory names, or official metadata. Do not accept the existing
  parser or configuration as evidence that its own mapping is correct.
- Subject identifiers are required for Leave-One-Subject-Out Cross-Validation.
- If several sensors were recorded simultaneously, align them by timestamp and
  stack them as channels of one multichannel time series.
- Treat timestamp units as dataset-specific source metadata. Determine whether
  raw values are seconds, milliseconds, microseconds, or nanoseconds before
  converting or comparing them; never compare a gap threshold in one unit to
  timestamps expressed in another.
- For simultaneous modalities, validate alignment with a bounded nearest-time
  or explicitly documented resampling operation. Report the tolerance, match
  rate, unmatched samples, duplicate timestamps, and whether alignment is
  constrained by subject, activity, source recording, or other provenance keys.
- Preserve source row/file order when timestamps reset, overlap, or are known
  to be unreliable. Use timestamp order only after verifying that it agrees
  with source chronology.
- Include only time-series sensor modalities, such as accelerometers,
  gyroscopes, magnetometers, motion capture, or physiological signals. Exclude
  audio, images, and video.
- If modalities were recorded at different times, report this and treat them as
  separate datasets rather than stacking them.
- Report every available activity-label scheme, such as coarse and fine-grained
  labels, and state which scheme the parser uses.

## Session definition

A session is one continuous recording of one subject performing one activity.
Create a new session whenever:

- the subject changes;
- the activity changes;
- the source recording/file changes when continuity cannot be established; or
- timestamps contain a large recording gap.

Never merge recordings merely because their subject and activity labels match.
Never split a recording into arbitrary chunks just to make windowing easier.
Short sessions can be valid, especially for brief events such as falls,
transitions, or freezing episodes.

## Cache and execution safety

- Audit one dataset at a time.
- Use `cfg.datasets_dir` as the cache root. It is commonly `datasets/`, but may
  be configured differently by the caller.
- Reuse existing downloads and extracted raw data. Do not redownload a dataset
  merely to run an audit.
- Do not use `force_recompute=True` until the raw layout and expected cache
  targets have been confirmed.
- Use `cfg.execution_backend = "sequential"` for the simplest diagnostic run.
  Check the process backend afterward for output equivalence.
- Use `cfg.num_workers = 1` when a deterministic single-core process-path check
  is useful.
- Keep generated audit artifacts outside committed source directories unless
  they are intentional test fixtures.

# Per-dataset audit procedure

## 1. Establish the external source of truth

Before changing code, inspect the dataset README, official metadata, label
files, raw filenames, and directory structure. Record:

- expected subjects and their raw identifiers;
- expected activities, raw labels, names, and any ignored/unknown class;
- expected recordings or repetitions per subject/activity, when documented;
- sensor modalities, sensor locations, axes, units, and channel order;
- nominal and timestamp-derived sampling rates;
- whether modalities are simultaneous;
- timestamp units and whether timestamps reset between files; and
- known missing, corrupt, or intentionally excluded recordings.

For every timestamped source, inspect timestamp magnitude and successive-step
quantiles against the documented sampling rate. Test at least one known
seconds/milliseconds/microseconds/nanoseconds value and confirm that timestamp
resets and large gaps are handled before interpolation or session construction.

Enumerate the actual raw files and compare them with documented combinations.
Report missing and unexpected files explicitly. A missing source recording is
not automatically a parser bug, but it must not disappear silently.

## 2. Audit the `WHARConfig`

Check the dataset configuration against the external source of truth:

- `dataset_id`, URLs, and parser callback refer to the correct dataset;
- `sampling_freq` represents the intended processing frequency;
- `num_of_subjects`, `num_of_activities`, and `num_of_channels` match what the
  parser actually returns;
- `available_activities` contains every supported label exactly once;
- `selected_activities` is a documented subset of available activities;
- `available_channels` is unique, ordered, and names every included
  time-series channel;
- `selected_channels` is a valid subset with the intended sensor locations and
  axes;
- `window_time`, `window_overlap`, resampling, and maximum-gap settings are
  plausible for the activity durations and sampling rate; and
- optional transforms and normalization do not change parser semantics.

Do not “fix” count mismatches by changing the configured number until the raw
mapping has been verified.

## 3. Audit raw-to-metadata mapping

Trace several representative files end to end, including the first and last
subject, multiple activities, and unusual filename patterns. Verify:

- filename/directory subject IDs map to the correct `subject_id`;
- filename, metadata, or row labels map to the correct `activity_id` and name;
- repetitions remain distinct sessions;
- label changes inside a recording start new sessions;
- timestamps and sensor columns are not accidentally exchanged;
- timestamp units, resets, duplicate timestamps, and large gaps are handled in
  the source unit before conversion to datetime;
- one-based raw IDs are converted consistently when zero-based IDs are used;
- factorization does not make label IDs depend accidentally on filesystem
  iteration order; and
- excluded labels are intentionally documented rather than silently lost.

For datasets containing row-level labels, inspect transition locations and
confirm that every resulting session contains only one activity. For
file-labelled datasets, compare parsed metadata directly with the filename.

## 4. Validate parser output contract

Call `cfg.parse(raw_dir, cfg.activity_id_col)` without downloading data and
verify all of the following:

- `activity_df` contains unique, non-null `activity_id` and `activity_name`;
- `session_df` contains unique, non-null `session_id`, `subject_id`, and
  `activity_id`;
- identifiers are integer typed, zero-based, contiguous where required by the
  library, and consistent across both metadata tables;
- every key in `sessions` has exactly one row in `session_df`, and vice versa;
- every configured subject and activity is represented, unless an externally
  verified source omission is reported;
- every session contains `timestamp` plus exactly the configured sensor
  channels in the configured order;
- timestamps are datetime typed, strictly increasing, and contain no gap larger
  than the configured session-gap threshold;
- when modalities are aligned, the output retains the intended reference
  samples, does not rely on exact timestamp equality unless documented, and
  contains no out-of-tolerance or cross-recording matches;
- sensor values are floating point and contain no NaN or infinity after the
  parser's documented missing-data policy; and
- no non-time-series modality leaked into the channel matrix.

Do not silently interpolate long recording gaps. Split the session first.
Interpolation of short sensor dropouts must be documented and tested at the
beginning, middle, and end of a recording. Verify that interpolation does not
extrapolate beyond the available source range and that duplicate/conflicting
timestamps have an explicit policy.

## 5. Measure session and sampling behavior

Generate and report, overall and by activity:

- number of sessions;
- sample-count and duration quantiles;
- minimum, median, and maximum timestamp step;
- timestamp-step quantiles in raw units and after conversion;
- timestamp-derived sampling-rate distribution;
- count of non-increasing timestamps and large gaps;
- for each multimodal join, matched/unmatched counts and alignment-offset
  quantiles;
- count of sessions shorter than one configured window; and
- counts of sessions producing 0, 1, 2, 3, and more windows.

Investigate clusters of very short sessions. Determine whether they correspond
to real brief activities, label noise, timestamp-unit mistakes, excessive gap
splitting, or incorrect activity transitions. Short sessions alone do not prove
that a parser is broken.

## 6. Validate cache and windowing output

Run preprocessing from the existing raw cache and check:

- schema-v2 Parquet/NumPy artifacts are readable;
- cached activity/session metadata equals parser output;
- cached session IDs, subjects, activities, channels, dtypes, and timestamps are
  unchanged;
- `window_id` is unique and maps to the correct session;
- `start_index` and `end_index` describe the exact source interval;
- every window has the configured sample count and channel order;
- windows never cross a session boundary;
- the final exactly fitting window is retained;
- selected activities/channels are applied without corrupting IDs; and
- sequential and process backends produce equivalent metadata and arrays.

For a regular session with `N` samples, window size `W`, and stride `S`, compare
the observed count with `1 + floor((N - W) / S)` when `N >= W`; otherwise expect
zero windows. Account explicitly for resampling before applying this formula.

## 7. Validate splitting, normalization, and loading

For at least one LOSO fold and preferably all folds, verify:

- the held-out subject appears only in test;
- every other eligible subject is represented in training;
- train, validation, and test index sets are disjoint;
- strict train/validation mode has no overlapping raw intervals across those
  partitions;
- unsplittable short sessions are assigned wholly to one partition and the
  aggregate assignment log is plausible;
- global normalization parameters are fitted only from training indices and
  separately per channel;
- validation and test use the training-derived normalization parameters; and
- cached split directories differ when their normalized training-index sets
  differ and are reused when the same split is requested again.

Test both `in_memory=True` and memory-mapped loading for several windows. Compare
loaded samples with their cached rows and verify labels against `session_df`.

# Tests to add or run

## Fast configuration and synthetic-contract tests

Run:

```bash
uv run --with pytest python -m pytest tests/test_dataset_consistency.py -q
```

These tests should cover configuration semantics, parser signatures, registry
membership, common-format validation, activity selection, and synthetic
windowing contracts. Add a dataset-specific regression whenever a bug could
recur without access to the full raw dataset.

## Cached artifact tests

Run the cached common-format and windowing checks in:

```bash
uv run --with pytest python -m pytest tests/test_dataset_requirements.py -q
```

Before running, confirm that the test's dataset root matches `cfg.datasets_dir`.
Do not copy or redownload datasets solely to satisfy a hard-coded test path.
When using a non-default cache root, set `WHAR_DATASETS_DIR` to that root; the
tests use it for cached artifacts and raw parser checks.

## Full parser end-to-end tests

When the raw cache is already present, enable parser checks with:

```bash
WHAR_RUN_PARSE_E2E=1 uv run --with pytest python -m pytest \
  tests/test_dataset_requirements.py -q
```

The test must skip unavailable raw datasets rather than downloading them.

## Dataset-specific regression tests

Add focused tests for the parser's actual risk areas, such as:

- filename-to-subject/activity/repetition parsing;
- one-based to zero-based ID conversion;
- alternate or malformed filename handling;
- row-level activity transitions;
- timestamp unit conversion and timestamp resets;
- large-gap session splitting;
- bounded nearest-timestamp alignment, tolerance, and cross-label/source
  protection;
- missing-value and sensor-dropout handling;
- simultaneous multi-sensor alignment and channel order;
- documented missing recordings;
- ignored/unknown activity behavior;
- multiple label schemes; and
- a known raw file's expected subject, activity, sample count, duration, and
  channel values.

Prefer tiny synthetic fixtures for edge cases and a small representative slice
of cached raw data for integration checks. Do not commit copyrighted or large
raw recordings as fixtures.

# Required audit report

Finish every dataset audit with a concise report containing:

1. **Verdict:** pass, pass with documented source limitations, or fail.
2. **Source evidence:** metadata/files used to verify subjects and activities.
3. **Labels:** available schemes, selected scheme, mappings, and exclusions.
4. **Subjects/recordings:** expected versus observed counts and missing files.
5. **Sensors:** modalities, simultaneity, channels, units, and sampling rates.
6. **Sessions:** boundary policy, counts, duration distribution, and gap results.
7. **Windows:** expected versus observed counts and short-session distribution.
8. **Splits/loading:** LOSO isolation, strict overlap result, and cache checks.
9. **Tests:** commands run, pass/skip/fail counts, and new regressions added.
10. **Open issues:** uncertainties that require documentation or user input.

If activity labels or subject identifiers are missing or ambiguous, stop and
state exactly what evidence is absent. Do not guess.

# Implementation requirements

- Use Python type hints in all implementations.
- Preserve raw data and unrelated user changes.
- Prefer deterministic ordering of files and labels.
- Raise descriptive errors for violated parser contracts.
- Keep sequential execution available even when adding multiprocessing.
- Verify process and sequential implementations produce equivalent results.

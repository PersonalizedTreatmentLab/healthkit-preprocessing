# HealthKit Preprocessing Pipeline Documentation

**Version**: 0.1.2  
**Last Updated**: February 3, 2026


**NOTE: This documentation was generated using AI isn't entirely accurate. Additionally, small changes have been made since then.**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Supported Data Streams](#supported-data-streams)
   - [Activity Streams](#activity-streams)
   - [Vital Sign Streams](#vital-sign-streams)
   - [Audio Exposure Streams](#audio-exposure-streams)
   - [Categorical Streams](#categorical-streams)
   - [Stream Processing Summary](#stream-processing-summary)
4. [Module Reference](#module-reference)
   - [healthkit_loader.py](#healthkit_loaderpy)
   - [healthkit_cleaner.py](#healthkit_cleanerpy)
   - [healthkit_cleaner_specs.py](#healthkit_cleaner_specspy)
   - [healthkit_aggregator.py](#healthkit_aggregatorpy)
   - [healthkit_aggregator_specs.py](#healthkit_aggregator_specspy)
   - [healthkit_aggregator_time.py](#healthkit_aggregator_timepy)
5. [Workflow Examples](#workflow-examples)
6. [Configuration Guide](#configuration-guide)
7. [Advanced Topics](#advanced-topics)

---

## Overview

The HealthKit Preprocessing Pipeline is a comprehensive framework for loading, cleaning, and aggregating Apple HealthKit data exports. It provides:

- **Flexible Data Loading**: Support for both CSV and XML (Apple Health export) formats
- **Robust Cleaning**: Type-specific data validation, deduplication, and outlier detection
- **Timezone Handling**: Advanced timezone support with DST awareness and travel schedule management
- **Time-Based Aggregation**: Configurable windowing (fixed bins or assessment-anchored) with hierarchical rollups
- **Wear Detection**: Heart rate-based device wear computation
- **Quality Control**: Multi-level QC (bin, day, participant) with configurable thresholds

---

## Architecture

### Pipeline Stages

```
Raw HealthKit Data (CSV/XML)
         ↓
    [LOADER]  ← healthkit_loader.py
         ↓
   Normalized DataFrame
         ↓
   [CLEANER]  ← healthkit_cleaner.py + healthkit_cleaner_specs.py
         ↓
  Clean, Validated Data
         ↓
 [AGGREGATOR]  ← healthkit_aggregator.py + healthkit_aggregator_specs.py + healthkit_aggregator_time.py
         ↓
  Time-Windowed Features
```
`
### Data Flow

1. **Loading**: Parse raw files, normalize schema, handle timezone metadata
2. **Cleaning**: Type-specific validation, overlap resolution, plausibility filtering
3. **Aggregation**: Window construction, observation assignment, per-window statistics
4. **QC**: Wear-based quality gates with drop/mask actions

---

## Supported Data Streams

The pipeline processes four categories of HealthKit data streams, each with specific time behaviors and aggregation strategies.

### Activity Streams

Activity streams represent cumulative quantities or totals over time intervals (e.g., steps taken, energy burned).

#### 1. StepCount
**Description**: Number of steps taken  
**Time Behavior**: `interval_total`  
**Units**: count  
**Device Sources**: Apple Watch, iPhone  

**Cleaning Options**:
- ✅ Watch device filter (default: enabled)
- ✅ Numeric coercion
- ✅ Min threshold: 0
- ✅ Overlap resolution (redistributes totals)
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `sum`: Total steps in window
- `duration`: Total duration of step observations (seconds)
- `rate_per_min`: Steps per minute (sum / duration * 60)

**Zero-Fill**: Eligible when watch worn (gate: `watch_on`)

---

#### 2. ActiveEnergyBurned
**Description**: Energy burned during physical activity  
**Time Behavior**: `interval_total`  
**Units**: kcal (converted from Joules if needed)  
**Device Sources**: Apple Watch, iPhone  

**Cleaning Options**:
- ✅ Watch device filter (default: enabled)
- ✅ Numeric coercion
- ✅ Unit conversion (Joules → kcal, divide by 4184)
- ✅ Rate capping: max 30 kcal/min (configurable)
  - **Policy**: `"cap"` (rescale) or `"drop"` (remove)
- ✅ Min threshold: 0
- ✅ Overlap resolution (redistributes totals)
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `sum`: Total kcal burned in window
- `duration`: Total duration (seconds)
- `rate_per_min`: kcal per minute

**Zero-Fill**: Eligible when watch worn

**Special Handling**: 
- Implausibly high kcal/min rates are capped or dropped
- Protects against sensor errors or data corruption

---

#### 3. BasalEnergyBurned
**Description**: Resting metabolic rate energy expenditure  
**Time Behavior**: `interval_total`  
**Units**: kcal  
**Device Sources**: Apple Watch, iPhone  

**Cleaning Options**:
- ✅ Watch device filter (default: enabled)
- ✅ Numeric coercion
- ✅ Unit conversion (Joules → kcal)
- ✅ Min threshold: 0
- ✅ Overlap resolution
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `sum`: Total basal kcal in window
- `duration`: Duration (seconds)
- `rate_per_min`: Basal kcal per minute

**Zero-Fill**: Eligible when watch worn

---

#### 4. AppleExerciseTime
**Description**: Minutes of exercise intensity activity  
**Time Behavior**: `interval_total`  
**Units**: minutes  
**Device Sources**: Apple Watch  

**Cleaning Options**:
- ✅ Watch device filter (default: enabled)
- ✅ Numeric coercion
- ✅ Min threshold: 0
- ✅ Overlap resolution
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `sum`: Total exercise minutes in window

**Zero-Fill**: Eligible when watch worn

---

#### 5. AppleStandTime
**Description**: Minutes spent standing  
**Time Behavior**: `interval_total`  
**Units**: minutes  
**Device Sources**: Apple Watch  

**Cleaning Options**:
- ✅ Watch device filter (default: enabled)
- ✅ Numeric coercion
- ✅ Min threshold: 0
- ✅ Overlap resolution
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `sum`: Total stand minutes in window

**Zero-Fill**: Eligible when watch worn

---

#### 6. DistanceWalkingRunning
**Description**: Distance covered by walking or running  
**Time Behavior**: `interval_total`  
**Units**: meters (or miles/km depending on export)  
**Device Sources**: Apple Watch, iPhone  

**Cleaning Options**:
- ✅ Watch device filter (default: enabled)
- ✅ Numeric coercion
- ✅ Min threshold: 0
- ✅ Overlap resolution
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `sum`: Total distance in window
- `duration`: Duration (seconds)
- `rate_per_min`: Distance per minute

**Zero-Fill**: Eligible when watch worn

---

#### 7. DistanceCycling
**Description**: Distance covered by cycling  
**Time Behavior**: `interval_total`  
**Units**: meters (or miles/km)  
**Device Sources**: Apple Watch, iPhone  

**Cleaning Options**:
- ✅ Watch device filter (default: enabled)
- ✅ Numeric coercion
- ✅ Min threshold: 0
- ✅ Overlap resolution
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `sum`: Total distance in window
- `duration`: Duration (seconds)
- `rate_per_min`: Distance per minute

**Zero-Fill**: Eligible when watch worn

---

#### 8. FlightsClimbed
**Description**: Number of flights of stairs climbed  
**Time Behavior**: `interval_total`  
**Units**: count  
**Device Sources**: Apple Watch, iPhone  

**Cleaning Options**:
- ✅ Watch device filter (default: enabled)
- ✅ Numeric coercion
- ✅ Min threshold: 0
- ✅ Overlap resolution
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `sum`: Total flights in window
- `duration`: Duration (seconds)
- `rate_per_min`: Flights per minute

**Zero-Fill**: Eligible when watch worn

---

### Vital Sign Streams

Vital sign streams represent point-in-time physiological measurements.

#### 9. HeartRate
**Description**: Instantaneous heart rate measurement  
**Time Behavior**: `point`  
**Units**: beats per minute (bpm)  
**Device Sources**: Apple Watch only  

**Cleaning Options**:
- ✅ Numeric coercion
- ✅ Deduplication (exact matches on user/type/time/value)
- ✅ Plausibility range: 25-230 bpm
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `mean`: Average heart rate in window
- `median`: Median heart rate
- `std`: Standard deviation
- `max`: Maximum heart rate
- `count`: Number of measurements

**Zero-Fill**: Eligible when watch worn

**Special Role**: 
- Used for wear detection computation
- Each point "covers" 5 minutes (configurable via `hr_coverage_window_seconds`)
- Drives QC thresholds for other watch-exclusive metrics

---

#### 10. HeartRateVariabilitySDNN
**Description**: Heart rate variability (SDNN metric)  
**Time Behavior**: `point`  
**Units**: seconds  
**Device Sources**: Apple Watch only  

**Cleaning Options**:
- ✅ Numeric coercion
- ✅ Deduplication
- ✅ Plausibility range: 0.005-0.3 seconds (5-300 ms)
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `mean`: Average HRV
- `median`: Median HRV
- `count`: Number of measurements

**Zero-Fill**: Eligible when watch worn

---

#### 11. RespiratoryRate
**Description**: Breathing rate  
**Time Behavior**: `point`  
**Units**: breaths per minute  
**Device Sources**: Apple Watch only  

**Cleaning Options**:
- ✅ Numeric coercion
- ✅ Deduplication
- ✅ Plausibility range: 4-80 breaths/min
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `mean`: Average respiratory rate
- `median`: Median respiratory rate
- `count`: Number of measurements

**Zero-Fill**: Eligible when watch worn

---

#### 12. OxygenSaturation
**Description**: Blood oxygen saturation (SpO2)  
**Time Behavior**: `point`  
**Units**: fraction (0.0-1.0) or percentage  
**Device Sources**: Apple Watch only  

**Cleaning Options**:
- ✅ Numeric coercion
- ✅ Deduplication
- ✅ Plausibility range: 0.50-1.0 (50%-100%)
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `mean`: Average SpO2
- `median`: Median SpO2
- `count`: Number of measurements

**Zero-Fill**: Eligible when watch worn

---

### HRV Derived Features (Optional)

**NEW in v0.1.2**: The pipeline can now compute advanced Heart Rate Variability (HRV) features from HeartRate observations within each time window. This is an **opt-in** feature controlled by configuration flags.

#### Overview
**Source Data**: HeartRate observations  
**Computation Method**: RR interval analysis (time-domain and frequency-domain)  
**Feature Prefix**: `HRV__`  
**Total Features**: 19 (10 time-domain + 9 frequency-domain)  

**Configuration**:
```python
config = HKAggConfig(
    bin_size="5min",
    compute_hrv=True,              # Enable HRV computation (default: False)
    hrv_time_domain=True,          # Include time-domain features (default: True)
    hrv_freq_domain=True,          # Include frequency-domain features (default: True)
    hrv_min_samples=10,            # Min HR samples for time-domain (default: 10)
    hrv_freq_min_samples=30,       # Min HR samples for freq-domain (default: 30)
    hrv_interpolation_fs=4.0,      # Interpolation rate Hz (default: 4.0)
)
```

#### Time-Domain Features (10 metrics)

These features are computed from NN intervals (time between consecutive heartbeats):

1. **HRV__nn_count** (count)  
   Number of valid NN intervals in the window

2. **HRV__mean_nn_ms** (milliseconds)  
   Mean NN interval duration

3. **HRV__median_nn_ms** (milliseconds)  
   Median NN interval duration

4. **HRV__mean_hr_bpm** (beats per minute)  
   Mean heart rate derived from NN intervals

5. **HRV__hr_range_bpm** (beats per minute)  
   Heart rate range (max - min)

6. **HRV__sdnn_ms** (milliseconds)  
   Standard deviation of NN intervals (classic HRV metric)

7. **HRV__rmssd_ms** (milliseconds)  
   Root mean square of successive differences (parasympathetic activity)

8. **HRV__pnn50_pct** (percentage)  
   Percentage of NN intervals differing by >50ms from adjacent interval

9. **HRV__pnn20_pct** (percentage)  
   Percentage of NN intervals differing by >20ms from adjacent interval

10. **HRV__hrv_tri_index** (dimensionless)  
    HRV triangular index (total NN count / mode bin count)

**Requirements**:
- Minimum `hrv_min_samples` HR observations (default: 10)
- No scipy dependency required

#### Frequency-Domain Features (9 metrics)

These features are computed from power spectral density (PSD) using Welch's method after cubic spline interpolation:

1. **HRV__vlf_power_ms2** (ms²)  
   Very low frequency power (0.003-0.04 Hz)

2. **HRV__lf_power_ms2** (ms²)  
   Low frequency power (0.04-0.15 Hz) - sympathetic + parasympathetic

3. **HRV__lf_peak_hz** (Hz)  
   Frequency of peak power in LF band

4. **HRV__lf_power_nu** (normalized units)  
   LF power normalized: LF / (LF + HF) × 100

5. **HRV__hf_power_ms2** (ms²)  
   High frequency power (0.15-0.4 Hz) - respiratory/parasympathetic

6. **HRV__hf_peak_hz** (Hz)  
   Frequency of peak power in HF band

7. **HRV__hf_power_nu** (normalized units)  
   HF power normalized: HF / (LF + HF) × 100

8. **HRV__lf_hf_ratio** (dimensionless)  
   Ratio of LF to HF power (sympatho-vagal balance)

9. **HRV__total_power_ms2** (ms²)  
   Total spectral power (VLF + LF + HF)

**Requirements**:
- Minimum `hrv_freq_min_samples` HR observations (default: 30)
- Scipy library (gracefully skipped if not available)
- NN intervals are interpolated to uniform sampling rate (`hrv_interpolation_fs` Hz, default: 4.0 Hz)

#### Computation Details

**RR Interval Cleaning**:
- Calculates inter-beat intervals from HeartRate observations
- Removes physiologically implausible intervals (<300ms or >2000ms)
- Applies adaptive outlier detection (>3 SD from median)
- Requires minimum sample count after cleaning

**Interpolation** (for frequency-domain):
- Cubic spline interpolation to uniform sampling
- Default 4 Hz sampling rate (recommended for HRV analysis)
- Detrends signal (removes linear trend)

**PSD Estimation**:
- Welch's method with Hann window
- Window length: 256 samples
- 50% overlap between windows
- Band power integration using trapezoidal rule

#### Clinical Interpretation

- **SDNN**: Overall HRV; higher values indicate better autonomic regulation
- **RMSSD**: Short-term HRV; primarily reflects parasympathetic activity
- **pNN50/pNN20**: Parasympathetic indices; higher = better vagal tone
- **LF Power**: Mixed sympathetic + parasympathetic modulation
- **HF Power**: Respiratory sinus arrhythmia; reflects vagal activity
- **LF/HF Ratio**: Sympatho-vagal balance; lower = more parasympathetic dominance

**Typical Values** (5-minute windows, healthy adults at rest):
- SDNN: 20-80 ms
- RMSSD: 15-60 ms
- pNN50: 5-50%
- LF Power: 500-2000 ms²
- HF Power: 500-2000 ms²
- LF/HF Ratio: 0.5-3.0

#### Zero-Fill Behavior

HRV features follow the same zero-fill policy as HeartRate:
- **Eligible**: When `watch_on` gate is true
- **Fill Value**: `NaN` (not zero) - indicates no data
- **Rationale**: Cannot infer HRV from absence of HR data

#### Performance Considerations

- **Time-domain only**: ~0.5-1.0 second per 1000 windows
- **Time + Frequency**: ~1.5-2.5 seconds per 1000 windows
- **Recommendation**: Use `hrv_freq_domain=False` for faster processing if frequency metrics not needed

#### Example Usage

```python
from healthkit_preprocessing import HealthKitAggregator, HKAggConfig

# Enable all HRV features
config = HKAggConfig(
    bin_size="5min",
    compute_hrv=True,
)

aggregator = HealthKitAggregator(config)
wide, report, aux = aggregator.run_aggregation(df)

# Check HRV columns
hrv_cols = [c for c in wide.columns if c.startswith('HRV__')]
print(f"HRV features: {hrv_cols}")
print(f"HRV report: {report['hrv']}")

# Example output:
# HRV features: ['HRV__nn_count', 'HRV__sdnn_ms', 'HRV__rmssd_ms', ...]
# HRV report: {
#   'enabled': True,
#   'time_domain': True,
#   'freq_domain': True,
#   'n_features': 19,
#   'n_windows_with_hrv': 1200,
#   'coverage_pct': 83.33,
#   'elapsed_seconds': 2.451
# }
```

#### Module Reference

See `healthkit_hrv.py` for implementation details:
- `HRVComputer` class: Main computation engine
- `compute_hrv_features_for_windows()`: Batch processing function
- Helper functions for RR cleaning, interpolation, and PSD estimation

---

### Audio Exposure Streams

Audio streams represent sound level measurements over time intervals.

#### 13. EnvironmentalAudioExposure
**Description**: Ambient environmental sound levels  
**Time Behavior**: `interval_level`  
**Units**: dB (A-weighted decibels)  
**Device Sources**: Apple Watch, iPhone  

**Cleaning Options**:
- ✅ Numeric coercion
- ✅ Plausibility range: 0-120 dB
- ✅ Overlap resolution (preserves level, not total)
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `leq`: Equivalent continuous sound level (energy-weighted)
- `energy_seconds`: Total acoustic energy × duration
- `time_weighted_mean`: Duration-weighted mean dB
- `max`: Peak sound level
- `duration`: Total duration (seconds)
- `median`: Median sound level
- `median_duration`: Median observation duration
- `count`: Number of observations

**Zero-Fill**: Not eligible (no zero-fill gate)

**Special Handling**: 
- Level aggregation (not total): overlaps preserve the winning interval's dB level
- LEQ computation: `10 * log10(sum(10^(dB/10) * duration) / sum(duration))`

---

#### 14. HeadphoneAudioExposure
**Description**: Sound levels from headphone/earphone audio  
**Time Behavior**: `interval_level`  
**Units**: dB (A-weighted)  
**Device Sources**: iPhone, iPad (not Apple Watch)  

**Cleaning Options**:
- ✅ Numeric coercion
- ✅ Min threshold: 0 dB
- ✅ Overlap resolution (level preservation)
- ⚠️ Quantile filter (default: disabled)

**Aggregation Statistics**:
- `leq`: Equivalent continuous level
- `energy_seconds`: Acoustic energy × duration
- `time_weighted_mean`: Duration-weighted mean
- `max`: Peak level
- `duration`: Total duration
- `median`: Median level
- `median_duration`: Median observation duration
- `count`: Number of observations

**Zero-Fill**: Not eligible

**Device Class**: `non_watch_exclusive` (excluded from watch-based QC)

---

### Categorical Streams

Categorical streams represent state or event labels over time intervals.

#### 15. SleepAnalysis
**Description**: Sleep stage classification  
**Time Behavior**: `categorical_interval`  
**Units**: Categorical labels  
**Device Sources**: Apple Watch, iPhone (inferred from motion/HR)  

**Labels**:
- `InBed`: User in bed (may or may not be asleep)
- `Asleep`: Confirmed sleep state
- `Awake`: Awake during sleep period
- `AwakeUnspecified`: Awake state (less certain)
- Additional labels may exist in some exports

**Cleaning Options**:
- ✅ Label normalization (strips "HKCategoryValueSleepAnalysis" prefix)
- ✅ Categorical overlap resolution
  - Priority: device_source_id > shorter duration > recency
  - Adjacent same-label segments merged

**Aggregation Statistics** (per label):
- `sum_label_<label>`: Total duration of label in window (seconds)
- `mean_label_<label>`: Mean observation duration for label
- `count_label_<label>`: Number of observations with label
- `start_label_<label>`: Earliest start time of label in window
- `end_label_<label>`: Latest end time of label in window

**Example Features**:
- `SleepAnalysis__sum_label_in_bed`
- `SleepAnalysis__sum_label_asleep`
- `SleepAnalysis__count_label_awake`
- `SleepAnalysis__start_label_asleep` (sleep onset)
- `SleepAnalysis__end_label_asleep` (sleep offset)

**Zero-Fill**: Eligible for sum/mean/count stats when watch worn

**Special Sleep Features**: 
- Daily sleep aggregation with 3pm-3pm windows (configurable)
- Sleep efficiency: `Asleep / InBed`
- Bedrest duration, sleep onset/offset times

---

#### 16. EnvironmentalAudioExposureEvent
**Description**: Loud environmental sound event detection  
**Time Behavior**: `categorical_interval`  
**Units**: Categorical (event type)  
**Device Sources**: Apple Watch, iPhone  

**Labels**:
- `MomentaryLimit`: Sound level exceeded safe threshold

**Cleaning Options**:
- ✅ Categorical overlap resolution

**Aggregation Statistics** (per label):
- `sum_label_<label>`: Total duration
- `mean_label_<label>`: Mean event duration
- `count_label_<label>`: Event count
- `start_label_<label>`: First event time
- `end_label_<label>`: Last event time

**Zero-Fill**: Eligible for sum/mean/count when watch worn

---

### Stream Processing Summary

| Stream | Time Behavior | Handler | Overlap Resolution | Zero-Fill | Device Filter |
|--------|--------------|---------|-------------------|-----------|---------------|
| StepCount | interval_total | Activity | ✅ Proportional | ✅ watch_on | ✅ Watch only |
| ActiveEnergyBurned | interval_total | Activity | ✅ Proportional | ✅ watch_on | ✅ Watch only |
| BasalEnergyBurned | interval_total | Activity | ✅ Proportional | ✅ watch_on | ✅ Watch only |
| AppleExerciseTime | interval_total | Activity | ✅ Proportional | ✅ watch_on | ✅ Watch only |
| AppleStandTime | interval_total | Activity | ✅ Proportional | ✅ watch_on | ✅ Watch only |
| DistanceWalkingRunning | interval_total | Activity | ✅ Proportional | ✅ watch_on | ✅ Watch only |
| DistanceCycling | interval_total | Activity | ✅ Proportional | ✅ watch_on | ✅ Watch only |
| FlightsClimbed | interval_total | Activity | ✅ Proportional | ✅ watch_on | ✅ Watch only |
| HeartRate | point | Vitals | N/A | ✅ watch_on | Watch only |
| **HRV (19 features)** | **derived** | **HRV Computer** | **N/A** | **✅ watch_on** | **Derived from HR** |
| HeartRateVariabilitySDNN | point | Vitals | N/A | ✅ watch_on | Watch only |
| RespiratoryRate | point | Vitals | N/A | ✅ watch_on | Watch only |
| OxygenSaturation | point | Vitals | N/A | ✅ watch_on | Watch only |
| EnvironmentalAudioExposure | interval_level | Audio | ✅ Level | ❌ None | Either |
| HeadphoneAudioExposure | interval_level | Audio | ✅ Level | ❌ None | Non-watch |
| SleepAnalysis | categorical_interval | Categorical | ✅ Label | ✅ watch_on | Either |
| EnvironmentalAudioExposureEvent | categorical_interval | Categorical | ✅ Label | ✅ watch_on | Either |

**Legend**:
- **interval_total**: Cumulative quantity (redistribute on overlap)
- **interval_level**: Measurement level (preserve level on overlap)
- **point**: Instantaneous measurement
- **categorical_interval**: State/event label over time
- **derived**: Computed from other streams (HRV from HeartRate)
- **Proportional**: Total redistributed by duration fraction
- **Level**: Winning interval's level preserved
- **Label**: Winning label based on priority rules

**Note**: HRV features are **optional** and must be enabled via `compute_hrv=True` in configuration (default: False).

---

## Module Reference

### healthkit_loader.py

**Purpose**: Load and normalize HealthKit data from CSV and XML sources.

#### Class: `HealthKitLoader`

Main class for loading HealthKit data with timezone normalization.

##### Configuration

```python
TZ_LOCALIZE_AMBIGUOUS = "infer"  # How to handle ambiguous times (DST transitions)
TZ_LOCALIZE_NONEXISTENT = "shift_forward"  # How to handle nonexistent times
```

##### Methods

###### `__init__(logger: Optional[logging.Logger] = None)`

Initialize the loader with optional logger.

###### `load_file(path: str | Path, *, user_id: str | int | None = None) -> pd.DataFrame`

Load a single HealthKit file (CSV or XML).

**Parameters**:
- `path`: File path to HealthKit data
- `user_id`: User identifier (required for XML, optional for CSV which should have its own user_id column)

**Returns**: DataFrame with normalized schema

**Schema**:
- `user_id`: Unique identifier for the user
- `type`: HealthKit data type (e.g., "HeartRate", "StepCount", "SleepAnalysis")
- `utc_start`, `utc_end`: UTC timestamps (tz-aware)
- `local_start`, `local_end`: Local timestamps (tz-naive)
- `timezone`: IANA timezone identifier
- `value`: Measurement value (numeric or categorical)
- `device.name`: Device model name
- `device.name_id`: Encoded device type (0=other, 1=iPhone, 2=Apple Watch)

**CSV Handling**:
- Expects existing columns with proper schema
- Fills/validates timezone column
- Localizes naive local times per timezone with configurable policies for ambiguous/nonexistent times
- Computes UTC timestamps if missing

**XML Handling**:
- Parses Apple Health export format
- Strips "HKQuantityTypeIdentifier", "HKCategoryTypeIdentifier" prefixes from type names
- Normalizes SleepAnalysis values (strips "HKCategoryValueSleepAnalysis" prefix)
- Infers timezone from `HKTimeZone` metadata with nearest-neighbor daily interpolation
- Falls back to "America/Los_Angeles" for missing timezones

**Timezone Handling Logic**:
- Groups by timezone for per-zone localization
- Uses `ambiguous="infer"` to detect DST transitions
- Uses `nonexistent="shift_forward"` for spring-forward gaps
- Falls back to `ambiguous=False` (standard time) if inference fails
- Raises detailed errors with row indices and user_id on failures

###### `load_participant(data_folder: str | Path, user_id: int | str, sensor_folders: Iterable[str] | Iterable[Path] | None = None) -> pd.DataFrame`

Load all HealthKit CSVs for a single participant.

**File Pattern**: `<data_folder>/<sensor_folder>/<user_id>-<sensor_folder>.csv`

**Parameters**:
- `data_folder`: Root directory containing sensor subfolders
- `user_id`: Participant identifier
- `sensor_folders`: List of sensor folders to search (if None, searches all subfolders)

**Returns**: Concatenated DataFrame of all sensor data for the participant

**Example**:
```python
loader = HealthKitLoader()
data = loader.load_participant(
    data_folder="/data/healthkit",
    user_id=12345,
    sensor_folders=["HeartRate", "StepCount", "SleepAnalysis"]
)
```

###### `load_bulk(root: str | Path, *, patterns: Sequence[str] = ("**/*.csv", "**/*.xml"), user_id_resolver: Optional[Callable[[Path], str | int]] = None, default_xml_user: str | int = "anon", strict: bool = False) -> pd.DataFrame`

Recursively load many HealthKit files under a root directory.

**Parameters**:
- `root`: Root directory to search
- `patterns`: Glob patterns to match files (default: CSV and XML)
- `user_id_resolver`: Function to extract user_id from file path (XML only)
- `default_xml_user`: Fallback user_id for XML when resolver fails
- `strict`: If True, raise on first file error; if False, log and continue

**Returns**: Concatenated DataFrame of all loaded files

**User ID Resolution** (XML only):
1. Use `user_id_resolver` if provided
2. Try to infer from digits in filename/path
3. Fall back to `default_xml_user`

**Example**:
```python
loader = HealthKitLoader()
data = loader.load_bulk(
    root="/data/healthkit_exports",
    patterns=["**/*.xml"],
    user_id_resolver=lambda p: int(p.parent.name),
    strict=False
)
```

##### Private Methods

###### `_load_healthkit_xml(path: Path, user_id: str = "anon") -> pd.DataFrame`

Parse XML export into DataFrame.

**Process**:
1. Parse XML tree
2. Flatten `<Record>` elements into dict rows
3. Merge `<MetadataEntry>` key-value pairs
4. Coerce timestamp columns to UTC
5. Normalize type strings (strip HK prefixes)
6. Extract device name from device string
7. Set body.quantity.count=1 for all records

###### `_add_local_time(hk_data: pd.DataFrame, default_tz: str = "America/Los_Angeles") -> pd.DataFrame`

Add timezone and local time columns to XML data.

**Process**:
1. Extract `HKTimeZone` metadata per day
2. Resample to daily frequency
3. Interpolate missing timezones (nearest + forward/backward fill)
4. Convert UTC timestamps to local per timezone
5. Create both tz-aware (`local_start_tz`, `local_end_tz`) and naive (`local_start`, `local_end`) columns

---

### healthkit_cleaner.py

**Purpose**: Clean and validate HealthKit data with type-specific rules.

#### Class: `HealthKitCleaner`

Configurable cleaning pipeline with per-type handlers.

##### Constants

```python
ACTIVITY_TYPES = {"StepCount", "AppleExerciseTime", "ActiveEnergyBurned", 
                  "BasalEnergyBurned", "AppleStandTime", "DistanceCycling", 
                  "DistanceWalkingRunning", "FlightsClimbed"}

VITAL_TYPES = {"HeartRate", "HeartRateVariabilitySDNN", 
               "RespiratoryRate", "OxygenSaturation"}

AUDIO_TYPES = {"EnvironmentalAudioExposure", "HeadphoneAudioExposure"}

CAT_TYPES = {"SleepAnalysis", "EnvironmentalAudioExposureEvent"}
```

##### Methods

###### `__init__(rules: Optional[HKCleanConfig] = None, logger: Optional[logging.Logger] = None)`

Initialize cleaner with configuration and logging.

**Parameters**:
- `rules`: Cleaning configuration (defaults to `HKCleanConfig()`)
- `logger`: Logger instance

###### `clean(df: pd.DataFrame, return_report: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]`

Main cleaning pipeline.

**Process**:
1. Convert time columns to datetime
2. Drop rows with missing/negative time intervals (UTC-based)
3. Drop rows with missing `type` or `user_id`
4. Group by `type` and route to appropriate handler
5. Concatenate cleaned groups
6. Sort by `(user_id, type, utc_start, utc_end)`
7. Generate cleaning report

**Parameters**:
- `df`: Raw HealthKit DataFrame
- `return_report`: If True, return `(cleaned_df, report_dict)`

**Returns**: Cleaned DataFrame or tuple with report

**Report Structure**:
```python
{
    "__global__": {
        "rows_in": int,
        "dropped_missing_or_negative_time": int,
        "dropped_missing_type": int,
        "dropped_missing_user_id": int,
        "rows_out": int,
        "types_seen": List[str],
        "types_without_handler": List[str]  # if any
    },
    "<type_name>": {
        "rows_in": int,
        "rows_out": int,
        "handler": str,
        "non_numeric_dropped": int,  # if applicable
        "out_of_range_dropped": int,  # if applicable
        "deduplicated": int,  # if applicable
        "kcal_per_min_capped": int,  # ActiveEnergyBurned only
        "converted_to_kcal": bool,  # Energy types only
        # ... handler-specific metrics
    }
}
```

###### `register_handler(hk_type: str, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> None`

Register or override a handler for a specific type.

**Example**:
```python
def custom_sleep_handler(df: pd.DataFrame) -> pd.DataFrame:
    # Custom logic
    return df

cleaner.register_handler("SleepAnalysis", custom_sleep_handler)
```

##### Type Handlers

###### `_handle_activity(df: pd.DataFrame) -> pd.DataFrame`

Handler for activity types (steps, distance, energy, etc.).

**Process**:
1. **Device Filter**: Keep only Apple Watch observations with `body.quantity.count == 1`
2. **Numeric Coercion**: Convert value to numeric, drop unparsable rows
3. **Energy Normalization**: Convert Joules to kcal (divide by 4184) if unit != "kcal"
4. **Rate Capping** (ActiveEnergyBurned only):
   - Compute kcal/min rate
   - If > threshold (default 30), either drop or cap value
   - Policy: `"cap"` (rescale value) or `"drop"` (remove row)
5. **Theory Thresholds**: Apply min/max value ranges from config
6. **Overlap Resolution**: Combine overlapping intervals per user using `combine_overlaps`
7. **Quantile Filter** (optional): Drop extreme values outside quantile range

**Overlap Resolution Strategy**:
- Preserves total quantity (redistributes proportionally)
- Priority: has device_source_id > higher device_source_id > shorter duration > later row

###### `_handle_vitals(df: pd.DataFrame) -> pd.DataFrame`

Handler for vital sign types (HR, HRV, SpO2, RR).

**Process**:
1. **Numeric Coercion**: Convert value to numeric
2. **Deduplication**: Remove exact duplicates `(user_id, type, utc_start, utc_end, value)`
3. **Theory Thresholds**: Apply plausibility ranges
4. **Quantile Filter** (optional): Drop extreme values

**Note**: No overlap resolution (point measurements don't overlap)

###### `_handle_audio(df: pd.DataFrame) -> pd.DataFrame`

Handler for audio exposure types.

**Process**:
1. **Numeric Coercion**: Convert value to numeric
2. **Theory Thresholds**: Apply dB limits (0-120 dB)
3. **Overlap Resolution**: Combine overlapping intervals with `value_kind="level"` (preserves level, not total)
4. **Quantile Filter** (optional)

**Level Aggregation**: When overlaps resolved, the winning interval's level is preserved (not time-weighted)

###### `_handle_cat(df: pd.DataFrame) -> pd.DataFrame`

Handler for categorical interval types (sleep stages, audio events).

**Process**:
1. **Label Normalization**: Strip "HKCategoryValueSleepAnalysis" prefix (e.g., "InBed", "Asleep")
2. **Categorical Overlap Resolution**: Use `combine_overlaps_categorical` per user

**Sleep Labels**: `InBed`, `Asleep`, `Awake`, `AwakeUnspecified` (may include others)

###### `_handle_unknown(df: pd.DataFrame) -> pd.DataFrame`

Fallback handler for unregistered types.

**Process**:
1. **Best-Effort Numeric Coercion**: Try to convert, don't fail if non-numeric
2. **Deduplication**: Remove exact duplicates

##### Overlap Resolution

###### `combine_overlaps(user_hk_data: pd.DataFrame, value_col: str, value_kind: str = "total") -> pd.DataFrame`

Resolve overlapping intervals using sweep-line algorithm.

**Parameters**:
- `user_hk_data`: Single-user data (pre-filtered)
- `value_col`: Column containing measurement value
- `value_kind`: `"total"` (redistribute proportionally) or `"level"` (preserve level)

**Algorithm**:
1. Create atomic time segments from all boundary events
2. For each segment, determine winning interval by priority:
   - Has `device_source_id` > higher `device_source_id` > shorter duration > later row
3. Coalesce adjacent segments with same winner
4. Recompute value:
   - `"total"`: `original_value * (segment_duration / original_duration)`
   - `"level"`: keep original value unchanged

**Returns**: DataFrame with non-overlapping intervals

###### `combine_overlaps_categorical(user_hk_data: pd.DataFrame, value_col: str) -> pd.DataFrame`

Resolve overlapping categorical intervals (similar to combine_overlaps).

**Priority Rules**:
1. If both have different `device_source_id`: higher wins
2. Otherwise: shorter duration wins
3. Ties: later row wins

**Label Handling**: Adjacent segments with same label are merged

##### Utility Methods

###### `_coerce_value_numeric(df, rep, hk_type, *, handler_label, best_effort=False) -> pd.DataFrame`

Convert value column to numeric, drop or warn on failures.

###### `_filter_range(df, min_val, max_val, col="value") -> Tuple[pd.DataFrame, int]`

Inclusive range filter, returns `(filtered_df, n_dropped)`.

###### `_filter_quantiles(df, col, lower_q, upper_q) -> Tuple[pd.DataFrame, int]`

Remove values outside quantile range.

###### `_apply_per_user(df, fn, value_col, value_kind=None) -> pd.DataFrame`

Apply a function per user group with optional parallelism.

**Features**:
- Detects CPU count
- Uses ThreadPoolExecutor with `min(cpu_count - 20, n_users)` workers
- Falls back to single-threaded if `max_workers <= 1` or `n_users <= 1`
- Error-resilient: logs errors and returns raw data for failed users

##### Summary Statistics

###### `summarize_by_type(df: pd.DataFrame, *, percentiles=(0.25, 0.75), duration_unit="minutes", round_digits=3) -> pd.DataFrame`

Generate comprehensive per-type summary statistics.

**Computed Metrics**:
- **Basic Counts**: `n_records`, `n_users`
- **Coverage**: `abs_start`, `abs_end`, `abs_span_days` (overall time range)
- **Value Distribution**: mean, median, std, min, max, sum, count, percentiles
- **Duration Distribution**: mean, median, std, min, max, sum, count (in specified unit)
- **Per-User Span**: mean, median, std, min, max span days
- **Records Per User**: mean, median, std, min, max

**Returns**: One row per type with all metrics

---

### healthkit_cleaner_specs.py

**Purpose**: Configuration dataclass for cleaning pipeline.

#### Class: `HKCleanConfig`

Configuration container with sensible defaults.

##### Fields

###### Required Columns
```python
required_cols: Set[str] = {
    "user_id", "type", "local_start", "local_end", 
    "value", "utc_start", "utc_end"
}
```

###### Thresholds
```python
thresholds: Dict[str, Dict[str, float]] = {
    "EnvironmentalAudioExposure": {"min": 0, "max": 120},
    "HeadphoneAudioExposure": {"min": 0},
    "StepCount": {"min": 0},
    "AppleExerciseTime": {"min": 0},
    "ActiveEnergyBurned": {"min": 0, "kcal_per_min_max": 30},
    "BasalEnergyBurned": {"min": 0},
    "AppleStandTime": {"min": 0},
    "DistanceCycling": {"min": 0},
    "DistanceWalkingRunning": {"min": 0},
    "FlightsClimbed": {"min": 0},
    "HeartRate": {"min": 25, "max": 230},
    "HeartRateVariabilitySDNN": {"min": 0.005, "max": 0.3},
    "RespiratoryRate": {"min": 4, "max": 80},
    "OxygenSaturation": {"min": 0.50, "max": 1.0},
}
```

###### Operation Toggles
```python
ops: Dict[str, bool] = {
    "coerce_numeric": True,
    "convert_energy_to_kcal": True,
    "cap_kcal_per_min": True,
    "apply_theory_thresholds": True,
    "deduplicate": True,
    "normalize_sleep_labels": True,
    "apply_quantile_filter": False,  # Off by default
    "apply_watch_device_filter": True,  # Apple Watch only
    "combine_overlaps": True,
    "combine_overlaps_cat": True,
}
```

###### Quantile Options
```python
quantile_lower: float = 0.01  # Lower bound (1st percentile)
quantile_upper: float = 0.99  # Upper bound (99th percentile)
```

###### Energy Rate Policy
```python
kcal_per_min_policy: Literal["cap", "drop"] = "cap"
```
- `"cap"`: Rescale excessive values to threshold
- `"drop"`: Remove rows exceeding threshold

##### Usage Example
```python
from healthkit_cleaner_specs import HKCleanConfig
from healthkit_cleaner import HealthKitCleaner

# Custom configuration
config = HKCleanConfig(
    kcal_per_min_policy="drop",
    ops={
        **HKCleanConfig().ops,
        "apply_quantile_filter": True,
        "apply_watch_device_filter": False,
    },
    quantile_lower=0.05,
    quantile_upper=0.95,
)

cleaner = HealthKitCleaner(rules=config)
cleaned_data, report = cleaner.clean(raw_data, return_report=True)
```

---

### healthkit_aggregator.py

**Purpose**: Aggregate cleaned HealthKit data into time-windowed feature matrices.

#### Class: `HealthKitAggregator`

Main aggregation engine with window construction, observation assignment, and feature computation.

##### Initialization

###### `__init__(config: Optional[HKAggConfig] = None)`

**Setup Process**:
1. Validate configuration
2. Cache zero-fill eligibility per type
3. Build operation registry for hierarchical aggregation
4. Set bin anchoring parameters (epoch + origin)
5. Initialize report container

**Key Attributes**:
- `cfg`: Configuration instance
- `op_registry`: Aggregation operations (sum, mean, weighted_mean, etc.)
- `_epoch_utc`: Reference epoch (1970-01-01 UTC)
- `_anchor_utc`: Bin grid anchor (epoch + origin offset)
- `_bin_seconds_nominal`: Expected bin width in seconds
- `report`: Nested dict tracking all pipeline metrics

##### Core Aggregation

###### `run_aggregation(df: pd.DataFrame, schedule: Optional[pd.DataFrame] = None, assessments: Optional[pd.DataFrame] = None, duration_before: Optional[Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]`

Main entry point for aggregation pipeline.

**Parameters**:
- `df`: Cleaned HealthKit data (output from HealthKitCleaner)
- `schedule`: Per-user timezone schedule (optional for UTC mode)
- `assessments`: Assessment metadata for anchored windows (optional)
- `duration_before`: Lookback duration for assessment windows

**Returns**: `(wide_df, report, aux)`
- `wide_df`: Feature matrix (one row per window)
- `report`: Pipeline metrics and timing
- `aux`: Auxiliary artifacts (`windows`, `wear`, `schedule`)

**Process**:
1. **Input Validation**: Normalize timestamps, validate schema
2. **Window Construction**: Build bin grid or assessment-anchored windows
3. **Observation Assignment**: Split observations into window-aligned fragments
4. **Per-Type Aggregation**: Compute statistics within windows
5. **Wide Pivot**: Reshape to one row per window with feature columns
6. **Wear Computation**: Calculate HR-based device wear coverage
7. **Presence Flags**: Identify windows with data per type
8. **Zero-Fill**: Apply zero-fill policy where appropriate
9. **Local Time Labels**: Attach date/hour/timezone metadata

**Schema**:
```python
# Required input columns
["user_id", "type", "value"]
# Plus either:
["utc_start", "utc_end"]  # UTC mode
# Or:
["local_start", "local_end", "timezone"]  # Local mode
```

**Output Schema** (wide_df):
```python
# Window identifiers
["user_id", "window_id", "window_start", "window_end", "window_seconds"]

# Bin metadata (grid windows only)
["bin_start", "bin_end", "bin_seconds", "bin_index"]

# Local time labels
["window_start_local", "window_end_local", "local_date", "anchor_local_date",
 "local_hour", "minute_of_day", "local_minute_index", "tz_name", "is_dst",
 "fold", "utc_offset_seconds", "local_weekday", "iso_week"]

# Wear metrics
["HR_coverage_seconds", "HR_wear_fraction"]

# Per-type features (example for HeartRate)
["HeartRate__mean", "HeartRate__median", "HeartRate__std", "HeartRate__max", 
 "HeartRate__count", "HeartRate__present"]

# Per-type features (example for StepCount)
["StepCount__sum", "StepCount__duration", "StepCount__rate_per_min", "StepCount__present"]

# ... all configured types ...
```

##### Quality Control

###### `apply_quality_control(wide: pd.DataFrame, aux: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]`

Apply multi-level wear-based QC gates.

**QC Levels**:
1. **Bin-Level**: Per-window HR wear fraction < `bin_frac_min`
2. **Day-Level**: Per-day average wear < `day_frac_min`
3. **Participant-Level**: Overall average wear < `overall_frac_min`

**Actions**:
- `"drop"`: Remove rows/users failing threshold
- `"mask"`: Set features to NaN, preserve structure

**QC Selectors**: Control which features are affected by each level
```python
qc_mask_selectors = {
    "bin": {"device_class": ["watch_exclusive", "either"]},
    "day": {"all": True},
    "participant": {"all": True},
}
```

**Returns**: `(wide_qc, report_qc)`

##### Hierarchical Aggregation

###### `aggregate_layer(child: pd.DataFrame, spec: LayerSpec, prev_layer_name: Optional[str] = None) -> pd.DataFrame`

Aggregate child layer into parent using LayerSpec rules.

**Example**: Hourly → Daily rollup
```python
daily_spec = LayerSpec(
    name="daily",
    freq="1D",
    origin="00:00:00",
    strict=True,
    tz_mode="local",
    rules=[
        LayerRule(out="HeartRate__mean", op="weighted_mean", 
                  inputs={"col": "HeartRate__mean", "weight": "HeartRate__count"}),
        LayerRule(out="StepCount__sum", op="sum", 
                  inputs={"col": "StepCount__sum"}),
    ]
)

daily = aggregator.aggregate_layer(hourly, daily_spec, prev_layer_name="hourly")
```

###### `run_hierarchical_aggregation(wide_l0: pd.DataFrame, specs: List[LayerSpec], apply_qc_during: bool = False, return_final_only: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]`

Execute multi-level aggregation chain.

**Parameters**:
- `wide_l0`: Layer 0 (finest granularity)
- `specs`: Ordered list of layer specifications
- `apply_qc_during`: Apply QC between layers
- `return_final_only`: If True, return only final layer; else return all layers dict

**Returns**: Final DataFrame or dict of `{layer_name: DataFrame}`

##### Sleep Features

###### `compute_daily_sleep_features(df: pd.DataFrame, schedule: Optional[pd.DataFrame] = None, bin_origin: str = "15:00:00") -> pd.DataFrame`

Compute sleep metrics on daily (3pm-3pm) windows.

**Features**:
- Sleep onset/offset times (first/last sleep event)
- Bedrest onset/offset (first/last InBed event)
- Duration totals per stage (InBed, Asleep, Awake, etc.)
- Time in each stage
- Sleep efficiency: `Asleep / InBed`

**Returns**: One row per user per day (3pm anchor)

###### `apply_sleep_quality_control(sleep_df: pd.DataFrame, bedrest_min_minutes: float = 180.0) -> Tuple[pd.DataFrame, Dict[str, Any]]`

Filter unreliable sleep days.

**Criteria**: Drop days with `< bedrest_min_minutes` of InBed time

##### Window Construction

###### `build_bin_windows(df: pd.DataFrame, schedule: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]`

Build fixed-frequency bin grid (UTC or local wall clock).

**Modes**:
- **UTC**: Constant-width bins anchored to epoch + origin
- **Local**: Variable-width bins following wall-clock time per schedule (handles DST)

**Returns**: `(windows, schedule)`

###### `build_assessment_anchored_windows(assessments: pd.DataFrame, schedule: Optional[pd.DataFrame], duration_before: Optional[Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame]`

Build windows ending at assessment times.

**Window Definition**: `[anchor_time - duration_before, anchor_time)`

**Assessments Schema**:
```python
{
    "user_id": int/str,
    "anchor_time": datetime (UTC),
    "assessment_id": int/str (optional, used as window_id),
    "duration_before": timedelta/str (optional, overrides config)
}
```

##### Observation Assignment

###### `assign_observations_to_windows(df: pd.DataFrame, windows: pd.DataFrame, types: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]`

Route raw observations into window-aligned intermediate frames.

**Time Behaviors**:
- **interval_total** (e.g., StepCount): Split proportionally across windows
- **interval_level** (e.g., audio exposure): Preserve level, track duration
- **point** (e.g., HeartRate): Assign to containing window(s)
- **categorical_interval** (e.g., SleepAnalysis): Track label and duration

**Returns**: Dict of `{type_name: intermediate_df}`

**Intermediate Schema**:
```python
# interval_total
["user_id", "window_id", "window_start", "window_end", "window_seconds",
 "value", "duration", "utc_start", "utc_end"]

# interval_level
["user_id", "window_id", ..., "value_x_seconds", "duration", "value", ...]

# point
["user_id", "window_id", ..., "value", "coverage_seconds"]  # HR only

# categorical_interval
["user_id", "window_id", ..., "label", "duration", ...]
```

###### `aggregate_type_within_windows(tname: str, intermediate: pd.DataFrame) -> pd.DataFrame`

Reduce intermediate fragments into per-window aggregates.

**Examples**:
```python
# StepCount (interval_total)
StepCount__sum = sum(value)
StepCount__duration = sum(duration)
StepCount__rate_per_min = sum / (duration / 60)

# EnvironmentalAudioExposure (interval_level)
Audio__time_weighted_mean = sum(value * duration) / sum(duration)
Audio__leq = 10 * log10(sum(10^(value/10) * duration) / sum(duration))
Audio__max = max(value)

# HeartRate (point)
HeartRate__mean = mean(value)
HeartRate__count = count(value)

# SleepAnalysis (categorical_interval)
SleepAnalysis__sum_label_asleep = sum(duration where label="Asleep")
SleepAnalysis__mean_label_asleep = mean(duration where label="Asleep")
```

##### Wear Computation

Wear detection uses HeartRate point coverage:

```python
# For each HR point:
coverage_window = [point_ts, point_ts + hr_coverage_window_seconds]
coverage_seconds = overlap(coverage_window, analysis_window)

# Per window:
HR_coverage_seconds = sum(coverage_seconds)
HR_wear_fraction = HR_coverage_seconds / window_seconds
```

**Default**: `hr_coverage_window_seconds = 300` (5 minutes)

##### Configuration

See `healthkit_aggregator_specs.py` for full `HKAggConfig` reference.

---

### healthkit_aggregator_specs.py

**Purpose**: Configuration and operation registry for aggregation.

#### Functions

###### `build_default_type_registry() -> Dict[str, Dict[str, Any]]`

Define per-type metadata.

**Registry Keys**:
```python
{
    "time_behavior": str,  # interval_total, interval_level, point, categorical_interval
    "aggregates": Tuple[str],  # Requested output statistics
    "device_class": str,  # watch_exclusive, non_watch_exclusive, either
    "zero_fill_gate": Tuple[str, Dict],  # Gate spec: (op, params)
    "zero_fill_stats": Set[str],  # Stats eligible for zero-fill
}
```

**Examples**:
```python
"HeartRate": {
    "time_behavior": "point",
    "aggregates": ("mean", "median", "std", "max", "count"),
    "device_class": "watch_exclusive",
    "zero_fill_gate": ("watch_on", {}),
    "zero_fill_stats": {"mean", "median", "std", "max", "count"},
}

"StepCount": {
    "time_behavior": "interval_total",
    "aggregates": ("sum", "duration", "rate_per_min"),
    "device_class": "either",
    "zero_fill_gate": ("watch_on", {}),
    "zero_fill_stats": {"sum", "duration", "rate_per_min"},
}
```

###### `build_default_op_registry() -> Dict[str, OpFn]`

Register aggregation operations for hierarchical layers.

**Available Operations**:
- **Basic**: `sum`, `count_nonnull`, `max`, `min`, `mean`, `std`, `median`
- **Weighted**: `weighted_mean`, `weighted_median`, `weighted_quantile`
- **Pooled**: `pooled_std`, `pooled_var_from_moments`
- **Alternative Means**: `geometric_mean`, `harmonic_mean`
- **Ratios**: `ratio`, `rate_from_totals`, `normalized_mean`
- **Logical**: `any`, `all`
- **Acoustic**: `leq_from_energy`
- **Distribution**: `quantile`, `skew`, `kurtosis`

**Operation Signature**:
```python
OpFn = Callable[[pd.core.groupby.generic.DataFrameGroupBy, Dict[str, Any]], pd.Series]
```

**Example Usage**:
```python
# Weighted mean
inputs = {"col": "HeartRate__mean", "weight": "HeartRate__count"}
result = op_registry["weighted_mean"](grouped, inputs)

# Pooled standard deviation
inputs = {
    "mean_col": "HeartRate__mean",
    "std_col": "HeartRate__std",
    "count_col": "HeartRate__count"
}
result = op_registry["pooled_std"](grouped, inputs)
```

#### Classes

###### `HKAggConfig`

Comprehensive configuration for aggregation pipeline.

**Key Fields**:

**Binning/Time**:
```python
bin_size: str = "1H"  # Pandas frequency string
bin_origin: str = "00:00:00"  # Time of day for bin anchoring
tz_mode_l0: str = "utc"  # "utc" or "local"
tz_policy: str = "strict"  # "strict" or "lenient"
output_timezone: str = "both"  # "UTC" (drop local labels), "local" (promote local window_start/window_end, keep UTC as *_utc), "both"
utc_schedule_inference_policy: str = "infer_fallback"  # "skip" | "infer_fallback" | "infer_fail" (UTC mode only)
schedule_point_policy: str = "include_epsilon"  # "include_epsilon" or "exclude"
schedule_point_epsilon: str = "1ns"  # positive timedelta when policy is include_epsilon
schedule_invalid_tz_policy: str = "strict"  # "strict" or "coerce_default"
```

Note: the field name is `tz_mode_l0` (letter `l` + zero), not `tz_mode_10`.

**Assessment Windows**:
```python
assessment_duration_before: Optional[str] = None  # e.g., "7D", "24H"
```

**Wear**:
```python
hr_type: str = "HeartRate"
hr_coverage_window_seconds: int = 300
```

**QC Thresholds**:
```python
bin_frac_min: Optional[float] = 0.80  # Minimum wear per bin
day_frac_min: Optional[float] = None  # Minimum wear per day
overall_frac_min: Optional[float] = None  # Minimum wear overall
```

**QC Toggles**:
```python
enable_bin_qc_drop: bool = False
enable_day_qc_drop: bool = False
enable_participant_qc_drop: bool = False
qc_action: str = "drop"  # "drop" or "mask"
```

**Zero-Fill**:
```python
apply_zero_fill: bool = False
```

**HRV Features** (NEW in v0.1.2):
```python
compute_hrv: bool = False  # Enable HRV computation
hrv_time_domain: bool = True  # Include time-domain features
hrv_freq_domain: bool = True  # Include frequency-domain features
hrv_min_samples: int = 10  # Minimum HR samples for time-domain
hrv_freq_min_samples: int = 30  # Minimum HR samples for frequency-domain
hrv_interpolation_fs: float = 4.0  # Interpolation frequency (Hz) for PSD
```

**Early Pruning**:
```python
early_prune_mode: Optional[str] = "any_data"  # Prune empty windows before expansion
early_pruning_types: Optional[List[str]] = None
early_prune_per_type: bool = True
```

**Parallelism**:
```python
max_workers: int = 1  # 1 = no parallelism
```

**Registries**:
```python
type_registry: Dict[str, Dict[str, Any]]  # Per-type metadata
qc_mask_selectors: Dict[str, Dict[str, Any]]  # QC scope control
qc_by_layer: Dict[str, Dict[str, Any]]  # Layer-specific QC
```

**Methods**:
- `validate()`: Check configuration consistency

###### `LayerSpec`

Specification for hierarchical aggregation layer.

**Fields**:
```python
name: str  # Layer identifier
freq: str  # Pandas frequency (e.g., "1D", "1W")
origin: str = "00:00:00"  # Time anchor
rules: List[LayerRule] = []  # Aggregation rules
strict: bool = True  # Require all inputs present
tz_mode: str = "utc"  # "utc" or "local"
```

###### `LayerRule`

Single aggregation rule within a layer.

**Fields**:
```python
out: str  # Output column name
op: str  # Operation key from op_registry
inputs: Dict[str, Any]  # Parameters for operation
strict: Optional[bool] = None  # Override layer strict
```

**Example**:
```python
LayerSpec(
    name="daily",
    freq="1D",
    origin="00:00:00",
    tz_mode="local",
    rules=[
        LayerRule(
            out="HeartRate__mean",
            op="weighted_mean",
            inputs={"col": "HeartRate__mean", "weight": "HeartRate__count"}
        ),
        LayerRule(
            out="HeartRate__std",
            op="pooled_std",
            inputs={
                "mean_col": "HeartRate__mean",
                "std_col": "HeartRate__std",
                "count_col": "HeartRate__count"
            }
        ),
    ]
)
```

---

### healthkit_hrv.py

**Purpose**: Compute Heart Rate Variability (HRV) features from HeartRate observations (NEW in v0.1.2).

#### Overview

This module provides time-domain and frequency-domain HRV analysis using RR interval methods. It's designed to integrate seamlessly with the aggregation pipeline but can also be used standalone.

#### Classes

###### `HRVComputer`

Main computation engine for HRV features.

**Constructor**:
```python
HRVComputer(
    min_samples: int = 10,
    time_domain: bool = True,
    freq_domain: bool = True,
    freq_min_samples: int = 30,
    interpolation_fs: float = 4.0,
    rr_min_ms: float = 300.0,
    rr_max_ms: float = 2000.0,
    rr_outlier_sd: float = 3.0
)
```

**Parameters**:
- `min_samples`: Minimum HR samples needed for time-domain analysis
- `time_domain`: Enable time-domain features (SDNN, RMSSD, etc.)
- `freq_domain`: Enable frequency-domain features (LF, HF, etc.)
- `freq_min_samples`: Minimum HR samples needed for frequency analysis
- `interpolation_fs`: Sampling rate for interpolation (Hz), recommended 4.0
- `rr_min_ms`: Minimum physiologically plausible RR interval (ms)
- `rr_max_ms`: Maximum physiologically plausible RR interval (ms)
- `rr_outlier_sd`: Standard deviation threshold for outlier detection

**Methods**:

**`compute_window_hrv(hr_observations: pd.DataFrame) -> Dict[str, float]`**

Compute HRV features for a single window's HR observations.

**Input DataFrame Columns**:
- `value`: Heart rate in bpm
- `utc_start`: Observation timestamp

**Output Dictionary Keys** (time-domain):
```python
{
    'nn_count': int,          # Number of valid NN intervals
    'mean_nn_ms': float,      # Mean NN interval
    'median_nn_ms': float,    # Median NN interval
    'mean_hr_bpm': float,     # Mean heart rate
    'hr_range_bpm': float,    # HR range (max - min)
    'sdnn_ms': float,         # SDNN
    'rmssd_ms': float,        # RMSSD
    'pnn50_pct': float,       # pNN50 percentage
    'pnn20_pct': float,       # pNN20 percentage
    'hrv_tri_index': float,   # HRV triangular index
}
```

**Output Dictionary Keys** (frequency-domain, if enabled and sufficient data):
```python
{
    'vlf_power_ms2': float,   # VLF power (0.003-0.04 Hz)
    'lf_power_ms2': float,    # LF power (0.04-0.15 Hz)
    'lf_peak_hz': float,      # LF peak frequency
    'lf_power_nu': float,     # LF normalized units
    'hf_power_ms2': float,    # HF power (0.15-0.4 Hz)
    'hf_peak_hz': float,      # HF peak frequency
    'hf_power_nu': float,     # HF normalized units
    'lf_hf_ratio': float,     # LF/HF ratio
    'total_power_ms2': float, # Total power
}
```

**Returns**: Dictionary with computed HRV metrics (returns empty dict if insufficient data)

#### Functions

###### `compute_hrv_features_for_windows(windows_df: pd.DataFrame, hr_long: pd.DataFrame, hrv_computer: HRVComputer) -> pd.DataFrame`

Batch compute HRV features for multiple windows.

**Parameters**:
- `windows_df`: DataFrame with window definitions (`user_id`, `window_id`, `window_start`, `window_end`)
- `hr_long`: Long-format HR observations with columns (`user_id`, `window_id`, `window_start`, `window_end`, `value`, `utc_start`)
- `hrv_computer`: Configured `HRVComputer` instance

**Returns**: Wide DataFrame with HRV features
```python
["user_id", "window_id", "window_start", "window_end",
 "HRV__nn_count", "HRV__mean_nn_ms", "HRV__sdnn_ms", ...]
```

**Example**:
```python
from healthkit_hrv import HRVComputer, compute_hrv_features_for_windows

# Create computer
computer = HRVComputer(
    min_samples=10,
    time_domain=True,
    freq_domain=True,
)

# Compute for all windows
hrv_features = compute_hrv_features_for_windows(
    windows_df=windows,
    hr_long=hr_intermediate_data,
    hrv_computer=computer
)

# Result: DataFrame with 19 HRV feature columns
```

#### Helper Methods (Internal)

These are private methods of `HRVComputer` but documented for understanding:

**`_clean_rr_intervals(rr_intervals: np.ndarray) -> np.ndarray`**
- Removes physiologically implausible intervals
- Applies adaptive outlier detection
- Returns cleaned array

**`_interpolate_nn(nn_intervals_ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`**
- Cubic spline interpolation to uniform sampling
- Detrends signal (removes linear trend)
- Returns (time_uniform, nn_interpolated)

**`_calculate_psd_welch(nn_uniform: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]`**
- Welch's method for power spectral density
- Hann window with 50% overlap
- Returns (frequencies, power)

**`_band_power(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float`**
- Integrate PSD over frequency band using trapezoidal rule
- Returns band power in ms²

**`_compute_time_domain(nn_ms: np.ndarray) -> Dict[str, float]`**
- Computes all time-domain metrics
- Returns dictionary with 10 features

**`_compute_freq_domain(nn_ms: np.ndarray) -> Dict[str, float]`**
- Computes all frequency-domain metrics
- Requires scipy (graceful fallback if missing)
- Returns dictionary with 9 features

#### Integration with Aggregation Pipeline

The HRV module is automatically invoked by `HealthKitAggregator.run_aggregation()` when `cfg.compute_hrv=True`:

1. **After feature assembly**: HRV computed from HeartRate intermediate data
2. **Before zero-fill**: HRV features added to wide DataFrame
3. **Automatic reporting**: Execution statistics added to report['hrv']

**Execution Flow**:
```python
# In run_aggregation()
if self.cfg.compute_hrv:
    # 1. Initialize computer (done in __init__)
    # 2. Get HeartRate intermediate data
    hr_long = inter[self.cfg.hr_type]
    
    # 3. Compute HRV features
    hrv_features = compute_hrv_features_for_windows(
        windows_df=windows,
        hr_long=hr_long,
        hrv_computer=self.hrv_computer
    )
    
    # 4. Merge into wide DataFrame
    wide = wide.merge(hrv_features, on=['user_id', 'window_id', ...])
    
    # 5. Update report
    report['hrv'] = {...}
```

#### Standalone Usage

The module can also be used independently:

```python
from healthkit_hrv import HRVComputer
import pandas as pd

# Your HR data
hr_df = pd.DataFrame({
    'utc_start': pd.date_range('2024-01-01', periods=50, freq='30S'),
    'value': [70, 72, 68, 71, 69, ...]  # HR in bpm
})

# Compute HRV
computer = HRVComputer()
hrv_metrics = computer.compute_window_hrv(hr_df)

print(f"SDNN: {hrv_metrics['sdnn_ms']:.2f} ms")
print(f"RMSSD: {hrv_metrics['rmssd_ms']:.2f} ms")
print(f"LF/HF: {hrv_metrics['lf_hf_ratio']:.2f}")
```

#### Dependencies

**Required**:
- numpy
- pandas

**Optional**:
- scipy (for frequency-domain analysis)
  - If scipy not available, frequency-domain features are skipped with a warning
  - Time-domain features work without scipy

#### Performance

**Typical execution times** (per 1000 windows):
- Time-domain only: 0.5-1.0 seconds
- Time + Frequency: 1.5-2.5 seconds

**Optimization tips**:
- Set `hrv_freq_domain=False` if frequency metrics not needed
- Increase `hrv_min_samples` threshold to reduce computation on sparse windows
- Use appropriate `bin_size` (5-10 minutes recommended for HRV)

---

### healthkit_aggregator_time.py

**Purpose**: Window construction and timezone utilities.

#### Functions

###### `build_schedule_from_input(df: pd.DataFrame, user_col: str = "user_id", start_col: str = "utc_start", end_col: str = "utc_end", tz_col: str = "timezone", default_tz: str = "UTC", fill_gaps: str = "ffill", coverage_bounds: Optional[pd.DataFrame] = None, point_policy: str = "include_epsilon", point_epsilon: str = "1ns", invalid_tz_policy: str = "strict", report: Optional[Dict[str, Any]] = None) -> pd.DataFrame`

Infer per-user timezone schedule from observed intervals using sweep-line algorithm.

**Algorithm**:
1. Sort all interval boundaries per user
2. Maintain active timezone counts
3. Winner at each span: argmax(count), ties favor previous winner
4. Emit schedule segments when winner changes
5. Fill gaps by policy: `"ffill"`, `"bfill"`, `"default"`
6. Coalesce adjacent segments with same timezone
7. Enforce per-user span against canonical normalized source bounds

**Canonical source normalization**:
- Drops invalid rows (`missing ts`, `end < start`)
- Handles point-like rows (`end == start`) via `point_policy`
- Validates timezone labels via `schedule_invalid_tz_policy`
- Emits diagnostics in `report["schedule"]`:
  - `source_rows_total`
  - `source_rows_epsilon_promoted`
  - `source_rows_dropped_invalid`
  - `invalid_tz_rows`

**Complexity**: O(N log N) time, O(N) memory

**Returns**: Schedule DataFrame
```python
["user_id", "start_utc", "end_utc", "tz_name"]
```

###### `build_bin_windows(...) -> Tuple[pd.DataFrame, pd.DataFrame]`

Construct per-user fixed-frequency bin grid.

**Modes**:
- **UTC**: Uniform bins anchored to epoch + origin
- **Local**: Variable bins following wall-clock per schedule (handles DST)

**Returns**: `(windows, validated_schedule)`

###### `build_assessment_anchored_windows(...) -> Tuple[pd.DataFrame, pd.DataFrame]`

Construct windows ending at assessment anchor times.

**Returns**: `(windows, validated_schedule)`

###### `validate_schedule(schedule: Optional[pd.DataFrame], users: Iterable[Any], utc_min: pd.Timestamp, utc_max: pd.Timestamp, tz_policy: str, report: Optional[Dict] = None, coverage_bounds: Optional[pd.DataFrame] = None) -> pd.DataFrame`

Validate and normalize timezone schedule.

**Checks**:
- Required columns present
- No non-positive ranges
- No overlaps (strict) or resolve overlaps (lenient)
- Coverage spans expected analysis interval (strict only, per-user when `coverage_bounds` is provided)

**Returns**: Validated schedule

###### `compute_schedule_coverage(windows: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame`

Attach schedule coverage metrics to windows.

**Adds Columns**:
- `schedule_covered_seconds`: Seconds of window covered by schedule
- `schedule_covered_fraction`: Fraction covered (0.0-1.0)

###### `add_local_time_labels(windows: pd.DataFrame, schedule: pd.DataFrame, tz_policy: str = "auto") -> pd.DataFrame`

Attach local time metadata to UTC windows.

**Reference Instant Policy**:
- `"start"` / `"window_start"`: Use window start
- `"end"` / `"window_end"` / `"anchor"`: Use window end
- `"auto"`: Start for grid, end for anchored

**Added Columns**:
```python
[
    "window_start_local", "window_end_local",
    "local_date", "anchor_local_date",
    "local_hour", "minute_of_day", "local_minute_index",
    "tz_name", "is_dst", "fold", "utc_offset_seconds",
    "local_weekday", "iso_week"
]
```

**DST Handling**:
- `is_dst`: Boolean flag for daylight saving time
- `fold`: Duplicate local time counter (0 or 1)
- `utc_offset_seconds`: Raw offset from UTC

###### `build_utc_bin_grid(...) -> pd.DataFrame`

Low-level UTC bin builder (constant width).

###### `build_local_bin_grid(...) -> pd.DataFrame`

Low-level local bin builder (variable width with DST).

**Process**:
1. Per schedule segment, generate naive local datetime grid
2. Localize to timezone (handling DST ambiguity)
3. Convert to UTC
4. Compute actual duration in seconds

###### `validate_window_table(windows: pd.DataFrame, window_kind: str = "grid", strict_nonoverlap: bool = True) -> None`

Validate window schema and invariants.

###### `count_window_overlaps(windows: pd.DataFrame) -> int`

Count overlapping windows per user.

###### `compute_window_overlap_depth_stats(windows: pd.DataFrame) -> Dict[str, Any]`

Compute overlap diagnostics (max depth, users with overlaps).

---

## Workflow Examples

### Basic Pipeline

```python
from healthkit_loader import HealthKitLoader
from healthkit_cleaner import HealthKitCleaner
from healthkit_aggregator import HealthKitAggregator
from healthkit_aggregator_specs import HKAggConfig

# 1. Load
loader = HealthKitLoader()
raw = loader.load_participant(
    data_folder="/data/healthkit",
    user_id=12345
)

# 2. Clean
cleaner = HealthKitCleaner()
clean, clean_report = cleaner.clean(raw, return_report=True)
print(f"Cleaned {clean_report['__global__']['rows_in']} → {clean_report['__global__']['rows_out']} rows")

# 3. Aggregate
config = HKAggConfig(
    bin_size="1H",
    tz_mode_l0="utc",
    bin_frac_min=0.8,
    enable_bin_qc_drop=True,
)
aggregator = HealthKitAggregator(config)
wide, report, aux = aggregator.run_aggregation(clean)

# 4. Apply QC
wide_qc, qc_report = aggregator.apply_quality_control(wide, aux)
print(f"QC: {qc_report['qc']['bin']['dropped_rows']} rows dropped")
```

### With Travel Schedule

```python
from healthkit_aggregator_time import build_schedule_from_input

# Infer schedule from timezone metadata
schedule = build_schedule_from_input(
    clean,
    user_col="user_id",
    start_col="utc_start",
    end_col="utc_end",
    tz_col="timezone",
    default_tz="America/New_York",
    fill_gaps="ffill",
)

# Aggregate with local bins
config = HKAggConfig(
    bin_size="1H",
    tz_mode_l0="local",
    tz_policy="lenient",
    output_timezone="both",
)
aggregator = HealthKitAggregator(config)
wide, report, aux = aggregator.run_aggregation(clean, schedule=schedule)
```

If `schedule` is not passed, `run_aggregation(...)` infers it from `timezone` + UTC bounds.
Local mode inference is fail-fast with clear errors; UTC mode behavior is controlled by `utc_schedule_inference_policy`.

### Assessment-Anchored Windows

```python
import pandas as pd

# Define assessments
assessments = pd.DataFrame({
    "user_id": [12345, 12345, 12345],
    "anchor_time": pd.to_datetime([
        "2024-01-01 09:00:00",
        "2024-01-08 09:00:00",
        "2024-01-15 09:00:00",
    ], utc=True),
    "assessment_id": [1, 2, 3],
})

# 7-day lookback windows
config = HKAggConfig(
    assessment_duration_before="7D",
    bin_frac_min=0.7,
)
aggregator = HealthKitAggregator(config)
wide, report, aux = aggregator.run_aggregation(
    clean,
    assessments=assessments,
    schedule=schedule,
)
```

### Hierarchical Aggregation

```python
from healthkit_aggregator_specs import LayerSpec, LayerRule

# Hourly → Daily → Weekly
specs = [
    LayerSpec(
        name="daily",
        freq="1D",
        origin="00:00:00",
        tz_mode="local",
        rules=[
            LayerRule(out="HeartRate__mean", op="weighted_mean",
                     inputs={"col": "HeartRate__mean", "weight": "HeartRate__count"}),
            LayerRule(out="StepCount__sum", op="sum",
                     inputs={"col": "StepCount__sum"}),
        ]
    ),
    LayerSpec(
        name="weekly",
        freq="1W",
        origin="00:00:00",
        tz_mode="local",
        rules=[
            LayerRule(out="HeartRate__mean", op="weighted_mean",
                     inputs={"col": "HeartRate__mean", "weight": "HeartRate__count"}),
            LayerRule(out="StepCount__sum", op="sum",
                     inputs={"col": "StepCount__sum"}),
        ]
    ),
]

# Run hierarchical
layers = aggregator.run_hierarchical_aggregation(
    hourly,
    specs,
    apply_qc_during=False,
    return_final_only=False,
)

daily = layers["daily"]
weekly = layers["weekly"]
```

### Sleep Features

```python
# Daily sleep metrics (3pm-3pm windows)
sleep = aggregator.compute_daily_sleep_features(
    clean,
    schedule=schedule,
    bin_origin="15:00:00",
)

# Apply sleep QC
sleep_qc, sleep_report = aggregator.apply_sleep_quality_control(
    sleep,
    bedrest_min_minutes=180.0,
)
```

---

## Configuration Guide

### Cleaning Configuration

```python
from healthkit_cleaner_specs import HKCleanConfig

config = HKCleanConfig(
    # Threshold overrides
    thresholds={
        "HeartRate": {"min": 30, "max": 200},  # Tighter HR range
        "ActiveEnergyBurned": {"min": 0, "kcal_per_min_max": 25},
    },
    
    # Energy policy
    kcal_per_min_policy="drop",  # Drop excessive values
    
    # Operation toggles
    ops={
        "coerce_numeric": True,
        "convert_energy_to_kcal": True,
        "apply_theory_thresholds": True,
        "deduplicate": True,
        "combine_overlaps": True,
        "combine_overlaps_cat": True,
        "apply_watch_device_filter": False,  # Include iPhone data
        "apply_quantile_filter": True,  # Enable quantile filtering
    },
    
    # Quantile bounds
    quantile_lower=0.05,
    quantile_upper=0.95,
)
```

### Aggregation Configuration

```python
from healthkit_aggregator_specs import HKAggConfig

config = HKAggConfig(
    # Binning
    bin_size="1H",
    bin_origin="00:00:00",
    tz_mode_l0="local",  # Local wall-clock bins
    tz_policy="lenient",  # Handle DST gracefully
    output_timezone="both",  # Include both UTC and local timestamps
    
    # Wear
    hr_type="HeartRate",
    hr_coverage_window_seconds=300,
    
    # QC thresholds
    bin_frac_min=0.80,  # 80% wear per bin
    day_frac_min=0.60,  # 60% average wear per day
    overall_frac_min=0.50,  # 50% overall wear
    
    # QC actions
    enable_bin_qc_drop=True,
    enable_day_qc_drop=True,
    enable_participant_qc_drop=False,
    qc_action="drop",  # "drop" or "mask"
    
    # Zero-fill
    apply_zero_fill=True,  # Fill zeros where appropriate
    
    # Pruning
    early_prune_mode="any_data",
    prune_empty_rows=True,
    
    # Parallelism
    max_workers=8,  # Use 8 threads
)
```

### Custom Type Registry

```python
from healthkit_aggregator_specs import build_default_type_registry

registry = build_default_type_registry()

# Add custom type
registry["CustomMetric"] = {
    "time_behavior": "point",
    "aggregates": ("mean", "median", "count"),
    "device_class": "either",
    "zero_fill_gate": ("expr_false", {}),
    "zero_fill_stats": set(),
}

config = HKAggConfig(type_registry=registry)
```

---

## Advanced Topics

### Timezone Handling

`tz_mode_l0` controls how windows are built, while `output_timezone` controls which timestamp columns are returned.
Coverage validation in strict mode is performed against **per-user canonical schedule-source bounds** (normalized from input timestamps), not global batch bounds.

1. **UTC windows**
   ```python
   config = HKAggConfig(tz_mode_l0="utc", tz_policy="strict")
   ```
   Window construction is UTC-anchored.

2. **Local wall-clock windows**
   ```python
   config = HKAggConfig(tz_mode_l0="local", tz_policy="lenient")
   ```
   Window construction follows local wall-clock schedule (DST-aware). Requires schedule input or timezone signal for inference.

3. **Output projection (`output_timezone`)**
   ```python
   config = HKAggConfig(tz_mode_l0="utc", output_timezone="both")
   ```
   - `output_timezone="both"`: keep UTC window columns and local label columns.
   - `output_timezone="UTC"`: keep UTC window columns and drop local label columns.
   - `output_timezone="local"`: promote `window_start_local/window_end_local` into `window_start/window_end`, and preserve original UTC values in `window_start_utc/window_end_utc`.

Schedule inference is fail-fast for local mode: malformed timezone inputs raise clear errors with hints about required columns (`timezone`, `utc_start`, `utc_end`) and valid IANA timezone names.

UTC-mode inference behavior is configurable via `utc_schedule_inference_policy`:
- `"skip"`: do not attempt schedule inference in UTC mode.
- `"infer_fallback"`: try inference; on failure, continue with UTC fallback schedule.
- `"infer_fail"`: try inference; on failure, raise.

Field-name reminder: use `tz_mode_l0` (letter `l` + zero), not `tz_mode_10`.

**DST Policies**:
- `"strict"`: Fail on ambiguous/nonexistent times
- `"lenient"`: Infer DST fold, shift forward spring gaps

**Schedule Behavior Matrix**:
- `tz_mode_l0="utc"`: windows are UTC-anchored; schedule is still validated for metadata/local projection.
- `tz_mode_l0="local"`: windows are local wall-clock by schedule (DST-aware); requires schedule input or inferable timezone signal.
- `tz_policy="strict"`: overlap or true per-user coverage gaps raise errors.
- `tz_policy="lenient"`: overlap/gap issues are reported but do not raise.
- `schedule_point_policy="include_epsilon"`: point-like rows (`end == start`) are promoted by `schedule_point_epsilon` before schedule build/coverage checks.
- `schedule_point_policy="exclude"`: point-like rows are excluded from canonical source and from expected strict coverage bounds.
- `schedule_invalid_tz_policy="strict"`: invalid timezone labels raise during schedule-source normalization.
- `schedule_invalid_tz_policy="coerce_default"`: invalid timezone labels are replaced with default timezone and counted in diagnostics.
- `utc_schedule_inference_policy="infer_fallback"`: in UTC mode, schedule inference errors do not stop aggregation.

**Time Contract Across Pipeline**:
- Loader: timezone labels are normalized/validated for CSV and XML paths before downstream use.
- Cleaner: `utc_start`/`utc_end` are re-coerced to UTC-aware datetimes and postcondition diagnostics are emitted.
- Aggregator: schedule creation, strict coverage bounds, and validation all run on UTC-aware canonical source bounds.

**Possible Schedule-Related Errors**:
- `schedule missing required columns`: user-provided schedule schema is incomplete.
- `schedule has non-positive ranges`: user-provided schedule has `end_utc <= start_utc`.
- `schedule has overlapping ranges for user_id=...`: strict overlap failure.
- `schedule does not fully cover analysis interval (...)`: true per-user coverage gap versus canonical bounds.
- `invalid timezone labels in schedule source`: malformed timezone values in input.
- `local mode requires a travel schedule with tz ranges`: local bins requested without schedule coverage.

**Schedule Inference Diagnostics (`report["schedule"]`)**:
- `inference_attempted`
- `inference_mode`
- `inference_fallback_used`
- `inference_error`
- `source_rows_total`
- `source_rows_epsilon_promoted`
- `source_rows_dropped_invalid`
- `invalid_tz_rows`

**Debug snippet (per-user end-gap inspection)**:
```python
src = cleaned[["user_id", "utc_start", "utc_end"]].copy()
src["utc_start"] = pd.to_datetime(src["utc_start"], utc=True, errors="coerce")
src["utc_end"] = pd.to_datetime(src["utc_end"], utc=True, errors="coerce")
raw_max = src.groupby("user_id", as_index=False).agg(raw_max_end=("utc_end", "max"))

norm_src, _diag = hk_time._normalize_schedule_source(  # internal helper
    cleaned, user_col="user_id", start_col="utc_start", end_col="utc_end",
    tz_col="timezone", point_policy="include_epsilon", point_epsilon="1ns",
    default_tz="UTC", invalid_tz_policy="strict",
)
norm_max = norm_src.groupby("user_id", as_index=False).agg(norm_max_end=("end_utc", "max"))
sch_max = aux["schedule"].groupby("user_id", as_index=False).agg(schedule_max_end=("end_utc", "max"))

dbg = raw_max.merge(norm_max, on="user_id", how="outer").merge(sch_max, on="user_id", how="outer")
dbg["gap_seconds"] = (dbg["norm_max_end"] - dbg["schedule_max_end"]).dt.total_seconds()
display(dbg.sort_values(["gap_seconds", "user_id"], ascending=[False, True]))
```

### Overlap Resolution

**Activity Types** (interval_total):
- Proportional redistribution of totals
- Priority: device_source_id > duration > recency

**Audio Types** (interval_level):
- Level preservation (not total)
- Same priority rules

**Categorical Types**:
- Label-based segmentation
- Adjacent same-label segments merged

**Algorithm**: O(N log N) sweep-line with atomic time segments

### Zero-Fill Policy

Zero-fill replaces missing values with zeros when appropriate:

**Gate Types**:
- `"watch_on"`: Fill only when watch was worn (HR coverage > 0)
- `"expr_false"`: Never fill (always leave NaN)

**Eligible Stats**: Configured per type (e.g., sums, durations, rates)

**Example**:
```python
# StepCount with zero-fill
# Window with no steps but HR present → StepCount__sum = 0
# Window with no steps and no HR → StepCount__sum = NaN
```

### Wear Detection

Wear fraction computed from HeartRate point coverage:

```python
# Per HR point:
coverage_start = point_timestamp
coverage_end = point_timestamp + coverage_window_seconds

# Overlap with analysis window:
overlap = max(0, min(coverage_end, window_end) - max(coverage_start, window_start))

# Aggregate:
HR_coverage_seconds = sum(overlap_per_point)
HR_wear_fraction = HR_coverage_seconds / window_seconds
```

**Default Coverage Window**: 5 minutes (300 seconds)

**Use Case**: A single HR measurement "covers" the 5-minute period after it

### Quality Control Strategies

**Three-Level QC**:

1. **Bin-Level** (per-window):
   - Threshold: `bin_frac_min` (e.g., 0.80)
   - Scope: Watch-exclusive features
   - Action: Drop/mask windows with insufficient wear

2. **Day-Level** (per local day):
   - Threshold: `day_frac_min` (e.g., 0.60)
   - Scope: All features
   - Action: Drop/mask all windows in low-wear days

3. **Participant-Level** (per user):
   - Threshold: `overall_frac_min` (e.g., 0.50)
   - Scope: All features
   - Action: Drop/mask entire users with low overall wear

**QC Actions**:
- `"drop"`: Remove rows/users from output
- `"mask"`: Set feature values to NaN, preserve structure

**Selectors**: Control which features are affected
```python
qc_mask_selectors = {
    "bin": {"device_class": ["watch_exclusive", "either"]},
    "day": {"all": True},
    "participant": {"all": True},
}
```

### Performance Optimization

**Parallelism**:
- Per-user parallelism in interval assignment (grid windows only)
- Per-user parallelism in overlap resolution
- ThreadPoolExecutor with configurable `max_workers`

**Memory Management**:
- Intermediate frames pruned early with `early_prune_mode`
- Empty rows dropped with `prune_empty_rows`
- Chunked processing for large datasets

**Indexing**:
- Bin index computation via integer arithmetic (fast)
- Two-pointer sweep for grid assignment (O(n+m))
- Overlap join for anchored windows (necessary for overlaps)

**Tips**:
```python
# Enable parallelism
config = HKAggConfig(max_workers=8)

# Early pruning for HR-sparse data
config = HKAggConfig(
    early_prune_mode="hr_present",
    early_pruning_types=["HeartRate"],
)

# Reduce memory with selective type processing
types_of_interest = ["HeartRate", "StepCount", "SleepAnalysis"]
intermediate = aggregator.assign_observations_to_windows(
    clean, windows, types=types_of_interest
)
```

### Custom Operations

Register custom aggregation operations:

```python
def custom_op(grouped, inputs):
    col = inputs["col"]
    threshold = inputs.get("threshold", 0)
    return grouped[col].apply(lambda x: (x > threshold).sum())

aggregator.register_operation("count_above_threshold", custom_op)

# Use in layer spec
LayerRule(
    out="HeartRate__count_high",
    op="count_above_threshold",
    inputs={"col": "HeartRate__max", "threshold": 100}
)
```

### Error Handling

**Loader**:
- `FileNotFoundError`: File not found
- `ValueError`: Invalid file format, schema violations
- Detailed timezone localization errors with row indices

**Cleaner**:
- Graceful per-type error handling (logs error, passes through raw data)
- Per-user error resilience in overlap resolution

**Aggregator**:
- Schedule validation failures (strict vs lenient)
- Window overlap detection
- Missing input columns for operations

**Best Practices**:
```python
# Enable logging
import logging
logging.basicConfig(level=logging.INFO)

# Validate incrementally
try:
    raw = loader.load_file(path, user_id=123)
    clean, report = cleaner.clean(raw, return_report=True)
    print(report)  # Check cleaning metrics
    wide, agg_report, aux = aggregator.run_aggregation(clean)
except Exception as e:
    logging.error(f"Pipeline failed: {e}")
    raise
```

### User ID Contract

- `user_id` is a numeric participant key across loader, cleaner, and aggregator.
- Rows with missing/non-numeric/non-integer-like `user_id` values are dropped during normalization.
- Loader XML fallback user id is `-999` when no participant id is resolved.
- Aggregator report includes `input.dropped_invalid_user_id_non_numeric`.
- Cleaner global report includes `dropped_invalid_user_id_non_numeric`.

### Schedule/Timezone Config Matrix

- `tz_mode_l0`:
  - `utc`: builds UTC bins; schedule inference may fallback per `utc_schedule_inference_policy`.
  - `local`: requires schedule coverage and valid timezone signal.
- `tz_policy`:
  - `strict`: raises on schedule overlaps/coverage gaps.
  - `lenient`: resolves overlaps/gaps and records diagnostics.
- `output_timezone`:
  - `both`: keeps UTC + local projection columns.
  - `UTC`: drops local projection columns.
  - `local`: promotes `window_start_local/window_end_local` into `window_start/window_end` and preserves UTC in `window_start_utc/window_end_utc`.

### Schedule Troubleshooting

- `merge on string and int64 columns for key 'user_id'`:
  - Cause: inconsistent user id dtypes between source rows and windows.
  - Fix: enforce numeric `user_id` contract before aggregation.
- `schedule does not fully cover analysis interval (...)`:
  - Cause: true per-user coverage gap under strict policy.
  - Fix: verify inferred/provided schedule spans each user’s analysis range, or use `tz_policy='lenient'` for triage.
- `invalid timezone labels in schedule source`:
  - Cause: non-IANA timezone values in source data.
  - Fix: correct timezone labels or use `schedule_invalid_tz_policy='coerce_default'` for controlled fallback.

---

## Summary

The HealthKit Preprocessing Pipeline provides a production-ready framework for:

✅ **Loading**: Multi-format support with robust timezone handling  
✅ **Cleaning**: Type-specific validation with configurable rules  
✅ **Aggregation**: Flexible windowing with hierarchical rollups  
✅ **QC**: Multi-level wear-based quality control  
✅ **Performance**: Parallel processing and memory optimization  
✅ **Extensibility**: Custom handlers, operations, and type registries  

**Key Features**:
- Handles DST transitions and international travel
- Resolves overlapping observations with principled priority rules
- Computes device wear from heart rate coverage
- Supports both fixed-frequency and assessment-anchored windows
- Generates comprehensive reporting at each stage

**Recommended Workflow**:
1. Load raw data (CSV or XML)
2. Clean with type-specific handlers
3. Optionally summarize by type
4. Aggregate into time windows
5. Apply quality control
6. Optionally compute hierarchical rollups

For questions or contributions, see the main project repository.

---

**End of Documentation**

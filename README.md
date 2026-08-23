# HealthKit Preprocessing Pipeline

**Credits:** Scripts in this repository were developed over several months by **Dawson Haddox** (Clinical Psychology Ph.D. Student, Personalized Treatment Lab, University of Arizona), drawing inspiration from prior work by **Dr. Samir Akre**. The experimental HRV features were designed and implemented primarily by **Dr. Agrim Sachdeva** (Assistant Professor, Management Information Systems, University of Arizona).

## Overview

A Python pipeline that converts raw Apple HealthKit CSV exports into analysis-ready, wide-format feature matrices. The pipeline follows a three-stage architecture: **load**, **clean**, **aggregate**. It is fully configurable through dataclass-based configuration objects, so researchers can customize feature plausibility thresholds, fixed or assessment-relative windowing, granular sleep features, and output aggregates without modifying core code. Optional branches provide quality control masking, hierarchical aggregation (e.g., hourly to daily), HRV, participant-aware zero filling, and assessment-safe daily sleep features.

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [HealthKit Datastreams Supported by Default](#healthkit-datastreams-supported-by-default)
- [Stage 1 — Loading](#stage-1--loading)
- [Stage 2 — Cleaning](#stage-2--cleaning)
- [Stage 3 — Aggregation](#stage-3--aggregation)
- [End-to-End Tutorial](#end-to-end-tutorial)
- [Data Flow](#data-flow)

---

## Tech Stack

| Package | Version    | Role                                |
|---------|------------|-------------------------------------|
| Python  | >= 3.10    | Runtime                             |
| pandas  | >= 2.0     | Core data manipulation              |
| numpy   | >= 1.24    | Numeric operations                  |
| tqdm    | >= 4.66    | Progress bars for bulk loading      |
| scipy   | >= 1.10    | *Optional* — frequency-domain HRV   |

---

## Installation

```bash
# Clone the repository
git clone <repo-url> healthkit-preprocessing
cd healthkit-preprocessing # May need to configure specific path

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `scipy` is only required if you enable frequency-domain HRV features (`compute_hrv=True` and `hrv_freq_domain=True` in `HKAggConfig`). The pipeline runs without it, and these features aren't typically recommended by default due to data limitations.


---

## HealthKit Datastreams Supported by Default

The pipeline recognizes the following Apple HealthKit types out-of-the-box. Each type is routed through a cleaning handler (Stage 2) and an aggregation profile (Stage 3) that depend on its **time behavior**. Types not listed here still pass through loading and a minimal cleaning path. For more elaborate cleaning and aggregation, new types must be configured in appropriate registries.

| Type | Time behavior | Stage 2 handler | Device class | Default aggregates (Stage 3) |
|---|---|---|---|---|
| `StepCount` | interval_total | activity | either | `sum`, `duration`, `rate_per_min`, continuity indicators |
| `DistanceWalkingRunning` | interval_total | activity | either | `sum`, `duration`, `rate_per_min`, continuity indicators |
| `DistanceCycling` | interval_total | activity | either | `sum`, `duration`, `rate_per_min`, continuity indicators |
| `FlightsClimbed` | interval_total | activity | either | `sum`, `duration`, `rate_per_min`, continuity indicators |
| `ActiveEnergyBurned` | interval_total | activity | either | `sum`, `duration`, `rate_per_min`, continuity indicators |
| `BasalEnergyBurned` | interval_total | activity | either | `sum`, `duration`, `rate_per_min`, continuity indicators |
| `AppleExerciseTime` | interval_total | activity | either | `sum`, continuity indicators |
| `AppleStandTime` | interval_total | activity | either | `sum`, continuity indicators |
| `EnvironmentalAudioExposure` | interval_level | audio | either | `leq`, `energy_seconds`, `time_weighted_mean`, `max`, `duration`, `median`, `median_duration`, `count`, continuity indicators |
| `HeadphoneAudioExposure` | interval_level | audio | non_watch_exclusive | `leq`, `energy_seconds`, `time_weighted_mean`, `max`, `duration`, `median`, `median_duration`, `count`, continuity indicators |
| `HeartRate` | point | vitals | watch_exclusive | `mean`, `median`, `std`, `max`, `min`, `count` (also drives wear detection and optional HRV) |
| `HeartRateVariabilitySDNN` | point | vitals | watch_exclusive | `mean`, `median`, `std`, `max`, `min`, `count` |
| `RespiratoryRate` | point | vitals | watch_exclusive | `mean`, `median`, `std`, `max`, `min`, `count` |
| `OxygenSaturation` | point | vitals | watch_exclusive | `mean`, `median`, `std`, `max`, `min`, `count` |
| `SleepAnalysis` | categorical_interval | cat | either | 60 configurable label/union duration, count, and continuity features |
| `EnvironmentalAudioExposureEvent` | categorical_interval | cat | either | `sum`, `mean`, `count`, and continuity for `momentary_limit` |

**Time-behavior categories, at a glance:**
Assignment to aggregation windows works differently depending on the time behavior of the stream.
- **interval_total** — cumulative counts/totals over an interval (e.g., steps in a 10-minute window). Values are split proportionally across window boundaries when intervals straddle them.
- **interval_level** — a sustained level maintained over an interval (e.g., ambient dB). Aggregation is time-weighted.
- **point** — an instantaneous sample at one moment (e.g., a single heart-rate reading). Observations are assigned to whichever window contains the sample timestamp.
- **categorical_interval** — a labeled interval (e.g., a sleep stage). Observations are split at the window boundary and maintain the same label.

**Device-class meanings:**

- `watch_exclusive` — only Apple Watch sources are considered valid.
- `non_watch_exclusive` — watch sources are excluded (e.g., headphone audio, which is a phone/headphone signal).
- `either` — watch and non-watch sources both count.

---

## Stage 1 — Loading

### What it does

The loader reads raw Apple HealthKit CSV exports and normalizes them into a canonical long-form DataFrame suitable for cleaning and aggregation. A single loader instance carries one `HKLoadConfig` that governs timezone and DST behavior, and exposes three public entry points that share the same per-file processing pipeline.

At a high level, each CSV is put through these steps (in order):

1. **Read** the CSV with `pandas.read_csv` (low memory mode off) and parse `local_start` / `local_end` with `pd.to_datetime(errors="coerce")`.
2. **Merge value columns.** If `body.quantity.value` and/or `body.category.value` are present, they are combined into a single `value` column (category overrides quantity when both are populated).
3. **Normalize `type` strings.** If any `type` is all-uppercase (e.g., `"HEART_RATE"`), the whole column is converted to CamelCase; `Sdnn` is corrected to `SDNN`; for sleep rows, the `value` column is CamelCased too (e.g., `ASLEEP_CORE` → `AsleepCore`).
4. **Rename timezone column variants.** `HKTimezone` or `HKTimeZone` → `timezone`.
5. **Normalize timezone values.** Missing or non-IANA `timezone` entries are handled per `timezone_missing_policy`: filled with `default_tz`, dropped, or raised.
6. **Localize and compute UTC timestamps.** For each `timezone` group, naive `local_start` / `local_end` are localized into timezone-aware `local_start_tz` / `local_end_tz`, then converted to `utc_start` / `utc_end`. Ambiguous fall-back times and nonexistent spring-forward times are resolved per `dst_ambiguous` / `dst_nonexistent`. If pandas' `ambiguous="infer"` fails on a group, the loader falls back to the second fold (standard time) and records a diagnostic sample. This step is skipped for any timestamp column that already exists on the input.
7. **Validate schema.** The loader requires `user_id`, `local_start`, `local_end`, `timezone`, `type`, `utc_start`, and `utc_end` to be present (most of which may have been computed in step 6). Missing columns raise `ValueError`.
8. **Coerce `user_id`.** Non-numeric or non-integer IDs are dropped, and the column is cast to `int64`.
9. **Derive `device.name_id`.** If a `device.name` column is present and `device.name_id` is not, the loader maps `"Apple Watch" → 2`, `"iPhone" → 1`, all else → `0` (nullable `Int8`).

All other input columns (e.g., `device.name`, `device_source_id`, source metadata) pass through unchanged so they remain available to downstream stages.

### Scripts

| File | Role |
|------|------|
| `healthkit_loader.py` | Main loader class (`HealthKitLoader`) with three public entry points |
| `healthkit_loader_specs.py` | Configuration dataclass (`HKLoadConfig`) controlling timezone and DST behavior |
| `healthkit_loader_report_analyzer.py` | CLI that aggregates saved loader report JSON files into Markdown summaries |

### Expected input CSV format

Files must be `.csv` — the loader raises `ValueError("Unknown file format")` on any other suffix. The loader is tolerant of a few column-name variants (it renames them into the canonical form during loading), but the following **logical** columns must be resolvable for each file:

| Logical column | Accepted input names | Required? | Notes |
|---|---|---|---|
| `user_id` | `user_id` | yes | Must be coercible to an integer; non-numeric rows are dropped |
| `type` | `type` | yes | HealthKit type label. Accepts CamelCase (`HeartRate`, `StepCount`) or SHOUTY form (`HEART_RATE`); SHOUTY is auto-converted. |
| `value` | `value`, `body.quantity.value`, `body.category.value` | yes | Numeric for quantity types, string label for category types (e.g., sleep stages). Merged into `value`. |
| `local_start`, `local_end` | `local_start`, `local_end` | yes | Naive local timestamps (parseable by `pd.to_datetime`). |
| `timezone` | `timezone`, `HKTimezone`, `HKTimeZone` | yes | IANA string (e.g., `"America/Los_Angeles"`). Missing/invalid values are handled via `timezone_missing_policy`. |
| `utc_start`, `utc_end` | `utc_start`, `utc_end` | optional | If absent, computed from `local_start`/`local_end` + `timezone`. |
| `local_start_tz`, `local_end_tz` | `local_start_tz`, `local_end_tz` | optional | If absent, computed during localization. |
| `device.name` | `device.name` | optional | When present, the loader derives `device.name_id`. |

Any additional columns (e.g., `device_source_id`, source identifiers, metadata) are retained as-is.

### Public entry points

All three methods accept `return_report=True`, which returns `(DataFrame, report)` instead of a DataFrame alone.

#### `load_file(path, *, return_report=False)`

Loads a single CSV and returns the normalized long-form DataFrame. Runs the full pipeline described above and also initializes/populates a per-file report dict. Raises `FileNotFoundError` if the path is missing and `ValueError` on a non-`.csv` suffix or a missing required column.

```python
from healthkit_loader import HealthKitLoader

loader = HealthKitLoader()
df = loader.load_file("path/to/101-heart.csv")
df, report = loader.load_file("path/to/101-heart.csv", return_report=True)
```

#### `load_participant(data_folder, user_id, sensor_folders=None, return_report=False)`

Loads all HealthKit CSVs for a **single participant** from a conventional folder layout and concatenates them. Expected layout:

```text
<data_folder>/
    <sensor_folder_a>/
        <user_id>-<sensor_folder_a>.csv
    <sensor_folder_b>/
        <user_id>-<sensor_folder_b>.csv
    ...
```

For each subfolder of `data_folder` (or each entry in `sensor_folders` if provided), the loader looks for `<user_id>-<folder_name>.csv`. Missing files are recorded in the report under `missing_files` and skipped silently; present files are passed through `load_file`. Returns an empty DataFrame if no files are found.

```python
# Auto-discover subfolders under data_folder
df = loader.load_participant("/data/healthkit_exports", user_id=101)

# Restrict to a specific list of sensor folders
df = loader.load_participant(
    "/data/healthkit_exports",
    user_id=101,
    sensor_folders=["heart", "steps", "sleep"],
)
```

#### `load_bulk(root, *, patterns=("**/*.csv",), strict=False, return_report=False)`

Recursively discovers CSVs under `root` using one or more `rglob` patterns, loads each with `load_file`, and concatenates the results. When `strict=False` (default), per-file exceptions are logged and recorded under `failed_files` in the report rather than halting the run. `strict=True` re-raises the first failure.

```python
# Default: recursive glob for all CSVs
df = loader.load_bulk("/data/healthkit_exports")

# Custom patterns (still limited to CSV — non-CSV files error in load_file)
df = loader.load_bulk(
    "/data/healthkit_exports",
    patterns=("participants/*/heart/*.csv", "participants/*/steps/*.csv"),
)

# Fail fast on the first bad file
df = loader.load_bulk("/data/healthkit_exports", strict=True)

# With diagnostic report
df, report = loader.load_bulk("/data/healthkit_exports", return_report=True)
```

A tqdm progress bar is shown while files are being loaded.

### Configuration (`HKLoadConfig`)

When you create a `HealthKitLoader()` without arguments, it uses `HKLoadConfig()` with all default values. The config is validated immediately on construction. To customize, create an `HKLoadConfig` and set only the fields you want to change — all other fields keep their defaults:

```python
from healthkit_loader_specs import HKLoadConfig

# Only override what you need — everything else stays at defaults
loader = HealthKitLoader(HKLoadConfig(
    default_tz="America/New_York",
    timezone_missing_policy="drop_rows",
))
```

Instead of overriding defaults when you create the config, you can also change the default values in `healthkit_loader_specs.py`.

**All configuration fields:**

| Field | Type | Default | Options | Description |
|-------|------|---------|---------|-------------|
| `timezone_missing_policy` | str | `"fill_default"` | `"fill_default"`, `"drop_rows"`, `"error"` | What to do when a row's timezone is missing or invalid. `"fill_default"` substitutes `default_tz`; `"drop_rows"` removes the row; `"error"` raises immediately. |
| `default_tz` | str | `"America/Los_Angeles"` | Any valid IANA timezone | Fallback timezone string. Only used when `timezone_missing_policy` is `"fill_default"`. |
| `dst_ambiguous` | str | `"infer"` | `"infer"`, `"NaT"`, `"raise"` | How to handle ambiguous times during fall-back (same clock time occurs twice). `"infer"` picks the most likely offset; `"NaT"` sets the timestamp to missing; `"raise"` raises an error. If `"infer"` fails on a group, the loader falls back to the second fold (standard time) and records the event in the DST report section. |
| `dst_nonexistent` | str | `"shift_forward"` | `"shift_forward"`, `"NaT"`, `"raise"` | How to handle nonexistent times during spring-forward (clock skips ahead). `"shift_forward"` moves to the next valid time; `"NaT"` sets to missing; `"raise"` raises an error. |

### Output columns

The loader produces a long-form DataFrame containing at least these canonical columns:

| Column | Description |
|--------|-------------|
| `user_id` | Participant identifier (int64) |
| `type` | Normalized HealthKit type name (e.g., `"HeartRate"`, `"StepCount"`, `"SleepAnalysis"`) |
| `value` | Observation value — numeric for quantity types, string label for category types (e.g., `"InBed"`, `"Asleep"`) |
| `timezone` | IANA timezone string (e.g., `"America/Los_Angeles"`) |
| `local_start`, `local_end` | Naive local timestamps (datetime64, tz-stripped after localization) |
| `local_start_tz`, `local_end_tz` | Timezone-aware local timestamps (datetime64) |
| `utc_start`, `utc_end` | UTC timestamps (datetime64) |

Additional columns that are created conditionally:

| Column | When it appears |
|--------|-----------------|
| `device.name_id` | When the input CSV has a `device.name` column and does not already have `device.name_id`. Values: `2` (Apple Watch), `1` (iPhone), `0` (other); nullable `Int8`. |

Any other columns present in the input CSV (e.g., `device.name`, `device_source_id`, source metadata) pass through unchanged.

### Reports

Every loader method accepts `return_report=True`, which returns a `(DataFrame, report_dict)` tuple. Reports are nested dictionaries containing diagnostic information about the loading process.

**Single-file report** (returned by `load_file`; embedded in the `files` list of participant/bulk reports):

| Key | Contents |
|-----|----------|
| `__global__` | `path`, `file_type` (always `"csv"`), `rows_in_raw`, `rows_out`, `rows_dropped_total`, and `dropped_by_reason` (a dict keyed by reason: `timezone_missing_or_invalid`, `user_id_non_numeric`). |
| `timezone` | `missing_column_rows`, `missing_value_rows`, `invalid_value_rows`, `filled_default_rows`, `dropped_rows`, and `invalid_values_sample` (up to 5 unique bad TZ strings). |
| `dst` | `fallback_ambiguous_false_rows`, `nonexistent_detected_rows`, `nonexistent_resolved_policy`, `ambiguous_infer_fail_rows`, `fallback_used` (bool), and `fallback_samples` (list of up to 5 per-row diagnostics: `row_index`, `user_id`, `timezone`, `label`, `local_time`). |
| `user_id` | `rows_in`, `rows_dropped_non_numeric`, `sample_values` (up to 5 bad IDs). |

**Participant report** (returned by `load_participant`):

| Key | Contents |
|-----|----------|
| `__global__` | `data_folder`, `user_id`, `files_scanned`, `files_found`, `files_missing`, `rows_out`. |
| `files` | List of per-file reports (same structure as `load_file`). |
| `missing_files` | List of expected-but-absent file paths. |

**Bulk report** (returned by `load_bulk`):

| Key | Contents |
|-----|----------|
| `__global__` | `root`, `patterns`, `files_matched`, `files_loaded`, `files_failed`, `rows_out`. |
| `files` | List of per-file reports for successfully loaded files. |
| `failed_files` | List of `{"path": ..., "error_type": ..., "error": ...}` entries for files that raised (only populated when `strict=False`). |

Reports can be saved to JSON for later analysis:

```python
import json

df, report = loader.load_bulk("path/to/exports", return_report=True)

with open("load_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
```

### Report Analyzer CLI

The script `healthkit_loader_report_analyzer.py` is a command-line tool that reads saved loader report JSON files and generates Markdown summaries. It ranks files by issue severity, aggregates per-participant metrics, and builds event taxonomy tables. 

***Note**: In testing, warnings generally only arose during DST fall-back hours, where the script had to assume observations belonged to one hour over the other. By default, Python's timezone handling methods assign observations to the second hour.*

```bash
# Print summary to stdout
python healthkit_loader_report_analyzer.py load_report*.json --stdout

# Write Markdown summaries to a directory
python healthkit_loader_report_analyzer.py /path/to/reports/ --out-dir ./summaries

# Fail CI if warnings or errors are found
python healthkit_loader_report_analyzer.py reports/ --fail-on warning

# Escalate DST info events to warnings when count exceeds threshold
python healthkit_loader_report_analyzer.py reports/ --fail-on warning --dst-warning-threshold 100
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `inputs` | *(required)* | Report file(s), directory path(s), or glob(s) |
| `--pattern` | `load_report*.json` | Glob pattern used when an input is a directory |
| `--out-dir` | None | Output directory for Markdown files |
| `--stdout` | off | Print Markdown to stdout instead of writing files |
| `--max-files` | `10` | Max files listed in top/problem sections |
| `--max-participants` | `15` | Max participants listed |
| `--sample-limit` | `3` | Max sample items per metric |
| `--fail-on` | `none` | Exit non-zero if severity >= `none` \| `error` \| `warning` \| `info` |
| `--dst-warning-threshold` | None | Escalate DST info events to warning if count exceeds this value |

---

## Stage 2 — Cleaning

### What it does

The cleaner enforces data quality on the long-form observations produced by the loader. It performs a global cleaning pass (dropping rows with invalid timestamps, missing types, or non-numeric user IDs), then routes each HealthKit type to a specialized handler that applies type-appropriate cleaning steps. After per-type cleaning, the results are concatenated and sorted.

### Scripts

| File | Role |
|------|------|
| `healthkit_cleaner.py` | Main cleaner class (`HealthKitCleaner`) with the `clean()` method and per-type handlers |
| `healthkit_cleaner_specs.py` | Configuration dataclass (`HKCleanConfig`) with thresholds, operation toggles, and behavioral options |

### How to use

```python
from healthkit_cleaner import HealthKitCleaner

cleaner = HealthKitCleaner()  # uses default config
cleaned_df = cleaner.clean(df)

# With diagnostic report
cleaned_df, report = cleaner.clean(df, return_report=True)
```

### Configuration (`HKCleanConfig`)

When you create a `HealthKitCleaner()` without arguments, it uses `HKCleanConfig()` with all default values. The config is validated immediately. To customize, modify the default values in `healthkit_cleaner_specs.py`, or create an `HKCleanConfig` and set only the fields you want to change:

```python
from healthkit_cleaner_specs import HKCleanConfig

# Only override what you need
cleaner = HealthKitCleaner(HKCleanConfig(
    thresholds={
        "StepCount": {
            "min": {"limit": 0, "action": "drop"},
            "rate_per_min_max": {"limit": 300, "action": "cap"},
        },
    },
    ops={**HKCleanConfig().ops, "apply_quantile_filter": True},
    quantile_lower=0.02,            # tighter quantile bounds
    quantile_upper=0.98,
))
```

**General configuration fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `required_cols` | set | `{"user_id", "type", "local_start", "local_end", "value", "utc_start", "utc_end"}` | Columns that must be present in the input DataFrame |
| `max_workers` | int or None | `None` | Thread pool size for parallel overlap resolution. `None` auto-detects CPU count. |
| `quantile_lower` | float | `0.01` | Lower percentile bound for outlier removal (only used when `apply_quantile_filter` is enabled) |
| `quantile_upper` | float | `0.99` | Upper percentile bound for outlier removal |

#### Default plausibility thresholds

Each check has its own `limit` and `action`. A `drop` action removes the row. A `cap` action clamps a value bound or rescales an interval total to `rate limit × duration_minutes`. Value bounds run before rate checks. Rate checks with zero, missing, or invalid durations are skipped and reported.

| Type | Minimum | Maximum | Rate per minute maximum |
|------|---------|---------|-------------------------|
| HeartRate | 25, drop | 240, drop | — |
| HeartRateVariabilitySDNN | 0, drop | 1, drop | — |
| RespiratoryRate | 0.1, drop | 100, drop | — |
| OxygenSaturation | 0.50, drop | 1.0, drop | — |
| EnvironmentalAudioExposure | 0, drop | — | — |
| HeadphoneAudioExposure | 0, drop | — | — |
| StepCount | 0, drop | — | 350, cap |
| AppleExerciseTime | 0, drop | — | — |
| ActiveEnergyBurned | 0, drop | — | 30, cap |
| BasalEnergyBurned | 0, drop | — | 10, cap |
| AppleStandTime | 0, drop | — | — |
| DistanceCycling | 0, drop | — | 2000, cap |
| DistanceWalkingRunning | 0, drop | — | 750, cap |
| FlightsClimbed | 0, drop | 8000, drop | — |

To override thresholds:

```python
cleaner = HealthKitCleaner(HKCleanConfig(
    thresholds={
        "HeartRate": {
            "min": {"limit": 40, "action": "drop"},
            "max": {"limit": 180, "action": "drop"},
        },
        "StepCount": {
            "min": {"limit": 0, "action": "drop"},
            "rate_per_min_max": {"limit": 300, "action": "cap"},
        },
    },
))
```

> **Note:** When you provide a `thresholds` dict, it replaces the entire default dict. Include all types you want filtered, not just the ones you changed. If you want to make targeted changes, it may be helpful to simply modify the default values in `healthkit_cleaner_specs.py` instead.

#### Operation toggles

Each cleaning step can be independently enabled or disabled via the `ops` dict. Default values are shown below:

| Operation | Default | Description |
|-----------|---------|-------------|
| `coerce_numeric` | **on** | Convert the `value` column to numeric; required by most downstream operations |
| `convert_energy_to_kcal` | **on** | Detect and convert joule-based energy readings to kilocalories |
| `apply_theory_thresholds` | **on** | Apply configured minimum, maximum, and rate rules independently |
| `deduplicate` | **on** | Remove exact duplicate rows (same user/type/interval/value) |
| `normalize_sleep_labels` | **on** | Strip the `HKCategoryValueSleepAnalysis` prefix from sleep values, producing clean labels like `"InBed"`, `"Asleep"`, `"Awake"` |
| `apply_quantile_filter` | **off** | Drop values outside `quantile_lower`/`quantile_upper` bounds (extra per-type scan) |
| `apply_watch_device_filter` | **on** | Keep only Apple Watch observations (drops iPhone/iPad sources) |
| `combine_overlaps` | **on** | Resolves overlapping numeric intervals per user using a sweep-line algorithm. All interval endpoints are collected into atomic time segments; each segment is assigned the "winning" active interval by priority: (1) has `device_source_id` > (2) higher `device_source_id` > (3) shorter duration > (4) later row. For `interval_total` types (activity), the output value is rescaled proportionally to the winning segment's actual duration relative to the original interval's duration. For `interval_level` types (audio), the value is kept as-is. Adjacent segments from the same winning interval are coalesced into one output row. |
| `combine_overlaps_cat` | **on** | Resolves overlapping categorical intervals (e.g., sleep stages) per user using the same sweep-line boundary approach. Priority when multiple intervals cover the same atomic segment: (1) higher `device_source_id` — only when both intervals have one and they differ > (2) shorter duration — shorter interval assumed more specific > (3) later row. Each atomic segment is assigned the winning interval's label; adjacent segments with the same label are merged into one output row. |

To toggle specific operations:

```python
cleaner = HealthKitCleaner(HKCleanConfig(
    ops={
        "coerce_numeric": True,
        "convert_energy_to_kcal": True,
        "apply_theory_thresholds": True,
        "deduplicate": True,
        "normalize_sleep_labels": True,
        "apply_quantile_filter": True,        # enable quantile filtering
        "apply_watch_device_filter": False,    # keep all device sources
        "combine_overlaps": True,
        "combine_overlaps_cat": True,
    },
))
```

> **Note:** When you provide an `ops` dict, it replaces the entire default dict. Include all keys, or copy the default mapping and change selected entries as shown above.

> **Dependency constraints** (enforced by `validate()`):
> - `convert_energy_to_kcal`, `apply_theory_thresholds`, and `apply_quantile_filter` require `coerce_numeric=True`.

### Type routing and cleaning handlers

The cleaner groups rows by `type` and routes them to one of five handlers. Each handler applies a specific sequence of cleaning steps appropriate for that data category:

**Activity types** → `_handle_activity`
Types: `StepCount`, `AppleExerciseTime`, `ActiveEnergyBurned`, `BasalEnergyBurned`, `AppleStandTime`, `DistanceCycling`, `DistanceWalkingRunning`, `FlightsClimbed`

Pipeline: watch device filter → numeric coercion → energy unit conversion (Joules → kcal) → value and rate rules → overlap resolution → quantile filtering

**Vital types** → `_handle_vitals`
Types: `HeartRate`, `HeartRateVariabilitySDNN`, `RespiratoryRate`, `OxygenSaturation`

Pipeline: numeric coercion → deduplication → plausibility thresholds → quantile filtering

**Audio types** → `_handle_audio`
Types: `EnvironmentalAudioExposure`, `HeadphoneAudioExposure`

Pipeline: numeric coercion → plausibility thresholds → overlap resolution → quantile filtering

**Categorical types** → `_handle_cat`
Types: `SleepAnalysis`, `EnvironmentalAudioExposureEvent`

Pipeline: sleep label normalization → categorical overlap resolution (with priority rules for sleep stages)

**Unknown types** → `_handle_unknown`
Any type not listed above falls through to this handler.

Pipeline: numeric coercion (best-effort, does not drop on failure) → deduplication

There's also a function for creating custom handlers if desired.

### Input and output

- **Input:** Long-form DataFrame from the loader (must contain the `required_cols`).
- **Output:** Cleaned long-form DataFrame sorted by `(user_id, type, utc_start, utc_end)`, with the same column schema as the input.

### Reports

When `return_report=True`, the cleaner returns a `(DataFrame, report_dict)` tuple. The report contains:

| Key | Contents |
|-----|----------|
| `__global__` | `rows_in`, `rows_out`, `dropped_missing_or_negative_time`, `dropped_missing_type`, `dropped_missing_user_id`, `dropped_invalid_user_id_non_numeric`, `types_seen`, `types_without_handler` |
| `<TypeName>` | Per-type entry with `rows_in`, `rows_out`, `handler` name, and handler-specific counters such as `min_dropped`, `max_capped`, `rate_per_min_max_capped`, `rate_per_min_max_skipped_invalid_duration`, `deduplicated`, overlap counts, and quantile-filter counts |

---

## Stage 3 — Aggregation

### What it does

The aggregator converts cleaned long-form observations into a wide feature matrix with one row per `(user_id, window_id)`. It:

1. Validates and normalizes the input DataFrame
2. Infers a per-user timezone schedule (or uses one you provide -- recommended to have it inferred)
3. Builds fixed-width bins or one of three explicitly configured assessment lookback layouts
4. Assigns each observation to windows based on its type's time behavior (how the data overlaps with window boundaries)
5. Derives watch-wear metrics from heart rate coverage
6. Aggregates each type within its windows to produce named feature columns (e.g., `HeartRate__mean`, `StepCount__sum`)
7. Optionally computes HRV, applies participant-aware per-statistic filling, and merges daily sleep features without using records after an assessment

### Scripts

| File | Role |
|------|------|
| `healthkit_aggregator.py` | Main aggregator class (`HealthKitAggregator`) with all public methods |
| `healthkit_aggregator_specs.py` | `HKAggConfig`, `AssessmentWindowSpec`, `SleepFeatureSpec`, the type registry, hierarchical rollup specs, and the operator registry |
| `healthkit_aggregator_time.py` | *Internal* — window construction, timezone schedule building, local-time projection. Used by the aggregator automatically. |
| `healthkit_hrv.py` | *Internal* — HRV feature computation (time-domain and frequency-domain). Used when `compute_hrv=True`. |

### How to use

```python
from healthkit_aggregator import HealthKitAggregator

aggregator = HealthKitAggregator()  # uses default config
wide, report, aux = aggregator.run_aggregation(cleaned_df)
```

`run_aggregation` returns three objects:

| Return value | Description |
|--------------|-------------|
| `wide` | DataFrame with one row per `(user_id, window_id)`. Feature columns are named `<Type>__<stat>` (e.g., `HeartRate__mean`, `StepCount__sum`). Also includes `window_start`, `window_end`, `window_seconds`, and (depending on `output_timezone`) UTC and/or local timestamp columns. |
| `report` | Nested dict with sections: `input` (normalization counts), `bins` (window counts per user), `wear` (watch-on rates), `fills` (zero-fill counts by column), `qc` (dropped/masked counts), `timing`, and `config_snapshot` (full config as a dict). |
| `aux` | Dict of supporting artifacts: `windows`, `win_wear`, `day_wear`, `participant_wear`, `presence`, `schedule`, and `daily_sleep` (empty unless requested). Pass `aux` to `apply_quality_control`. |

### Configuration (`HKAggConfig`)

When you create a `HealthKitAggregator()` without arguments, it uses `HKAggConfig()` with all default values. The config is validated immediately. To customize, create an `HKAggConfig` and set only the fields you want to change:

```python
from healthkit_aggregator_specs import HKAggConfig

# Only override what you need
aggregator = HealthKitAggregator(HKAggConfig(
    bin_size="30min",       # 30-minute windows instead of 1-hour
    compute_hrv=True,       # enable HRV features
))
```

You may also simply change the default values in `healthkit_aggregator_specs.py`.

#### Time and binning

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bin_size` | str | `"1H"` | Window width as a pandas frequency string (e.g., `"1H"`, `"30min"`, `"1D"`) |
| `bin_origin` | str | `"00:00:00"` | Time-of-day that windows align to (e.g., `"00:00:00"` = midnight) |
| `tz_mode_l0` | str | `"utc"` | `"utc"` bins by UTC timestamps; `"local"` bins by local wall-clock time |
| `output_timezone` | str | `"both"` | Which timestamps appear in the output: `"UTC"`, `"local"`, or `"both"` |

#### Assessment lookback windows

Assessment aggregation requires an `assessments` DataFrame with unique, nonmissing `(user_id, assessment_id)` pairs and a UTC-coercible `anchor_time`, plus one `AssessmentWindowSpec`. The spec rejects fields that are missing or irrelevant for its mode.

| Mode | Required fields | Exact behavior |
|------|-----------------|----------------|
| `whole_lookback` | `duration_before` | One window from `anchor_time - duration_before` through the assessment. |
| `split_lookback` | `duration_before`, `n_bins` | Divides the full assessment-relative lookback into `n_bins` equal windows. |
| `aligned_grid` | `n_bins`, `bin_size`; optional `bin_origin`, `tz_mode` | Selects the last `n_bins` canonical bins and truncates only the terminal bin when the assessment is off-grid. An assessment exactly on a boundary receives `n_bins` complete preceding bins. |

Assessment windows carry `assessment_id`, `anchor_time`, and zero-based `bin_in_lookback`. `window_id` is an independent internal key, and `window_seconds` records the actual duration of a shortened terminal bin. Assessment windows do not use `bin_index`.

```python
import pandas as pd

from healthkit_aggregator import HealthKitAggregator
from healthkit_aggregator_specs import AssessmentWindowSpec

assessments = pd.DataFrame({
    "user_id": [101],
    "assessment_id": ["ema-001"],
    "anchor_time": ["2025-01-01T17:36:00Z"],
})

window_spec = AssessmentWindowSpec(
    mode="aligned_grid",
    n_bins=4,
    bin_size="1H",
)

wide, report, aux = HealthKitAggregator().run_aggregation(
    cleaned_df,
    assessments=assessments,
    assessment_window=window_spec,
    include_daily_sleep=True,
)
```

With a UTC hourly grid, this example produces 14:00–15:00, 15:00–16:00, 16:00–17:00, and 17:00–17:36. `aligned_grid` is generally the most convenient assessment mode for sequence models because positions represent canonical clock intervals while `window_seconds` preserves the terminal-bin exposure.

#### Wear detection

Watch wear is derived from heart rate coverage: if HR data covers at least `bin_frac_min` of a window, that window is marked as "watch on."

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hr_type` | str | `"HeartRate"` | HealthKit type used to detect watch wear |
| `hr_coverage_window_seconds` | int | `300` | Sliding window size (seconds) for HR coverage calculation |
| `bin_frac_min` | float or None | `0.80` | Min fraction of a window with HR data to count as "watch on" (0.0–1.0). `None` to disable. |
| `day_frac_min` | float or None | `None` | Min fraction of a day's windows with watch wear for the day to be valid. `None` to disable. |
| `overall_frac_min` | float or None | `None` | Min overall participant wear fraction. `None` to disable. |

#### Quality control

QC can drop or mask rows at three levels. All three are disabled by default. Enable them and call `apply_quality_control(wide, aux)` after aggregation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_bin_qc` | bool | `False` | Apply QC at the window (bin) level |
| `enable_day_qc` | bool | `False` | Apply QC at the day level |
| `enable_participant_qc` | bool | `False` | Apply QC at the participant level |
| `qc_action` | str | `"drop"` | `"drop"` removes rows failing QC; `"mask"` sets feature values to NaN |

```python
# Apply QC after aggregation
wide_qc, report_qc = aggregator.apply_quality_control(wide, aux)
```

#### Pruning

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `early_prune_mode` | str or None | `"any_data"` | Drop empty windows early to save memory. `"any_data"` keeps windows with any observations; `"wear_seconds_gt0"` keeps windows with watch wear > 0; `"any_non_watch_data"` keeps windows with non-watch data; `"none"` / `None` keeps all windows. |
| `prune_empty_rows` | bool | `True` | Remove rows with all feature values missing after QC/masking |

#### Zero fill

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `apply_zero_fill` | bool | `False` | Enables the per-statistic `zero_fill_rules` in the type registry. No granular values are filled by default. |
| `apply_sleep_qc_fill_nan` | bool | `False` | When enabled, apply_sleep_quality_control() will fill missing daily sleep stage/category aggregates with zeros when sleep duration is present |

Each granular rule has an independent gate, fill value, and availability policy:

```python
step_count_registry = {
    "StepCount": {
        "time_behavior": "interval_total",
        "aggregates": ("sum", "duration", "rate_per_min"),
        "device_class": "either",
        "zero_fill_rules": {
            "sum": {
                "gate": ("watch_on", {}),
                "fill_value": 0.0,
                "availability": "participant_evidence",
            },
        },
    }
}
```

`participant_evidence` requires evidence in the complete validated input, before window pruning, that the participant produces the type. Fine-grained sleep labels require the exact label; sleep unions require any constituent label. `assume_available` explicitly permits filling without participant evidence. The gate must also pass. This keeps a stream that the participant's device never emits as NaN while allowing explicit zero semantics for streams known to be available.


#### HRV

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `compute_hrv` | bool | `False` | Compute heart rate variability features from HeartRate data |
| `hrv_time_domain` | bool | `True` | Compute time-domain HRV metrics (SDNN, RMSSD, pNN50, pNN20, TriangularIndex) |
| `hrv_freq_domain` | bool | `True` | Compute frequency-domain HRV metrics (VLF, LF, HF power, LF/HF ratio). Requires `scipy`. |
| `hrv_min_samples` | int | `10` | Min HeartRate samples per window for time-domain HRV |
| `hrv_freq_min_samples` | int | `30` | Min HeartRate samples per window for frequency-domain HRV |
| `hrv_interpolation_fs` | float | `4.0` | Interpolation sampling rate (Hz) for frequency-domain HRV |

#### Parallelism

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_workers` | int or None | `None` | Thread pool size for parallel observation assignment. `None` auto-detects CPU count. |

### Type registry

The type registry (defined in `healthkit_aggregator_specs.py` via `build_default_type_registry()`) tells the aggregator how to handle each HealthKit data type. Each entry specifies the type's time behavior and which statistics to compute. Output columns are named `<Type>__<stat>`.

To modify the registry, call `build_default_type_registry()`, edit the dict, and pass it via `HKAggConfig(type_registry=...)`.

#### Interval total types

Values represent totals accumulated across an interval (e.g., total steps). The aggregator sums values proportionally when intervals overlap window boundaries.

| Type | Aggregates | Device class |
|------|------------|--------------|
| StepCount | `sum`, `duration`, `rate_per_min`, continuity indicators | either |
| DistanceCycling | `sum`, `duration`, `rate_per_min`, continuity indicators | either |
| DistanceWalkingRunning | `sum`, `duration`, `rate_per_min`, continuity indicators | either |
| FlightsClimbed | `sum`, `duration`, `rate_per_min`, continuity indicators | either |
| ActiveEnergyBurned | `sum`, `duration`, `rate_per_min`, continuity indicators | either |
| BasalEnergyBurned | `sum`, `duration`, `rate_per_min`, continuity indicators | either |
| AppleExerciseTime | `sum`, continuity indicators | either |
| AppleStandTime | `sum`, continuity indicators | either |

#### Interval level types

Values represent a sustained level across an interval (e.g., dB exposure). The aggregator computes time-weighted statistics.

| Type | Aggregates | Special aggregates | Device class |
|------|------------|-------------------|--------------|
| EnvironmentalAudioExposure | `leq`, `energy_seconds`, `time_weighted_mean`, `max`, `duration`, `median`, `median_duration`, `count`, continuity indicators | `p95` (95th percentile), `time_above_85` (seconds above 85 dB) | either |
| HeadphoneAudioExposure | `leq`, `energy_seconds`, `time_weighted_mean`, `max`, `duration`, `median`, `median_duration`, `count`, continuity indicators | `p95`, `time_above_85` | non_watch_exclusive |

Special aggregates are defined in `special_aggregates`; add their names to the type's `aggregates` tuple when you want them emitted.

#### Point types

Instantaneous measurements (e.g., a single heart rate reading at a point in time).

| Type | Aggregates | Special aggregates | Device class |
|------|------------|-------------------|--------------|
| HeartRate | `mean`, `median`, `std`, `max`, `min`, `count` | — | watch_exclusive |
| HeartRateVariabilitySDNN | `mean`, `median`, `std`, `max`, `min`, `count` | `p75` (75th percentile) | watch_exclusive |
| RespiratoryRate | `mean`, `median`, `std`, `max`, `min`, `count` | — | watch_exclusive |
| OxygenSaturation | `mean`, `median`, `std`, `max`, `min`, `count` | `pct_below_92` (% below 92) | watch_exclusive |

#### Categorical interval types

Labeled time intervals (e.g., sleep stages). The continuity indicators are 1 only when the original interval extends strictly across the relevant boundary. An interval that starts or ends exactly on a boundary has indicator 0.

| Type | Aggregates (per label) | Device class |
|------|----------------------|--------------|
| SleepAnalysis | Configurable `sum`, `mean`, `count`, `continues_from_previous_bin`, and `continues_into_next_bin` for nine exact labels and three unions | either |
| EnvironmentalAudioExposureEvent | Duration, count, and continuity for `momentary_limit` | either |

`SleepFeatureSpec()` emits all 60 granular sleep features by default: five statistics for each of nine normalized labels and five statistics for each of the `in_bed`, `asleep`, and `awake` unions. `AwakeUnspecified` belongs to both the asleep and awake unions. To request a smaller deterministic schema:

```python
from healthkit_aggregator_specs import HKAggConfig, SleepFeatureSpec, build_default_type_registry

sleep_features = SleepFeatureSpec(
    labels=("asleep_core", "asleep_deep", "asleep_rem"),
    unions=("asleep",),
    statistics=("sum", "count", "continues_into_next_bin"),
)
config = HKAggConfig(type_registry=build_default_type_registry(sleep_features))
```

### Hierarchical aggregation

To roll up L0 windows into coarser time periods (e.g., hourly → daily), define `LayerSpec` rules:

```python
from healthkit_aggregator_specs import LayerSpec, LayerRule

day_spec = LayerSpec(
    name="daily",               # layer name (used as key in output dict)
    freq="1D",                  # aggregation frequency (pandas string)
    origin="00:00:00",          # time-of-day alignment
    tz_mode="local",            # "utc" or "local"
    strict=False,               # False = skip missing columns; True = error
    rules=[
        LayerRule(out="HeartRate__mean_day", op="mean",
                  inputs={"col": "HeartRate__mean"}, strict=False),
        LayerRule(out="StepCount__sum_day", op="sum",
                  inputs={"col": "StepCount__sum"}, strict=False),
    ],
)

layers, layer_report = aggregator.run_hierarchical_aggregation(wide_qc, [day_spec])
daily_df = layers["daily"]
```

Each `LayerRule` specifies an output column name (`out`), an operator (`op`), and the input column(s) (`inputs`). Available operators from the default registry include: `sum`, `mean`, `weighted_mean`, `min`, `max`, `median`, `stdev`, `count_nonnull`.

### Daily sleep features

A daily sleep row is separate from the 60 granular sleep features. Granular features describe one aggregation window and split intervals at window boundaries. Daily sleep uses complete source records, assigns each record wholly to the 15:00-anchored day containing its `local_start`, and never splits a record at 15:00.

Set `include_daily_sleep=True` to calculate and merge daily features into the returned `wide` matrix:

```python
wide, report, aux = aggregator.run_aggregation(
    cleaned_df,
    include_daily_sleep=True,
)

daily_sleep = aux["daily_sleep"]
```

The merge key is derived from each granular row's local window start:

```python
sleep_date = (window_start_local + pd.Timedelta(hours=9)).date()
```

Daily rows merge on `(user_id, sleep_date)` for fixed grids and `(user_id, assessment_id, sleep_date)` for assessment windows. The same daily values repeat across all granular rows whose starts belong to that 15:00–15:00 sleep day. If no daily sleep row exists, the merged daily columns remain NaN.

For an assessment at 17:36, a 14:00–15:00 row receives the sleep day ending at 15:00. Rows beginning at 15:00, 16:00, and 17:00 receive the assessment-truncated sleep day beginning at 15:00. `sleep_day_elapsed_fraction` distinguishes that shorter current day from complete historical days.

For assessment runs, daily features are calculated independently for each assessment. Sleep intervals whose `utc_end` is after the assessment are excluded in full; they are not shortened. Point samples after the assessment are also excluded. For fixed-grid runs, the supplied data is treated as complete for its daily rows.

`InBed` coverage is the union of all sleep-analysis labels (`InBed`, any `Asleep*`, any `Awake*`, `CategoryValueUnknown`). The asleep union contains `Asleep`, `AsleepUnspecified`, `AsleepCore`, `AsleepDeep`, `AsleepREM`, `CategoryValueUnknown`, and `AwakeUnspecified`. The awake union contains `Awake` and `AwakeUnspecified`.

The direct method remains available when a separate daily table is useful:

```python
sleep_df = aggregator.compute_daily_sleep_features(
    cleaned_df,
    assessments=assessments,  # omit for complete fixed-grid days
    schedule=timezone_schedule,
    qc=True,
)
```

Output has one row per `(user_id, sleep_date)` or `(user_id, assessment_id, sleep_date)`.

**Identifiers**

| Column | Description |
|---|---|
| `user_id` | Participant identifier carried through from the input. |
| `assessment_id` | Present for assessment-specific daily rows. |
| `sleep_date` | Local calendar date on which the 15:00-anchored sleep day ends. |

**Durations and timing (hours)**

| Column | Description |
|---|---|
| `sleep_bedrest_duration_day` | Total time in bed — sum of durations of all in-bed-category segments on the day. |
| `sleep_sleep_duration_day` | Total sleep time (TST) — sum of durations of all asleep-category segments. |
| `sleep_awake_duration_day` | Total time labeled `Awake`/`AwakeUnspecified` during the in-bed window. |
| `sleep_wake_after_sleep_onset_day` | Sum of complete awake-category records starting at or after first sleep onset and ending no later than the last sleep offset. |
| `sleep_sleep_onset_latency_day` | Time from bedrest onset to first sleep onset. |
| `sleep_first_bedrest_start_hours_day` | Clock hour of the first in-bed record's start, expressed relative to the prior local midnight; values above 24 cross midnight. |
| `sleep_last_bedrest_end_hours_day` | Clock hour of the last in-bed record's end. |
| `sleep_first_onset_hours_day` | Clock hour of the first asleep record's start. |
| `sleep_last_offset_hours_day` | Clock hour of the last asleep record's end. |
| `sleep_day_elapsed_fraction` | 1 for a complete historical sleep day; otherwise the fraction of the assessment-containing 15:00 day elapsed at the assessment. |

**Quality and context**

| Column | Description |
|---|---|
| `sleep_efficiency_day` | `sleep_duration / bedrest_duration`, clipped to [0, 1]; 0 is coerced to NaN. |
| `sleep_sleep_hr_median_day` | Median `HeartRate` among point samples that fall inside at least one actual asleep-category record. BPM. |
| `sleep_sleep_hrv_median_day` | Median `HeartRateVariabilitySDNN` among point samples inside actual asleep-category records. |
| `sleep_bedrest_audio_median_day` | Median `EnvironmentalAudioExposure` among samples inside actual in-bed-category records. No LEQ substitution is applied. |

**Per-stage statistics**
For each sleep-analysis label `<stage>` observed anywhere in the supplied dataset, three columns are emitted. The daily stage schema is therefore data-dependent, while the core daily columns are fixed.

| Column pattern | Description |
|---|---|
| `sleep_<stage>_sum_day` | Total duration of segments with this label (hours). |
| `sleep_<stage>_mean_day` | Mean segment duration for this label (hours). |
| `sleep_<stage>_count_day` | Number of segments with this label. |

Stage columns only appear when that label is present in the input, so the exact column set varies by device and participant (e.g., watchOS ≥ 9 devices emit `AsleepCore`/`AsleepDeep`/`AsleepREM`; older sources emit only `Asleep`).

#### QC behavior (when `qc=True`)

`apply_sleep_quality_control` runs after feature assembly and applies these adjustments in place:

- `sleep_efficiency_day`: 0 → NaN; values > 1 clipped to 1.
- `sleep_sleep_duration_day`, `sleep_bedrest_duration_day`: 0 → NaN.
- `sleep_Awake_sum_day`: values > 20 h → NaN.
- If `HKAggConfig.apply_sleep_qc_fill_nan` is set, rows that have a non-NaN `sleep_sleep_duration_day` get missing `InBed`/`Asleep`/`Awake`/`CategoryValueUnknown` stage stats (sum/mean/count) filled with 0.

### Feature manifest and model preparation

`get_feature_manifest()` returns the configured granular feature schema in deterministic order. After a run with daily sleep enabled, it also records the core daily features and the stage columns observed in that run.

```python
manifest = aggregator.get_feature_manifest()
```

Each row includes the output name, cadence, source type, aggregation, label or union, units, value type, device class, and zero-fill semantics. Configured granular columns are retained in `wide` even when an input batch has no observations for that feature, so batching does not change granular schema or order.

For deep learning, `aligned_grid` usually provides the clearest sequence representation. Keep `window_seconds` as an input or mask so the model can identify the shortened terminal bin, and use `sleep_day_elapsed_fraction` when daily sleep is present. Fit normalization, imputation, learned missingness handling, and any outcome-informed feature selection on training data only, then apply those fitted transformations unchanged to validation and test data.

### Reports

The `report` dict from `run_aggregation` contains:

| Section | Contents |
|---------|----------|
| `input` | Normalization counts: ambiguous local times inferred, nonexistent times shifted, invalid user IDs dropped, split expansion factor |
| `bins` | `bin_seconds_nominal`, `bin_origin_utc`, `counts_per_user` (dict of user_id → window count), `non_empty_bin_counts_by_type`, `early_pruned_bins` |
| `wear` | `bin_seconds_worn_total`, `bin_watch_on_rate`, `day_valid_rate`, `participant_overall_frac_mean`, `day_valid_counts`, `participant_valid_counts` |
| `fills` | `counts_by_column` — number of values zero-filled per feature column |
| `daily_sleep` | Daily row count and the exact daily feature columns produced when `include_daily_sleep=True` |
| `qc` | `dropped_counts` and `masked_counts` at bin/day/participant levels, `pruned_empty_rows` |
| `config_snapshot` | Full `HKAggConfig` as a dict (for reproducibility) |

---

## End-to-End Tutorial

### Basic workflow

A complete pipeline from raw exports to granular and hierarchical feature CSVs:

```python
from healthkit_loader import HealthKitLoader
from healthkit_cleaner import HealthKitCleaner
from healthkit_aggregator import HealthKitAggregator
from healthkit_aggregator_specs import HKAggConfig, LayerSpec, LayerRule

# ── 1. Load ──────────────────────────────────────────────────
loader = HealthKitLoader()
raw_df, load_report = loader.load_bulk(
    "/path/to/healthkit_exports",
    return_report=True,
)
print(f"Loaded {len(raw_df)} rows")

# ── 2. Clean ─────────────────────────────────────────────────
cleaner = HealthKitCleaner()
clean_df, clean_report = cleaner.clean(raw_df, return_report=True)
print(f"Cleaned: {len(clean_df)} rows remaining")

# ── 3. Aggregate (1-hour windows + merged daily sleep) ──────
aggregator = HealthKitAggregator(HKAggConfig(bin_size="1H"))
wide, agg_report, aux = aggregator.run_aggregation(
    clean_df,
    include_daily_sleep=True,
)

# ── 4. Quality control ───────────────────────────────────────
wide_qc, qc_report = aggregator.apply_quality_control(wide, aux)

# ── 5. Roll up hourly → daily ────────────────────────────────
day_spec = LayerSpec(
    name="daily", freq="1D", tz_mode="local", strict=False,
    rules=[
        LayerRule(out="HeartRate__mean_day", op="mean",
                  inputs={"col": "HeartRate__mean"}, strict=False),
        LayerRule(out="StepCount__sum_day", op="sum",
                  inputs={"col": "StepCount__sum"}, strict=False),
        LayerRule(out="ActiveEnergyBurned__sum_day", op="sum",
                  inputs={"col": "ActiveEnergyBurned__sum"}, strict=False),
    ],
)
layers, _ = aggregator.run_hierarchical_aggregation(wide_qc, [day_spec])
daily = layers["daily"]

# ── 6. Export ─────────────────────────────────────────────────
wide_qc.to_csv("features_hourly_with_daily_sleep.csv", index=False)
daily.to_csv("features_daily.csv", index=False)
aggregator.get_feature_manifest().to_csv("feature_manifest.csv", index=False)
```

### Inspecting reports

Reports are plain dicts that you can print, inspect, or save:

```python
import json

# Save all three reports to JSON
for name, rep in [("load", load_report), ("clean", clean_report), ("agg", agg_report)]:
    with open(f"{name}_report.json", "w") as f:
        json.dump(rep, f, indent=2, default=str)

# Quick summary from the cleaner report
print(f"Rows in:  {clean_report['__global__']['rows_in']}")
print(f"Rows out: {clean_report['__global__']['rows_out']}")

# Check wear metrics from the aggregator report
print(f"Watch-on rate: {agg_report['wear']['bin_watch_on_rate']:.1%}")
print(f"Day valid rate: {agg_report['wear']['day_valid_rate']:.1%}")
```

---

## Data Flow

```text
Apple HealthKit CSV exports
              │
              ▼
┌─────────────────────────────────┐
│  HealthKitLoader                │  healthkit_loader.py
│  HKLoadConfig                   │  healthkit_loader_specs.py
│                                 │
│  • Parse CSV files              │
│  • Normalize types & values     │
│  • Resolve timezones & DST      │
│  • Produce UTC + local times    │
└────────────┬────────────────────┘
             │
             ▼
  Canonical long-form observations
  [user_id, type, value, utc_start, utc_end, local_start, local_end, timezone]
             │
             ▼
┌─────────────────────────────────┐
│  HealthKitCleaner               │  healthkit_cleaner.py
│  HKCleanConfig                  │  healthkit_cleaner_specs.py
│                                 │
│  • Global validity filters      │
│  • Per-type cleaning handlers   │
│  • Plausibility thresholds      │
│  • Overlap resolution           │
└────────────┬────────────────────┘
             │
             ▼
  Cleaned long-form observations
             │
             ▼
┌─────────────────────────────────┐
│  HealthKitAggregator            │  healthkit_aggregator.py
│  HKAggConfig                    │  healthkit_aggregator_specs.py
│                                 │
│  • Build time windows           │  (healthkit_aggregator_time.py)
│  • Assign observations          │
│  • Compute per-type features    │
│  • Derive wear metrics          │
│  • Optional HRV features        │  (healthkit_hrv.py)
│  • Optional zero-fill           │
│  • Optional daily sleep merge   │
└────────────┬────────────────────┘
             │
             ▼
  Wide feature matrix
  [user_id, window_id, HeartRate__mean, StepCount__sum, ...]
             │
    ┌────────┴───────────────────┐
    ▼                            ▼
  QC Pass                 Hierarchical Rollup
  (drop / mask)           (hourly → daily)
                         LayerSpec / LayerRule
```

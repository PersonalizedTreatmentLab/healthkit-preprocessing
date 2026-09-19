"""
User configuration for HealthKit aggregation (Stage 3 of the pipeline).

HOW TO CUSTOMIZE THIS STAGE
- Edit HKAggConfig to control windows, wear detection, quality control,
  zero filling, pruning, HRV features, and parallelism.
- Edit build_default_type_registry() to select HealthKit types and the
  output features constructed for each type.
- Import and instantiate SleepFeatureSpec, AssessmentWindowSpec, LayerRule,
  and LayerSpec when supplying specialized configurations in pipeline code;
  do not edit their class definitions for a particular analysis.
- Edit the custom-operation section only when adding a new hierarchical
  aggregation operator.
- Do not edit the internal validation and helper functions unless you are
  changing how aggregation works.

NOTES
- The current aggregation code does not reproduce Samir's time-domain and
  circadian features. Generate those separately and merge them with these
  features, or extend the pipeline implementation.
- Plausibility thresholds are applied during cleaning. Aggregated outputs may
  still require study-specific review, especially for types without thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

# =============================================================================
# USER-EDITABLE DEFAULTS — EDIT VALUES IN THIS SECTION
# =============================================================================
# Edit HKAggConfig to change pipeline defaults. Edit the entries in
# build_default_type_registry() to change default types and output features.

@dataclass
class HKAggConfig:
    """Settings that users can change for the HealthKit aggregation stage."""

    # -------------------------------------------------------------------------
    # Time and binning
    # -------------------------------------------------------------------------
    # Change these values to control the width, alignment, and displayed
    # timezone of the output windows.
    bin_size: str = "1H"               # Window width as pandas frequency string (e.g."1H", "30min", "1D")
    bin_origin: str = "00:00:00"       # Time-of-day that windows align to (e.g., "00:00:00" = midnight)
    tz_mode_l0: Literal["utc", "local"] = "utc"  # "utc" bins by UTC timestamps; "local" bins by local time
    output_timezone: Literal["UTC", "local", "both"] = "both"  # Which timestamps appear in output columns

    # -------------------------------------------------------------------------
    # Timezone schedule inference
    # -------------------------------------------------------------------------
    tz_policy: Literal["strict", "lenient"] = "strict"
    utc_schedule_inference_policy: Literal["skip", "infer_fallback", "infer_fail"] = "infer_fail"
    schedule_point_policy: Literal["include_epsilon", "exclude"] = "include_epsilon"
    schedule_point_epsilon: str = "1ns"
    schedule_invalid_tz_policy: Literal["strict", "coerce_default"] = "strict"

    # -------------------------------------------------------------------------
    # Wear detection
    # -------------------------------------------------------------------------
    hr_type: str = "HeartRate"                  # HealthKit type used to detect watch wear
    hr_coverage_window_seconds: int = 300       # Sliding window size (seconds) for heart rate coverage
    bin_frac_min: Optional[float] = 0.80        # Min fraction of window with HR data to count as "watch on" (0.0-1.0; None to disable)
    day_frac_min: Optional[float] = None        # Min fraction of day with watch wear (None to disable)
    overall_frac_min: Optional[float] = None    # Min overall participant wear fraction (None to disable)

    # -------------------------------------------------------------------------
    # Quality control
    # -------------------------------------------------------------------------.
    enable_bin_qc: bool = False                 # Apply QC at the window (bin) level
    enable_day_qc: bool = False                 # Apply QC at the day level
    enable_participant_qc: bool = False          # Apply QC at the participant level
    
    qc_action: Literal["drop", "mask"] = "drop"  # "drop" removes rows; "mask" sets feature values to NaN
    qc_by_layer: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Per-layer early-pruning rules
    
    # Select which feature groups are affected when qc_action="mask".
    qc_mask_selectors: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "bin": {"device_class": ["watch_exclusive", "either"]},
        "day": {"all": True},
        "participant": {"all": True},})

    # -------------------------------------------------------------------------
    # Pruning
    # -------------------------------------------------------------------------
    # Change early_prune_mode to control which empty windows are removed early.
    #   "any_data"            — keep window if any type has data; when using this method, early pruning has no effect on the final output, but it can reduce memory usage and speed up processing.
    #   "wear_seconds_gt0"    — keep window if watch wear > 0
    #   "any_non_watch_data"  — keep window if any non-watch type has data
    #   "none" / None         — keep all windows
    early_prune_mode: Optional[Literal["none", "wear_seconds_gt0", "any_data", "any_non_watch_data"]] = "any_data"
    early_pruning_types: Optional[List[str]] = None  # Restrict pruning check to specific types (None = all)
    prune_empty_rows: bool = True

    # -------------------------------------------------------------------------
    # Type registry
    # -------------------------------------------------------------------------
    # To edit, see build_default_type_registry() below.
    type_registry: Dict[str, Dict[str, Any]] = field(default_factory=lambda: build_default_type_registry())

    # -------------------------------------------------------------------------
    # Zero fill
    # -------------------------------------------------------------------------
    # Set apply_zero_fill=True to apply each configured statistic's gate and
    # availability rule.
    apply_zero_fill: bool = True
    apply_sleep_qc_fill_nan: bool = False        # Fill missing daily sleep stage/category aggregates with zeros when sleep duration is present

    # -------------------------------------------------------------------------
    # Heart rate variability
    # -------------------------------------------------------------------------
    # Set compute_hrv=True to compute additional HRV features from HeartRate.
    compute_hrv: bool = False
    hrv_time_domain: bool = False        # Compute time-domain HRV metrics (SDNN, RMSSD, etc.)
    hrv_freq_domain: bool = False        # Compute frequency-domain HRV metrics (LF, HF, etc.)
    hrv_min_samples: int = 10           # Min HeartRate samples per window for time-domain HRV
    hrv_freq_min_samples: int = 30      # Min HeartRate samples per window for frequency-domain HRV
    hrv_interpolation_fs: float = 4.0   # Interpolation sampling rate (Hz) for frequency-domain HRV

    # -------------------------------------------------------------------------
    # Parallelism
    # -------------------------------------------------------------------------
    # Thread pool size for interval assignment parallelization.
    # None = auto-detect CPU count via os.cpu_count() or 1.
    max_workers: int | None = None

    def validate(self):
        """Validate the configured values."""
        _validate_agg_config(self)

# -----------------------------------------------------------------------------
# HealthKit types and output features
# -----------------------------------------------------------------------------
# Edit the entries in build_default_type_registry() to change which HealthKit
# types are aggregated and which output statistics are constructed.
#
# To add a type:
# 1. Copy an entry with the same time_behavior.
# 2. Replace the HealthKit type name.
# 3. Select the desired aggregates and device_class.


def build_default_type_registry(
    sleep_features: Optional[SleepFeatureSpec] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build the default HealthKit type registry.

    The type registry tells the aggregator how to handle each HealthKit data type.
    Each entry produces output columns named ``<type>__<stat>`` (e.g., ``HeartRate__mean``).

    To add a new type: copy an existing entry with the same time_behavior,
    change the key to your HealthKit type name, and adjust the aggregates tuple.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of HealthKit type name -> metadata dict.
    """
    # Each type entry has these fields:
    #
    #   time_behavior      — How the pipeline treats this data temporally.
    #                        One of: "interval_total", "interval_level", "point", "categorical_interval"
    #   aggregates         — Tuple of stat names to compute. Each produces a column: <type>__<stat>
    #   device_class       — Which device sources are valid: "either", "watch_exclusive", "non_watch_exclusive"
    #   zero_fill_rules    — Per-stat rules with gate, fill value, and participant
    #                        availability policy (optional; omit to leave missing)
    #   special_aggregates — Dict of custom aggregation specs (optional). See interval_level examples.

    reg: Dict[str, Dict[str, Any]] = {}
    sleep_features = sleep_features or SleepFeatureSpec()
    sleep_aggregates = sleep_features.aggregate_names()

    # -----------------------------------------------------------------------
    # INTERVAL TOTAL — Cumulative measures recorded over time intervals
    # -----------------------------------------------------------------------
    # Values represent totals accumulated across an interval (e.g., total steps, total kcal).
    # The pipeline sums values proportionally when intervals overlap window boundaries.
    #
    # Available aggregates: sum, duration, rate_per_min
    #   sum          — total value within the window
    #   duration     — total seconds of observation within the window
    #   rate_per_min — sum / (duration in minutes)
    #
    # To add a new interval_total type, copy any entry below, change the key, and adjust aggregates.

    reg["StepCount"] = {
        "time_behavior": "interval_total",
        "aggregates": ("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin"),
        "device_class": "either",
        "zero_fill_rules": _participant_zero_rules(("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin")),
    }
    reg["DistanceCycling"] = {
        "time_behavior": "interval_total",
        "aggregates": ("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin"),
        "device_class": "either",
        "zero_fill_rules": _participant_zero_rules(("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin")),
    }
    reg["DistanceWalkingRunning"] = {
        "time_behavior": "interval_total",
        "aggregates": ("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin"),
        "device_class": "either",
        "zero_fill_rules": _participant_zero_rules(("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin")),
    }
    reg["FlightsClimbed"] = {
        "time_behavior": "interval_total",
        "aggregates": ("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin"),
        "device_class": "either",
        "zero_fill_rules": _participant_zero_rules(("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin")),
    }
    reg["ActiveEnergyBurned"] = {
        "time_behavior": "interval_total",
        "aggregates": ("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin"),
        "device_class": "either",
        "zero_fill_rules": _participant_zero_rules(("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin")),
    }
    reg["BasalEnergyBurned"] = {
        "time_behavior": "interval_total",
        "aggregates": ("sum", "duration", "rate_per_min", "continues_from_previous_bin", "continues_into_next_bin"),
        "device_class": "either",
        "zero_fill_rules": _participant_zero_rules(("duration", "continues_from_previous_bin", "continues_into_next_bin")),
    }

    # interval_total — duration-only types (no rate needed)
    reg["AppleExerciseTime"] = {
        "time_behavior": "interval_total",
        "aggregates": ("sum", "continues_from_previous_bin", "continues_into_next_bin"),
        "device_class": "either",
        "zero_fill_rules": _participant_zero_rules(("sum", "continues_from_previous_bin", "continues_into_next_bin")),
    }
    reg["AppleStandTime"] = {
        "time_behavior": "interval_total",
        "aggregates": ("sum", "continues_from_previous_bin", "continues_into_next_bin"),
        "device_class": "either",
        "zero_fill_rules": _participant_zero_rules(("sum", "continues_from_previous_bin", "continues_into_next_bin")),
    }

    # -----------------------------------------------------------------------
    # INTERVAL LEVEL — Continuous level measurements over time intervals
    # -----------------------------------------------------------------------
    # Values represent a sustained level across an interval (e.g., audio exposure in dB).
    # Aggregation weights values by interval duration.
    #
    # Available aggregates: leq, energy_seconds, time_weighted_mean, max, duration,
    #                       median, median_duration, count
    # Special aggregates (via special_aggregates dict):
    #   weighted_quantile — duration-weighted percentile (params: op, q)
    #   time_above        — seconds above a threshold (params: op, threshold)

    reg["EnvironmentalAudioExposure"] = {
        "time_behavior": "interval_level",
        "aggregates": ("leq", "energy_seconds", "time_weighted_mean", "max", "duration",
                       "median", "median_duration", "count", "p95", "time_above_85",
                       "continues_from_previous_bin", "continues_into_next_bin"),
        "device_class": "either",
        "special_aggregates": {
            "p95": {"op": "weighted_quantile", "q": 0.95},
            "time_above_85": {"op": "time_above", "threshold": 85.0},
        },
    }
    reg["HeadphoneAudioExposure"] = {
        "time_behavior": "interval_level",
        "aggregates": ("leq", "energy_seconds", "time_weighted_mean", "max", "duration",
                       "median", "median_duration", "count", "p95", "time_above_85",
                       "continues_from_previous_bin", "continues_into_next_bin"),
        "device_class": "non_watch_exclusive",
        "special_aggregates": {
            "p95": {"op": "weighted_quantile", "q": 0.95},
            "time_above_85": {"op": "time_above", "threshold": 85.0},
        },
    }

    # -----------------------------------------------------------------------
    # POINT — Instantaneous measurements at specific moments
    # -----------------------------------------------------------------------
    # Each observation is a single value at a point in time (e.g., a heart rate sample).
    #
    # Available aggregates: mean, median, std, max, min, count
    # Special aggregates (via special_aggregates dict):
    #   quantile  — unweighted percentile (params: op, q)
    #   pct_below — percentage of values below a threshold (params: op, threshold)

    reg["HeartRate"] = {
        "time_behavior": "point",
        "aggregates": ("mean", "median", "std", "max", "min", "count", "p25", "p75"),
        "device_class": "watch_exclusive",
        "special_aggregates": {
            "p25": {"op": "quantile", "q": 0.25},
            "p75": {"op": "quantile", "q": 0.75},
        },
    }
    reg["HeartRateVariabilitySDNN"] = {
        "time_behavior": "point",
        "aggregates": ("mean", "median", "std", "max", "min", "count", "p25", "p75"),
        "device_class": "watch_exclusive",
        "special_aggregates": {
            "p25": {"op": "quantile", "q": 0.25},
            "p75": {"op": "quantile", "q": 0.75},
        },
    }
    reg["RespiratoryRate"] = {
        "time_behavior": "point",
        "aggregates": ("mean", "median", "std", "max", "min", "count", "p25", "p75"),
        "device_class": "watch_exclusive",
        "special_aggregates": {
            "p25": {"op": "quantile", "q": 0.25},
            "p75": {"op": "quantile", "q": 0.75},
        },
    }
    reg["OxygenSaturation"] = {
        "time_behavior": "point",
        "aggregates": ("mean", "median", "std", "max", "min", "count", "p25", "p75", "pct_below_92"),
        "device_class": "watch_exclusive",
        "special_aggregates": {
            "p25": {"op": "quantile", "q": 0.25},
            "p75": {"op": "quantile", "q": 0.75},
            "pct_below_92": {"op": "pct_below", "threshold": 0.92},
        },
    }

    # -----------------------------------------------------------------------
    # CATEGORICAL INTERVAL — Labeled time intervals
    # -----------------------------------------------------------------------
    # Values are category labels across time intervals (e.g., sleep stages).
    # The pipeline tracks duration, count, and strict boundary continuity.
    #
    # Aggregate naming pattern:
    #   sum_label_<label>   — total seconds with that label in the window
    #   mean_label_<label>  — mean bout duration (seconds) for that label
    #   count_label_<label> — number of bouts with that label
    #   continues_from_previous_bin_label_<label> — original interval crosses
    #                                                the window-start boundary
    #   continues_into_next_bin_label_<label>     — original interval crosses
    #                                                the window-end boundary

    reg["EnvironmentalAudioExposureEvent"] = {
        "time_behavior": "categorical_interval",
        "aggregates": (
            "sum_label_momentary_limit",
            "mean_label_momentary_limit",
            "count_label_momentary_limit",
            "continues_from_previous_bin_label_momentary_limit",
            "continues_into_next_bin_label_momentary_limit",
        ),
        "device_class": "either",
        "zero_fill_rules": _participant_zero_rules((
            "sum_label_momentary_limit",
            "mean_label_momentary_limit",
            "count_label_momentary_limit",
            "continues_from_previous_bin_label_momentary_limit",
            "continues_into_next_bin_label_momentary_limit",
        )),
    }
    reg["SleepAnalysis"] = {
        "time_behavior": "categorical_interval",
        "aggregates": sleep_aggregates,
        "device_class": "either",
        "zero_fill_rules": _participant_zero_rules(sleep_aggregates),
    }

    return reg





# =============================================================================
# REUSABLE CONFIGURATION OBJECTS — IMPORT AND INSTANTIATE; DO NOT EDIT HERE
# =============================================================================
# These public classes describe optional configurations supplied elsewhere in
# pipeline code. Import and instantiate them rather than editing their class
# definitions for a particular analysis.

# -----------------------------------------------------------------------------
# Supported sleep-feature values
# -----------------------------------------------------------------------------
# These tuples define the supported labels, unions, and statistics and also
# serve as the defaults for SleepFeatureSpec().
#
# For a particular analysis, do not edit these tuples. Instead, construct
# SleepFeatureSpec(labels=..., unions=..., statistics=...) and pass it to
# build_default_type_registry(). Edit these tuples only when extending the
# pipeline's supported sleep schema.
SLEEP_LABEL_SUFFIXES = (
    "in_bed",
    "asleep",
    "asleep_unspecified",
    "asleep_core",
    "asleep_deep",
    "asleep_rem",
    "awake",
    "awake_unspecified",
    "sleep_unknown",
)
SLEEP_UNION_SUFFIXES = ("in_bed", "asleep", "awake")
SLEEP_WINDOW_STATISTICS = (
    "sum",
    "mean",
    "count",
    "continues_from_previous_bin",
    "continues_into_next_bin",
)


@dataclass(frozen=True)
class SleepFeatureSpec:
    """Select granular sleep labels, unions, and window statistics."""

    labels: tuple[str, ...] = SLEEP_LABEL_SUFFIXES
    unions: tuple[str, ...] = SLEEP_UNION_SUFFIXES
    statistics: tuple[str, ...] = SLEEP_WINDOW_STATISTICS

    def validate(self) -> None:
        """Validate the selected sleep features."""
        _validate_sleep_feature_spec(self)

    def aggregate_names(self) -> tuple[str, ...]:
        """Return the selected sleep feature names."""
        return _build_sleep_aggregate_names(self)

# -----------------------------------------------------------------------------
# Assessment-window specification
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AssessmentWindowSpec:
    """Define how lookback windows are constructed for each assessment.

    ``aligned_grid`` requires ``bin_size`` and exactly one lookback definition:
    ``n_bins`` selects a fixed number of canonical bins, whereas
    ``duration_before`` clips the first and final bins to cover an exact elapsed
    duration before the assessment.
    """

    mode: Literal["whole_lookback", "split_lookback", "aligned_grid"]
    duration_before: Optional[Any] = None
    n_bins: Optional[int] = None
    bin_size: Optional[Any] = None
    bin_origin: Optional[Any] = None
    tz_mode: Optional[Literal["utc", "local"]] = None

    def validate(self) -> None:
        """Validate the configured assessment window."""
        _validate_assessment_window_spec(self)

# -----------------------------------------------------------------------------
# Hierarchical-rollup specifications
# -----------------------------------------------------------------------------


@dataclass
class LayerRule:
    """
    One aggregation rule within a hierarchical layer.

    Example:
        LayerRule(out="HeartRate__median_mean_day", op="mean",
                  inputs={"col": "HeartRate__median"}, strict=False)
    """
    out: str                        # Output column name in the rolled-up layer
    op: str                         # Operator name from the op registry (e.g., "mean", "sum", "weighted_mean")
    inputs: Dict[str, Any]          # Parameters passed to the operator (e.g., {"col": "HeartRate__median"})
    strict: Optional[bool] = None   # True = error if input column missing; False = skip silently; None = use layer default


@dataclass
class LayerSpec:
    """
    Specification for one hierarchical aggregation layer (e.g., hourly -> daily).

    Example:
        LayerSpec(name="daily", freq="1D", rules=[
            LayerRule(out="HeartRate__mean_day", op="mean", inputs={"col": "HeartRate__mean"}),
            LayerRule(out="StepCount__sum_day", op="sum", inputs={"col": "StepCount__sum"}),
        ])
    """
    name: str                                       # Layer name (e.g., "daily", "weekly")
    freq: str                                       # Aggregation frequency as pandas string (e.g., "1D", "7D")
    origin: str = "00:00:00"                        # Time-of-day alignment anchor
    rules: List[LayerRule] = field(default_factory=list)  # List of aggregation rules
    strict: bool = True                             # Default strict mode for rules that don't specify their own
    tz_mode: str = "utc"                            # Timezone mode for window boundaries

# =============================================================================
# ADVANCED EXTENSION POINT — CUSTOM AGGREGATION OPERATIONS
# =============================================================================
# Most users do not need to edit this section. Edit build_default_op_registry()
# to add an operation that LayerRule.op should reference by default, or call
# register_feature_operation(aggregator.op_registry, ...) after construction.

OpFn = Callable[[pd.core.groupby.generic.DataFrameGroupBy, Dict[str, Any]], pd.Series]


def register_feature_operation(registry: Dict[str, OpFn], key: str, fn: OpFn) -> None:
    """Register a new operator callable(groupby, inputs) -> Series."""
    registry[key] = fn


def build_default_op_registry() -> Dict[str, OpFn]:
    """
    Build the default registry of aggregation operators for hierarchical layers.

    These operators are referenced by name in LayerRule.op. Each operator is a
    callable(groupby, inputs_dict) -> pd.Series.

    Available operators and their required inputs:
      sum                    — Sum of values                           (inputs: col)
      count_nonnull          — Count of non-null values                (inputs: col)
      max / min              — Maximum / minimum                       (inputs: col)
      mean / median          — Arithmetic mean / median                (inputs: col)
      std                    — Standard deviation                      (inputs: col, ddof=1)
      skew / kurtosis        — Skewness / excess kurtosis              (inputs: col)
      quantile               — Unweighted quantile                     (inputs: col, q)
      weighted_mean          — Duration-weighted mean                  (inputs: col, weight)
      weighted_median        — Duration-weighted median                (inputs: col, weight)
      weighted_quantile      — Duration-weighted quantile              (inputs: col, weight, q)
      ratio                  — Ratio of two column sums                (inputs: num_col, den_col)
      pooled_std             — Pooled standard deviation               (inputs: mean_col, std_col, count_col)
      pooled_var_from_moments — Pooled variance from raw moments       (inputs: wxsq, wx, w)
      rate_from_totals       — Rate: sum(num) / (sum(den) / k)        (inputs: num, den, k=1.0)
      normalized_mean        — Weighted sum normalized to a target     (inputs: col, weight, norm_to)
      leq_from_energy        — Acoustic LEQ from energy + duration     (inputs: energy_col, duration_col)
      geometric_mean         — Geometric mean                          (inputs: col or log_col)
      harmonic_mean          — Harmonic mean                           (inputs: col or inv_col)
      any / all              — Logical OR / AND on a boolean column    (inputs: flag_col)
    """
    registry: Dict[str, OpFn] = {}

    # Basic aggregations
    _sum  = lambda g, i: g[i["col"]].sum()
    _cnt  = lambda g, i: g[i["col"]].count()
    _max  = lambda g, i: g[i["col"]].max()
    _min  = lambda g, i: g[i["col"]].min()
    _mean = lambda g, i: g[i["col"]].mean()

    # Weighted/pooled metrics
    def _weighted_mean(g, i):
        c, w = i["col"], i["weight"]
        return g.apply(lambda df: (df[c] * df[w]).sum() / df[w].sum() if df[w].sum() != 0 else np.nan)

    def _ratio(g, i):
        n, d = i["num_col"], i["den_col"]
        return g.apply(lambda df: df[n].sum() / df[d].sum() if df[d].sum() != 0 else np.nan)

    def _pooled_std(g, i):
        mc, sc, nc = i["mean_col"], i["std_col"], i["count_col"]
        def _one(df):
            counts = df[nc].fillna(0).astype(float); means = df[mc].astype(float)
            stds = df[sc].astype(float).fillna(0.0).where(counts > 1, 0.0)
            N = counts.sum()
            if N <= 1: return np.nan
            mu = (means * counts).sum() / N
            ss_within  = ((counts - 1.0) * (stds ** 2)).sum()
            ss_between = (counts * (means - mu) ** 2).sum()
            return np.sqrt((ss_within + ss_between) / (N - 1.0))
        return g.apply(_one)

    def _pooled_var_from_moments(g, i):
        wxsq, wx, w = i["wxsq"], i["wx"], i["w"]
        return g.apply(lambda df: np.nan if df[w].sum() <= 1 else (df[wxsq].sum() - (df[wx].sum() ** 2) / df[w].sum()) / (df[w].sum() - 1.0))

    # Means on transformed scales
    def _geometric_mean(g, i):
        if "log_col" in i:
            lc = i["log_col"]
            return g.apply(lambda df: np.exp(df[lc].sum() / df[lc].count()) if df[lc].count() > 0 else np.nan)
        c = i["col"]
        return g.apply(lambda df: (lambda x: np.exp(np.log(x).mean()) if len(x) else np.nan)(df[c][df[c] > 0]))

    def _harmonic_mean(g, i):
        if "inv_col" in i:
            ic = i["inv_col"]
            return g.apply(lambda df: df[ic].count() / df[ic].sum() if df[ic].sum() != 0 else np.nan)
        c = i["col"]
        def _one(df):
            x = df[c]; x = x[x != 0]; s = (1.0 / x).sum()
            return (len(x) / s) if (len(x) and s != 0) else np.nan
        return g.apply(_one)

    # Logical reducers
    _any = lambda g, i: g[i["flag_col"]].astype(bool).any()
    _all = lambda g, i: g[i["flag_col"]].astype(bool).all()

    # Quantiles (weighted & unweighted)
    _median   = lambda g, i: g[i["col"]].median()
    _quantile = lambda g, i: g[i["col"]].quantile(float(i["q"]))

    def _weighted_quantile(g, i):
        col, w, q = i["col"], i["weight"], float(i["q"])
        def _one(df):
            d = df[[col, w]].dropna()
            if d.empty:
                return np.nan
            d = d.sort_values(col)
            cw = d[w].cumsum()
            tot = d[w].sum()
            if tot <= 0:
                return np.nan
            cut = q * tot
            idx = int(np.searchsorted(cw.to_numpy(), cut, side="left"))
            idx = min(idx, len(d) - 1)
            return d.iloc[idx][col]
        return g.apply(_one)

    _weighted_median = lambda g, i: _weighted_quantile(g, {"col": i["col"], "weight": i["weight"], "q": 0.5})

    # Rates/normalization
    def _rate_from_totals(g, i):
        num, den, k = i["num"], i["den"], float(i.get("k", 1.0))
        return g.apply(lambda df: df[num].sum() / (df[den].sum() / k) if df[den].sum() != 0 else np.nan)

    def _normalized_mean(g, i):
        c, w, norm_to = i["col"], i["weight"], float(i["norm_to"])
        return g.apply(lambda df: (df[c] * df[w]).sum() / norm_to if norm_to != 0 else np.nan)

    # Acoustic LEQ from energy
    def _leq_from_energy(g, i):
        e_col, d_col = i["energy_col"], i["duration_col"]
        def _calc(df):
            dur_sum = df[d_col].sum()
            en_sum  = df[e_col].sum()
            if pd.isna(dur_sum) or pd.isna(en_sum) or dur_sum <= 0 or en_sum <= 0:
                return np.nan
            return 10.0 * np.log10(en_sum / dur_sum)
        return g.apply(_calc)

    def _std(g, i):
        col = i["col"]
        ddof = int(i.get("ddof", 1))
        return g[col].agg(lambda s: pd.to_numeric(s, errors="coerce").std(ddof=ddof))

    def _skew(g, i):
        col = i["col"]
        return g[col].agg(lambda s: pd.to_numeric(s, errors="coerce").skew())

    def _kurtosis(g, i):
        col = i["col"]
        return g[col].agg(lambda s: pd.to_numeric(s, errors="coerce").kurt())


    # Build default operations registry
    defaults = {
        "sum": _sum,
        "count_nonnull": _cnt,
        "max": _max,
        "min": _min,
        "mean": _mean,
        "weighted_mean": _weighted_mean,
        "ratio": _ratio,
        "pooled_std": _pooled_std,
        "pooled_var_from_moments": _pooled_var_from_moments,
        "geometric_mean": _geometric_mean,
        "harmonic_mean": _harmonic_mean,
        "any": _any,
        "all": _all,
        "median": _median,
        "weighted_median": _weighted_median,
        "quantile": _quantile,
        "weighted_quantile": _weighted_quantile,
        "rate_from_totals": _rate_from_totals,
        "normalized_mean": _normalized_mean,
        "leq_from_energy": _leq_from_energy,
        "std": _std,
        "skew": _skew,
        "kurtosis": _kurtosis,
    }

    for k, v in defaults.items():
        register_feature_operation(registry, k, v)

    return registry





# =============================================================================
# INTERNAL VALIDATION AND HELPERS — DO NOT EDIT
# =============================================================================
# The helpers below validate and support the public settings above.
# Change them only when modifying pipeline behavior itself.

def _validate_agg_config(self: HKAggConfig) -> None:
    """Light config validation; raise early for schema errors."""
    if hasattr(self, "tz_mode_10"):
        raise ValueError("Use `tz_mode_l0` (letter l + zero), not `tz_mode_10`.")
    bin_duration = pd.to_timedelta(self.bin_size)
    bin_seconds = bin_duration.total_seconds()
    if pd.isna(bin_duration) or bin_seconds < 1 or not float(bin_seconds).is_integer():
        raise ValueError("bin_size must be a fixed duration in whole seconds")
    origin = pd.to_timedelta(self.bin_origin)
    if pd.isna(origin) or origin < pd.Timedelta(0) or origin >= pd.Timedelta(days=1):
        raise ValueError("bin_origin must be within [00:00:00, 24:00:00)")
    for t, meta in self.type_registry.items():
        if "time_behavior" not in meta: raise ValueError(f"type_registry entry '{t}' missing 'time_behavior'")
        rules = meta.get("zero_fill_rules", {})
        if not isinstance(rules, dict):
            raise ValueError(f"type_registry entry '{t}' zero_fill_rules must be a mapping")
        configured = set(meta.get("aggregates", ()))
        for statistic, rule in rules.items():
            if statistic not in configured:
                raise ValueError(
                    f"type_registry entry '{t}' has a zero-fill rule for unconfigured statistic '{statistic}'"
                )
            if not isinstance(rule, dict):
                raise ValueError(f"zero-fill rule '{t}__{statistic}' must be a mapping")
            unknown = set(rule) - {"gate", "fill_value", "availability"}
            missing = {"gate", "fill_value", "availability"} - set(rule)
            if unknown or missing:
                raise ValueError(
                    f"zero-fill rule '{t}__{statistic}' has unknown={sorted(unknown)} missing={sorted(missing)}"
                )
            gate = rule["gate"]
            if not (
                isinstance(gate, tuple)
                and len(gate) == 2
                and isinstance(gate[0], str)
                and isinstance(gate[1], dict)
            ):
                raise ValueError(
                    f"zero-fill rule '{t}__{statistic}' gate must be a (name, parameters) tuple"
                )
            supported_gates = {
                "expr_false", "watch_on", "source_present",
                "any_source_present", "any_non_watch_present", "presence_types",
            }
            if gate[0] not in supported_gates:
                raise ValueError(
                    f"zero-fill rule '{t}__{statistic}' has unsupported gate '{gate[0]}'"
                )
            try:
                fill_value = float(rule["fill_value"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"zero-fill rule '{t}__{statistic}' fill_value must be numeric"
                ) from exc
            if not np.isfinite(fill_value):
                raise ValueError(
                    f"zero-fill rule '{t}__{statistic}' fill_value must be finite"
                )
            if rule["availability"] not in {"participant_evidence", "assume_available"}:
                raise ValueError(
                    f"zero-fill rule '{t}__{statistic}' availability must be "
                    "'participant_evidence' or 'assume_available'"
                )
    if self.qc_action not in {"drop","mask"}: raise ValueError("qc_action must be 'drop' or 'mask'")
    if self.tz_mode_l0 not in {"utc","local"}: raise ValueError("tz_mode_l0 must be one of {'utc','local'}")
    if self.tz_policy not in {"strict","lenient"}: raise ValueError("tz_policy must be one of {'strict','lenient'}")
    if self.output_timezone not in {"UTC","local","both"}: raise ValueError("output_timezone must be one of {'UTC','local','both'}")
    if self.utc_schedule_inference_policy not in {"skip","infer_fallback","infer_fail"}:
        raise ValueError("utc_schedule_inference_policy must be one of {'skip','infer_fallback','infer_fail'}")
    if self.schedule_point_policy not in {"include_epsilon","exclude"}:
        raise ValueError("schedule_point_policy must be one of {'include_epsilon','exclude'}")
    if self.schedule_invalid_tz_policy not in {"strict","coerce_default"}:
        raise ValueError("schedule_invalid_tz_policy must be one of {'strict','coerce_default'}")
    eps = pd.to_timedelta(self.schedule_point_epsilon)
    if eps <= pd.Timedelta(0):
        raise ValueError("schedule_point_epsilon must be > 0")
    if self.hrv_min_samples < 2: raise ValueError("hrv_min_samples must be >= 2")
    if self.hrv_freq_min_samples < 4: raise ValueError("hrv_freq_min_samples must be >= 4")
    if self.hrv_interpolation_fs <= 0: raise ValueError("hrv_interpolation_fs must be > 0")
    _valid_prune = {"none", "wear_seconds_gt0", "any_data", "any_non_watch_data", None}
    if self.early_prune_mode not in _valid_prune:
        raise ValueError(f"early_prune_mode must be one of {_valid_prune!r}, got {self.early_prune_mode!r}")
    for _name, _val in [("bin_frac_min", self.bin_frac_min), ("day_frac_min", self.day_frac_min), ("overall_frac_min", self.overall_frac_min)]:
        if _val is not None and not (0.0 <= _val <= 1.0):
            raise ValueError(f"{_name} must be in [0, 1], got {_val}")


def _validate_sleep_feature_spec(self: SleepFeatureSpec) -> None:
    selections = (
        ("labels", self.labels, SLEEP_LABEL_SUFFIXES),
        ("unions", self.unions, SLEEP_UNION_SUFFIXES),
        ("statistics", self.statistics, SLEEP_WINDOW_STATISTICS),
    )
    for name, values, allowed in selections:
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise ValueError(f"SleepFeatureSpec.{name} contains unsupported value(s): {unknown}")
        if len(values) != len(set(values)):
            raise ValueError(f"SleepFeatureSpec.{name} must not contain duplicates")


def _build_sleep_aggregate_names(self: SleepFeatureSpec) -> tuple[str, ...]:
    self.validate()
    label_features = tuple(
        f"{stat}_label_{label}"
        for label in self.labels
        for stat in self.statistics
    )
    union_features = tuple(
        f"{stat}_any_{union}"
        for union in self.unions
        for stat in self.statistics
    )
    return label_features + union_features


def _validate_assessment_window_spec(self: AssessmentWindowSpec) -> None:
    if self.mode not in {"whole_lookback", "split_lookback", "aligned_grid"}:
        raise ValueError(
            "AssessmentWindowSpec.mode must be 'whole_lookback', "
            "'split_lookback', or 'aligned_grid'"
        )

    populated = {
        name
        for name in ("duration_before", "n_bins", "bin_size", "bin_origin", "tz_mode")
        if getattr(self, name) is not None
    }
    required = {
        "whole_lookback": {"duration_before"},
        "split_lookback": {"duration_before", "n_bins"},
        "aligned_grid": {"bin_size"},
    }[self.mode]
    allowed = {
        "whole_lookback": {"duration_before"},
        "split_lookback": {"duration_before", "n_bins"},
        "aligned_grid": {
            "duration_before", "n_bins", "bin_size", "bin_origin", "tz_mode"
        },
    }[self.mode]
    missing = required - populated
    irrelevant = populated - allowed
    if missing or irrelevant:
        raise ValueError(
            f"AssessmentWindowSpec(mode={self.mode!r}) has "
            f"missing={sorted(missing)} irrelevant={sorted(irrelevant)}"
        )

    if self.mode == "aligned_grid" and (
        (self.duration_before is None) == (self.n_bins is None)
    ):
        raise ValueError(
            "aligned_grid requires exactly one of duration_before or n_bins"
        )

    if self.duration_before is not None:
        duration = pd.to_timedelta(self.duration_before)
        if pd.isna(duration) or duration <= pd.Timedelta(0):
            raise ValueError("duration_before must be a positive duration")
    if self.n_bins is not None:
        if isinstance(self.n_bins, bool) or int(self.n_bins) != self.n_bins or int(self.n_bins) < 1:
            raise ValueError("n_bins must be a positive integer")
    if self.bin_size is not None:
        bin_size = pd.to_timedelta(self.bin_size)
        seconds = bin_size.total_seconds()
        if pd.isna(bin_size) or seconds < 1 or not float(seconds).is_integer():
            raise ValueError("bin_size must be a fixed duration in whole seconds")
    if self.bin_origin is not None:
        origin = pd.to_timedelta(self.bin_origin)
        if pd.isna(origin) or origin < pd.Timedelta(0) or origin >= pd.Timedelta(days=1):
            raise ValueError("bin_origin must be within [00:00:00, 24:00:00)")
    if self.tz_mode is not None and self.tz_mode not in {"utc", "local"}:
        raise ValueError("tz_mode must be 'utc' or 'local'")


def _participant_zero_rules(statistics: tuple[str, ...]) -> Dict[str, Dict[str, Any]]:
    return {
        statistic: {
            "gate": ("watch_on", {}),
            "fill_value": 0.0,
            "availability": "participant_evidence",
        }
        for statistic in statistics
    }

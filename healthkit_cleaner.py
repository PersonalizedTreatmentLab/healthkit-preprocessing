"""
Tasks:
- Refine plausibility ranges
- Plausibility ranges for categorical variables based on duration?
Misc. Ideas
- Implement Hampel filter: Detect outliers using median absolute deviation (MAD) within moving window (threshold ~3 SD)
- Sensor drift detection: Identify unreliable data when device is removed
  - Use IQR ranges on sample participants to set thresholds
  - Fit linear trend on first 5 minutes; flag large deviations
  - Consider Kalman Filter or bypass window approaches
- Consider options for collapsing gaps
#TODO Add other cardio variables (waiting on Agrim)
Note. Quantile filters and operations often apply to entire types but this isn't always desirable.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union
from tqdm.auto import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
from healthkit_cleaner_specs import HKCleanConfig



# ===================================================
# Class: HealthKitCleaner
# ===================================================
class HealthKitCleaner:
    """
    Provides a configurable, per-type cleaning pipeline for HealthKit data.

    Main responsibilities:
      1. Define known HealthKit categories (activity, vitals, audio, categorical).
      2. Enforce basic validity rules (time, user_id, type).
      3. Register and route each type to the appropriate cleaning handler.
      4. Apply per-type processing, deduplication, overlap resolution, and plausibility filtering.
      5. Produce a structured report summarizing operations performed, rows dropped, and transformations applied per type.
    """

    # Constants retained for reference; defaults come from HKCleanConfig in __init__
    ACTIVITY_TYPES = {"StepCount", "AppleExerciseTime", "ActiveEnergyBurned", "BasalEnergyBurned", "AppleStandTime", "DistanceCycling", "DistanceWalkingRunning", "FlightsClimbed"}
    VITAL_TYPES    = {"HeartRate", "HeartRateVariabilitySDNN", "RespiratoryRate", "OxygenSaturation"}
    AUDIO_TYPES    = {"EnvironmentalAudioExposure", "HeadphoneAudioExposure"}
    CAT_TYPES    = {"SleepAnalysis", "EnvironmentalAudioExposureEvent"} # "AppleStandHour"

    def __init__(self, rules: Optional[HKCleanConfig] = None, logger: Optional[logging.Logger] = None):
        self.rules = rules or HKCleanConfig()
        self.rules.validate()
        self.logger = logger or logging.getLogger(__name__)
        self.handlers: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}

        # Per-type handlers
        for t in self.AUDIO_TYPES:
            self.handlers[t] = self._handle_audio
        for t in self.ACTIVITY_TYPES:
            self.handlers[t] = self._handle_activity
        for t in self.VITAL_TYPES:
            self.handlers[t] = self._handle_vitals
        for t in self.CAT_TYPES:
            self.handlers[t] = self._handle_cat

        # Internal report store
        self._report: Dict[str, Dict[str, Any]] = {}

    # ===================================================
    # Internal report helper
    # ===================================================
    def _get_type_report(self, hk_type: str) -> Dict[str, Any]:
        """Return (and create if needed) the report dict entry for hk_type."""
        return self._report.setdefault(hk_type, {})

    # ===================================================
    # Main Cleaning Function
    # ===================================================
    def clean(self, df: pd.DataFrame, return_report: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Clean a HealthKit dataframe by applying global and per-type rules.

        Steps performed:
        1. Copy the input dataframe and record initial row count.
        2. Validate required columns and fail fast if any are missing.
        3. Convert time columns to datetime.
        4. Drop rows with missing/negative time intervals (based on UTC columns).
        5. Drop rows with missing `type`.
        6. Drop rows with missing `user_id`.
        7. Initialize the global cleaning report.
        8. Group rows by `type` and route each group to its handler.
        9. For each type:
            - Apply handler-specific cleaning logic.
            - Track rows in/out and any errors.
        10. Concatenate all cleaned groups into a unified dataframe.
        11. Sort by (`user_id`, `type`, `utc_start`, `utc_end`) and reset index.
        12. Finalize the global report including observed types and handlers used.
        13. Return the cleaned dataframe, or (df, report) if `return_report=True`.
        """
        
        # Copy df, Record original length
        df = df.copy()
        required = set(self.rules.required_cols)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Dataframe missing required columns: {missing}")
        if self.rules.ops.get("apply_watch_device_filter", True):
            watch_required = {"device.name", "body.quantity.count"}
            missing_watch = sorted(watch_required - set(df.columns))
            if missing_watch:
                raise ValueError(
                    "Dataframe missing required columns for watch filter: "
                    f"{missing_watch}"
                )
        original_rows = int(len(df))

        # Convert time columns to DateTime
        if "utc_start" in df.columns:
            df["utc_start"] = pd.to_datetime(df["utc_start"], errors="coerce", utc=True)
        if "utc_end" in df.columns:
            df["utc_end"] = pd.to_datetime(df["utc_end"], errors="coerce", utc=True)
        if "local_start" in df.columns:
            df["local_start"] = pd.to_datetime(df["local_start"], errors="coerce")
        if "local_end" in df.columns:
            df["local_end"] = pd.to_datetime(df["local_end"], errors="coerce")

        # Drop rows with missing or negative time data (UTC-based)
        time_valid = (
            df.get("utc_start").notna() 
            & df.get("utc_end").notna() 
            & (df["utc_end"] >= df["utc_start"])
        )
        dropped_time = int((~time_valid).sum())
        if dropped_time:
            self.logger.info("Dropping %d rows with invalid or non-positive UTC intervals", dropped_time)
            df = df[time_valid].copy()

        # Drop rows with missing type
        missing_type_mask = df["type"].isna()
        dropped_type = int(missing_type_mask.sum())
        if dropped_type:
            self.logger.info("Dropping %d rows with missing type", dropped_type)
            df = df[~missing_type_mask].copy()

        # Coerce user_id to numeric participant ids and drop invalid values.
        missing_user_mask = df["user_id"].isna()
        dropped_missing_user = int(missing_user_mask.sum())
        uid_num = pd.to_numeric(df["user_id"], errors="coerce")
        finite_user_mask = uid_num.notna() & np.isfinite(uid_num)
        integer_like_user_mask = finite_user_mask & (np.floor(uid_num) == uid_num)
        invalid_user_mask = ~integer_like_user_mask
        dropped_invalid_user_non_numeric = int((invalid_user_mask & ~missing_user_mask).sum())
        dropped_invalid_user_total = int(invalid_user_mask.sum())
        if dropped_invalid_user_total:
            self.logger.info("Dropping %d rows with invalid/non-numeric user_id", dropped_invalid_user_total)
            df = df.loc[~invalid_user_mask].copy()
            uid_num = uid_num.loc[df.index]
        if not df.empty:
            df["user_id"] = uid_num.astype("int64")
        else:
            df["user_id"] = pd.Series(dtype="int64")

        self.logger.info("Converted columns to datetime, dropped rows with invalid time, type, or user_id data")

        # Reset & initialize report
        self._report = {}
        global_rep = self._report.setdefault("__global__", {})
        global_rep.update({
            "rows_in": original_rows,
            "dropped_missing_or_negative_time": dropped_time,
            "dropped_missing_type": dropped_type,
            "dropped_missing_user_id": dropped_missing_user,
            "dropped_invalid_user_id_non_numeric": dropped_invalid_user_non_numeric,
        })

        # Setup per-type variables
        cleaned_frames: list[pd.DataFrame] = []
        types_seen: list[str] = []
        types_without_handler: Set[str] = set()

        # Optional tqdm progress bar for per-type cleaning
        try:
            from tqdm.auto import tqdm  # type: ignore[import]
        except ImportError:  # pragma: no cover
            def tqdm(x, **kwargs):
                return x

        grouped = df.groupby("type", sort=False)

        # Group by type and route to handler
        for hk_type, group_df in tqdm(grouped, total=getattr(grouped, "ngroups", None), desc="Cleaning types"):
            hk_type_str = str(hk_type)
            self.logger.info("Processing type '%s' with %d rows", hk_type_str, len(group_df))
            types_seen.append(hk_type_str)
            handler = self.handlers.get(hk_type_str, self._handle_unknown)
            try:
                cleaned_group = handler(group_df)
                cleaned_frames.append(cleaned_group)

                rep = self._get_type_report(hk_type_str)
                rep.setdefault("rows_in", int(len(group_df)))
                rep["rows_out"] = int(len(cleaned_group))
                rep["handler"] = handler.__name__
            except Exception as exc:  # defensive fallback
                self.logger.exception(
                    "Error cleaning type '%s'; passing raw data through. Error: %s",
                    hk_type_str, exc,
                )
                rep = self._get_type_report(hk_type_str)
                rep.update({
                    "error": str(exc),
                    "rows_in": int(len(group_df)),
                    "rows_out": int(len(group_df)),
                    "handler": handler.__name__,
                })
                cleaned_frames.append(group_df)

            if handler is self._handle_unknown:
                types_without_handler.add(hk_type_str)

        # Concatenate all cleaned groups, sort & reindex
        cleaned_all = (pd.concat(cleaned_frames, ignore_index=True) if cleaned_frames else df.iloc[0:0].copy()).sort_values(
            ["user_id", "type", "utc_start", "utc_end"]).reset_index(drop=True)
        self.logger.info("Concatenated all cleaned groups")

        # Postconditions: enforce UTC-aware interval columns and emit diagnostics.
        cleaned_all["utc_start"] = pd.to_datetime(cleaned_all.get("utc_start"), errors="coerce", utc=True)
        cleaned_all["utc_end"] = pd.to_datetime(cleaned_all.get("utc_end"), errors="coerce", utc=True)

        invalid_utc_mask = cleaned_all["utc_start"].isna() | cleaned_all["utc_end"].isna()
        post_invalid_utc_rows = int(invalid_utc_mask.sum())
        if post_invalid_utc_rows:
            cleaned_all = cleaned_all.loc[~invalid_utc_mask].copy()

        negative_interval_mask = cleaned_all["utc_end"] < cleaned_all["utc_start"]
        point_interval_mask = cleaned_all["utc_end"] == cleaned_all["utc_start"]
        post_negative_interval_rows = int(negative_interval_mask.sum())
        post_point_interval_rows = int(point_interval_mask.sum())
        if post_negative_interval_rows:
            cleaned_all = cleaned_all.loc[~negative_interval_mask].copy()

        post_missing_timezone_rows = 0
        post_invalid_timezone_rows = 0
        if "timezone" in cleaned_all.columns:
            tz_series = cleaned_all["timezone"].astype("string").str.strip()
            missing_tz_mask = tz_series.isna() | tz_series.eq("")
            post_missing_timezone_rows = int(missing_tz_mask.sum())

            invalid_vals = set()
            for tz in tz_series[~missing_tz_mask].dropna().unique():
                tz_str = str(tz)
                if not self._is_valid_timezone_name(tz_str):
                    invalid_vals.add(tz_str)
            invalid_tz_mask = tz_series.isin(list(invalid_vals)) if invalid_vals else pd.Series(False, index=cleaned_all.index)
            post_invalid_timezone_rows = int(invalid_tz_mask.sum())
            cleaned_all["timezone"] = tz_series

        # Finish the global report
        global_rep.update(
            {
                "rows_out": int(len(cleaned_all)),
                "types_seen": types_seen,
                "post_invalid_utc_rows": post_invalid_utc_rows,
                "post_negative_interval_rows": post_negative_interval_rows,
                "post_point_interval_rows": post_point_interval_rows,
                "post_missing_timezone_rows": post_missing_timezone_rows,
                "post_invalid_timezone_rows": post_invalid_timezone_rows,
            }
        )
        if types_without_handler:
            global_rep["types_without_handler"] = sorted(types_without_handler)
        self.logger.info("Finished global report")

        # Return cleaned data, optionally return report
        if return_report:
            return cleaned_all, self._report
        return cleaned_all

    # ===================================================
    # Register new / override handlers
    # ===================================================
    def register_handler(self, hk_type: str, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """
        Register/override a handler for a given HealthKit type.
        """
        if not callable(fn):
            raise TypeError("Handler must be callable: (pd.DataFrame) -> pd.DataFrame")
        self.handlers[hk_type] = fn
        self.logger.info("Registered handler '%s' for HealthKit type '%s'", getattr(fn, "__name__", repr(fn)), hk_type,)

    # ===================================================
    # Handlers (per-type pipelines)
    # ===================================================
    def _handle_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        0. Filter to observations recorded by Apple Watch.
        1. Coerce `value` to numeric and drop rows that cannot be parsed.
        2. If type==EnergyBurned: Convert Joules to kcal.
        3. Drop/clip outliers based on theory thresholds
            - if type is ActiveEnergyBurned or BasalEnergyBurned: cap implausible kcal/min (not drop)
            - apply value range thresholds (min/max)
        4. Drop outliers based on quantile
        5. Calls combine_overlaps per user for value
        """
        if df.empty:
            return df
        df = df.copy()
        hk_type = str(df["type"].iloc[0])
        rep = self._get_type_report(hk_type)
        rep["rows_in"] = int(len(df))
        self.logger.info("[Activity handler] Starting preprocessing for type '%s' with %d rows", hk_type, rep['rows_in'])

        # 0. Filter to observations recorded by Apple Watch
        if self.rules.ops.get("apply_watch_device_filter", True):
                before_watch = len(df)
                df = df[
                    (df["device.name"] == "Apple Watch") &
                    (df["body.quantity.count"] == 1)].copy()
                rep["watch_filter_dropped"] = before_watch - len(df)
                self.logger.info(
                    "[Activity handler] Watch device filter for '%s': dropped %d rows, %d remain",
                    hk_type, rep["watch_filter_dropped"], len(df),
                )

        # 1. Coerce `value` to numeric and drop unparsable rows
        df = self._coerce_value_numeric(df, rep, hk_type, handler_label="Activity handler")

        # 2. Normalize energy values: convert to kcal if needed
        if self.rules.ops.get("convert_energy_to_kcal", True) and "EnergyBurned" in hk_type:
            duration_min = (df["utc_end"] - df["utc_start"]) / pd.Timedelta("1m")
            valid_duration = duration_min.gt(0) & duration_min.notna()
            cap_raw = self.rules.thresholds.get(hk_type, {}).get("kcal_per_min_max", 30.0)
            try:
                cap = float(cap_raw)
            except (TypeError, ValueError):
                cap = 30.0
            if cap <= 0:
                cap = 30.0

            if valid_duration.any():
                floor = 0.01
                value_valid = df.loc[valid_duration, "value"]
                dur_valid = duration_min.loc[valid_duration]
                rate_if_joules = (value_valid / 4184.0) / dur_valid
                rate_if_kcal = value_valid / dur_valid
                plausible_j = rate_if_joules.ge(floor) & rate_if_joules.le(cap)
                plausible_k = rate_if_kcal.ge(floor) & rate_if_kcal.le(cap)
                share_j = float(plausible_j.mean())
                share_k = float(plausible_k.mean())
                n_valid = int(valid_duration.sum())
                if n_valid >= 20 and share_k >= 0.8 and (share_k - share_j) >= 0.5:
                    self.logger.warning(
                        "[Activity handler] Energy values for type '%s' may already be in kcal; "
                        "Joules plausibility=%.3f, kcal plausibility=%.3f, n=%d",
                        hk_type,
                        share_j,
                        share_k,
                        n_valid,
                    )

            df["value"] = df["value"] / 4184.0
            rep["converted_to_kcal"] = True
            self.logger.info(
                "[Activity handler] Energy normalization for type '%s': converted_to_kcal=%s",
                hk_type, rep['converted_to_kcal'],
            )

        # 3a. Cap kcal/min for ActiveEnergyBurned and BasalEnergyBurned
        if self.rules.ops.get("cap_kcal_per_min", True) and hk_type in {"ActiveEnergyBurned", "BasalEnergyBurned"}:
            duration_min = (df["utc_end"] - df["utc_start"]) / pd.Timedelta("1m")
            safe_dur = duration_min.replace(0, np.nan)
            rate = df["value"] / safe_dur
            cap_raw = self.rules.thresholds.get(hk_type, {}).get("kcal_per_min_max", 30.0)
            try:
                cap = float(cap_raw)
            except (TypeError, ValueError):
                cap = 30.0
            if cap <= 0:
                cap = 30.0
            over = rate > cap
            rep["kcal_per_min_capped"] = int(over.fillna(False).sum())

            policy = getattr(self.rules, "kcal_per_min_policy", "cap")
            if policy == "drop":
                # Drop intervals above the cap
                df = df.loc[~over.fillna(False)].copy()
                self.logger.info(
                    "[Activity handler] Dropped kcal/min for type '%s' on %d intervals",
                    hk_type, rep['kcal_per_min_capped'],
                )
            else:
                # For rows above the cap, rescale `value` to cap * duration
                df.loc[over, "value"] = duration_min.loc[over] * cap
                self.logger.info(
                    "[Activity handler] Capped kcal/min for type '%s' on %d intervals",
                    hk_type, rep['kcal_per_min_capped'],
                )

        # 3b. Drop values outside configured [min, max] range for this type
        if self.rules.ops.get("apply_theory_thresholds", True):
            thr = self.rules.thresholds.get(hk_type, {})
            if thr:
                df, dropped = self._filter_range(df, thr.get("min"), thr.get("max"), col="value")
                rep["out_of_range_dropped"] = int(dropped)
                self.logger.info(
                    "[Activity handler] Applied theory thresholds for type '%s', dropped %d out-of-range rows; %d rows remain",
                    hk_type, rep['out_of_range_dropped'], len(df),
                )

        # 4. Combine overlapping intervals per user (uses `combine_overlaps`)
        if self.rules.ops.get("combine_overlaps", True):
            df = self._apply_per_user(df, self.combine_overlaps, value_col="value")
            if not df.empty:
                df = self._add_local_times(df)
            rep["per_user_fallback_count"] = getattr(self, "_last_per_user_fallback_count", 0)
            rep["overlap_input_intervals"] = getattr(self, "_last_overlap_input", 0)
            rep["overlap_output_intervals"] = getattr(self, "_last_overlap_output", 0)
            self.logger.info(
                "[Activity handler] Combined overlapping intervals for type '%s', resulting dataframe has %d rows",
                hk_type, len(df),
            )

        # 5. Apply quantile range filtering
        if self.rules.ops.get("apply_quantile_filter", True) and not df.empty:
            ql = float(self.rules.quantile_lower)
            qh = float(self.rules.quantile_upper)
            df, q_dropped = self._filter_quantiles(df, col="value", lower_q=ql, upper_q=qh)
            rep["quantile_filtered_dropped"] = int(q_dropped)
            self.logger.info(
                "[Activity handler] Applied quantile filter for type '%s', dropped %d rows; %d rows remain",
                hk_type, rep['quantile_filtered_dropped'], len(df),
            )

        rep["rows_out"] = int(len(df))
        self.logger.info("[Activity handler] Finished preprocessing for type '%s', output rows: %d", hk_type, rep['rows_out'])
        return df

    def _handle_vitals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Coerce `value` to numeric and drop rows that cannot be parsed.
        2. Drop exact duplicate observations (same user, type, interval, value).
        3. Apply type-specific plausibility thresholds (if configured)
        4. Apply quantile thresholds
        """

        if df.empty:
            return df
        df = df.copy()
        hk_type = str(df["type"].iloc[0])
        rep = self._get_type_report(hk_type)
        rep["rows_in"] = int(len(df))
        self.logger.info("[Vitals handler] Starting preprocessing for type '%s' with %d rows", hk_type, rep['rows_in'])

        # 1. Coerce `value` to numeric and drop unparsable rows
        df = self._coerce_value_numeric(df, rep, hk_type, handler_label="Vitals handler")

        # 2. Remove exact duplicates across user/type/interval/value
        if self.rules.ops.get("deduplicate", True):
            before_dd = len(df)
            df = df.drop_duplicates(subset=["user_id", "type", "utc_start", "utc_end", "value"], keep="last")
            rep["deduplicated"] = int(before_dd - len(df))
            self.logger.info(
                "[Vitals handler] Deduplicated records for type '%s', removed %d rows; %d rows remain",
                hk_type, rep['deduplicated'], len(df),
            )

        # 3. Apply configured [min, max] value thresholds for this type
        if self.rules.ops.get("apply_theory_thresholds", True):
            thr = self.rules.thresholds.get(hk_type, {})
            if thr:
                df, dropped = self._filter_range(df, thr.get("min"), thr.get("max"), col="value")
                rep["out_of_range_dropped"] = int(dropped)
                self.logger.info(
                    "[Vitals handler] Applied theory thresholds for type '%s', dropped %d out-of-range rows; %d rows remain",
                    hk_type, rep['out_of_range_dropped'], len(df),
                )

        # 4. Apply quantile range filtering
        if (self.rules.ops.get("apply_quantile_filter", True)
            and not df.empty):
            ql = float(self.rules.quantile_lower)
            qh = float(self.rules.quantile_upper)
            df, q_dropped = self._filter_quantiles(df, col="value", lower_q=ql, upper_q=qh)
            rep["quantile_filtered_dropped"] = int(q_dropped)
            self.logger.info(
                "[Vitals handler] Applied quantile filter for type '%s', dropped %d rows; %d rows remain",
                hk_type, rep['quantile_filtered_dropped'], len(df),
            )

        rep["rows_out"] = int(len(df))
        self.logger.info("[Vitals handler] Finished preprocessing for type '%s', output rows: %d", hk_type, rep['rows_out'])
        return df

    def _handle_audio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Coerce `value` to numeric and drop rows that cannot be parsed.
        2. Apply type-specific plausibility thresholds (if configured)
        3. For each user, merge overlapping intervals by calling `combine_overlaps` with `value_col="value"`.        
        4. Apply quantile thresholds
        """
        if df.empty:
            return df
        df = df.copy()
        hk_type = str(df["type"].iloc[0])
        rep = self._get_type_report(hk_type)
        rep["rows_in"] = int(len(df))
        self.logger.info("[Audio handler] Starting preprocessing for type '%s' with %d rows", hk_type, rep['rows_in'])

        # 1. Coerce `value` to numeric and drop unparsable rows
        df = self._coerce_value_numeric(df, rep, hk_type, handler_label="Audio handler")

        # 2. Apply configured [min, max] value thresholds for this type
        if self.rules.ops.get("apply_theory_thresholds", True):
            thr = self.rules.thresholds.get(hk_type, {})
            if thr:
                df, dropped = self._filter_range(df, thr.get("min"), thr.get("max"), col="value")
                rep["out_of_range_dropped"] = int(dropped)
                self.logger.info(
                    "[Audio handler] Applied theory thresholds for type '%s', dropped %d out-of-range rows; %d rows remain",
                    hk_type, rep['out_of_range_dropped'], len(df),
                )

        # 3. Combine overlapping intervals per user (uses `combine_overlaps`)
        if self.rules.ops.get("combine_overlaps", True):
            df = self._apply_per_user(df, self.combine_overlaps, value_col="value", value_kind="level")
            if not df.empty:
                df = self._add_local_times(df)
            rep["per_user_fallback_count"] = getattr(self, "_last_per_user_fallback_count", 0)
            rep["overlap_input_intervals"] = getattr(self, "_last_overlap_input", 0)
            rep["overlap_output_intervals"] = getattr(self, "_last_overlap_output", 0)
            self.logger.info(
                "[Audio handler] Combined overlapping intervals for type '%s', resulting dataframe has %d rows",
                hk_type, len(df),
            )

        # 4. Apply quantile range filtering
        if self.rules.ops.get("apply_quantile_filter", True) and not df.empty:
            ql = float(self.rules.quantile_lower)
            qh = float(self.rules.quantile_upper)
            df, q_dropped = self._filter_quantiles(df, col="value", lower_q=ql, upper_q=qh)
            rep["quantile_filtered_dropped"] = int(q_dropped)
            self.logger.info(
                "[Audio handler] Applied quantile filter for type '%s', dropped %d rows; %d rows remain",
                hk_type, rep['quantile_filtered_dropped'], len(df),
            )

        rep["rows_out"] = int(len(df))
        self.logger.info("[Audio handler] Finished preprocessing for type '%s', output rows: %d", hk_type, rep['rows_out'])
        return df

    def _handle_cat(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Normalize `value` labels by stripping the `HKCategoryValueSleepAnalysis` prefix,
        leaving simple category names (e.g., "InBed", "Asleep").
        2. For each user, resolve overlaps by calling `combine_overlaps_categorical` via `_apply_per_user`.
        """
        if df.empty:
            return df

        df = df.copy()
        hk_type = str(df["type"].iloc[0])
        rep = self._get_type_report(hk_type)
        rep["rows_in"] = int(len(df))

        self.logger.info("[Categorical handler] Starting preprocessing for type '%s' with %d rows", hk_type, rep['rows_in'])

        # 1. Strip the verbose HK sleep prefix from category labels
        if "SleepAnalysis" in hk_type and self.rules.ops.get("normalize_sleep_labels", True):
            df["value"] = df["value"].astype(str).str.replace(
                "HKCategoryValueSleepAnalysis", "", regex=False
            )
            self.logger.info("[Categorical handler] Normalized sleep labels for type '%s'", hk_type)

        # 2. Per-user overlap resolution for categorical intervals
        if self.rules.ops.get("combine_overlaps_cat", True):
            # Helpful stats before the heavy operation
            try:
                n_users = df["user_id"].nunique(dropna=False)
                max_rows_per_user = df.groupby("user_id", dropna=False).size().max()
                self.logger.info(
                    "[Categorical handler] Combining overlaps for type '%s' across %d users (max rows per user: %d)",
                    hk_type, n_users, max_rows_per_user,
                )
            except Exception as stats_exc:
                self.logger.warning(
                    "[Categorical handler] Could not compute user stats before combine_overlaps_categorical: %r",
                    stats_exc,
                )

            try:
                df = self._apply_per_user(df, self.combine_overlaps_categorical, value_col="value")
                if not df.empty:
                    df = self._add_local_times(df)
                rep["per_user_fallback_count"] = getattr(self, "_last_per_user_fallback_count", 0)
                rep["overlap_input_intervals"] = getattr(self, "_last_overlap_input", 0)
                rep["overlap_output_intervals"] = getattr(self, "_last_overlap_output", 0)
            except Exception as exc:
                # If something goes wrong, log and fall back to raw intervals
                self.logger.error("[Categorical handler] ERROR during combine_overlaps_categorical: %r", exc)
                rep["combine_overlaps_cat_error"] = str(exc)

        rep["rows_out"] = int(len(df))
        self.logger.info("[Categorical handler] Finished preprocessing for type '%s', output rows: %d", hk_type, rep['rows_out'])
        return df

    def _handle_unknown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Coerce `value` to numeric
        2. Drop exact duplicate observations (same user, type, interval, value).
        """
        if df.empty:
            return df
        df = df.copy()
        hk_type = str(df["type"].iloc[0])
        self.logger.info("[Unknown handler] Starting preprocessing for type '%s' with %d rows", hk_type, len(df))

        # 1. Best-effort numeric coercion; swallow errors to avoid data loss
        df = self._coerce_value_numeric(df, {}, hk_type, handler_label="Unknown handler", best_effort=True)

        # 2. Drop exact duplicates (user/type/interval/value)
        if self.rules.ops.get("deduplicate", True):
            before_dd = len(df)
            df = df.drop_duplicates(subset=["user_id", "type", "utc_start", "utc_end", "value"], keep="last")
            deduplicated = before_dd - len(df)
            self.logger.info(
                "[Unknown handler] Deduplicated records for type '%s', removed %d rows; %d rows remain",
                hk_type, deduplicated, len(df),
            )

        self.logger.info("[Unknown handler] Finished preprocessing for type '%s', output rows: %d", hk_type, len(df))
        return df

# ===================================================
# Shared Helpers
# ===================================================    
    def _apply_per_user(
        self,
        df: pd.DataFrame,
        fn: Callable[[pd.DataFrame, str], pd.DataFrame],
        value_col: str = "value",
        value_kind: str = None,
    ) -> pd.DataFrame:
        """
        Apply an overlap-combining function per user_id (NaN user_ids handled in their own group).

        This version:
        - Logs how many user groups exist.
        - Uses configured max_workers with user-group bounds.
        - Logs per-user processing errors and falls back to raw data for that user.
        """
        if df.empty:
            return df

        grouped = df.groupby("user_id", dropna=False, sort=False)
        ngroups = int(grouped.ngroups)

        def _process_group(uid, g):
            if value_kind is not None:
                return fn(g.copy(), value_col, value_kind)
            else:
                return fn(g.copy(), value_col)

        cpu_count = os.cpu_count() or 1
        configured_cap = self.rules.max_workers
        try:
            worker_cap = int(cpu_count if configured_cap is None else configured_cap)
        except (TypeError, ValueError):
            worker_cap = 1
        worker_cap = max(1, worker_cap)
        workers = min(worker_cap, ngroups)
        self.logger.info(
            "[_apply_per_user] %d user groups, worker_cap=%d, os.cpu_count()=%d",
            ngroups, worker_cap, cpu_count,
        )

        fallback_count = 0
        total_input_rows = 0
        total_output_rows = 0

        # Single-threaded path (or only one group)
        if workers <= 1 or ngroups <= 1:
            parts: list[pd.DataFrame] = []
            for uid, g in tqdm(grouped, total=ngroups, desc="Per-user processing"):
                total_input_rows += len(g)
                try:
                    result = _process_group(uid, g)
                    parts.append(result)
                    total_output_rows += len(result)
                except Exception as exc:
                    self.logger.warning(
                        "[_apply_per_user] Error processing user_id=%r: %r; falling back to raw data",
                        uid, exc,
                    )
                    parts.append(g.copy())
                    total_output_rows += len(g)
                    fallback_count += 1
        else:
            # Use up to configured worker cap and never more than ngroups
            self.logger.info("[_apply_per_user] Using ThreadPoolExecutor with %d workers", workers)

            groups = list(grouped)
            parts = [None] * ngroups  # type: ignore[list-item]
            for _, g in groups:
                total_input_rows += len(g)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_group, uid, g): idx
                    for idx, (uid, g) in enumerate(groups)
                }
                for future in tqdm(
                    as_completed(futures),
                    total=ngroups,
                    desc="Per-user processing",
                ):
                    idx = futures[future]
                    try:
                        result = future.result()
                        parts[idx] = result
                        total_output_rows += len(result)
                    except Exception as exc:
                        uid, g = groups[idx]
                        self.logger.warning(
                            "[_apply_per_user] Error processing user_id=%r (group index %d): %r; falling back to raw data",
                            uid, idx, exc,
                        )
                        # Fall back to unprocessed group so we don't lose that user's data
                        parts[idx] = g.copy()
                        total_output_rows += len(g)
                        fallback_count += 1

        self._last_per_user_fallback_count = fallback_count
        self._last_overlap_input = total_input_rows
        self._last_overlap_output = total_output_rows
        return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()

    def _coerce_value_numeric(
        self,
        df: pd.DataFrame,
        rep: Dict[str, Any],
        hk_type: str,
        *,
        handler_label: str,
        best_effort: bool = False
    ) -> pd.DataFrame:
        """
        Coerce `value` to numeric, drop rows with non-numeric values, and record count.
        """
        if not self.rules.ops.get("coerce_numeric", True):
            return df
        try:
            before = len(df)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            dropped = int(before - len(df))
            if rep is not None:
                rep["non_numeric_dropped"] = dropped
            self.logger.info(
                "[%s] Coerced 'value' to numeric for type '%s', dropped %d rows; %d rows remain",
                handler_label, hk_type, dropped, len(df),
            )
            return df
        except Exception:
            if best_effort:
                self.logger.info(
                    "[%s] Numeric coercion failed for type '%s', continuing without dropping non-numeric values",
                    handler_label, hk_type,
                )
                return df
            raise

    @staticmethod
    def _filter_range(
        df: pd.DataFrame, 
        min_val: Optional[float] = None, max_val: Optional[float] = None, 
        col: str = "value",
    ) -> Tuple[pd.DataFrame, int]:
        """
        Inclusive filter on `col` with optional bounds. Returns (filtered_df, n_dropped).
        """
        if df.empty or (min_val is None and max_val is None):
            return df, 0

        # boolean mask for inclusive range filter
        mask = pd.Series(True, index=df.index)
        if min_val is not None:
            mask &= df[col] >= min_val
        if max_val is not None:
            mask &= df[col] <= max_val

        n_dropped = int((~mask).sum())
        return df[mask].copy(), n_dropped

    @staticmethod
    def _is_valid_timezone_name(tz_name: str) -> bool:
        try:
            ZoneInfo(str(tz_name))
            return True
        except Exception:
            return False

    @staticmethod
    def _add_local_times(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["utc_start"] = pd.to_datetime(df["utc_start"], utc=True, errors="coerce")
        df["utc_end"]   = pd.to_datetime(df["utc_end"],   utc=True, errors="coerce")

        if "timezone" in df.columns:
            tz_series = df["timezone"].astype("string")
            local_start = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
            local_end = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

            for tz in tz_series.dropna().astype(str).unique():
                idx = tz_series == tz
                local_start.loc[idx] = df.loc[idx, "utc_start"].dt.tz_convert(tz).dt.tz_localize(None)
                local_end.loc[idx] = df.loc[idx, "utc_end"].dt.tz_convert(tz).dt.tz_localize(None)

            missing_tz = tz_series.isna() | tz_series.eq("")
            if bool(missing_tz.any()):
                local_start.loc[missing_tz] = df.loc[missing_tz, "utc_start"].dt.tz_convert("UTC").dt.tz_localize(None)
                local_end.loc[missing_tz] = df.loc[missing_tz, "utc_end"].dt.tz_convert("UTC").dt.tz_localize(None)

            df["local_start"] = local_start
            df["local_end"] = local_end
            return df

        df["local_start"] = df["utc_start"].dt.tz_convert("UTC").dt.tz_localize(None)
        df["local_end"]   = df["utc_end"].dt.tz_convert("UTC").dt.tz_localize(None)
        return df

    @staticmethod
    def combine_overlaps(user_hk_data: pd.DataFrame, value_col: str, value_kind: str = "total") -> pd.DataFrame:
        """
        Combine overlapping activities by resolving conflicts with a sweep-line
        algorithm instead of iterative pairwise trimming.

        Priority is preserved:
        1) has device_source_id
        2) higher device_source_id
        3) shorter duration
        4) later row (recency)
        """
        if value_kind not in {"level", "total"}:
            raise ValueError('value_kind must be "level" or "total"')

        minute = pd.Timedelta("1m")
        df = user_hk_data.copy()

        # --- basic preprocessing
        df["utc_start"] = pd.to_datetime(df["utc_start"], errors="coerce", utc=True)
        df["utc_end"] = pd.to_datetime(df["utc_end"], errors="coerce", utc=True)
        # drop non-positive intervals (UTC)
        df = df[df["utc_end"] > df["utc_start"]]
        # de-dupe, sort by time
        dedupe_subset = ["utc_start", "user_id", "utc_end", value_col, "type"]
        if "device_source_id" in df.columns:
            dedupe_subset.append("device_source_id")
        df = (
            df.drop_duplicates(subset=dedupe_subset, keep="last")
            .sort_values("utc_start", kind="mergesort")
            .reset_index(drop=True)
        )
        
        # nothing left
        if df.empty:
            return df

        # add duration and immutable bases (UTC)
        df["duration"] = (df["utc_end"] - df["utc_start"]) / minute
        df["_base_total"] = df[value_col]
        df["_base_duration"] = df["duration"]
        n = len(df)
        if n <= 1:
            df.drop(columns=["_base_total", "_base_duration"], inplace=True, errors="ignore")
            return df
        
        starts = df["utc_start"].to_numpy()
        ends = df["utc_end"].to_numpy()

        # no-overlap check
        if not (starts[1:] < ends[:-1]).any():
            df.drop(columns=["_base_total", "_base_duration"], inplace=True, errors="ignore")
            return df

        # --- build priority info ---------------------------------------------------
        has_dev_col = "device_source_id" in df.columns

        if has_dev_col:
            dev = df["device_source_id"].to_numpy()
            has_dev = ~pd.isna(dev)
            dev_id = np.zeros(n, dtype="int64")
            dev_id[has_dev] = dev[has_dev].astype("int64")
        else:
            has_dev = np.zeros(n, dtype=bool)
            dev_id = np.zeros(n, dtype="int64")

        duration = df["duration"].to_numpy()
        recency = np.arange(n, dtype="int64")  # later index = more recent

        # Shorter duration is better (more likely to be precise to time period)
        def priority_key(j: int):
            return (
                bool(has_dev[j]),                    # has device flag?
                int(dev_id[j]) if has_dev[j] else 0, # higher device_source_id wins
                float(-duration[j]),                 # shorter duration wins
                int(recency[j]),)                     # later row wins

        # --- construct atomic time segments ----------------------------------------
        boundaries = np.unique(np.concatenate([starts, ends]))
        if len(boundaries) <= 1:
            df.drop(columns=["_base_total", "_base_duration"], inplace=True, errors="ignore")
            return df

        K = len(boundaries)
        seg_starts = boundaries[:-1]
        seg_ends = boundaries[1:]
        seg_dur = (seg_ends - seg_starts) / minute  # float minutes

        # map each interval to boundary indices (inclusive start, exclusive end)
        start_pos = np.searchsorted(boundaries, starts, side="left")
        end_pos = np.searchsorted(boundaries, ends, side="left")

        # for each boundary index, which intervals start/end here?
        start_at = [[] for _ in range(K)]
        end_at = [[] for _ in range(K)]
        for j in range(n):
            start_at[start_pos[j]].append(j)
            end_at[end_pos[j]].append(j)

        # --- sweep line: choose winner for each segment ----------------------------
        active = set()
        seg_owner = np.full(K - 1, -1, dtype="int64")
        for i in range(K - 1):
           
            # intervals ending exactly at this boundary stop covering [t_i, t_{i+1})
            for j in end_at[i]:
                active.discard(j)
            # intervals starting at this boundary start covering [t_i, t_{i+1})
            for j in start_at[i]:
                active.add(j)
            if not active or seg_dur[i] <= 0:
                continue

            # best interval according to priority rules
            best_j = max(active, key=priority_key)
            seg_owner[i] = best_j

        # --- build output, coalescing adjacent segments with same owner ------------
        out_rows = []
        base_total = df["_base_total"].to_numpy()
        base_duration = df["_base_duration"].to_numpy()

        i = 0
        while i < K - 1:
            j = seg_owner[i]
            if j == -1 or seg_dur[i] <= 0:
                i += 1
                continue

            start = seg_starts[i]
            end = seg_ends[i]

            # merge consecutive segments with the same winner j
            k = i + 1
            while k < K - 1 and seg_owner[k] == j:
                end = seg_ends[k]
                k += 1

            dur_min = (end - start) / minute

            row = df.iloc[j].copy()
            row["utc_start"] = start
            row["utc_end"] = end
            row["duration"] = dur_min

            if value_kind == "level":
                row[value_col] = base_total[j]
            else:  # "total"
                if base_duration[j] > 0 and dur_min > 0:
                    row[value_col] = base_total[j] * (dur_min / base_duration[j])
                else:
                    row[value_col] = np.nan

            out_rows.append(row)
            i = k

        if out_rows:
            result = pd.DataFrame(out_rows)
        else:
            result = df.iloc[0:0].copy()

        # clean up
        result.drop(columns=["_base_total", "_base_duration"], inplace=True, errors="ignore")
        result.sort_values("utc_start", inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    @staticmethod
    def combine_overlaps_categorical(user_hk_data: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """
        Combines and splits overlapping categorical variables durations in the given user health data
        using a sweep-line algorithm.

        Resolution rules when multiple intervals cover the same time:
            1) If both have device_source_id and they differ: higher device_source_id wins.
            2) Otherwise, shorter duration wins.
            3) If duration ties, later row (by input order after sort) wins.

        For the final result we:
            - assign each atomic time segment a winning label according to the rules above
            - merge adjacent segments that have the same label
        """

        minute = pd.Timedelta("1m")

        # --- basic preprocessing ---------------------------------
        activity = user_hk_data.copy()
        activity["utc_start"] = pd.to_datetime(activity["utc_start"], errors="coerce", utc=True)
        activity["utc_end"] = pd.to_datetime(activity["utc_end"], errors="coerce", utc=True)

        # drop non-positive intervals
        activity = activity[activity["utc_end"] > activity["utc_start"]]

        # dedupe and sort by start time
        dedupe_subset = ["utc_start", "user_id", "utc_end", value_col, "type"]
        if "device_source_id" in activity.columns:
            dedupe_subset.append("device_source_id")
        activity = (
            activity
            .drop_duplicates(
                subset=dedupe_subset,
                keep="last",
            )
            .sort_values(by="utc_start", kind="mergesort")
            .reset_index(drop=True)
        )

        # add duration in minutes for priority logic
        activity["duration"] = (activity["utc_end"] - activity["utc_start"]) / minute

        # early exit when no overlaps or single row
        if len(activity) <= 1:
            return activity

        starts = activity["utc_start"].to_numpy()
        ends = activity["utc_end"].to_numpy()
        if not (starts[1:] < ends[:-1]).any():
            return activity

        # --- build arrays for fast priority decisions ------------------------------
        n = len(activity)
        labels = activity[value_col].to_numpy()
        duration = activity["duration"].to_numpy(dtype="float64")
        recency = np.arange(n, dtype="int64")  # later index = more recent

        if "device_source_id" in activity.columns:
            dev_series = activity["device_source_id"]
            have_dev = dev_series.notna().to_numpy()
            dev_id = np.zeros(n, dtype="int64")
            dev_id[have_dev] = dev_series[have_dev].astype("int64").to_numpy()
        else:
            have_dev = np.zeros(n, dtype=bool)
            dev_id = np.zeros(n, dtype="int64")

        def is_better(a: int, b: int) -> bool:
            """
            True if interval `a` should win over interval `b`
            according to the original resolution rules.
            """
            if b == -1:
                return True
            if a == -1:
                return False

            # device_source_id priority only when both present and differ
            if have_dev[a] and have_dev[b] and dev_id[a] != dev_id[b]:
                return dev_id[a] > dev_id[b]

            # otherwise: shorter duration wins
            if duration[a] < duration[b]:
                return True
            if duration[b] < duration[a]:
                return False

            # duration tie: later row wins
            return recency[a] > recency[b]

        # --- construct atomic time segments (boundaries) ---------------------------
        boundaries = np.unique(np.concatenate([starts, ends]))
        if len(boundaries) <= 1:
            return activity

        K = len(boundaries)
        seg_starts = boundaries[:-1]
        seg_ends = boundaries[1:]

        # map each interval to boundary indices
        start_pos = np.searchsorted(boundaries, starts, side="left")
        end_pos = np.searchsorted(boundaries, ends, side="left")

        start_at = [[] for _ in range(K)]
        end_at = [[] for _ in range(K)]
        for j in range(n):
            start_at[start_pos[j]].append(j)
            end_at[end_pos[j]].append(j)

        # --- sweep line: choose winning interval for each segment ------------------
        active = set()
        seg_owner = np.full(K - 1, -1, dtype="int64")

        for i in range(K - 1):
            # intervals ending at this boundary stop covering [t_i, t_{i+1})
            for j in end_at[i]:
                active.discard(j)

            # intervals starting at this boundary start covering [t_i, t_{i+1})
            for j in start_at[i]:
                active.add(j)

            if not active:
                continue

            seg_start = seg_starts[i]
            seg_end = seg_ends[i]
            if seg_end <= seg_start:
                continue

            # pick best interval among active using the same pairwise rules
            best = -1
            # sort for deterministic behavior
            for j in sorted(active):
                if is_better(j, best):
                    best = j

            seg_owner[i] = best

        # --- build output: coalesce adjacent segments with the same label ----------
        out_rows = []

        i = 0
        while i < K - 1:
            j = seg_owner[i]
            if j == -1:
                i += 1
                continue

            curr_label = labels[j]
            start = seg_starts[i]
            end = seg_ends[i]
            candidate_indices = {j}

            k = i + 1
            while k < K - 1:
                j2 = seg_owner[k]
                if j2 == -1 or labels[j2] != curr_label:
                    break
                end = seg_ends[k]
                candidate_indices.add(j2)
                k += 1

            # choose a representative row for metadata among all contributing intervals
            base_j = -1
            for j2 in candidate_indices:
                if is_better(j2, base_j):
                    base_j = j2

            row = activity.iloc[base_j].copy()
            row["utc_start"] = start
            row["utc_end"] = end
            row["duration"] = (end - start) / minute
            out_rows.append(row)

            i = k

        if out_rows:
            result = pd.DataFrame(out_rows)
        else:
            # no surviving segments -> empty frame with same columns
            result = activity.iloc[0:0].copy()

        # final sort & clean index
        result.sort_values("utc_start", inplace=True)
        result.reset_index(drop=True, inplace=True)

        return result

    @staticmethod
    def _filter_quantiles(
            df: pd.DataFrame, 
            col: str, 
            lower_q: float, 
            upper_q: float
        ) -> Tuple[pd.DataFrame, int]:
        """Filter rows outside [lower_q, upper_q] quantiles for the given column."""
        
        if df.empty or col not in df.columns:
            return df, 0
        series = df[col]
        if series.empty:
            return df, 0
        
        q_low = series.quantile(lower_q)
        q_high = series.quantile(upper_q)
        mask = (series >= q_low) & (series <= q_high)
        dropped = int((~mask).sum())
        
        return df[mask].copy(), dropped
    
    # ===================================================
    # Summary Statistics
    # ===================================================
    @staticmethod
    def summarize_by_type(df: pd.DataFrame, *, percentiles: Iterable[float] = (0.25, 0.75), duration_unit: str = "minutes", round_digits: int = 3) -> pd.DataFrame:
        """
        Create per-type summary statistics for a processed HealthKit dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            A processed dataframe containing at least the columns: ['user_id', 'type', 'local_start', 'local_end', 'value'].
        percentiles : Iterable[float], default=(0.25, 0.75)
            Quantiles to compute for distributions (expressed as fractions in [0, 1]).
            The median (0.50) is always reported as 'median' regardless of this list.
        duration_unit : {'seconds', 'minutes', 'hours', 'days'}, default='minutes'
            The unit used when summarizing interval durations.
        round_digits : int, default=3
            Number of decimal places to round floating-point outputs.

        Returns
        -------
        pd.DataFrame
            One row per `type` with columns covering: 
            - n_records, n_users, abs_start, abs_end, abs_span_days
            - per-user span-days distribution (per_user_span_days_<min/perc/median/max/mean/std>)
            - records-per-user distribution (records_per_user_<min/perc/median/max/mean/std>)
            - value distribution (value_<mean/median/std/min/perc/max/sum/count>) when numeric
            - duration distribution in the requested unit (duration_<unit>_<mean/median/std/min/perc/max/sum/count>)
        
        Notes
        -----
        - Non-numeric `value` entries (e.g., Sleep labels) yield NaN for 'value_*' columns.
        - 'abs_span_days' is the overall coverage window for the type: (max(local_end) - min(local_start)) in days, irrespective of gaps.
        - Assumes the input has already passed cleaning.
        """
        required = {"user_id", "type", "local_start", "local_end", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataframe missing required columns: {sorted(missing)}")

        if df.empty:
            # Return an empty frame with stable columns for downstream code
            base_cols = [
                "type", "n_records", "n_users", "abs_start", "abs_end", "abs_span_days",
            ]
            return pd.DataFrame(columns=base_cols).astype({
                "type": "object",
                "n_records": "Int64",
                "n_users": "Int64",
            })

        # Coerce time columns and compute duration
        work = df.copy()
        work["local_start"] = pd.to_datetime(work["local_start"], errors="coerce")
        work["local_end"] = pd.to_datetime(work["local_end"], errors="coerce")

        unit_map = {
            "seconds": "1s",
            "minutes": "1m",
            "hours": "1h",
            "days": "1d",
        }
        if duration_unit not in unit_map:
            raise ValueError(f"duration_unit must be one of {sorted(unit_map)}, got: {duration_unit}")

        # Row-level duration in the requested unit
        work["__duration__"] = (work["local_end"] - work["local_start"]) / pd.Timedelta(unit_map[duration_unit])

        # Numeric-only series for 'value' distribution; keep NaN where not numeric
        work["__value_num__"] = pd.to_numeric(work["value"], errors="coerce")

        # Helper: compute a statistics dict for a numeric Series
        def _stats(series: pd.Series, *, include_sum: bool, qs: Iterable[float]) -> Dict[str, Any]:
            s = pd.to_numeric(series, errors="coerce").dropna()
            if s.empty:
                # maintain key set even if empty
                out: Dict[str, Any] = {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "p50": np.nan,
                    "max": np.nan,
                }
                for q in qs:
                    out[f"p{int(round(q*100))}"] = np.nan
                if include_sum:
                    out["sum"] = np.nan
                return out

            out = {
                "count": int(s.size),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.size > 1 else float(0.0),
                "min": float(s.min()),
                "p50": float(s.quantile(0.50)),
                "max": float(s.max()),
            }
            for q in qs:
                out[f"p{int(round(q*100))}"] = float(s.quantile(q))
            if include_sum:
                out["sum"] = float(s.sum())
            return out

        # Helper: flatten and round numbers for the final output row
        def _round_if_number(x: Any) -> Any:
            if isinstance(x, (int, np.integer)):
                return int(x)
            try:
                # treat NaN/inf gracefully
                xv = float(x)
                if np.isfinite(xv):
                    return round(xv, round_digits)
                return np.nan
            except Exception:
                return x

        rows = []
        for hk_type, g in work.groupby("type", sort=True):
            # Basic counts and absolute coverage
            n_records = int(len(g))
            n_users = int(g["user_id"].nunique(dropna=False))
            abs_start = pd.to_datetime(g["local_start"].min())
            abs_end = pd.to_datetime(g["local_end"].max())
            abs_span_days = ((abs_end - abs_start) / pd.Timedelta("1d")) if pd.notna(abs_start) and pd.notna(abs_end) else np.nan

            # Per-user spans and record counts
            per_user = g.groupby("user_id", dropna=False).agg(
                user_start=("local_start", "min"),
                user_end=("local_end", "max"),
                records=("user_id", "size"),
            )
            per_user["span_days"] = (per_user["user_end"] - per_user["user_start"]) / pd.Timedelta("1d")

            # Distributions
            value_stats = _stats(g["__value_num__"], include_sum=True, qs=tuple(q for q in percentiles if q != 0.5))
            duration_stats = _stats(g["__duration__"], include_sum=True, qs=tuple(q for q in percentiles if q != 0.5))
            span_stats = _stats(per_user["span_days"], include_sum=False, qs=tuple(q for q in percentiles if q != 0.5))
            rpu_stats = _stats(per_user["records"].astype(float), include_sum=False, qs=tuple(q for q in percentiles if q != 0.5))

            # Build a flat row
            row: Dict[str, Any] = {
                "type": hk_type,
                "n_records": n_records,
                "n_users": n_users,
                "abs_start": abs_start,
                "abs_end": abs_end,
                "abs_span_days": float(abs_span_days) if pd.notna(abs_span_days) else np.nan,
            }

            # Value stats
            for k, v in value_stats.items():
                # map p50 -> median for readability
                col = "value_median" if k == "p50" else f"value_{k}"
                row[col] = _round_if_number(v)

            # Duration stats
            for k, v in duration_stats.items():
                col = f"duration_{duration_unit}_{'median' if k=='p50' else k}"
                row[col] = _round_if_number(v)

            # Per-user span days stats
            for k, v in span_stats.items():
                col = f"per_user_span_days_{'median' if k=='p50' else k}"
                row[col] = _round_if_number(v)

            # Records-per-user stats
            for k, v in rpu_stats.items():
                col = f"records_per_user_{'median' if k=='p50' else k}"
                row[col] = _round_if_number(v)

            rows.append(row)

        out = pd.DataFrame(rows).sort_values("type").reset_index(drop=True)

        # Ensure dtypes look reasonable (counts as Int64 where possible)
        for c in ["n_records", "n_users"]:
            if c in out.columns:
                out[c] = out[c].astype("Int64")

        return out

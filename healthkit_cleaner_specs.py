"""
Configuration for HealthKit data cleaning (Stage 2 of the pipeline).

Modify this file to:
  - Adjust plausibility thresholds (min/max ranges) for each HealthKit type
  - Toggle cleaning operations on or off (overlap resolution, deduplication, etc.)
  - Change quantile filtering bounds for outlier removal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Literal


# ===================================================
# Dataclass: HKCleanConfig
# ===================================================
@dataclass
class HKCleanConfig:
    """Configuration for the HealthKit cleaning stage."""

    # =========================
    # Required columns
    # =========================
    # Columns that must be present in the input DataFrame. Rarely needs modification.
    required_cols: Set[str] = field(default_factory=lambda: {
        "user_id",
        "type",
        "local_start",
        "local_end",
        "value",
        "utc_start",
        "utc_end",
    })

    # =========================
    # Plausibility thresholds
    # =========================
    # Observations with values outside these ranges are dropped (when apply_theory_thresholds is True).
    # Format: "TypeName": {"min": lower_bound, "max": upper_bound}
    #   - Omit "max" to allow any upper value.
    #   - "kcal_per_min_max" is a special key for energy types only (see kcal_per_min_policy below).
    #
    # Units: HeartRate (bpm), HeartRateVariabilitySDNN (seconds), OxygenSaturation (fraction 0-1),
    #        RespiratoryRate (breaths/min), Audio types (dB), Activity types (native HealthKit units).
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        # Audio (dB)
        "EnvironmentalAudioExposure": {"min": 0},
        "HeadphoneAudioExposure": {"min": 0},

        # Activity (counts, minutes, kcal, meters, flights)
        "StepCount": {"min": 0},
        "AppleExerciseTime": {"min": 0},
        "ActiveEnergyBurned": {"min": 0, "kcal_per_min_max": 30},
        "BasalEnergyBurned": {"min": 0, "kcal_per_min_max": 30},
        "AppleStandTime": {"min": 0},
        "DistanceCycling": {"min": 0},
        "DistanceWalkingRunning": {"min": 0},
        "FlightsClimbed": {"min": 0},

        # Vitals
        "HeartRate": {"min": 30, "max": 200},                  # bpm
        "HeartRateVariabilitySDNN": {"min": 0, "max": 1},      # seconds
        "RespiratoryRate": {"min": 0.1, "max": 100},              # breaths/min
        "OxygenSaturation": {"min": 0.50, "max": 1.0},         # fraction
    })

    # =========================
    # Behavioral options
    # =========================
    # When energy burn rate exceeds kcal_per_min_max:
    #   "cap"  — clamp the value at the threshold
    #   "drop" — remove the observation entirely
    kcal_per_min_policy: Literal["cap", "drop"] = "cap"

    # Thread pool size for parallel overlap resolution. None = auto-detect CPU count.
    max_workers: int | None = None

    # =========================
    # Operation toggles
    # =========================
    # Each toggle enables/disables a cleaning operation. Set to False to skip that step.
    ops: Dict[str, bool] = field(default_factory=lambda: {
        "coerce_numeric": True,            # Convert the value column to numeric; required by most downstream ops
        "convert_energy_to_kcal": True,    # Convert joule-based energy readings to kilocalories
        "cap_kcal_per_min": True,          # Enforce kcal/min rate cap on energy types (see kcal_per_min_policy)
        "apply_theory_thresholds": True,   # Drop values outside the thresholds ranges above
        "deduplicate": True,               # Remove exact duplicate rows (same user/type/interval/value)
        "normalize_sleep_labels": True,    # Strip "HKCategoryValueSleepAnalysis" prefix from sleep values
        "apply_quantile_filter": False,    # Drop values outside quantile_lower/upper bounds (extra scan per type)
        "apply_watch_device_filter": True, # Keep only Apple Watch observations (drops iPhone/iPad sources)
        "combine_overlaps": True,          # Resolve overlapping numeric intervals (e.g., double-counted steps)
        "combine_overlaps_cat": True,      # Resolve overlapping categorical intervals (e.g., sleep stages)
    })

    # =========================
    # Quantile options
    # =========================
    # Percentile bounds for outlier removal. Only used when ops["apply_quantile_filter"] is True.
    # 0.01 = 1st percentile, 0.99 = 99th percentile.
    quantile_lower: float = 0.01
    quantile_upper: float = 0.99


    # =========================
    # Self-validate config values
    # =========================
    def validate(self):
        """Validate cross-field constraints."""
        ops = self.ops
        if not (0.0 <= self.quantile_lower < self.quantile_upper <= 1.0):
            raise ValueError(
                f"Require 0 <= quantile_lower < quantile_upper <= 1, "
                f"got ({self.quantile_lower}, {self.quantile_upper})"
            )
        if not ops.get("coerce_numeric", True):
            for dep in ("convert_energy_to_kcal", "cap_kcal_per_min",
                         "apply_theory_thresholds", "apply_quantile_filter"):
                if ops.get(dep, False):
                    raise ValueError(
                        f"ops['{dep}'] requires coerce_numeric=True"
                    )
        if ops.get("cap_kcal_per_min", False) and not ops.get("convert_energy_to_kcal", True):
            raise ValueError(
                "ops['cap_kcal_per_min']=True requires ops['convert_energy_to_kcal']=True"
            )

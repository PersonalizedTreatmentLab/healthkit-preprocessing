"""
User configuration for HealthKit cleaning (Stage 2 of the pipeline).

HOW TO CUSTOMIZE THIS STAGE
- Edit the default values in the USER CONFIGURATION section below, or
  pass overrides when constructing HKCleanConfig.
- Each configurable includes its available options and effect.
- Do not edit code below the INTERNAL IMPLEMENTATION boundary unless
  you are changing how the cleaning pipeline itself works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Set

# =============================================================================
# USER CONFIGURATION — EDIT VALUES IN THIS SECTION
# =============================================================================


@dataclass
class HKCleanConfig:
    """Settings that users can change for the HealthKit cleaning stage."""

    # -------------------------------------------------------------------------
    # Required input columns
    # -------------------------------------------------------------------------
    # Edit this set when the input schema intentionally uses additional or
    # different required columns.
    required_cols: Set[str] = field(default_factory=lambda: {
        "user_id",
        "type",
        "local_start",
        "local_end",
        "value",
        "utc_start",
        "utc_end",
    })

    # -------------------------------------------------------------------------
    # Plausibility thresholds
    # -------------------------------------------------------------------------
    # Edit limits or actions here to control how implausible values are handled.
    # Omit a check when that check should not be applied to a HealthKit type.
    #
    # Rule format: "TypeName": {
    #     "min" | "max" | "rate_per_min_max": {
    #         "limit": number,
    #         "action": "cap" | "drop",
    #     }
    # }
    #
    # Units: HeartRate (bpm), HeartRateVariabilitySDNN (seconds), OxygenSaturation (fraction 0-1),
    #        RespiratoryRate (breaths/min), Audio types (dB), Activity types (native HealthKit units).
    thresholds: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=lambda: {
        # Audio (dB)
        "EnvironmentalAudioExposure": {"min": {"limit": 0, "action": "drop"}},
        "HeadphoneAudioExposure": {"min": {"limit": 0, "action": "drop"}},

        # Activity (counts, minutes, kcal, meters, flights)
        "StepCount": {
            "min": {"limit": 0, "action": "drop"},
            "rate_per_min_max": {"limit": 350, "action": "cap"},
        },
        "AppleExerciseTime": {"min": {"limit": 0, "action": "drop"}},
        "ActiveEnergyBurned": {
            "min": {"limit": 0, "action": "drop"},
            "rate_per_min_max": {"limit": 30, "action": "cap"},
        },
        "BasalEnergyBurned": {
            "min": {"limit": 0, "action": "drop"},
            "rate_per_min_max": {"limit": 10, "action": "cap"},
        },
        "AppleStandTime": {"min": {"limit": 0, "action": "drop"}},
        "DistanceCycling": {
            "min": {"limit": 0, "action": "drop"},
            "rate_per_min_max": {"limit": 2000, "action": "cap"},
        },
        "DistanceWalkingRunning": {
            "min": {"limit": 0, "action": "drop"},
            "rate_per_min_max": {"limit": 750, "action": "cap"},
        },
        "FlightsClimbed": {
            "min": {"limit": 0, "action": "drop"},
            "max": {"limit": 8000, "action": "drop"},
        },

        # Vitals
        "HeartRate": {
            "min": {"limit": 25, "action": "drop"},
            "max": {"limit": 240, "action": "drop"},
        },
        "HeartRateVariabilitySDNN": {
            "min": {"limit": 0, "action": "drop"},
            "max": {"limit": 1, "action": "drop"},
        },
        "RespiratoryRate": {
            "min": {"limit": 0.1, "action": "drop"},
            "max": {"limit": 100, "action": "drop"},
        },
        "OxygenSaturation": {
            "min": {"limit": 0.50, "action": "drop"},
            "max": {"limit": 1.0, "action": "drop"},
        },
    })

    # -------------------------------------------------------------------------
    # Parallelism
    # -------------------------------------------------------------------------
    # Thread pool size for parallel overlap resolution. None = auto-detect CPU count.
    max_workers: int | None = None

    # -------------------------------------------------------------------------
    # Cleaning-operation toggles
    # -------------------------------------------------------------------------
    # Set an operation to False to skip it. Keep coerce_numeric=True when using
    # energy conversion, plausibility thresholds, or quantile filtering.
    ops: Dict[str, bool] = field(default_factory=lambda: {
        "coerce_numeric": True,            # Convert the value column to numeric; required by most downstream ops
        "convert_energy_to_kcal": True,    # Convert joule-based energy readings to kilocalories
        "apply_theory_thresholds": True,   # Apply configured min, max, and rate checks
        "deduplicate": True,               # Remove exact duplicate rows (same user/type/interval/value)
        "normalize_sleep_labels": True,    # Strip "HKCategoryValueSleepAnalysis" prefix from sleep values
        "apply_quantile_filter": False,    # Drop values outside quantile_lower/upper bounds (extra scan per type)
        "apply_watch_device_filter": True, # Keep only Apple Watch observations (drops iPhone/iPad sources)
        "combine_overlaps": True,          # Resolve overlapping numeric intervals (e.g., double-counted steps)
        "combine_overlaps_cat": True,      # Resolve overlapping categorical intervals (e.g., sleep stages)
    })

    # -------------------------------------------------------------------------
    # Quantile-filter settings
    # -------------------------------------------------------------------------
    # Percentile bounds for outlier removal. Only used when ops["apply_quantile_filter"] is True.
    # 0.01 = 1st percentile, 0.99 = 99th percentile.
    quantile_lower: float = 0.01
    quantile_upper: float = 0.99

    def validate(self):
        """Validate the configured values."""
        _validate_clean_config(self)

# =============================================================================
# INTERNAL IMPLEMENTATION — NO ROUTINE CONFIGURATION BELOW THIS LINE
# =============================================================================
# The helpers below validate and execute the settings above.
# Change them only when modifying pipeline behavior itself.


def _validate_clean_config(self: HKCleanConfig) -> None:
    """Validate cross-field constraints."""
    ops = self.ops
    if not (0.0 <= self.quantile_lower < self.quantile_upper <= 1.0):
        raise ValueError(
            f"Require 0 <= quantile_lower < quantile_upper <= 1, "
            f"got ({self.quantile_lower}, {self.quantile_upper})"
        )
    if not ops.get("coerce_numeric", True):
        for dep in ("convert_energy_to_kcal", "apply_theory_thresholds", "apply_quantile_filter"):
            if ops.get(dep, False):
                raise ValueError(
                    f"ops['{dep}'] requires coerce_numeric=True"
                )
    allowed_checks = {"min", "max", "rate_per_min_max"}
    for hk_type, checks in self.thresholds.items():
        unknown = sorted(set(checks) - allowed_checks)
        if unknown:
            raise ValueError(f"thresholds['{hk_type}'] has unknown check(s): {unknown}")
        limits: Dict[str, float] = {}
        for check_name, rule in checks.items():
            if not isinstance(rule, dict):
                raise ValueError(f"thresholds['{hk_type}']['{check_name}'] must be a rule mapping")
            unknown_rule = sorted(set(rule) - {"limit", "action"})
            if unknown_rule:
                raise ValueError(
                    f"thresholds['{hk_type}']['{check_name}'] has unknown key(s): {unknown_rule}"
                )
            if "limit" not in rule or "action" not in rule:
                raise ValueError(
                    f"thresholds['{hk_type}']['{check_name}'] requires 'limit' and 'action'"
                )
            try:
                limit = float(rule["limit"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"thresholds['{hk_type}']['{check_name}']['limit'] must be numeric"
                ) from exc
            if not math.isfinite(limit):
                raise ValueError(
                    f"thresholds['{hk_type}']['{check_name}']['limit'] must be finite"
                )
            if rule["action"] not in {"cap", "drop"}:
                raise ValueError(
                    f"thresholds['{hk_type}']['{check_name}']['action'] must be 'cap' or 'drop'"
                )
            if check_name == "rate_per_min_max" and limit <= 0:
                raise ValueError(
                    f"thresholds['{hk_type}']['rate_per_min_max']['limit'] must be > 0"
                )
            limits[check_name] = limit
        if "min" in limits and "max" in limits and limits["min"] > limits["max"]:
            raise ValueError(f"thresholds['{hk_type}'] min limit must be <= max limit")

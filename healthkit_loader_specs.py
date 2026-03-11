"""
Configuration for HealthKit data loading (Stage 1 of the pipeline).

Controls how raw HealthKit exports are ingested, including timezone handling
and daylight saving time (DST) resolution. Modify this file to change the
default timezone or how the loader handles missing/ambiguous time data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class HKLoadConfig:
    """Configuration for HealthKit data loading."""

    # --- Timezone Handling ---
    # What to do when a row's timezone is missing or invalid:
    #   "fill_default" — substitute default_tz (keeps the row)
    #   "drop_rows"    — remove rows with missing/invalid timezone
    #   "error"        — raise an error immediately
    timezone_missing_policy: Literal["fill_default", "drop_rows", "error"] = "fill_default"

    # Fallback IANA timezone string (e.g., "America/New_York", "Europe/London").
    # Only used when timezone_missing_policy is "fill_default".
    default_tz: str = "America/Los_Angeles"

    # --- DST Resolution ---
    # How to handle ambiguous times during fall-back (clocks go back, same time occurs twice):
    #   "infer" — pick the most likely offset automatically
    #   "NaT"   — set the timestamp to NaT (missing)
    #   "raise"  — raise an error
    dst_ambiguous: Literal["infer", "NaT", "raise"] = "infer"

    # How to handle nonexistent times during spring-forward (clocks skip ahead):
    #   "shift_forward" — move the time forward to the next valid time
    #   "NaT"           — set the timestamp to NaT (missing)
    #   "raise"          — raise an error
    dst_nonexistent: Literal["shift_forward", "NaT", "raise"] = "shift_forward"

    # =========================
    # Self-validate config values
    # =========================
    def validate(self):
        """Validate config fields."""
        from zoneinfo import ZoneInfo
        try:
            ZoneInfo(self.default_tz)
        except (KeyError, ValueError):
            raise ValueError(
                f"default_tz is not a valid IANA timezone: {self.default_tz!r}"
            )
        if self.timezone_missing_policy not in {"fill_default", "drop_rows", "error"}:
            raise ValueError(
                f"timezone_missing_policy must be one of "
                f"{{'fill_default', 'drop_rows', 'error'}}, got {self.timezone_missing_policy!r}"
            )
        if self.dst_ambiguous not in {"infer", "NaT", "raise"}:
            raise ValueError(
                f"dst_ambiguous must be one of {{'infer', 'NaT', 'raise'}}, got {self.dst_ambiguous!r}"
            )
        if self.dst_nonexistent not in {"shift_forward", "NaT", "raise"}:
            raise ValueError(
                f"dst_nonexistent must be one of {{'shift_forward', 'NaT', 'raise'}}, got {self.dst_nonexistent!r}"
            )

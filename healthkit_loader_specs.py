"""
User configuration for HealthKit loading (Stage 1 of the pipeline).

HOW TO CUSTOMIZE THIS STAGE
- Edit the default values in the USER CONFIGURATION section below, or
  pass overrides when constructing HKLoadConfig.
- Each configurable includes its available options and effect.
- Do not edit code below the INTERNAL IMPLEMENTATION boundary unless
  you are changing how the loading pipeline itself works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# =============================================================================
# USER CONFIGURATION — EDIT VALUES IN THIS SECTION
# =============================================================================


@dataclass
class HKLoadConfig:
    """Settings that users can change for HealthKit data loading."""

    # -------------------------------------------------------------------------
    # Missing or invalid timezones
    # -------------------------------------------------------------------------
    # Change timezone_missing_policy to choose whether affected rows use
    # default_tz, are dropped, or cause an error:
    #   "fill_default" — substitute default_tz (keeps the row)
    #   "drop_rows"    — remove rows with missing/invalid timezone
    #   "error"        — raise an error immediately
    timezone_missing_policy: Literal["fill_default", "drop_rows", "error"] = "fill_default"

    # Fallback IANA timezone string (e.g., "America/New_York", "Europe/London").
    # Only used when timezone_missing_policy is "fill_default".
    default_tz: str = "America/Los_Angeles"

    # -------------------------------------------------------------------------
    # Daylight saving time resolution
    # -------------------------------------------------------------------------
    # Change these values only to select how ambiguous fall-back times and
    # nonexistent spring-forward times are resolved.
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

    def validate(self):
        """Validate the configured values."""
        _validate_load_config(self)

# =============================================================================
# INTERNAL IMPLEMENTATION — NO ROUTINE CONFIGURATION BELOW THIS LINE
# =============================================================================
# The helpers below validate and execute the settings above.
# Change them only when modifying pipeline behavior itself.


def _validate_load_config(self: HKLoadConfig) -> None:
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

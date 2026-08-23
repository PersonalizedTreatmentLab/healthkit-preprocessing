import pandas as pd
import pytest

from healthkit_cleaner import HealthKitCleaner
from healthkit_cleaner_specs import HKCleanConfig


def _activity_rows(values, durations_minutes):
    start = pd.Timestamp("2025-01-01T12:00:00Z")
    rows = []
    for index, (value, duration) in enumerate(zip(values, durations_minutes)):
        row_start = start + pd.Timedelta(minutes=index * 10)
        row_end = row_start + pd.Timedelta(minutes=duration)
        rows.append(
            {
                "user_id": 1,
                "type": "StepCount",
                "utc_start": row_start,
                "utc_end": row_end,
                "local_start": row_start.tz_localize(None),
                "local_end": row_end.tz_localize(None),
                "value": value,
            }
        )
    return pd.DataFrame(rows)


def _ops_without_unrelated_cleaning():
    return {
        "coerce_numeric": True,
        "convert_energy_to_kcal": False,
        "apply_theory_thresholds": True,
        "deduplicate": False,
        "normalize_sleep_labels": True,
        "apply_quantile_filter": False,
        "apply_watch_device_filter": False,
        "combine_overlaps": False,
        "combine_overlaps_cat": False,
    }


def test_independent_threshold_actions_drop_minimum_and_cap_rate():
    config = HKCleanConfig(
        thresholds={
            "StepCount": {
                "min": {"limit": 0, "action": "drop"},
                "rate_per_min_max": {"limit": 350, "action": "cap"},
            }
        },
        ops=_ops_without_unrelated_cleaning(),
    )

    cleaned, report = HealthKitCleaner(config).clean(
        _activity_rows(values=[-5, 800, 100], durations_minutes=[1, 1, 1]),
        return_report=True,
    )

    assert cleaned["value"].tolist() == [350.0, 100.0]
    assert report["StepCount"]["min_dropped"] == 1
    assert report["StepCount"]["rate_per_min_max_capped"] == 1


def test_zero_duration_is_skipped_by_rate_check_and_reported():
    config = HKCleanConfig(
        thresholds={
            "StepCount": {
                "rate_per_min_max": {"limit": 350, "action": "drop"},
            }
        },
        ops=_ops_without_unrelated_cleaning(),
    )

    cleaned, report = HealthKitCleaner(config).clean(
        _activity_rows(values=[100], durations_minutes=[0]),
        return_report=True,
    )

    assert cleaned["value"].tolist() == [100.0]
    assert report["StepCount"]["rate_per_min_max_skipped_invalid_duration"] == 1


def test_minimum_and_maximum_caps_are_independent():
    config = HKCleanConfig(
        thresholds={
            "StepCount": {
                "min": {"limit": 0, "action": "cap"},
                "max": {"limit": 500, "action": "cap"},
            }
        },
        ops=_ops_without_unrelated_cleaning(),
    )

    cleaned, report = HealthKitCleaner(config).clean(
        _activity_rows(values=[-5, 600], durations_minutes=[1, 1]),
        return_report=True,
    )

    assert cleaned["value"].tolist() == [0.0, 500.0]
    assert report["StepCount"]["min_capped"] == 1
    assert report["StepCount"]["max_capped"] == 1


def test_rate_rule_can_drop_instead_of_cap():
    config = HKCleanConfig(
        thresholds={
            "StepCount": {
                "rate_per_min_max": {"limit": 350, "action": "drop"},
            }
        },
        ops=_ops_without_unrelated_cleaning(),
    )

    cleaned, report = HealthKitCleaner(config).clean(
        _activity_rows(values=[400, 100], durations_minutes=[1, 1]),
        return_report=True,
    )

    assert cleaned["value"].tolist() == [100.0]
    assert report["StepCount"]["rate_per_min_max_dropped"] == 1


@pytest.mark.parametrize(
    "thresholds, message",
    [
        ({"StepCount": {"minimum": {"limit": 0, "action": "drop"}}}, "unknown check"),
        ({"StepCount": {"min": {"limit": 0, "action": "ignore"}}}, "action"),
        ({"StepCount": {"min": {"action": "drop"}}}, "requires 'limit' and 'action'"),
        ({"StepCount": {"max": {"limit": float("inf"), "action": "drop"}}}, "finite"),
        ({"StepCount": {"rate_per_min_max": {"limit": 0, "action": "cap"}}}, "must be > 0"),
        (
            {
                "StepCount": {
                    "min": {"limit": 10, "action": "drop"},
                    "max": {"limit": 5, "action": "drop"},
                }
            },
            "min limit must be <= max limit",
        ),
    ],
)
def test_threshold_validation_rejects_ambiguous_rules(thresholds, message):
    with pytest.raises(ValueError, match=message):
        HKCleanConfig(thresholds=thresholds, ops=_ops_without_unrelated_cleaning()).validate()


def test_default_activity_thresholds_use_configured_rate_limits():
    thresholds = HKCleanConfig().thresholds

    assert thresholds["StepCount"]["rate_per_min_max"]["limit"] == 350
    assert thresholds["ActiveEnergyBurned"]["rate_per_min_max"]["limit"] == 30
    assert thresholds["BasalEnergyBurned"]["rate_per_min_max"]["limit"] == 10
    assert thresholds["DistanceCycling"]["rate_per_min_max"]["limit"] == 2000
    assert thresholds["DistanceWalkingRunning"]["rate_per_min_max"]["limit"] == 750

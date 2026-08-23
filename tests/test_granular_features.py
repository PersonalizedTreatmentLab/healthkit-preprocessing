import pandas as pd

from healthkit_aggregator import HealthKitAggregator
import healthkit_aggregator_specs as specs
from healthkit_aggregator_specs import HKAggConfig, build_default_type_registry


def _interval_fragment(start, end, window_start, window_end, value=1.0):
    return {
        "user_id": 1,
        "window_id": 10,
        "window_start": pd.Timestamp(window_start),
        "window_end": pd.Timestamp(window_end),
        "window_seconds": 3600.0,
        "duration": (
            min(pd.Timestamp(end), pd.Timestamp(window_end))
            - max(pd.Timestamp(start), pd.Timestamp(window_start))
        ).total_seconds(),
        "value": value,
        "utc_start": pd.Timestamp(start),
        "utc_end": pd.Timestamp(end),
    }


def test_default_sleep_feature_spec_builds_all_60_granular_features():
    registry = build_default_type_registry(specs.SleepFeatureSpec())
    aggregates = registry["SleepAnalysis"]["aggregates"]

    assert len(aggregates) == 60
    assert "sum_label_asleep_core" in aggregates
    assert "continues_from_previous_bin_label_awake_unspecified" in aggregates
    assert "continues_into_next_bin_any_asleep" in aggregates

    fragment = _interval_fragment(
        "2025-01-01T12:10:00Z",
        "2025-01-01T12:40:00Z",
        "2025-01-01T12:00:00Z",
        "2025-01-01T13:00:00Z",
        value="AsleepCore",
    )
    fragment["label"] = fragment.pop("value")
    output = HealthKitAggregator(HKAggConfig()).aggregate_type_within_windows(
        "SleepAnalysis", pd.DataFrame([fragment])
    )
    feature_columns = [
        column for column in output.columns if column.startswith("SleepAnalysis__")
    ]
    assert feature_columns == [f"SleepAnalysis__{name}" for name in aggregates]


def test_sleep_feature_spec_selects_labels_unions_and_statistics_independently():
    spec = specs.SleepFeatureSpec(
        labels=("asleep_core",),
        unions=("asleep",),
        statistics=("sum", "count"),
    )

    aggregates = build_default_type_registry(spec)["SleepAnalysis"]["aggregates"]

    assert aggregates == (
        "sum_label_asleep_core",
        "count_label_asleep_core",
        "sum_any_asleep",
        "count_any_asleep",
    )


def test_interval_continuity_requires_strict_crossing_of_window_boundary():
    aggregator = HealthKitAggregator(HKAggConfig())
    fragments = pd.DataFrame(
        [
            _interval_fragment(
                "2025-01-01T11:50:00Z",
                "2025-01-01T12:20:00Z",
                "2025-01-01T12:00:00Z",
                "2025-01-01T13:00:00Z",
            ),
            _interval_fragment(
                "2025-01-01T12:30:00Z",
                "2025-01-01T13:15:00Z",
                "2025-01-01T12:00:00Z",
                "2025-01-01T13:00:00Z",
            ),
        ]
    )

    out = aggregator.aggregate_type_within_windows("StepCount", fragments).iloc[0]

    assert out["StepCount__continues_from_previous_bin"] == 1.0
    assert out["StepCount__continues_into_next_bin"] == 1.0

    exact = pd.DataFrame(
        [
            _interval_fragment(
                "2025-01-01T12:00:00Z",
                "2025-01-01T13:00:00Z",
                "2025-01-01T12:00:00Z",
                "2025-01-01T13:00:00Z",
            )
        ]
    )
    exact_out = aggregator.aggregate_type_within_windows("StepCount", exact).iloc[0]

    assert exact_out["StepCount__continues_from_previous_bin"] == 0.0
    assert exact_out["StepCount__continues_into_next_bin"] == 0.0


def test_sleep_unions_retain_awake_unspecified_overlap():
    aggregator = HealthKitAggregator(HKAggConfig())
    fragment = _interval_fragment(
        "2025-01-01T11:50:00Z",
        "2025-01-01T13:15:00Z",
        "2025-01-01T12:00:00Z",
        "2025-01-01T13:00:00Z",
        value="AwakeUnspecified",
    )
    fragment["label"] = fragment.pop("value")

    out = aggregator.aggregate_type_within_windows(
        "SleepAnalysis", pd.DataFrame([fragment])
    ).iloc[0]

    assert out["SleepAnalysis__sum_any_asleep"] == 3600.0
    assert out["SleepAnalysis__sum_any_awake"] == 3600.0
    assert out["SleepAnalysis__continues_from_previous_bin_any_asleep"] == 1.0
    assert out["SleepAnalysis__continues_into_next_bin_any_awake"] == 1.0


def test_categorical_continuity_is_zero_at_exact_boundaries():
    aggregator = HealthKitAggregator(HKAggConfig())
    fragment = _interval_fragment(
        "2025-01-01T12:00:00Z",
        "2025-01-01T13:00:00Z",
        "2025-01-01T12:00:00Z",
        "2025-01-01T13:00:00Z",
        value="AsleepCore",
    )
    fragment["label"] = fragment.pop("value")

    out = aggregator.aggregate_type_within_windows(
        "SleepAnalysis", pd.DataFrame([fragment])
    ).iloc[0]

    assert out["SleepAnalysis__continues_from_previous_bin_label_asleep_core"] == 0.0
    assert out["SleepAnalysis__continues_into_next_bin_any_asleep"] == 0.0

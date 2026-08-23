from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from healthkit_aggregator import HealthKitAggregator
from healthkit_aggregator_specs import HKAggConfig, build_default_type_registry


def _wide(**features):
    row = {
        "user_id": 1,
        "window_id": 11,
        "window_start": pd.Timestamp("2025-01-01T12:00:00Z"),
        "window_end": pd.Timestamp("2025-01-01T13:00:00Z"),
        "window_seconds": 3600.0,
        "wear_seconds": 3600.0,
        "wear_fraction": 1.0,
        "watch_on": True,
    }
    row.update(features)
    return pd.DataFrame([row])


def _presence():
    return pd.DataFrame(
        [{
            "user_id": 1,
            "window_id": 11,
            "window_start": pd.Timestamp("2025-01-01T12:00:00Z"),
            "window_end": pd.Timestamp("2025-01-01T13:00:00Z"),
            "window_seconds": 3600.0,
            "presence_any": True,
            "presence_non_watch": False,
            "source_active__watch": True,
            "source_active__phone": False,
            "source_active__other": False,
        }]
    )


def test_zero_fill_is_disabled_by_default():
    aggregator = HealthKitAggregator(HKAggConfig())
    wide = _wide(StepCount__sum=np.nan)

    assert not aggregator.cfg.apply_zero_fill
    assert pd.isna(wide.loc[0, "StepCount__sum"])


def test_participant_evidence_and_assumed_availability_are_per_statistic():
    registry = deepcopy(build_default_type_registry())
    registry["StepCount"]["zero_fill_rules"] = {
        "sum": {
            "gate": ("watch_on", {}),
            "fill_value": 0.0,
            "availability": "participant_evidence",
        },
        "duration": {
            "gate": ("watch_on", {}),
            "fill_value": -1.0,
            "availability": "assume_available",
        },
    }
    aggregator = HealthKitAggregator(HKAggConfig(type_registry=registry, apply_zero_fill=True))
    wide = _wide(StepCount__sum=np.nan, StepCount__duration=np.nan)

    unavailable = aggregator.apply_zero_fill_policy(
        wide, _presence(), {},
        participant_availability={"type_users": {}, "label_users": {}},
    )
    assert pd.isna(unavailable.loc[0, "StepCount__sum"])
    assert unavailable.loc[0, "StepCount__duration"] == -1.0

    available = aggregator.apply_zero_fill_policy(
        wide, _presence(), {},
        participant_availability={
            "type_users": {"StepCount": {1}},
            "label_users": {},
        },
    )
    assert available.loc[0, "StepCount__sum"] == 0.0


def test_sleep_label_and_union_availability_use_distinct_evidence():
    aggregator = HealthKitAggregator(HKAggConfig(apply_zero_fill=True))
    wide = _wide(
        SleepAnalysis__sum_label_asleep_core=np.nan,
        SleepAnalysis__sum_label_awake=np.nan,
        SleepAnalysis__sum_any_asleep=np.nan,
    )
    availability = {
        "type_users": {"SleepAnalysis": {1}},
        "label_users": {
            ("SleepAnalysis", "AsleepCore"): {1},
        },
    }

    out = aggregator.apply_zero_fill_policy(
        wide, _presence(), {}, participant_availability=availability
    )

    assert out.loc[0, "SleepAnalysis__sum_label_asleep_core"] == 0.0
    assert pd.isna(out.loc[0, "SleepAnalysis__sum_label_awake"])
    assert out.loc[0, "SleepAnalysis__sum_any_asleep"] == 0.0


def test_participant_availability_is_built_from_complete_input():
    aggregator = HealthKitAggregator(HKAggConfig())
    complete = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "type": ["StepCount", "SleepAnalysis", "SleepAnalysis"],
            "value": [10.0, "AsleepCore", "Awake"],
        }
    )

    evidence = aggregator.build_participant_availability(complete)

    assert evidence["type_users"]["StepCount"] == {1}
    assert evidence["label_users"][("SleepAnalysis", "AsleepCore")] == {1}
    assert evidence["label_users"][("SleepAnalysis", "Awake")] == {2}


def test_manifest_and_assembled_granular_schema_follow_registry_order():
    registry = deepcopy(build_default_type_registry())
    registry = {
        "StepCount": registry["StepCount"],
        "SleepAnalysis": registry["SleepAnalysis"],
    }
    aggregator = HealthKitAggregator(HKAggConfig(type_registry=registry))
    windows = pd.DataFrame(
        [{
            "user_id": 1,
            "window_id": 4,
            "window_start": pd.Timestamp("2025-01-01T12:00:00Z"),
            "window_end": pd.Timestamp("2025-01-01T13:00:00Z"),
            "window_seconds": 3600.0,
            "bin_index": 4,
        }]
    )

    manifest = aggregator.get_feature_manifest()
    wide = aggregator.assemble_feature_matrix(windows, {})
    expected = [
        f"StepCount__{stat}" for stat in registry["StepCount"]["aggregates"]
    ] + [
        f"SleepAnalysis__{stat}"
        for stat in registry["SleepAnalysis"]["aggregates"]
    ]

    assert manifest["feature_name"].tolist() == expected
    assert manifest["cadence"].eq("granular").all()
    assert wide.columns.tolist()[-len(expected):] == expected
    assert wide[expected].isna().all().all()


@pytest.mark.parametrize(
    "rule",
    [
        {"gate": ("unknown", {}), "fill_value": 0.0, "availability": "participant_evidence"},
        {"gate": ("watch_on", {}), "fill_value": np.inf, "availability": "participant_evidence"},
        {"gate": ("watch_on", {}), "fill_value": 0.0, "availability": "sometimes"},
        {"gate": ("watch_on", {}), "fill_value": 0.0},
    ],
)
def test_invalid_zero_fill_rules_fail_configuration_validation(rule):
    registry = deepcopy(build_default_type_registry())
    registry["StepCount"]["zero_fill_rules"] = {"sum": rule}

    with pytest.raises(ValueError):
        HealthKitAggregator(HKAggConfig(type_registry=registry))

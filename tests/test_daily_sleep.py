import numpy as np
import pandas as pd

from healthkit_aggregator import HealthKitAggregator
from healthkit_aggregator_specs import AssessmentWindowSpec, HKAggConfig


def _row(type_name, value, start, end=None, user_id=1):
    end = start if end is None else end
    return {
        "user_id": user_id,
        "type": type_name,
        "value": value,
        "utc_start": pd.Timestamp(start),
        "utc_end": pd.Timestamp(end),
        "local_start": pd.Timestamp(start).tz_convert("UTC").tz_localize(None),
        "local_end": pd.Timestamp(end).tz_convert("UTC").tz_localize(None),
    }


def _sleep_fixture():
    return pd.DataFrame(
        [
            _row("SleepAnalysis", "InBed", "2025-01-01T22:00:00Z", "2025-01-01T22:30:00Z"),
            _row("SleepAnalysis", "AsleepCore", "2025-01-01T22:30:00Z", "2025-01-01T23:30:00Z"),
            _row("SleepAnalysis", "Awake", "2025-01-01T23:30:00Z", "2025-01-01T23:45:00Z"),
            _row("SleepAnalysis", "AsleepREM", "2025-01-01T23:45:00Z", "2025-01-02T00:45:00Z"),
            _row("SleepAnalysis", "Awake", "2025-01-02T00:45:00Z", "2025-01-02T01:00:00Z"),
            # Starts before 15:00, so the whole record remains on the day ending Jan 2.
            _row("SleepAnalysis", "InBed", "2025-01-02T14:30:00Z", "2025-01-02T15:30:00Z"),
            # Current assessment-containing sleep day.
            _row("SleepAnalysis", "AsleepCore", "2025-01-02T15:30:00Z", "2025-01-02T16:00:00Z"),
            _row("SleepAnalysis", "AsleepREM", "2025-01-02T16:30:00Z", "2025-01-02T17:00:00Z"),
            # Crosses the later assessment and must be excluded, not clipped.
            _row("SleepAnalysis", "AsleepCore", "2025-01-02T17:00:00Z", "2025-01-02T18:00:00Z"),
            _row("HeartRate", 60.0, "2025-01-01T23:00:00Z"),
            _row("HeartRate", 200.0, "2025-01-01T23:40:00Z"),  # awake gap
            _row("HeartRate", 80.0, "2025-01-02T00:00:00Z"),
            _row("HeartRate", 250.0, "2025-01-02T17:40:00Z"),  # post-assessment
            _row("HeartRateVariabilitySDNN", 40.0, "2025-01-01T23:10:00Z"),
            _row("HeartRateVariabilitySDNN", 90.0, "2025-01-01T23:40:00Z"),
            _row("EnvironmentalAudioExposure", 30.0, "2025-01-01T22:15:00Z"),
            _row("EnvironmentalAudioExposure", 50.0, "2025-01-01T23:40:00Z"),
            _row("EnvironmentalAudioExposure", 70.0, "2025-01-02T02:00:00Z"),
        ]
    )


def _assessments(*anchors):
    return pd.DataFrame(
        {
            "user_id": [1] * len(anchors),
            "assessment_id": [f"a{i + 1}" for i in range(len(anchors))],
            "anchor_time": pd.to_datetime(list(anchors), utc=True),
        }
    )


def _schedule():
    return pd.DataFrame(
        [{
            "user_id": 1,
            "start_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
            "end_utc": pd.Timestamp("2025-01-04T00:00:00Z"),
            "tz_name": "UTC",
        }]
    )


def test_daily_sleep_uses_whole_records_revised_waso_and_actual_context_intervals():
    aggregator = HealthKitAggregator(HKAggConfig())
    daily = aggregator.compute_daily_sleep_features(
        _sleep_fixture(),
        assessments=_assessments("2025-01-02T17:36:00Z"),
        schedule=_schedule(),
        qc=False,
    )
    prior = daily.loc[daily["sleep_date"] == pd.Timestamp("2025-01-02").date()].iloc[0]

    assert prior["sleep_sleep_duration_day"] == 2.0
    assert prior["sleep_InBed_sum_day"] == 1.5
    assert prior["sleep_wake_after_sleep_onset_day"] == 0.25
    assert prior["sleep_sleep_hr_median_day"] == 70.0
    assert prior["sleep_sleep_hrv_median_day"] == 40.0
    assert prior["sleep_bedrest_audio_median_day"] == 40.0
    assert "sleep_bedrest_audio_leq_day" not in daily.columns
    assert prior["sleep_first_bedrest_start_hours_day"] == 22.0
    assert prior["sleep_last_bedrest_end_hours_day"] == 39.5
    assert prior["sleep_first_onset_hours_day"] == 22.5
    assert prior["sleep_last_offset_hours_day"] == 24.75


def test_daily_sleep_excludes_crossing_records_and_is_assessment_specific():
    aggregator = HealthKitAggregator(HKAggConfig())
    daily = aggregator.compute_daily_sleep_features(
        _sleep_fixture(),
        assessments=_assessments(
            "2025-01-02T16:15:00Z", "2025-01-02T17:36:00Z"
        ),
        schedule=_schedule(),
        qc=False,
    )
    current = daily.loc[daily["sleep_date"] == pd.Timestamp("2025-01-03").date()]

    assert current.set_index("assessment_id")["sleep_sleep_duration_day"].to_dict() == {
        "a1": 0.5,
        "a2": 1.0,
    }
    assert np.isclose(
        current.set_index("assessment_id").loc["a1", "sleep_day_elapsed_fraction"],
        1.25 / 24.0,
    )
    assert np.isclose(
        current.set_index("assessment_id").loc["a2", "sleep_day_elapsed_fraction"],
        2.6 / 24.0,
    )


def test_daily_stage_columns_are_data_dependent_and_missing_days_are_not_created():
    aggregator = HealthKitAggregator(HKAggConfig())
    daily = aggregator.compute_daily_sleep_features(
        _sleep_fixture(),
        assessments=_assessments("2025-01-02T17:36:00Z"),
        schedule=_schedule(),
        qc=False,
    )

    assert "sleep_AsleepCore_sum_day" in daily.columns
    assert "sleep_Awake_sum_day" in daily.columns
    assert "sleep_AsleepDeep_sum_day" not in daily.columns
    assert pd.Timestamp("2025-01-04").date() not in set(daily["sleep_date"])


def test_daily_sleep_merges_by_window_start_sleep_date_without_leakage():
    aggregator = HealthKitAggregator(
        HKAggConfig(early_prune_mode="none", prune_empty_rows=False)
    )
    wide, _, aux = aggregator.run_aggregation(
        _sleep_fixture(),
        schedule=_schedule(),
        assessments=_assessments("2025-01-02T17:36:00Z"),
        assessment_window=AssessmentWindowSpec(
            mode="aligned_grid", n_bins=4, bin_size="1H"
        ),
        include_daily_sleep=True,
    )

    assert list(zip(wide["window_start"], wide["window_end"])) == [
        (pd.Timestamp("2025-01-02T14:00:00Z"), pd.Timestamp("2025-01-02T15:00:00Z")),
        (pd.Timestamp("2025-01-02T15:00:00Z"), pd.Timestamp("2025-01-02T16:00:00Z")),
        (pd.Timestamp("2025-01-02T16:00:00Z"), pd.Timestamp("2025-01-02T17:00:00Z")),
        (pd.Timestamp("2025-01-02T17:00:00Z"), pd.Timestamp("2025-01-02T17:36:00Z")),
    ]
    assert wide["sleep_date"].tolist() == [
        pd.Timestamp("2025-01-02").date(),
        pd.Timestamp("2025-01-03").date(),
        pd.Timestamp("2025-01-03").date(),
        pd.Timestamp("2025-01-03").date(),
    ]
    assert wide.loc[0, "sleep_sleep_duration_day"] == 2.0
    assert wide.loc[1:, "sleep_sleep_duration_day"].tolist() == [1.0, 1.0, 1.0]
    assert "daily_sleep" in aux
    assert not aux["daily_sleep"].empty
    manifest_names = aggregator.get_feature_manifest()["feature_name"].tolist()
    assert "sleep_sleep_duration_day" in manifest_names
    assert set(manifest_names).issubset(wide.columns)
    manifest = aggregator.get_feature_manifest().set_index("feature_name")
    assert manifest.loc["sleep_sleep_hr_median_day", "source_type"] == "HeartRate"
    assert manifest.loc["sleep_AsleepCore_count_day", "units"] == "count"
    assert manifest.loc["sleep_AsleepCore_count_day", "label_or_union"] == "label:AsleepCore"


def test_fixed_grid_daily_sleep_uses_complete_rows_and_window_start_merge():
    aggregator = HealthKitAggregator(
        HKAggConfig(
            bin_size="1H",
            early_prune_mode="none",
            prune_empty_rows=False,
        )
    )
    wide, _, aux = aggregator.run_aggregation(
        _sleep_fixture(),
        schedule=_schedule(),
        include_daily_sleep=True,
    )

    jan_two = wide.loc[wide["sleep_date"] == pd.Timestamp("2025-01-02").date()]
    assert not jan_two.empty
    assert jan_two["sleep_sleep_duration_day"].dropna().eq(2.0).all()
    assert aux["daily_sleep"]["sleep_day_elapsed_fraction"].eq(1.0).all()

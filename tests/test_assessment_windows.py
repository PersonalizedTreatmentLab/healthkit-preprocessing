import pandas as pd
import pytest

from healthkit_aggregator import HealthKitAggregator
from healthkit_aggregator_specs import AssessmentWindowSpec, HKAggConfig


def _assessments(*anchors):
    return pd.DataFrame(
        {
            "user_id": [1] * len(anchors),
            "assessment_id": [f"a{i + 1}" for i in range(len(anchors))],
            "anchor_time": list(anchors),
        }
    )


def _utc_schedule(start="2025-01-01T00:00:00Z", end="2026-01-01T00:00:00Z"):
    return pd.DataFrame(
        [{
            "user_id": 1,
            "start_utc": pd.Timestamp(start),
            "end_utc": pd.Timestamp(end),
            "tz_name": "UTC",
        }]
    )


@pytest.mark.parametrize(
    "spec",
    [
        AssessmentWindowSpec(mode="whole_lookback"),
        AssessmentWindowSpec(mode="whole_lookback", duration_before="6H", n_bins=2),
        AssessmentWindowSpec(mode="split_lookback", duration_before="6H"),
        AssessmentWindowSpec(mode="aligned_grid", n_bins=4),
        AssessmentWindowSpec(
            mode="aligned_grid", n_bins=4, bin_size="1H", duration_before="4H"
        ),
    ],
)
def test_assessment_window_spec_rejects_missing_or_irrelevant_fields(spec):
    with pytest.raises(ValueError):
        spec.validate()


def test_whole_and_split_lookback_have_clear_assessment_relative_semantics():
    aggregator = HealthKitAggregator(HKAggConfig())
    assessments = _assessments("2025-01-01T17:36:00Z")

    whole, _ = aggregator.build_assessment_windows(
        assessments,
        _utc_schedule(),
        AssessmentWindowSpec(mode="whole_lookback", duration_before="4H"),
    )
    assert whole[["window_start", "window_end"]].iloc[0].tolist() == [
        pd.Timestamp("2025-01-01T13:36:00Z"),
        pd.Timestamp("2025-01-01T17:36:00Z"),
    ]
    assert whole["bin_in_lookback"].tolist() == [0]

    split, _ = aggregator.build_assessment_windows(
        assessments,
        _utc_schedule(),
        AssessmentWindowSpec(
            mode="split_lookback", duration_before="4H", n_bins=4
        ),
    )
    assert split["window_start"].tolist() == list(
        pd.to_datetime(
            [
                "2025-01-01T13:36:00Z",
                "2025-01-01T14:36:00Z",
                "2025-01-01T15:36:00Z",
                "2025-01-01T16:36:00Z",
            ],
            utc=True,
        )
    )
    assert split["window_end"].iloc[-1] == pd.Timestamp("2025-01-01T17:36:00Z")
    assert split["bin_in_lookback"].tolist() == [0, 1, 2, 3]


def test_aligned_grid_truncates_only_off_grid_terminal_bin():
    aggregator = HealthKitAggregator(HKAggConfig())
    windows, _ = aggregator.build_assessment_windows(
        _assessments("2025-01-01T17:36:00Z"),
        _utc_schedule(),
        AssessmentWindowSpec(mode="aligned_grid", n_bins=4, bin_size="1H"),
    )

    assert list(zip(windows["window_start"], windows["window_end"])) == [
        (pd.Timestamp("2025-01-01T14:00:00Z"), pd.Timestamp("2025-01-01T15:00:00Z")),
        (pd.Timestamp("2025-01-01T15:00:00Z"), pd.Timestamp("2025-01-01T16:00:00Z")),
        (pd.Timestamp("2025-01-01T16:00:00Z"), pd.Timestamp("2025-01-01T17:00:00Z")),
        (pd.Timestamp("2025-01-01T17:00:00Z"), pd.Timestamp("2025-01-01T17:36:00Z")),
    ]
    assert windows["window_seconds"].tolist() == [3600.0, 3600.0, 3600.0, 2160.0]
    assert "bin_index" not in windows.columns


def test_aligned_grid_on_boundary_returns_complete_bins():
    aggregator = HealthKitAggregator(HKAggConfig())
    windows, _ = aggregator.build_assessment_windows(
        _assessments("2025-01-01T17:00:00Z"),
        _utc_schedule(),
        AssessmentWindowSpec(mode="aligned_grid", n_bins=4, bin_size="1H"),
    )

    assert windows["window_start"].tolist() == list(
        pd.to_datetime(
            [
                "2025-01-01T13:00:00Z",
                "2025-01-01T14:00:00Z",
                "2025-01-01T15:00:00Z",
                "2025-01-01T16:00:00Z",
            ],
            utc=True,
        )
    )
    assert windows["window_seconds"].eq(3600.0).all()


def test_assessment_identifiers_are_validated_and_internal_window_ids_are_unique():
    aggregator = HealthKitAggregator(HKAggConfig())
    duplicate = pd.concat(
        [_assessments("2025-01-01T17:00:00Z")] * 2, ignore_index=True
    )
    spec = AssessmentWindowSpec(mode="split_lookback", duration_before="2H", n_bins=2)

    with pytest.raises(ValueError, match="unique"):
        aggregator.build_assessment_windows(duplicate, _utc_schedule(), spec)

    windows, _ = aggregator.build_assessment_windows(
        _assessments("2025-01-01T17:00:00Z", "2025-01-01T18:00:00Z"),
        _utc_schedule(),
        spec,
    )
    assert windows["window_id"].is_unique
    assert windows["assessment_id"].tolist() == ["a1", "a1", "a2", "a2"]
    assert windows["anchor_time"].tolist() == [
        pd.Timestamp("2025-01-01T17:00:00Z"),
        pd.Timestamp("2025-01-01T17:00:00Z"),
        pd.Timestamp("2025-01-01T18:00:00Z"),
        pd.Timestamp("2025-01-01T18:00:00Z"),
    ]


def test_local_aligned_grid_handles_dst_and_preserves_schedule_coverage():
    aggregator = HealthKitAggregator(HKAggConfig(tz_policy="strict"))
    schedule = pd.DataFrame(
        [{
            "user_id": 1,
            "start_utc": pd.Timestamp("2025-11-01T00:00:00Z"),
            "end_utc": pd.Timestamp("2025-11-03T00:00:00Z"),
            "tz_name": "America/New_York",
        }]
    )
    windows, _ = aggregator.build_assessment_windows(
        _assessments("2025-11-02T10:30:00Z"),
        schedule,
        AssessmentWindowSpec(
            mode="aligned_grid", n_bins=4, bin_size="1H", tz_mode="local"
        ),
    )

    assert len(windows) == 4
    assert windows["window_end"].iloc[-1] == pd.Timestamp("2025-11-02T10:30:00Z")
    assert (windows["window_end"] > windows["window_start"]).all()
    assert windows["window_start_local"].notna().all()
    assert windows["schedule_covered_fraction"].eq(1.0).all()


def test_local_projection_preserves_all_existing_window_metadata():
    aggregator = HealthKitAggregator(HKAggConfig())
    windows, _ = aggregator.build_assessment_windows(
        _assessments("2025-01-01T17:36:00Z"),
        _utc_schedule(),
        AssessmentWindowSpec(mode="aligned_grid", n_bins=2, bin_size="1H"),
    )
    wide = aggregator.assemble_feature_matrix(windows, {})

    for column in (
        "assessment_id",
        "anchor_time",
        "bin_in_lookback",
        "schedule_covered_seconds",
        "schedule_covered_fraction",
    ):
        assert column in windows.columns
        assert column in wide.columns


def test_assessment_metadata_survives_end_to_end_aggregation_and_qc():
    aggregator = HealthKitAggregator(
        HKAggConfig(early_prune_mode="none", prune_empty_rows=False)
    )
    data = pd.DataFrame(
        [
            {
                "user_id": 1,
                "type": "HeartRate",
                "value": 70.0,
                "utc_start": "2025-01-01T15:30:00Z",
                "utc_end": "2025-01-01T15:30:00Z",
            },
            {
                "user_id": 1,
                "type": "StepCount",
                "value": 100.0,
                "utc_start": "2025-01-01T15:00:00Z",
                "utc_end": "2025-01-01T16:00:00Z",
            },
        ]
    )
    wide, _, aux = aggregator.run_aggregation(
        data,
        schedule=_utc_schedule(),
        assessments=_assessments("2025-01-01T17:36:00Z"),
        assessment_window=AssessmentWindowSpec(
            mode="aligned_grid", n_bins=4, bin_size="1H"
        ),
    )
    qc, _ = aggregator.apply_quality_control(wide, aux)

    for frame in (wide, qc):
        assert frame["assessment_id"].eq("a1").all()
        assert frame["anchor_time"].eq(pd.Timestamp("2025-01-01T17:36:00Z")).all()
        assert frame["bin_in_lookback"].tolist() == [0, 1, 2, 3]
        assert frame["schedule_covered_fraction"].eq(1.0).all()


def test_qc_masking_never_masks_assessment_or_window_metadata():
    aggregator = HealthKitAggregator(
        HKAggConfig(
            early_prune_mode="none",
            prune_empty_rows=False,
            enable_bin_qc=True,
            qc_action="mask",
            bin_frac_min=0.8,
            qc_mask_selectors={
                "bin": {"all": True},
                "day": {"all": True},
                "participant": {"all": True},
            },
        )
    )
    data = pd.DataFrame(
        [{
            "user_id": 1,
            "type": "StepCount",
            "value": 10.0,
            "utc_start": "2025-01-01T15:00:00Z",
            "utc_end": "2025-01-01T16:00:00Z",
        }]
    )
    wide, _, aux = aggregator.run_aggregation(
        data,
        schedule=_utc_schedule(),
        assessments=_assessments("2025-01-01T17:36:00Z"),
        assessment_window=AssessmentWindowSpec(
            mode="aligned_grid", n_bins=2, bin_size="1H"
        ),
    )
    masked, _ = aggregator.apply_quality_control(wide, aux)

    assert masked["assessment_id"].eq("a1").all()
    assert masked["anchor_time"].notna().all()
    assert masked["bin_in_lookback"].tolist() == [0, 1]
    assert masked["window_seconds"].tolist() == [3600.0, 2160.0]

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


"""
TEST WITH 5 PARTICIPANTS
The loader successfully processed 54 HealthKit files (3.37M rows) with no errors or warnings. 
However, 35 files contained info-level events caused by ambiguous timestamps during the DST fall-back on 2022-11-06 in the America/Los_Angeles timezone. 
During this transition the local clock repeats the 1:00–1:59 AM hour, producing timestamps that map to two possible UTC instants. 
The loader first attempted ambiguous='infer' to resolve these timestamps, but inference failed for 2,134 rows, 
after which the fallback policy ambiguous=False was applied, interpreting the timestamps as the second occurrence of the repeated hour (post-DST). 
These rows were still loaded successfully; the events simply document that automatic DST disambiguation was required.
"""

SEVERITY_RANK = {"none": 0, "info": 1, "warning": 2, "error": 3}


EVENT_DEFINITIONS: Dict[str, tuple[str, str]] = {
    # Error events
    "rows_dropped_total": (
        "error",
        "Rows dropped between file input and normalized output.",
    ),
    "timezone_dropped_rows": (
        "error",
        "Rows dropped due to missing or invalid timezone when fill_tz=False.",
    ),
    "user_id_rows_dropped_non_numeric": (
        "error",
        "Rows dropped because user_id could not be coerced to integer.",
    ),
    # Warning events
    "timezone_missing_column_rows": (
        "warning",
        "Rows in files that were missing a timezone column.",
    ),
    "timezone_missing_value_rows": (
        "warning",
        "Rows with missing timezone value.",
    ),
    "timezone_invalid_value_rows": (
        "warning",
        "Rows with invalid timezone labels.",
    ),
    "timezone_filled_default_rows": (
        "warning",
        "Rows where timezone was replaced with default timezone.",
    ),
    "xml_timezone_rows_inferred_nearest": (
        "warning",
        "XML rows timezone-filled from nearest day inference.",
    ),
    "xml_timezone_rows_inferred_ffill": (
        "warning",
        "XML rows timezone-filled with forward-fill.",
    ),
    "xml_timezone_rows_inferred_bfill": (
        "warning",
        "XML rows timezone-filled with backward-fill.",
    ),
    "xml_timezone_rows_defaulted": (
        "warning",
        "XML rows defaulted to fallback timezone after inference.",
    ),
    "xml_timezone_rows_invalid_tz_coerced_default": (
        "warning",
        "XML rows with invalid timezone labels coerced to default timezone.",
    ),
    "dst_escalated_from_info": (
        "warning",
        "DST informational events escalated to warning by threshold policy.",
    ),
    # Info events
    "dst_fallback_ambiguous_false_rows": (
        "info",
        "Rows localized using DST ambiguous=False fallback policy.",
    ),
    "dst_ambiguous_infer_fail_rows": (
        "info",
        "Rows where DST ambiguous='infer' failed before fallback.",
    ),
    "dst_nonexistent_detected_rows": (
        "info",
        "Rows with nonexistent local time detected during DST spring-forward transition.",
    ),
}

_EVENT_ORDER = {
    event_key: index
    for index, event_key in enumerate(
        [
            "rows_dropped_total",
            "timezone_dropped_rows",
            "user_id_rows_dropped_non_numeric",
            "timezone_missing_column_rows",
            "timezone_missing_value_rows",
            "timezone_invalid_value_rows",
            "timezone_filled_default_rows",
            "xml_timezone_rows_inferred_nearest",
            "xml_timezone_rows_inferred_ffill",
            "xml_timezone_rows_inferred_bfill",
            "xml_timezone_rows_defaulted",
            "xml_timezone_rows_invalid_tz_coerced_default",
            "dst_escalated_from_info",
            "dst_fallback_ambiguous_false_rows",
            "dst_ambiguous_infer_fail_rows",
            "dst_nonexistent_detected_rows",
        ]
    )
}


@dataclass
class FileMetrics:
    path: str
    sensor: str
    participant_id: Optional[str]
    rows_in_raw: int
    rows_out: int
    error_events: int
    warning_events: int
    info_events: int
    highest_severity_rank: int
    weighted_score: int
    issue_rate: float
    error_details: Dict[str, int]
    warning_details: Dict[str, int]
    info_details: Dict[str, int]
    sample_details: Dict[str, List[Any]]


@dataclass
class ParticipantMetrics:
    participant_id: str
    files: int
    files_affected: int
    rows_in_raw: int
    error_events: int
    warning_events: int
    info_events: int
    highest_severity_rank: int
    weighted_score: int
    issue_rate: float


@dataclass
class EventTaxonomyRow:
    severity: str
    event_key: str
    description: str
    event_count: int
    affected_files: int


@dataclass
class ReportSummary:
    source_path: Path
    report_kind: str
    files_total: int
    files_with_issues: int
    rows_out: int
    error_events: int
    warning_events: int
    info_events: int
    error_files: int
    warning_files: int
    info_files: int
    max_severity_rank: int
    failed_files: List[Dict[str, Any]]
    missing_files: List[str]
    file_metrics: List[FileMetrics]


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value).strip())
    except Exception:
        return 0


def _nested_int(mapping: Mapping[str, Any], *keys: str) -> int:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping):
            return 0
        cur = cur.get(key)
    return _safe_int(cur)


def _nested_list(mapping: Mapping[str, Any], *keys: str) -> List[Any]:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping):
            return []
        cur = cur.get(key)
    if isinstance(cur, list):
        return cur
    return []


def _derive_sensor(path_value: str) -> str:
    if not path_value:
        return "unknown"
    p = Path(path_value)
    parent = p.parent.name.strip()
    return parent or "unknown"


def _derive_participant_id(path_value: str) -> Optional[str]:
    if not path_value:
        return None
    match = re.match(r"^(\d+)-", Path(path_value).name)
    if match:
        return match.group(1)
    return None


def _participant_label(participant_id: Optional[str]) -> str:
    if participant_id is None:
        return "unknown"
    cleaned = str(participant_id).strip()
    return cleaned or "unknown"


def _looks_like_single_file_report(data: Mapping[str, Any]) -> bool:
    required = {"__global__", "timezone", "dst", "xml_timezone", "user_id"}
    return required.issubset(set(data.keys()))


def _discover_from_glob(raw_input: str) -> List[Path]:
    out: List[Path] = []
    for matched in glob.glob(raw_input, recursive=True):
        p = Path(matched).expanduser().resolve()
        if p.is_file():
            out.append(p)
    return sorted(out)


def discover_report_paths(inputs: Sequence[str], pattern: str = "load_report*.json") -> List[Path]:
    discovered: List[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        raw = raw.strip()
        if not raw:
            continue

        has_glob_chars = any(char in raw for char in "*?[]")
        candidate = Path(raw).expanduser()

        matches: List[Path] = []
        if has_glob_chars:
            matches = _discover_from_glob(raw)
        elif candidate.exists():
            candidate = candidate.resolve()
            if candidate.is_file():
                matches = [candidate]
            elif candidate.is_dir():
                matches = sorted(
                    p.resolve()
                    for p in candidate.rglob(pattern)
                    if p.is_file() and p.suffix.lower() == ".json"
                )
        else:
            # Fallback for shell patterns that were not expanded before invocation.
            matches = _discover_from_glob(raw)

        for p in matches:
            if p not in seen:
                seen.add(p)
                discovered.append(p)

    return discovered


def _extract_file_reports(data: Mapping[str, Any]) -> tuple[str, List[Mapping[str, Any]]]:
    if isinstance(data.get("files"), list):
        files = [entry for entry in data["files"] if isinstance(entry, Mapping)]
        if "failed_files" in data:
            return "bulk", files
        if "missing_files" in data:
            return "participant", files
        return "multi_file", files

    if _looks_like_single_file_report(data):
        return "single_file", [data]

    if isinstance(data.get("__global__"), Mapping):
        return "single_file", [data]

    return "unknown", []


def _compute_file_metrics(file_report: Mapping[str, Any], dst_warning_threshold: Optional[int]) -> FileMetrics:
    global_sec = file_report.get("__global__", {}) if isinstance(file_report.get("__global__"), Mapping) else {}
    path_value = str(global_sec.get("path", "")).strip()

    rows_in_raw = _nested_int(file_report, "__global__", "rows_in_raw")
    rows_out = _nested_int(file_report, "__global__", "rows_out")

    error_details = {
        "rows_dropped_total": _nested_int(file_report, "__global__", "rows_dropped_total"),
        "timezone_dropped_rows": _nested_int(file_report, "timezone", "dropped_rows"),
        "user_id_rows_dropped_non_numeric": _nested_int(file_report, "user_id", "rows_dropped_non_numeric"),
    }

    warning_details = {
        "timezone_missing_column_rows": _nested_int(file_report, "timezone", "missing_column_rows"),
        "timezone_missing_value_rows": _nested_int(file_report, "timezone", "missing_value_rows"),
        "timezone_invalid_value_rows": _nested_int(file_report, "timezone", "invalid_value_rows"),
        "timezone_filled_default_rows": _nested_int(file_report, "timezone", "filled_default_rows"),
        "xml_timezone_rows_inferred_nearest": _nested_int(file_report, "xml_timezone", "rows_inferred_nearest"),
        "xml_timezone_rows_inferred_ffill": _nested_int(file_report, "xml_timezone", "rows_inferred_ffill"),
        "xml_timezone_rows_inferred_bfill": _nested_int(file_report, "xml_timezone", "rows_inferred_bfill"),
        "xml_timezone_rows_defaulted": _nested_int(file_report, "xml_timezone", "rows_defaulted"),
        "xml_timezone_rows_invalid_tz_coerced_default": _nested_int(
            file_report,
            "xml_timezone",
            "rows_invalid_tz_coerced_default",
        ),
    }

    info_details = {
        "dst_fallback_ambiguous_false_rows": _nested_int(file_report, "dst", "fallback_ambiguous_false_rows"),
        "dst_ambiguous_infer_fail_rows": _nested_int(file_report, "dst", "ambiguous_infer_fail_rows"),
        "dst_nonexistent_detected_rows": _nested_int(file_report, "dst", "nonexistent_detected_rows"),
    }

    error_events = sum(error_details.values())
    warning_events = sum(warning_details.values())
    info_events = sum(info_details.values())

    if dst_warning_threshold is not None and info_events > dst_warning_threshold:
        warning_events += info_events
        warning_details["dst_escalated_from_info"] = info_events
        info_events = 0

    highest_severity_rank = 0
    if error_events > 0:
        highest_severity_rank = SEVERITY_RANK["error"]
    elif warning_events > 0:
        highest_severity_rank = SEVERITY_RANK["warning"]
    elif info_events > 0:
        highest_severity_rank = SEVERITY_RANK["info"]

    total_events = error_events + warning_events + info_events
    weighted_score = error_events * 100 + warning_events * 10 + info_events
    issue_rate = float(total_events) / float(max(rows_in_raw, 1))

    sample_details = {
        "timezone_invalid_values_sample": _nested_list(file_report, "timezone", "invalid_values_sample"),
        "user_id_sample_values": _nested_list(file_report, "user_id", "sample_values"),
        "dst_fallback_samples": _nested_list(file_report, "dst", "fallback_samples"),
    }

    return FileMetrics(
        path=path_value,
        sensor=_derive_sensor(path_value),
        participant_id=_derive_participant_id(path_value),
        rows_in_raw=rows_in_raw,
        rows_out=rows_out,
        error_events=error_events,
        warning_events=warning_events,
        info_events=info_events,
        highest_severity_rank=highest_severity_rank,
        weighted_score=weighted_score,
        issue_rate=issue_rate,
        error_details=error_details,
        warning_details=warning_details,
        info_details=info_details,
        sample_details=sample_details,
    )


def summarize_report(
    report_path: Path,
    data: Mapping[str, Any],
    dst_warning_threshold: Optional[int] = None,
) -> ReportSummary:
    report_kind, file_reports = _extract_file_reports(data)

    file_metrics = [_compute_file_metrics(fr, dst_warning_threshold) for fr in file_reports]

    failed_files_raw = data.get("failed_files") if isinstance(data.get("failed_files"), list) else []
    failed_files: List[Dict[str, Any]] = [
        entry for entry in failed_files_raw if isinstance(entry, Mapping)
    ]

    missing_files_raw = data.get("missing_files") if isinstance(data.get("missing_files"), list) else []
    missing_files: List[str] = [str(item) for item in missing_files_raw]

    rows_out = _nested_int(data, "__global__", "rows_out")
    files_with_issues = sum(1 for fm in file_metrics if (fm.error_events + fm.warning_events + fm.info_events) > 0)

    file_error_events = sum(fm.error_events for fm in file_metrics)
    file_warning_events = sum(fm.warning_events for fm in file_metrics)
    file_info_events = sum(fm.info_events for fm in file_metrics)

    error_events = file_error_events + len(failed_files)
    warning_events = file_warning_events + len(missing_files)
    info_events = file_info_events

    max_severity_rank = 0
    if error_events > 0:
        max_severity_rank = SEVERITY_RANK["error"]
    elif warning_events > 0:
        max_severity_rank = SEVERITY_RANK["warning"]
    elif info_events > 0:
        max_severity_rank = SEVERITY_RANK["info"]

    return ReportSummary(
        source_path=report_path,
        report_kind=report_kind,
        files_total=len(file_metrics),
        files_with_issues=files_with_issues,
        rows_out=rows_out,
        error_events=error_events,
        warning_events=warning_events,
        info_events=info_events,
        error_files=sum(1 for fm in file_metrics if fm.error_events > 0),
        warning_files=sum(1 for fm in file_metrics if fm.warning_events > 0),
        info_files=sum(1 for fm in file_metrics if fm.info_events > 0),
        max_severity_rank=max_severity_rank,
        failed_files=failed_files,
        missing_files=missing_files,
        file_metrics=file_metrics,
    )


def _severity_name(rank: int) -> str:
    if rank >= SEVERITY_RANK["error"]:
        return "error"
    if rank >= SEVERITY_RANK["warning"]:
        return "warning"
    if rank >= SEVERITY_RANK["info"]:
        return "info"
    return "none"


def _participant_sort_key(participant_id: str) -> tuple[int, int, str]:
    cleaned = participant_id.strip()
    if cleaned.isdigit():
        return (0, int(cleaned), "")
    return (1, 0, cleaned)


def _participant_rank_key(pm: ParticipantMetrics) -> tuple[int, int, float, tuple[int, int, str]]:
    return (
        -pm.highest_severity_rank,
        -pm.weighted_score,
        -pm.issue_rate,
        _participant_sort_key(pm.participant_id),
    )


def rank_problem_files(file_metrics: Iterable[FileMetrics], max_files: int) -> List[FileMetrics]:
    flagged = [fm for fm in file_metrics if (fm.error_events + fm.warning_events + fm.info_events) > 0]
    flagged.sort(
        key=lambda fm: (
            -fm.highest_severity_rank,
            -fm.weighted_score,
            -fm.issue_rate,
            fm.path,
        )
    )
    return flagged[: max(1, max_files)]


def aggregate_participant_metrics(file_metrics: Iterable[FileMetrics]) -> List[ParticipantMetrics]:
    aggregate: Dict[str, Dict[str, int]] = {}

    for fm in file_metrics:
        participant = _participant_label(fm.participant_id)
        if participant not in aggregate:
            aggregate[participant] = {
                "files": 0,
                "files_affected": 0,
                "rows_in_raw": 0,
                "error_events": 0,
                "warning_events": 0,
                "info_events": 0,
                "highest_severity_rank": 0,
                "weighted_score": 0,
            }

        row = aggregate[participant]
        row["files"] += 1
        if (fm.error_events + fm.warning_events + fm.info_events) > 0:
            row["files_affected"] += 1
        row["rows_in_raw"] += fm.rows_in_raw
        row["error_events"] += fm.error_events
        row["warning_events"] += fm.warning_events
        row["info_events"] += fm.info_events
        row["highest_severity_rank"] = max(row["highest_severity_rank"], fm.highest_severity_rank)
        row["weighted_score"] += fm.weighted_score

    participants: List[ParticipantMetrics] = []
    for participant_id, row in aggregate.items():
        total_events = row["error_events"] + row["warning_events"] + row["info_events"]
        issue_rate = float(total_events) / float(max(row["rows_in_raw"], 1))
        participants.append(
            ParticipantMetrics(
                participant_id=participant_id,
                files=row["files"],
                files_affected=row["files_affected"],
                rows_in_raw=row["rows_in_raw"],
                error_events=row["error_events"],
                warning_events=row["warning_events"],
                info_events=row["info_events"],
                highest_severity_rank=row["highest_severity_rank"],
                weighted_score=row["weighted_score"],
                issue_rate=issue_rate,
            )
        )

    return participants


def select_participants_for_rollup(
    participant_metrics: Sequence[ParticipantMetrics],
    max_participants: int,
) -> List[ParticipantMetrics]:
    ranked = sorted(participant_metrics, key=_participant_rank_key)
    return ranked[: max(1, max_participants)]


def rank_problem_participants(
    participant_metrics: Sequence[ParticipantMetrics],
    max_participants: int,
) -> List[ParticipantMetrics]:
    flagged = [
        pm
        for pm in participant_metrics
        if (pm.error_events + pm.warning_events + pm.info_events) > 0
    ]
    flagged.sort(key=_participant_rank_key)
    return flagged[: max(1, max_participants)]


def _iter_nonzero_event_counts(fm: FileMetrics) -> Iterable[tuple[str, str, int]]:
    for event_key, count in fm.error_details.items():
        if count > 0:
            yield "error", event_key, count
    for event_key, count in fm.warning_details.items():
        if count > 0:
            yield "warning", event_key, count
    for event_key, count in fm.info_details.items():
        if count > 0:
            yield "info", event_key, count


def _event_description(event_key: str) -> str:
    _, description = EVENT_DEFINITIONS.get(event_key, ("", "No description available."))
    return description


def _event_sort_key(row: EventTaxonomyRow) -> tuple[int, int, int, str]:
    return (
        -SEVERITY_RANK.get(row.severity, 0),
        -row.event_count,
        _EVENT_ORDER.get(row.event_key, 10_000),
        row.event_key,
    )


def aggregate_event_taxonomy(file_metrics: Iterable[FileMetrics]) -> List[EventTaxonomyRow]:
    aggregate: Dict[tuple[str, str], Dict[str, int]] = {}

    for fm in file_metrics:
        for severity, event_key, count in _iter_nonzero_event_counts(fm):
            key = (severity, event_key)
            if key not in aggregate:
                aggregate[key] = {"event_count": 0, "affected_files": 0}
            aggregate[key]["event_count"] += count
            aggregate[key]["affected_files"] += 1

    rows: List[EventTaxonomyRow] = []
    for (severity, event_key), vals in aggregate.items():
        rows.append(
            EventTaxonomyRow(
                severity=severity,
                event_key=event_key,
                description=_event_description(event_key),
                event_count=vals["event_count"],
                affected_files=vals["affected_files"],
            )
        )

    rows.sort(key=_event_sort_key)
    return rows


def aggregate_event_taxonomy_by_sensor(
    file_metrics: Iterable[FileMetrics],
) -> List[tuple[str, List[EventTaxonomyRow]]]:
    grouped: Dict[str, List[FileMetrics]] = {}
    for fm in file_metrics:
        grouped.setdefault(fm.sensor, []).append(fm)

    rows_by_sensor: List[tuple[str, List[EventTaxonomyRow]]] = []
    for sensor, items in grouped.items():
        rows = aggregate_event_taxonomy(items)
        if rows:
            rows_by_sensor.append((sensor, rows))

    rows_by_sensor.sort(
        key=lambda x: (
            -sum(r.event_count for r in x[1]),
            x[0],
        )
    )
    return rows_by_sensor


def aggregate_event_taxonomy_by_participant(
    file_metrics: Iterable[FileMetrics],
    participant_order: Sequence[str],
) -> List[tuple[str, List[EventTaxonomyRow]]]:
    grouped: Dict[str, List[FileMetrics]] = {}
    for fm in file_metrics:
        participant = _participant_label(fm.participant_id)
        grouped.setdefault(participant, []).append(fm)

    rows_by_participant: List[tuple[str, List[EventTaxonomyRow]]] = []
    for participant in participant_order:
        rows = aggregate_event_taxonomy(grouped.get(participant, []))
        if rows:
            rows_by_participant.append((participant, rows))
    return rows_by_participant


def _fmt_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(100.0 * float(numerator) / float(denominator)):.2f}%"


def _fmt_issue_rate(rate: float) -> str:
    return f"{rate * 100.0:.4f}%"


def render_report_markdown(
    summary: ReportSummary,
    max_files: int,
    sample_limit: int,
    max_participants: int = 15,
) -> str:
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    participant_metrics_all = aggregate_participant_metrics(summary.file_metrics)
    participant_rollup = select_participants_for_rollup(participant_metrics_all, max_participants)
    top_problem_participants = rank_problem_participants(participant_metrics_all, max_participants)
    selected_participant_ids = [pm.participant_id for pm in participant_rollup]

    global_taxonomy = aggregate_event_taxonomy(summary.file_metrics)
    sensor_taxonomy = aggregate_event_taxonomy_by_sensor(summary.file_metrics)
    participant_taxonomy = aggregate_event_taxonomy_by_participant(summary.file_metrics, selected_participant_ids)
    total_rows_in_raw = sum(fm.rows_in_raw for fm in summary.file_metrics)
    rows_with_errors = sum(fm.error_details.get("rows_dropped_total", 0) for fm in summary.file_metrics)

    lines: List[str] = []
    lines.append("# HealthKit Loader Report Summary")
    lines.append("")
    lines.append(f"Source report: `{summary.source_path}`  ")
    lines.append(f"Generated at (UTC): `{now_iso}`")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Report kind: `{summary.report_kind}`")
    lines.append(f"- Loaded files analyzed: `{summary.files_total}`")
    lines.append(f"- Loaded files with issues: `{summary.files_with_issues}` ({_fmt_ratio(summary.files_with_issues, summary.files_total)})")
    lines.append(f"- Rows in raw (loaded files): `{total_rows_in_raw}`")
    lines.append(f"- Rows with errors (dropped): `{rows_with_errors}` ({_fmt_ratio(rows_with_errors, total_rows_in_raw)})")
    lines.append(f"- Failed files: `{len(summary.failed_files)}`")
    lines.append(f"- Missing files: `{len(summary.missing_files)}`")
    lines.append(f"- Rows out: `{summary.rows_out}`")
    lines.append("")

    lines.append("## Severity Totals")
    lines.append("")
    lines.append("| Severity | Event Count | Affected Loaded Files |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| error | {summary.error_events} | {summary.error_files} |")
    lines.append(f"| warning | {summary.warning_events} | {summary.warning_files} |")
    lines.append(f"| info | {summary.info_events} | {summary.info_files} |")
    lines.append("")

    lines.append("## Failed/Missing File Highlights")
    lines.append("")
    if summary.failed_files:
        lines.append("### Failed Files")
        for failed in summary.failed_files[:max_files]:
            path_value = str(failed.get("path", ""))
            err_type = str(failed.get("error_type", "UnknownError"))
            err_msg = str(failed.get("error", ""))
            lines.append(f"- `{path_value}`: `{err_type}` - {err_msg}")
        if len(summary.failed_files) > max_files:
            lines.append(f"- ... {len(summary.failed_files) - max_files} additional failed files")
        lines.append("")

    if summary.missing_files:
        lines.append("### Missing Files")
        for missing in summary.missing_files[:max_files]:
            lines.append(f"- `{missing}`")
        if len(summary.missing_files) > max_files:
            lines.append(f"- ... {len(summary.missing_files) - max_files} additional missing files")
        lines.append("")

    if not summary.failed_files and not summary.missing_files:
        lines.append("- None")
        lines.append("")

    lines.append("## Sensor Rollup")
    lines.append("")
    if not summary.file_metrics:
        lines.append("- No loaded file reports available.")
        lines.append("")
    else:
        rollup: Dict[str, Dict[str, int]] = {}
        for fm in summary.file_metrics:
            sensor = fm.sensor
            if sensor not in rollup:
                rollup[sensor] = {
                    "files": 0,
                    "files_affected": 0,
                    "error_events": 0,
                    "warning_events": 0,
                    "info_events": 0,
                }
            row = rollup[sensor]
            row["files"] += 1
            if (fm.error_events + fm.warning_events + fm.info_events) > 0:
                row["files_affected"] += 1
            row["error_events"] += fm.error_events
            row["warning_events"] += fm.warning_events
            row["info_events"] += fm.info_events

        sorted_rollup = sorted(
            rollup.items(),
            key=lambda kv: (
                -kv[1]["error_events"],
                -kv[1]["warning_events"],
                -kv[1]["info_events"],
                kv[0],
            ),
        )

        lines.append("| Sensor | Files | Affected Files | Error Events | Warning Events | Info Events |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for sensor, row in sorted_rollup:
            lines.append(
                f"| {sensor} | {row['files']} | {row['files_affected']} | {row['error_events']} | {row['warning_events']} | {row['info_events']} |"
            )
        lines.append("")

    lines.append("## Participant Rollup")
    lines.append("")
    if not participant_metrics_all:
        lines.append("- No participant data derived from loaded files.")
        lines.append("")
    else:
        if len(participant_rollup) < len(participant_metrics_all):
            lines.append(
                f"- Showing top `{len(participant_rollup)}` of `{len(participant_metrics_all)}` participants by issue ranking (`--max-participants={max_participants}`)."
            )
            lines.append("")
        lines.append("| Participant | Files | Affected Files | Error Events | Warning Events | Info Events | Weighted Score | Issue Rate |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for pm in participant_rollup:
            lines.append(
                f"| {pm.participant_id} | {pm.files} | {pm.files_affected} | {pm.error_events} | {pm.warning_events} | {pm.info_events} | {pm.weighted_score} | {_fmt_issue_rate(pm.issue_rate)} |"
            )
        lines.append("")

    lines.append("## Top Problematic Participants")
    lines.append("")
    if not top_problem_participants:
        lines.append("- No participant-level issues detected.")
        lines.append("")
    else:
        lines.append("| Highest Severity | Participant | Files | Affected Files | Weighted Score | Issue Rate | E/W/I |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for pm in top_problem_participants:
            severity = _severity_name(pm.highest_severity_rank)
            lines.append(
                f"| {severity} | {pm.participant_id} | {pm.files} | {pm.files_affected} | {pm.weighted_score} | {_fmt_issue_rate(pm.issue_rate)} | {pm.error_events}/{pm.warning_events}/{pm.info_events} |"
            )
        lines.append("")

    lines.append("## Event Taxonomy (Global)")
    lines.append("")
    lines.append("- Event counts are additive counter totals from file reports (not deduplicated row ids).")
    lines.append("- Failed/missing files are tracked above and intentionally excluded from event taxonomy tables.")
    lines.append("")
    if not global_taxonomy:
        lines.append("- No event taxonomy entries found.")
        lines.append("")
    else:
        lines.append("| Severity | Event Key | Description | Event Count | Affected Files |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for row in global_taxonomy:
            lines.append(
                f"| {row.severity} | `{row.event_key}` | {row.description} | {row.event_count} | {row.affected_files} |"
            )
        lines.append("")

    lines.append("## Event Taxonomy by Sensor")
    lines.append("")
    if not sensor_taxonomy:
        lines.append("- No sensor-level event taxonomy entries found.")
        lines.append("")
    else:
        lines.append("| Sensor | Severity | Event Key | Description | Event Count | Affected Files |")
        lines.append("| --- | --- | --- | --- | ---: | ---: |")
        for sensor, rows in sensor_taxonomy:
            for row in rows:
                lines.append(
                    f"| {sensor} | {row.severity} | `{row.event_key}` | {row.description} | {row.event_count} | {row.affected_files} |"
                )
        lines.append("")

    lines.append("## Event Taxonomy by Participant")
    lines.append("")
    if not participant_taxonomy:
        lines.append("- No participant-level event taxonomy entries found.")
        lines.append("")
    else:
        if len(participant_rollup) < len(participant_metrics_all):
            lines.append(
                f"- Participant taxonomy is limited to the selected rollup participants (`--max-participants={max_participants}`)."
            )
            lines.append("")
        lines.append("| Participant | Severity | Event Key | Description | Event Count | Affected Files |")
        lines.append("| --- | --- | --- | --- | ---: | ---: |")
        for participant, rows in participant_taxonomy:
            for row in rows:
                lines.append(
                    f"| {participant} | {row.severity} | `{row.event_key}` | {row.description} | {row.event_count} | {row.affected_files} |"
                )
        lines.append("")

    lines.append("## Top Problematic Files")
    lines.append("")
    top_files = rank_problem_files(summary.file_metrics, max_files=max_files)
    if not top_files:
        lines.append("- No file-level issues detected.")
        lines.append("")
    else:
        lines.append("| Highest Severity | Weighted Score | Issue Rate | Rows In | E/W/I | File |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for fm in top_files:
            severity = _severity_name(fm.highest_severity_rank)
            lines.append(
                "| "
                f"{severity} | {fm.weighted_score} | {_fmt_issue_rate(fm.issue_rate)} | {fm.rows_in_raw} | "
                f"{fm.error_events}/{fm.warning_events}/{fm.info_events} | `{fm.path}` |"
            )
        lines.append("")

    lines.append("## Brief Samples")
    lines.append("")
    wrote_samples = False
    for fm in top_files:
        if sample_limit <= 0:
            break
        invalid_tz = fm.sample_details.get("timezone_invalid_values_sample", [])[:sample_limit]
        uid_samples = fm.sample_details.get("user_id_sample_values", [])[:sample_limit]
        dst_samples = fm.sample_details.get("dst_fallback_samples", [])[:sample_limit]

        has_any = bool(invalid_tz or uid_samples or dst_samples)
        if not has_any:
            continue

        wrote_samples = True
        lines.append(f"### `{fm.path}`")
        if invalid_tz:
            joined = ", ".join(f"`{str(v)}`" for v in invalid_tz)
            lines.append(f"- timezone.invalid_values_sample: {joined}")
        if uid_samples:
            joined = ", ".join(f"`{str(v)}`" for v in uid_samples)
            lines.append(f"- user_id.sample_values: {joined}")
        if dst_samples:
            for sample in dst_samples:
                if isinstance(sample, Mapping):
                    row_index = sample.get("row_index")
                    user_id = sample.get("user_id")
                    tz_name = sample.get("timezone")
                    label = sample.get("label")
                    local_time = sample.get("local_time")
                    lines.append(
                        "- dst.fallback_sample: "
                        f"row_index={row_index}, user_id={user_id}, timezone={tz_name}, label={label}, local_time={local_time}"
                    )
                else:
                    lines.append(f"- dst.fallback_sample: `{sample}`")
        lines.append("")

    if not wrote_samples:
        lines.append("- No sample payloads available for flagged files.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_batch_markdown(summaries: Sequence[ReportSummary]) -> str:
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    total_reports = len(summaries)
    total_files = sum(s.files_total for s in summaries)
    total_failed = sum(len(s.failed_files) for s in summaries)
    total_missing = sum(len(s.missing_files) for s in summaries)
    total_error_events = sum(s.error_events for s in summaries)
    total_warning_events = sum(s.warning_events for s in summaries)
    total_info_events = sum(s.info_events for s in summaries)

    lines: List[str] = []
    lines.append("# HealthKit Loader Batch Summary")
    lines.append("")
    lines.append(f"Generated at (UTC): `{now_iso}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Reports analyzed: `{total_reports}`")
    lines.append(f"- Loaded files analyzed: `{total_files}`")
    lines.append(f"- Failed files: `{total_failed}`")
    lines.append(f"- Missing files: `{total_missing}`")
    lines.append("")

    lines.append("## Severity Totals")
    lines.append("")
    lines.append("| Severity | Event Count |")
    lines.append("| --- | ---: |")
    lines.append(f"| error | {total_error_events} |")
    lines.append(f"| warning | {total_warning_events} |")
    lines.append(f"| info | {total_info_events} |")
    lines.append("")

    lines.append("## Per-Report Rollup")
    lines.append("")
    lines.append("| Report | Kind | Files | Failed | Missing | Error | Warning | Info |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in summaries:
        lines.append(
            f"| `{summary.source_path}` | {summary.report_kind} | {summary.files_total} | "
            f"{len(summary.failed_files)} | {len(summary.missing_files)} | "
            f"{summary.error_events} | {summary.warning_events} | {summary.info_events} |"
        )
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def determine_exit_code(summaries: Sequence[ReportSummary], fail_on: str) -> int:
    target_rank = SEVERITY_RANK.get(fail_on, 0)
    if target_rank <= 0:
        return 0
    for summary in summaries:
        if summary.max_severity_rank >= target_rank:
            return 1
    return 0


def _default_summary_output_path(report_path: Path, out_dir: Optional[Path]) -> Path:
    if out_dir is not None:
        return out_dir / f"{report_path.stem}.summary.md"
    return report_path.with_name(f"{report_path.stem}.summary.md")


def _default_batch_output_path(report_paths: Sequence[Path], out_dir: Optional[Path]) -> Path:
    if out_dir is not None:
        return out_dir / "load_report_batch.summary.md"
    parent_dirs = {p.parent.resolve() for p in report_paths}
    if len(parent_dirs) == 1:
        only_parent = next(iter(parent_dirs))
        return only_parent / "load_report_batch.summary.md"
    return Path.cwd() / "load_report_batch.summary.md"


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"JSON root is not an object: {path}")
    return data


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize HealthKit loader JSON reports into concise markdown.",
    )
    parser.add_argument("inputs", nargs="+", help="Report file(s), directory path(s), or glob(s).")
    parser.add_argument(
        "--pattern",
        default="load_report*.json",
        help="Directory scan pattern used when an input is a directory (default: load_report*.json).",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional directory for markdown outputs.")
    parser.add_argument("--stdout", action="store_true", help="Print markdown to stdout instead of writing files.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Max files listed in top/problem sections (default: 10).",
    )
    parser.add_argument(
        "--max-participants",
        type=int,
        default=15,
        help="Max participants listed in participant-heavy sections (default: 15).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=3,
        help="Max sample items per metric in the Brief Samples section (default: 3).",
    )
    parser.add_argument(
        "--fail-on",
        choices=("none", "error", "warning", "info"),
        default="none",
        help="Exit non-zero if any report meets/exceeds this severity.",
    )
    parser.add_argument(
        "--dst-warning-threshold",
        type=int,
        default=None,
        help="If set, files with DST info events > N are escalated to warning.",
    )

    args = parser.parse_args(argv)
    if args.max_files <= 0:
        parser.error("--max-files must be greater than 0")
    if args.max_participants <= 0:
        parser.error("--max-participants must be greater than 0")
    if args.sample_limit < 0:
        parser.error("--sample-limit must be 0 or greater")
    if args.dst_warning_threshold is not None and args.dst_warning_threshold < 0:
        parser.error("--dst-warning-threshold must be 0 or greater")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    report_paths = discover_report_paths(args.inputs, pattern=args.pattern)
    if not report_paths:
        print("No report JSON files found for the provided inputs.", file=sys.stderr)
        return 2

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[ReportSummary] = []
    for report_path in report_paths:
        data = _load_json(report_path)
        summary = summarize_report(
            report_path=report_path,
            data=data,
            dst_warning_threshold=args.dst_warning_threshold,
        )
        summaries.append(summary)

    report_markdowns = [
        render_report_markdown(
            s,
            max_files=args.max_files,
            sample_limit=args.sample_limit,
            max_participants=args.max_participants,
        )
        for s in summaries
    ]

    if args.stdout:
        for idx, markdown in enumerate(report_markdowns):
            if idx > 0:
                print("\n---\n")
            print(markdown, end="")
        if len(summaries) > 1:
            print("\n---\n")
            print(render_batch_markdown(summaries), end="")
    else:
        for summary, markdown in zip(summaries, report_markdowns):
            out_path = _default_summary_output_path(summary.source_path, args.out_dir)
            out_path.write_text(markdown, encoding="utf-8")
            print(f"Wrote summary: {out_path}")

        if len(summaries) > 1:
            batch_md = render_batch_markdown(summaries)
            batch_path = _default_batch_output_path(report_paths, args.out_dir)
            batch_path.write_text(batch_md, encoding="utf-8")
            print(f"Wrote batch summary: {batch_path}")

    return determine_exit_code(summaries, fail_on=args.fail_on)


if __name__ == "__main__":
    raise SystemExit(main())

"""
Utilities for constructing analysis windows and local-time projections for HealthKit aggregation.

Provides:
- build_schedule_from_input: infer a per-user timezone schedule from observed intervals.
- build_bin_windows: create per-user fixed-frequency bin grids (UTC or local-wall clock via schedule).
- build_assessment_windows: create configured lookback windows for assessments.
- validate_schedule / compute_schedule_coverage: schedule hygiene and coverage diagnostics.
- add_local_time_labels: attach local date/hour/DST/offset labels to UTC windows.

These helpers standardize downstream schemas:
['user_id','window_id','window_start','window_end','window_seconds', ...local labels...]
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from zoneinfo import ZoneInfo
from healthkit_aggregator_specs import AssessmentWindowSpec

_logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Internal Helpers
# -----------------------------------------------------------------------------
def _coerce_utc_series(values: Any, context: str, errors: str = "coerce") -> pd.Series:
    """
    Convert sequence-like values into a tz-aware UTC datetime Series.
    Raises ValueError with context if coercion/localization fails unexpectedly.
    """
    idx = getattr(values, "index", None)
    try:
        dt = pd.to_datetime(values, utc=True, errors=errors)
    except Exception as exc:
        raise ValueError(f"failed to parse datetime values for {context}") from exc

    if isinstance(dt, pd.Series):
        out = dt
    else:
        out = pd.Series(dt, index=idx)

    try:
        if getattr(out.dt, "tz", None) is None:
            out = out.dt.tz_localize("UTC")
        else:
            out = out.dt.tz_convert("UTC")
    except Exception as exc:
        raise ValueError(f"failed to normalize timezone to UTC for {context}") from exc

    return out


def _normalize_user_id_series(values: Any) -> pd.Series:
    """Normalize user_id values to nullable integer ids; invalid/non-integer-like values become <NA>."""
    idx = getattr(values, "index", None)
    if isinstance(values, pd.Series):
        ser = values.copy()
    else:
        ser = pd.Series(values, index=idx)
    numeric = pd.to_numeric(ser, errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    integer_like = finite & (np.floor(numeric) == numeric)
    out = pd.Series(pd.array(np.where(integer_like, numeric, np.nan), dtype="Int64"), index=ser.index)
    return out


def _validate_timezone_name(name: str) -> bool:
    """Return True when name resolves to an IANA timezone via zoneinfo."""
    try:
        ZoneInfo(str(name))
        return True
    except Exception:
        return False


def _normalize_coverage_bounds(coverage_bounds: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalize optional per-user coverage bounds to UTC min/max rows."""
    if coverage_bounds is None or coverage_bounds.empty:
        return pd.DataFrame(columns=["user_id", "utc_min", "utc_max"])

    req_bounds = {"user_id", "utc_min", "utc_max"}
    miss_bounds = req_bounds - set(coverage_bounds.columns)
    if miss_bounds:
        raise ValueError(f"coverage_bounds missing required columns: {sorted(miss_bounds)}")

    cb = coverage_bounds[["user_id", "utc_min", "utc_max"]].copy()
    cb["user_id"] = _normalize_user_id_series(cb["user_id"])
    cb["utc_min"] = _coerce_utc_series(cb["utc_min"], context="coverage_bounds.utc_min", errors="coerce")
    cb["utc_max"] = _coerce_utc_series(cb["utc_max"], context="coverage_bounds.utc_max", errors="coerce")
    cb = cb.dropna(subset=["user_id", "utc_min", "utc_max"])
    if cb.empty:
        return cb
    cb["user_id"] = cb["user_id"].astype("int64")

    cb = (
        cb.groupby("user_id", as_index=False)
        .agg(utc_min=("utc_min", "min"), utc_max=("utc_max", "max"))
    )
    cb = cb[cb["utc_max"] >= cb["utc_min"]].copy()
    return cb


def _normalize_schedule_source(
    df: pd.DataFrame,
    user_col: str,
    start_col: str,
    end_col: str,
    tz_col: str = "timezone",
    point_policy: str = "include_epsilon",
    point_epsilon: str = "1ns",
    default_tz: str = "UTC",
    invalid_tz_policy: str = "strict",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Canonical normalization for schedule source rows used by inference and strict coverage.

    Returns:
      normalized source columns: [user_col, start_utc, end_utc, tz_name]
      diagnostics dict with stable counters
    """
    empty_source = pd.DataFrame(columns=[user_col, "start_utc", "end_utc", "tz_name"])
    diagnostics = {
        "source_rows_total": int(0 if df is None else len(df)),
        "source_rows_epsilon_promoted": 0,
        "source_rows_dropped_invalid": 0,
        "invalid_tz_rows": 0,
    }

    if point_policy not in {"include_epsilon", "exclude"}:
        raise ValueError("schedule_point_policy must be one of {'include_epsilon','exclude'}")
    if invalid_tz_policy not in {"strict", "coerce_default"}:
        raise ValueError("schedule_invalid_tz_policy must be one of {'strict','coerce_default'}")

    if df is None or df.empty:
        return empty_source, diagnostics

    if user_col not in df.columns:
        raise ValueError(f"schedule source missing required column: '{user_col}'")
    if start_col not in df.columns or end_col not in df.columns:
        raise ValueError(f"schedule source requires '{start_col}' and '{end_col}'")

    if not _validate_timezone_name(default_tz):
        raise ValueError(f"default_tz is not a valid IANA timezone: {default_tz}")

    cols = [user_col, start_col, end_col]
    if tz_col in df.columns:
        cols.append(tz_col)
    src = df[cols].copy()
    src[user_col] = _normalize_user_id_series(src[user_col])
    src["start_utc"] = _coerce_utc_series(src[start_col], context=start_col, errors="coerce")
    src["end_utc"] = _coerce_utc_series(src[end_col], context=end_col, errors="coerce")

    valid_base = src[user_col].notna() & src["start_utc"].notna() & src["end_utc"].notna()
    diagnostics["source_rows_dropped_invalid"] += int((~valid_base).sum())
    src = src.loc[valid_base].copy()
    if src.empty:
        return empty_source, diagnostics
    src[user_col] = src[user_col].astype("int64")

    neg_mask = src["end_utc"] < src["start_utc"]
    if neg_mask.any():
        diagnostics["source_rows_dropped_invalid"] += int(neg_mask.sum())
        src = src.loc[~neg_mask].copy()
    if src.empty:
        return empty_source, diagnostics

    eq_mask = src["end_utc"] == src["start_utc"]
    if eq_mask.any():
        if point_policy == "include_epsilon":
            eps = pd.to_timedelta(point_epsilon)
            if eps <= pd.Timedelta(0):
                raise ValueError("schedule_point_epsilon must be > 0 when schedule_point_policy='include_epsilon'")
            src.loc[eq_mask, "end_utc"] = src.loc[eq_mask, "start_utc"] + eps
            diagnostics["source_rows_epsilon_promoted"] += int(eq_mask.sum())
        elif point_policy == "exclude":
            diagnostics["source_rows_dropped_invalid"] += int(eq_mask.sum())
            src = src.loc[~eq_mask].copy()
        else:
            raise ValueError("schedule_point_policy must be one of {'include_epsilon','exclude'}")
    if src.empty:
        return empty_source, diagnostics

    if tz_col in src.columns:
        tz_raw = src[tz_col].astype("string").str.strip()
    else:
        tz_raw = pd.Series(pd.NA, index=src.index, dtype="string")

    tz_name = tz_raw.where(tz_raw.notna() & tz_raw.ne(""), default_tz)

    candidates = tz_name.dropna().unique().tolist()
    invalid_values: Set[str] = set()
    for tz_val in candidates:
        tz_txt = str(tz_val)
        if not _validate_timezone_name(tz_txt):
            invalid_values.add(tz_txt)

    invalid_mask = tz_name.astype("string").isin(list(invalid_values))
    invalid_count = int(invalid_mask.sum())
    diagnostics["invalid_tz_rows"] = invalid_count

    if invalid_count:
        if invalid_tz_policy == "strict":
            sample = sorted(list(invalid_values))[:5]
            raise ValueError(f"invalid timezone labels in schedule source; sample={sample}")
        if invalid_tz_policy == "coerce_default":
            tz_name = tz_name.mask(invalid_mask, default_tz)

    out = src[[user_col, "start_utc", "end_utc"]].copy()
    out["start_utc"] = _coerce_utc_series(out["start_utc"], context="schedule_source.start_utc", errors="coerce")
    out["end_utc"] = _coerce_utc_series(out["end_utc"], context="schedule_source.end_utc", errors="coerce")
    out["tz_name"] = tz_name.astype("string").fillna(default_tz)

    positive = out["end_utc"] > out["start_utc"]
    if (~positive).any():
        diagnostics["source_rows_dropped_invalid"] += int((~positive).sum())
        out = out.loc[positive].copy()
    if out.empty:
        return empty_source, diagnostics
    out[user_col] = out[user_col].astype("int64")

    out = out.sort_values([user_col, "start_utc", "end_utc"]).reset_index(drop=True)
    return out, diagnostics


def _coalesce_schedule_rows(schedule: pd.DataFrame, user_col: str = "user_id") -> pd.DataFrame:
    """Coalesce adjacent schedule rows with identical timezone labels."""
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=[user_col, "start_utc", "end_utc", "tz_name"])

    rows: List[Tuple[Any, pd.Timestamp, pd.Timestamp, str]] = []
    s = schedule.sort_values([user_col, "start_utc", "end_utc"]).copy()
    s[user_col] = _normalize_user_id_series(s[user_col])
    s["start_utc"] = _coerce_utc_series(s["start_utc"], context="coalesce.start_utc", errors="coerce")
    s["end_utc"] = _coerce_utc_series(s["end_utc"], context="coalesce.end_utc", errors="coerce")
    s = s.dropna(subset=[user_col, "start_utc", "end_utc"])
    if s.empty:
        return pd.DataFrame(columns=[user_col, "start_utc", "end_utc", "tz_name"])
    s[user_col] = s[user_col].astype("int64")
    for uid, g in s.groupby(user_col, sort=False):
        g = g[g["end_utc"] > g["start_utc"]].sort_values("start_utc")
        if g.empty:
            continue
        cs = g["start_utc"].iloc[0]
        ce = g["end_utc"].iloc[0]
        cl = str(g["tz_name"].iloc[0])
        for st, en, label in zip(g["start_utc"].iloc[1:], g["end_utc"].iloc[1:], g["tz_name"].iloc[1:]):
            lab = str(label)
            if lab == cl and st <= ce:
                ce = max(ce, en)
            else:
                rows.append((uid, cs, ce, cl))
                cs, ce, cl = st, en, lab
        rows.append((uid, cs, ce, cl))
    return pd.DataFrame(rows, columns=[user_col, "start_utc", "end_utc", "tz_name"])


# -----------------------------------------------------------------------------
# Public APIs
# -----------------------------------------------------------------------------
def build_schedule_from_input(
    df: pd.DataFrame,
    user_col: str = "user_id",
    start_col: str = "utc_start",
    end_col: str = "utc_end",
    tz_col: str = "timezone",
    default_tz: str = "UTC",
    fill_gaps: str = "ffill",  # "ffill", "bfill", or "default"
    coverage_bounds: Optional[pd.DataFrame] = None,
    point_policy: str = "include_epsilon",
    point_epsilon: str = "1ns",
    invalid_tz_policy: str = "strict",
    report: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Sweep-line schedule builder (memory-lean).

    Output schema (required by downstream code):
      ['user_id','start_utc','end_utc','tz_name'] with tz-aware UTC bounds.

    Key idea:
      - Sort all interval boundary events per user.
      - Maintain active counts per tz.
      - The winning tz on any open span is argmax(active_count[tz]).
      - Emit a new segment only when the winner changes.
      - Gaps (no active tz) are filled by policy (ffill/bfill/default).

    Complexity:
      - Time: O(N log N) for sorting 2N events per user; updates are O(1) expected.
      - Memory: O(N) for events + O(#distinct_tz) active counters.
    """

    if report is None:
        report = {}

    empty = pd.DataFrame(columns=[user_col, "start_utc", "end_utc", "tz_name"])
    schedule_report = report.setdefault("schedule", {})
    schedule_report["source_rows_total"] = int(0 if df is None else len(df))
    schedule_report["source_rows_epsilon_promoted"] = 0
    schedule_report["source_rows_dropped_invalid"] = 0
    schedule_report["invalid_tz_rows"] = 0

    d, source_diag = _normalize_schedule_source(
        df=df,
        user_col=user_col,
        start_col=start_col,
        end_col=end_col,
        tz_col=tz_col,
        point_policy=point_policy,
        point_epsilon=point_epsilon,
        default_tz=default_tz,
        invalid_tz_policy=invalid_tz_policy,
    )

    schedule_report["source_rows_total"] = int(source_diag.get("source_rows_total", 0))
    schedule_report["source_rows_epsilon_promoted"] = int(source_diag.get("source_rows_epsilon_promoted", 0))
    schedule_report["source_rows_dropped_invalid"] = int(source_diag.get("source_rows_dropped_invalid", 0))
    schedule_report["invalid_tz_rows"] = int(source_diag.get("invalid_tz_rows", 0))

    if d.empty:
        return empty

    out_rows = []

    for uid, g in d.groupby(user_col, sort=False):
        # Build event array: (time_ns, type, tz)
        # type: 0=end, 1=start so that we process interval ends before starts at the same instant ([start,end) half-open)
        starts = pd.DataFrame({
            "t": g["start_utc"].astype("int64").to_numpy(),
            "typ": np.ones(len(g), dtype=np.int8),
            "tz": g["tz_name"].to_numpy(object)
        })
        ends = pd.DataFrame({
            "t": g["end_utc"].astype("int64").to_numpy(),
            "typ": np.zeros(len(g), dtype=np.int8),
            "tz": g["tz_name"].to_numpy(object)
        })
        ev = pd.concat([starts, ends], ignore_index=True)
        ev = ev.sort_values(["t", "typ"])  # ends (0) before starts (1)

        active = {}          # tz -> active count
        cur_t = None         # current span start (ns)
        cur_winner = None    # current winning tz (or None)
        segments = []        # (start_ns, end_ns, tz_or_None)

        def winner_from_counts(prev: Optional[str]) -> Optional[str]:
            if not active:
                return None
            # Find max count; prefer previous winner if tied; else deterministic by tz string
            maxc = max(active.values()) if active else 0
            if maxc <= 0:
                return None
            tied = [tz for tz, c in active.items() if c == maxc]
            if not tied:
                return None
            if prev in tied:
                return prev
            return sorted(tied)[0]  # deterministic tie-break

        for _, row in ev.iterrows():
            t = int(row["t"]); typ = int(row["typ"]); tz = row["tz"]

            # Close span if winner across (cur_t, t) was set
            if cur_t is not None and t > cur_t:
                segments.append((cur_t, t, cur_winner))

            # Apply event
            if typ == 0:
                # end
                active[tz] = active.get(tz, 0) - 1
                if active[tz] <= 0:
                    active.pop(tz, None)
            else:
                # start
                active[tz] = active.get(tz, 0) + 1

            # Determine new winner for next span
            new_winner = winner_from_counts(cur_winner)
            cur_t = t
            cur_winner = new_winner

        # No more events; nothing to extend after last boundary

        if not segments:
            g_start = g["start_utc"].min()
            g_end = g["end_utc"].max()
            if pd.notna(g_start) and pd.notna(g_end) and g_end > g_start:
                out_rows.append((uid, g_start, g_end, default_tz))
            continue

        # Convert to DataFrame and fill gaps according to policy
        seg = pd.DataFrame(segments, columns=["start_ns", "end_ns", "label"])
        seg = seg[seg["end_ns"] > seg["start_ns"]].copy()

        if fill_gaps in ("ffill", "bfill"):
            # First pass: fill None with NaN for convenience
            seg["label"] = seg["label"].astype(object)
            if fill_gaps == "ffill":
                seg["label"] = seg["label"].replace({None: np.nan}).ffill().bfill().fillna(default_tz).astype(object)
            else:
                seg["label"] = seg["label"].replace({None: np.nan}).bfill().ffill().fillna(default_tz).astype(object)
        else:
            seg["label"] = seg["label"].apply(lambda x: x if x is not None else default_tz)

        # Coalesce adjacent segments with same label
        seg = seg.sort_values("start_ns").reset_index(drop=True)
        coalesced = []
        cs, ce, cl = int(seg.loc[0, "start_ns"]), int(seg.loc[0, "end_ns"]), str(seg.loc[0, "label"])
        for i in range(1, len(seg)):
            s, e, l = int(seg.loc[i, "start_ns"]), int(seg.loc[i, "end_ns"]), str(seg.loc[i, "label"])
            if l == cl and s == ce:
                ce = e
            else:
                coalesced.append((cs, ce, cl))
                cs, ce, cl = s, e, l
        coalesced.append((cs, ce, cl))

        # Emit
        for s, e, l in coalesced:
            out_rows.append((uid, pd.to_datetime(s, utc=True), pd.to_datetime(e, utc=True), l))

    if not out_rows:
        schedule = empty.copy()
    else:
        schedule = pd.DataFrame(out_rows, columns=[user_col, "start_utc", "end_utc", "tz_name"])

    schedule[user_col] = _normalize_user_id_series(schedule[user_col])
    schedule["start_utc"] = _coerce_utc_series(schedule["start_utc"], context="schedule.start_utc", errors="coerce")
    schedule["end_utc"] = _coerce_utc_series(schedule["end_utc"], context="schedule.end_utc", errors="coerce")
    schedule = schedule.dropna(subset=[user_col, "start_utc", "end_utc"])
    if not schedule.empty:
        schedule[user_col] = schedule[user_col].astype("int64")
    schedule = schedule[schedule["end_utc"] > schedule["start_utc"]].copy()

    # Optional explicit bounds can extend schedule edges where needed.
    cb = _normalize_coverage_bounds(coverage_bounds)
    if not cb.empty:
        sch_groups = {uid: g for uid, g in schedule.groupby(user_col)}
        pad_rows: List[Tuple[Any, pd.Timestamp, pd.Timestamp, str]] = []
        for _, row in cb.iterrows():
            uid = row["user_id"]
            exp_min = row["utc_min"]
            exp_max = row["utc_max"]
            if pd.isna(exp_min) or pd.isna(exp_max) or exp_max <= exp_min:
                continue
            g = sch_groups.get(uid)
            if g is None or g.empty:
                pad_rows.append((uid, exp_min, exp_max, default_tz))
                continue
            smin = g["start_utc"].min()
            smax = g["end_utc"].max()
            if smin > exp_min:
                pad_rows.append((uid, exp_min, smin, default_tz))
            if smax < exp_max:
                pad_rows.append((uid, smax, exp_max, default_tz))
        if pad_rows:
            schedule = pd.concat(
                [schedule, pd.DataFrame(pad_rows, columns=[user_col, "start_utc", "end_utc", "tz_name"])],
                ignore_index=True,
            )

    # Guarantee per-user span against canonical source bounds.
    norm_bounds = (
        d.groupby(user_col, as_index=False)
        .agg(utc_min=("start_utc", "min"), utc_max=("end_utc", "max"))
    )
    if not norm_bounds.empty:
        norm_bounds[user_col] = _normalize_user_id_series(norm_bounds[user_col])
        norm_bounds["utc_min"] = _coerce_utc_series(norm_bounds["utc_min"], context="norm_bounds.utc_min", errors="coerce")
        norm_bounds["utc_max"] = _coerce_utc_series(norm_bounds["utc_max"], context="norm_bounds.utc_max", errors="coerce")
        bad_norm = norm_bounds[user_col].isna() | norm_bounds["utc_min"].isna() | norm_bounds["utc_max"].isna()
        if bad_norm.any():
            schedule_report["source_rows_dropped_invalid"] = int(
                schedule_report.get("source_rows_dropped_invalid", 0) + int(bad_norm.sum())
            )
            norm_bounds = norm_bounds.loc[~bad_norm].copy()
        if not norm_bounds.empty:
            norm_bounds[user_col] = norm_bounds[user_col].astype("int64")
    if not norm_bounds.empty:
        sch_groups = {uid: g for uid, g in schedule.groupby(user_col)}
        fallback_rows: List[Tuple[Any, pd.Timestamp, pd.Timestamp, str]] = []
        for _, row in norm_bounds.iterrows():
            uid = row[user_col]
            exp_min = row["utc_min"]
            exp_max = row["utc_max"]
            g = sch_groups.get(uid)
            if g is None or g.empty:
                fallback_rows.append((uid, exp_min, exp_max, default_tz))
                continue
            smin = g["start_utc"].min()
            smax = g["end_utc"].max()
            if smin > exp_min:
                fallback_rows.append((uid, exp_min, smin, default_tz))
            if smax < exp_max:
                fallback_rows.append((uid, smax, exp_max, default_tz))
        if fallback_rows:
            schedule = pd.concat(
                [schedule, pd.DataFrame(fallback_rows, columns=[user_col, "start_utc", "end_utc", "tz_name"])],
                ignore_index=True,
            )

    schedule = _coalesce_schedule_rows(schedule, user_col=user_col)
    schedule = schedule[schedule["end_utc"] > schedule["start_utc"]]
    schedule = schedule.sort_values([user_col, "start_utc", "end_utc"]).reset_index(drop=True)
    return schedule


def build_bin_windows(
    df: pd.DataFrame,
    schedule: Optional[pd.DataFrame],
    tz_mode_l0: str,
    tz_policy: str,
    bin_size: Any,
    bin_origin: Any,
    anchor_utc: pd.Timestamp,
    bin_seconds_nominal: int,
    bin_index_from_start: Any,
    report: Optional[Dict[str, Any]] = None,
    schedule_point_policy: str = "include_epsilon",
    schedule_point_epsilon: str = "1ns",
    schedule_invalid_tz_policy: str = "strict",
    schedule_default_tz: str = "UTC",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct per-user bin grid; attach local projection labels once."""
    if report is None:
        report = {}
    df = df.copy()
    df["user_id"] = _normalize_user_id_series(df["user_id"])
    df = df[df["user_id"].notna()].copy()
    if not df.empty:
        df["user_id"] = df["user_id"].astype("int64")
    report.setdefault("schedule", {})
    report["schedule"]["source_rows_total"] = int(0 if df is None else len(df))
    report["schedule"]["source_rows_epsilon_promoted"] = 0
    report["schedule"]["source_rows_dropped_invalid"] = 0
    report["schedule"]["invalid_tz_rows"] = 0

    # UTC mode can operate without schedule or timezone-column normalization.
    if schedule is None and tz_mode_l0 == "utc":
        users = df["user_id"].dropna().unique()
        _starts = pd.to_datetime(df["utc_start"], utc=True, errors="coerce")
        _ends = pd.to_datetime(df["utc_end"], utc=True, errors="coerce")
        coverage_bounds = (
            df.assign(_s=_starts, _e=_ends)
            .groupby("user_id", as_index=False)
            .agg(utc_min=("_s", "min"), utc_max=("_e", "max"))
        )
        utc_min = coverage_bounds["utc_min"].min() if not coverage_bounds.empty else _starts.min()
        utc_max = coverage_bounds["utc_max"].max() if not coverage_bounds.empty else _ends.max()
    else:
        source_norm, source_diag = _normalize_schedule_source(
            df=df,
            user_col="user_id",
            start_col="utc_start",
            end_col="utc_end",
            tz_col="timezone",
            point_policy=schedule_point_policy,
            point_epsilon=schedule_point_epsilon,
            default_tz=schedule_default_tz,
            invalid_tz_policy=schedule_invalid_tz_policy,
        )
        report["schedule"]["source_rows_total"] = int(source_diag.get("source_rows_total", 0))
        report["schedule"]["source_rows_epsilon_promoted"] = int(source_diag.get("source_rows_epsilon_promoted", 0))
        report["schedule"]["source_rows_dropped_invalid"] = int(source_diag.get("source_rows_dropped_invalid", 0))
        report["schedule"]["invalid_tz_rows"] = int(source_diag.get("invalid_tz_rows", 0))

        coverage_bounds = (
            source_norm.groupby("user_id", as_index=False)
            .agg(utc_min=("start_utc", "min"), utc_max=("end_utc", "max"))
            if not source_norm.empty
            else pd.DataFrame(columns=["user_id", "utc_min", "utc_max"])
        )
        users = (
            source_norm["user_id"].dropna().unique()
            if not source_norm.empty
            else df["user_id"].dropna().unique()
        )
        utc_min = coverage_bounds["utc_min"].min() if not coverage_bounds.empty else pd.to_datetime(df["utc_start"], utc=True, errors="coerce").min()
        utc_max = coverage_bounds["utc_max"].max() if not coverage_bounds.empty else pd.to_datetime(df["utc_end"], utc=True, errors="coerce").max()

    if tz_mode_l0 == "local" and (schedule is None or schedule.empty):
        raise ValueError("local mode requires a travel schedule with tz ranges")
    sch = validate_schedule(
        schedule,
        users,
        utc_min,
        utc_max,
        tz_policy,
        report=report,
        coverage_bounds=coverage_bounds,
    )

    if tz_mode_l0 == "utc":
        bins_base = build_utc_bin_grid(df, anchor_utc, bin_seconds_nominal, bin_index_from_start)
    else:
        bins_base = build_local_bin_grid(df, sch, bin_seconds_nominal, bin_size, bin_origin, anchor_utc, bin_index_from_start, tz_policy)

    bounds_for_clip = (
        coverage_bounds.copy()
        if not coverage_bounds.empty
        else df.groupby("user_id").agg(utc_min=("utc_start", "min"), utc_max=("utc_end", "max")).reset_index()
    )
    bins_base = bins_base.merge(bounds_for_clip, on="user_id", how="left")
    bins_base = bins_base[(bins_base["bin_end"] > bins_base["utc_min"]) & (bins_base["bin_start"] < bins_base["utc_max"])]
    bins_base = bins_base.drop(columns=["utc_min", "utc_max"])

    windows = bins_base.rename(
        columns={
            "bin_start": "window_start",
            "bin_end": "window_end",
            "bin_seconds": "window_seconds",
            "bin_index": "window_id",
        }
    ).copy()
    windows["window_kind"] = "grid"

    windows = add_local_time_labels(windows, sch, report=report)

    windows["bin_start"] = windows["window_start"]
    windows["bin_end"] = windows["window_end"]
    windows["bin_seconds"] = windows["window_seconds"]
    windows["bin_index"] = windows["window_id"]

    report.setdefault("bins", {})
    report["bins"]["mode"] = tz_mode_l0
    report["bins"]["window_kind"] = "grid"
    report["bins"]["non_nominal_seconds_bins"] = int((windows["bin_seconds"] != bin_seconds_nominal).sum()) if len(windows) else 0
    report["bins"]["count"] = int(len(windows))
    report["bins"]["seconds_total"] = float((windows["window_end"] - windows["window_start"]).dt.total_seconds().sum()) if len(windows) else 0.0

    validate_window_table(windows, window_kind="grid", strict_nonoverlap=True)
    return windows, sch


def build_assessment_windows(
    assessments: pd.DataFrame,
    schedule: Optional[pd.DataFrame],
    tz_policy: str,
    window_spec: AssessmentWindowSpec,
    default_bin_origin: str = "00:00:00",
    default_tz_mode: str = "utc",
    report: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build validated whole, assessment-relative, or aligned lookback bins."""
    if report is None:
        report = {}
    window_spec.validate()

    if assessments is None or len(assessments) == 0:
        empty = pd.DataFrame(columns=[
            "user_id", "window_id", "window_start", "window_end", "window_seconds",
            "window_kind", "assessment_id", "anchor_time", "bin_in_lookback",
        ])
        sch = validate_schedule(schedule, [], pd.Timestamp.now(tz="UTC"), pd.Timestamp.now(tz="UTC"), tz_policy, report=report)
        return empty, sch

    req = {"user_id", "assessment_id", "anchor_time"}
    missing = req - set(assessments.columns)
    if missing:
        raise ValueError(f"assessments missing required columns: {sorted(missing)}")

    a = assessments[["user_id", "assessment_id", "anchor_time"]].copy()
    a["assessment_order"] = np.arange(len(a), dtype="int64")
    normalized_users = _normalize_user_id_series(a["user_id"])
    if normalized_users.isna().any():
        raise ValueError("assessments.user_id must contain only nonmissing integer ids")
    a["user_id"] = normalized_users.astype("int64")
    if a["assessment_id"].isna().any():
        raise ValueError("assessments.assessment_id must be nonmissing")
    if a.duplicated(["user_id", "assessment_id"]).any():
        raise ValueError("(user_id, assessment_id) must be unique")
    a["anchor_time"] = _coerce_utc_series(
        a["anchor_time"], context="assessments.anchor_time", errors="coerce"
    )
    if a["anchor_time"].isna().any():
        raise ValueError("assessments.anchor_time must be nonmissing and UTC-coercible")

    rows: List[Dict[str, Any]] = []
    if window_spec.mode in {"whole_lookback", "split_lookback"}:
        duration = pd.to_timedelta(window_spec.duration_before)
        n_bins = 1 if window_spec.mode == "whole_lookback" else int(window_spec.n_bins)
        bin_duration = duration / n_bins
        for assessment in a.itertuples(index=False):
            lookback_start = assessment.anchor_time - duration
            for bin_number in range(n_bins):
                start = lookback_start + bin_duration * bin_number
                end = (
                    assessment.anchor_time
                    if bin_number == n_bins - 1
                    else start + bin_duration
                )
                rows.append({
                    "user_id": assessment.user_id,
                    "assessment_id": assessment.assessment_id,
                    "anchor_time": assessment.anchor_time,
                    "bin_in_lookback": bin_number,
                    "window_start": start,
                    "window_end": end,
                    "assessment_order": assessment.assessment_order,
                })
    else:
        n_bins = int(window_spec.n_bins) if window_spec.n_bins is not None else None
        duration = (
            pd.to_timedelta(window_spec.duration_before)
            if window_spec.duration_before is not None
            else None
        )
        step = pd.to_timedelta(window_spec.bin_size)
        origin = pd.to_timedelta(
            default_bin_origin if window_spec.bin_origin is None else window_spec.bin_origin
        )
        tz_mode = window_spec.tz_mode or default_tz_mode
        if tz_mode == "utc":
            epoch_anchor = pd.Timestamp("1970-01-01", tz="UTC") + origin
            step_ns = int(step.value)
            for assessment in a.itertuples(index=False):
                if duration is None:
                    delta_ns = int(assessment.anchor_time.value - epoch_anchor.value)
                    on_boundary = delta_ns % step_ns == 0
                    terminal_start = epoch_anchor + (
                        (delta_ns // step_ns) - int(on_boundary)
                    ) * step
                    first_start = terminal_start - (n_bins - 1) * step
                    boundaries = [
                        (first_start + bin_number * step,
                         min(first_start + (bin_number + 1) * step, assessment.anchor_time))
                        for bin_number in range(n_bins)
                    ]
                else:
                    lookback_start = assessment.anchor_time - duration
                    first_index = int(
                        (lookback_start.value - epoch_anchor.value) // step_ns
                    )
                    stop_index = int(
                        (assessment.anchor_time.value - epoch_anchor.value + step_ns - 1)
                        // step_ns
                    )
                    boundaries = []
                    for grid_index in range(first_index, stop_index):
                        grid_start = epoch_anchor + grid_index * step
                        grid_end = grid_start + step
                        start = max(grid_start, lookback_start)
                        end = min(grid_end, assessment.anchor_time)
                        if end > start:
                            boundaries.append((start, end))

                for bin_number, (start, end) in enumerate(boundaries):
                    rows.append({
                        "user_id": assessment.user_id,
                        "assessment_id": assessment.assessment_id,
                        "anchor_time": assessment.anchor_time,
                        "bin_in_lookback": bin_number,
                        "window_start": start,
                        "window_end": end,
                        "assessment_order": assessment.assessment_order,
                    })
        else:
            if schedule is None or schedule.empty:
                raise ValueError("aligned_grid with tz_mode='local' requires a schedule")
            sch_for_grid = schedule.copy()
            sch_for_grid["user_id"] = _normalize_user_id_series(sch_for_grid["user_id"])
            sch_for_grid["start_utc"] = _coerce_utc_series(
                sch_for_grid["start_utc"], "schedule.start_utc", errors="coerce"
            )
            sch_for_grid["end_utc"] = _coerce_utc_series(
                sch_for_grid["end_utc"], "schedule.end_utc", errors="coerce"
            )
            sch_for_grid = sch_for_grid.dropna(
                subset=["user_id", "start_utc", "end_utc", "tz_name"]
            ).copy()
            sch_for_grid["user_id"] = sch_for_grid["user_id"].astype("int64")
            for assessment in a.itertuples(index=False):
                lookback_start = (
                    assessment.anchor_time - duration
                    if duration is not None
                    else None
                )
                candidate_start = (
                    assessment.anchor_time
                    - (duration if duration is not None else n_bins * step)
                    - pd.Timedelta(days=2)
                )
                candidate_end = assessment.anchor_time + step
                user_schedule = sch_for_grid.loc[
                    (sch_for_grid["user_id"] == assessment.user_id)
                    & (sch_for_grid["end_utc"] > candidate_start)
                    & (sch_for_grid["start_utc"] < candidate_end)
                ].copy()
                if not user_schedule.empty:
                    user_schedule["start_utc"] = user_schedule["start_utc"].clip(
                        lower=candidate_start
                    )
                    user_schedule["end_utc"] = user_schedule["end_utc"].clip(
                        upper=candidate_end
                    )
                dummy = pd.DataFrame([{
                    "user_id": assessment.user_id,
                    "utc_start": assessment.anchor_time,
                    "utc_end": assessment.anchor_time,
                }])
                candidates = build_local_bin_grid(
                    dummy,
                    user_schedule,
                    int(step.total_seconds()),
                    step,
                    str(default_bin_origin if window_spec.bin_origin is None else window_spec.bin_origin),
                    pd.Timestamp("1970-01-01", tz="UTC") + origin,
                    lambda starts: pd.Series(np.arange(len(starts)), index=starts.index),
                    tz_policy,
                )
                candidates = candidates.loc[
                    candidates["bin_start"] < assessment.anchor_time
                ].sort_values(["bin_start", "bin_end"])
                candidates = candidates.drop_duplicates("bin_start", keep="last")
                if duration is None:
                    candidates = candidates.tail(n_bins)
                    if len(candidates) != n_bins:
                        raise ValueError(
                            f"schedule does not provide {n_bins} aligned local bins for "
                            f"assessment {assessment.assessment_id!r}"
                        )
                else:
                    candidates = candidates.loc[
                        candidates["bin_end"] > lookback_start
                    ]

                boundaries = []
                for candidate in candidates.itertuples(index=False):
                    start = (
                        max(candidate.bin_start, lookback_start)
                        if lookback_start is not None
                        else candidate.bin_start
                    )
                    end = min(candidate.bin_end, assessment.anchor_time)
                    if end <= start:
                        continue
                    boundaries.append((start, end))

                if duration is not None:
                    is_contiguous = all(
                        previous_end == next_start
                        for (_, previous_end), (next_start, _) in zip(
                            boundaries, boundaries[1:]
                        )
                    )
                    spans_requested_interval = (
                        bool(boundaries)
                        and boundaries[0][0] == lookback_start
                        and boundaries[-1][1] == assessment.anchor_time
                        and is_contiguous
                    )
                    if not spans_requested_interval:
                        raise ValueError(
                            "schedule does not provide aligned local bins spanning "
                            "the full exact-duration interval for assessment "
                            f"{assessment.assessment_id!r}"
                        )

                for bin_number, (start, end) in enumerate(boundaries):
                    rows.append({
                        "user_id": assessment.user_id,
                        "assessment_id": assessment.assessment_id,
                        "anchor_time": assessment.anchor_time,
                        "bin_in_lookback": bin_number,
                        "window_start": start,
                        "window_end": end,
                        "assessment_order": assessment.assessment_order,
                    })

    windows = pd.DataFrame(rows)
    windows = windows.sort_values(
        ["assessment_order", "bin_in_lookback"], kind="stable"
    ).reset_index(drop=True)
    windows["window_id"] = np.arange(len(windows), dtype="int64")
    windows["window_seconds"] = (
        windows["window_end"] - windows["window_start"]
    ).dt.total_seconds()
    windows["window_kind"] = "anchored"
    windows = windows[[
        "user_id", "window_id", "window_start", "window_end", "window_seconds",
        "window_kind", "assessment_id", "anchor_time", "bin_in_lookback",
    ]]

    users = windows["user_id"].unique()
    utc_min = windows["window_start"].min()
    utc_max = windows["window_end"].max()
    coverage_bounds_raw = (
        windows.groupby("user_id")
        .agg(utc_min=("window_start", "min"), utc_max=("window_end", "max"))
        .reset_index()
    )
    coverage_bounds = _normalize_coverage_bounds(coverage_bounds_raw)
    if not coverage_bounds.empty:
        utc_min = coverage_bounds["utc_min"].min()
        utc_max = coverage_bounds["utc_max"].max()
    sch = validate_schedule(
        schedule,
        users,
        utc_min,
        utc_max,
        tz_policy,
        report=report,
        coverage_bounds=coverage_bounds,
    )

    windows = compute_schedule_coverage(windows, sch)
    if tz_policy == "strict":
        bad = windows["schedule_covered_fraction"] < 1.0
        if bad.any():
            raise ValueError("schedule does not fully cover one or more assessment windows under strict policy")

    windows = add_local_time_labels(windows, sch, report=report)

    windows["bin_start"] = windows["window_start"]
    windows["bin_end"] = windows["window_end"]
    windows["bin_seconds"] = windows["window_seconds"]

    report.setdefault("bins", {})
    report["bins"]["window_kind"] = "anchored"
    report["bins"]["assessment_window_mode"] = window_spec.mode
    if window_spec.mode == "aligned_grid" and window_spec.duration_before is not None:
        actual_counts = windows.groupby(["user_id", "assessment_id"]).size()
        report["bins"]["lookback_seconds"] = float(
            pd.to_timedelta(window_spec.duration_before).total_seconds()
        )
        report["bins"]["actual_bins_per_assessment_min"] = int(actual_counts.min())
        report["bins"]["actual_bins_per_assessment_max"] = int(actual_counts.max())
        report["bins"].pop("n_bins_per_assessment", None)
    else:
        report["bins"].pop("lookback_seconds", None)
        report["bins"].pop("actual_bins_per_assessment_min", None)
        report["bins"].pop("actual_bins_per_assessment_max", None)
        report["bins"]["n_bins_per_assessment"] = n_bins
    report.setdefault("windows", {})["overlap_stats"] = compute_window_overlap_depth_stats(windows)
    report.setdefault("windows", {})["schedule_coverage_fraction_lt1"] = int((windows["schedule_covered_fraction"] < 1.0).sum())

    validate_window_table(windows, window_kind="anchored", strict_nonoverlap=False)
    return windows, sch


def get_tz_localize_kwargs(tz_policy: str) -> Dict[str, str]:
    return {"ambiguous": "NaT", "nonexistent": "NaT"} if tz_policy == "strict" else {"ambiguous": "infer", "nonexistent": "shift_forward"}


def _resolve_lenient_overlaps_for_user(g: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve overlapping schedule rows for one user using half-open span slicing.

    Deterministic precedence for conflicting overlap spans: later input row wins.
    Expects columns: ['user_id','start_utc','end_utc','tz_name','_row_order'].
    """
    if g.empty:
        return pd.DataFrame(columns=["user_id", "start_utc", "end_utc", "tz_name"])

    uid = int(g["user_id"].iloc[0])
    work = g.copy()
    work = work[work["end_utc"] > work["start_utc"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["user_id", "start_utc", "end_utc", "tz_name"])

    starts = pd.DataFrame(
        {
            "t": work["start_utc"].astype("int64").to_numpy(),
            "typ": np.ones(len(work), dtype=np.int8),
            "_row_order": work["_row_order"].to_numpy(dtype=np.int64),
        }
    )
    ends = pd.DataFrame(
        {
            "t": work["end_utc"].astype("int64").to_numpy(),
            "typ": np.zeros(len(work), dtype=np.int8),
            "_row_order": work["_row_order"].to_numpy(dtype=np.int64),
        }
    )
    events = pd.concat([starts, ends], ignore_index=True)
    events = events.sort_values(["t", "typ", "_row_order"], kind="mergesort")

    row_to_tz = dict(zip(work["_row_order"].astype("int64"), work["tz_name"].astype(str)))
    active_rows: set[int] = set()
    prev_t: Optional[int] = None
    spans: List[Tuple[int, int, int, str]] = []

    for t, eg in events.groupby("t", sort=False):
        t_i = int(t)

        if prev_t is not None and t_i > prev_t and active_rows:
            winner = int(max(active_rows))
            winner_tz = str(row_to_tz[winner])
            if spans and spans[-1][2] == winner and spans[-1][3] == winner_tz and spans[-1][1] == prev_t:
                last_start, _, _, _ = spans[-1]
                spans[-1] = (last_start, t_i, winner, winner_tz)
            else:
                spans.append((prev_t, t_i, winner, winner_tz))

        for row_id in eg.loc[eg["typ"] == 0, "_row_order"].to_numpy(dtype=np.int64):
            active_rows.discard(int(row_id))
        for row_id in eg.loc[eg["typ"] == 1, "_row_order"].to_numpy(dtype=np.int64):
            active_rows.add(int(row_id))

        prev_t = t_i

    if not spans:
        return pd.DataFrame(columns=["user_id", "start_utc", "end_utc", "tz_name"])

    out = pd.DataFrame(spans, columns=["start_ns", "end_ns", "_winner_row", "tz_name"])
    out = out[out["end_ns"] > out["start_ns"]].copy()
    if out.empty:
        return pd.DataFrame(columns=["user_id", "start_utc", "end_utc", "tz_name"])

    out["user_id"] = uid
    out["start_utc"] = pd.to_datetime(out["start_ns"], utc=True)
    out["end_utc"] = pd.to_datetime(out["end_ns"], utc=True)
    return out[["user_id", "start_utc", "end_utc", "tz_name"]]


# -----------------------------------------------------------------------------
# Schedule Validation
# -----------------------------------------------------------------------------
def validate_schedule(
    schedule: Optional[pd.DataFrame],
    users: Iterable[Any],
    utc_min: pd.Timestamp,
    utc_max: pd.Timestamp,
    tz_policy: str,
    report: Optional[Dict[str, Any]] = None,
    coverage_bounds: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Validate per-user schedule coverage/overlaps and return normalized schedule."""
    if report is None:
        report = {}

    if schedule is None:
        u_ser = _normalize_user_id_series(pd.Series(list(users) if users is not None else []))
        u = pd.Index(u_ser.dropna().astype("int64")).unique().tolist()
        n = len(u)
        return pd.DataFrame(
            {
                "user_id": u,
                "start_utc": [pd.Timestamp.min.tz_localize("UTC")] * n,
                "end_utc": [pd.Timestamp.max.tz_localize("UTC")] * n,
                "tz_name": ["UTC"] * n,
            }
        )

    req = {"user_id", "start_utc", "end_utc", "tz_name"}
    missing = req - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule missing required columns: {sorted(missing)}")

    sch = schedule.copy()
    sch["user_id"] = _normalize_user_id_series(sch["user_id"])
    sch["start_utc"] = _coerce_utc_series(sch["start_utc"], context="schedule.start_utc", errors="coerce")
    sch["end_utc"] = _coerce_utc_series(sch["end_utc"], context="schedule.end_utc", errors="coerce")
    sch = sch.dropna(subset=["user_id", "start_utc", "end_utc"]).copy()
    if not sch.empty:
        sch["user_id"] = sch["user_id"].astype("int64")
    if (sch["end_utc"] <= sch["start_utc"]).any():
        raise ValueError("schedule has non-positive ranges")

    sch["_row_order"] = np.arange(len(sch), dtype=np.int64)

    out, overlaps = [], 0
    per_user = sch.sort_values(["user_id", "start_utc", "end_utc", "_row_order"], kind="mergesort")
    for uid, g in per_user.groupby("user_id", sort=False):
        g = g.copy()
        g_scan = g.sort_values(["start_utc", "end_utc", "_row_order"], kind="mergesort")
        over_mask = g_scan["start_utc"].shift(-1) < g_scan["end_utc"]
        if over_mask.any():
            overlaps += int(over_mask.sum())
            if tz_policy == "strict":
                raise ValueError(f"schedule has overlapping ranges for user_id={uid}")
            g = _resolve_lenient_overlaps_for_user(g_scan)
        else:
            g = g_scan[["user_id", "start_utc", "end_utc", "tz_name"]].copy()
        out.append(g)

    sch2 = pd.concat(out, ignore_index=True) if out else sch.iloc[0:0][["user_id", "start_utc", "end_utc", "tz_name"]].copy()
    sch2 = sch2.sort_values(["user_id", "start_utc", "end_utc"], kind="mergesort").reset_index(drop=True)

    if not sch2.empty:
        remaining_overlaps = int(
            sch2.sort_values(["user_id", "start_utc", "end_utc"], kind="mergesort")
            .groupby("user_id", sort=False)
            .apply(lambda g: int((g["start_utc"].shift(-1) < g["end_utc"]).sum()))
            .sum()
        )
        if remaining_overlaps and tz_policy == "lenient":
            raise ValueError("internal error: lenient schedule overlap resolution left overlapping ranges")

    report.setdefault("schedule", {})["overlap_ranges"] = overlaps

    # Build optional per-user expected bounds map.
    bounds_map: Dict[Any, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    cb = _normalize_coverage_bounds(coverage_bounds)
    if not cb.empty:
        bounds_map = {
            row["user_id"]: (row["utc_min"], row["utc_max"])
            for _, row in cb.iterrows()
        }

    user_list = (
        pd.Index(_normalize_user_id_series(pd.Series(list(users) if users is not None else [])).dropna().astype("int64"))
        .dropna()
        .unique()
        .tolist()
    )
    sch_by_user = {uid: g for uid, g in sch2.groupby("user_id")}

    gaps = 0
    gap_samples: List[Dict[str, Any]] = []
    for uid in user_list:
        exp_min, exp_max = bounds_map.get(uid, (utc_min, utc_max))
        g = sch_by_user.get(uid)
        if g is None or g.empty:
            gaps += 1
            if len(gap_samples) < 5:
                gap_samples.append(
                    {
                        "user_id": uid,
                        "reason": "missing_schedule",
                        "expected_min": str(exp_min),
                        "expected_max": str(exp_max),
                        "schedule_min": None,
                        "schedule_max": None,
                    }
                )
            continue

        smin, smax = g["start_utc"].min(), g["end_utc"].max()
        under_start = pd.notna(exp_min) and (smin > exp_min)
        under_end = pd.notna(exp_max) and (smax < exp_max)
        if under_start or under_end:
            gaps += 1
            if len(gap_samples) < 5:
                if under_start and under_end:
                    reason = "under_covered_start_and_end"
                elif under_start:
                    reason = "under_covered_start"
                else:
                    reason = "under_covered_end"
                gap_samples.append(
                    {
                        "user_id": uid,
                        "reason": reason,
                        "expected_min": str(exp_min),
                        "expected_max": str(exp_max),
                        "schedule_min": str(smin),
                        "schedule_max": str(smax),
                    }
                )
    if gaps and tz_policy == "strict":
        sample_txt = "; ".join(
            [
                f"user_id={d['user_id']} reason={d['reason']} expected=[{d['expected_min']}, {d['expected_max']}] "
                f"schedule=[{d['schedule_min']}, {d['schedule_max']}]"
                for d in gap_samples
            ]
        )
        raise ValueError(f"schedule does not fully cover analysis interval ({gaps} users). sample: {sample_txt}")
    report["schedule"]["coverage_gaps_users"] = gaps
    report["schedule"]["coverage_gap_users_sample"] = gap_samples
    return sch2

# -----------------------------------------------------------------------------
# Bin Grid Internals
# -----------------------------------------------------------------------------
def build_utc_bin_grid(
    df_utc: pd.DataFrame,
    anchor_utc: pd.Timestamp,
    bin_seconds_nominal: int,
    bin_index_from_start: Any,
) -> pd.DataFrame:
    """Build per-user UTC grid anchored to epoch+origin; constant bin seconds."""
    if df_utc.empty:
        return pd.DataFrame(columns=["user_id", "bin_start", "bin_end", "bin_seconds", "bin_index"])

    bounds = df_utc.groupby("user_id").agg(utc_min=("utc_start", "min"), utc_max=("utc_end", "max")).reset_index()
    frames = []
    step = pd.to_timedelta(bin_seconds_nominal, unit="s")

    for _, r in bounds.iterrows():
        uid, start, end = r["user_id"], r["utc_min"], r["utc_max"]
        s_idx = ((start.value - anchor_utc.value) // int(step.value))
        e_idx = ((end.value - anchor_utc.value + int(step.value) - 1) // int(step.value))
        starts = anchor_utc + np.arange(int(s_idx), int(e_idx)) * step
        if len(starts) == 0:
            continue
        frame = pd.DataFrame({"user_id": uid, "bin_start": starts})
        frame["bin_end"] = frame["bin_start"] + step
        frame["bin_seconds"] = (frame["bin_end"] - frame["bin_start"]).dt.total_seconds()
        frame["bin_index"] = bin_index_from_start(frame["bin_start"])
        frames.append(frame)

    bins = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["user_id", "bin_start", "bin_end", "bin_seconds", "bin_index"])

    if bins.empty:
        return bins
    if bins.duplicated(subset=["user_id", "bin_start"]).any():
        raise ValueError("Duplicate (user_id, bin_start) in bins")
    if (bins["bin_end"] <= bins["bin_start"]).any():
        raise ValueError("Non-positive bin width")

    return bins


def build_local_bin_grid(
    df_utc: pd.DataFrame,
    schedule: pd.DataFrame,
    bin_seconds_nominal: int,
    bin_size: Any,
    bin_origin: Any,
    anchor_utc: pd.Timestamp,
    bin_index_from_start: Any,
    tz_policy: str,
) -> pd.DataFrame:
    """Build per-user local-wall grid by schedule; convert edges to UTC and compute seconds."""
    if df_utc.empty:
        return pd.DataFrame(columns=["user_id", "bin_start", "bin_end", "bin_seconds", "bin_index"])
    if schedule is None or schedule.empty:
        raise ValueError("local mode requires a travel schedule with tz ranges")

    step = pd.to_timedelta(bin_seconds_nominal, unit="s")
    origin_delta = pd.to_timedelta(bin_origin)
    origin_time = (pd.Timestamp("1970-01-01") + origin_delta).time()
    amb = get_tz_localize_kwargs(tz_policy)
    pieces = []

    sch = schedule.sort_values(["user_id", "start_utc"]).copy()
    for uid, sg in sch.groupby("user_id"):
        for _, row in sg.iterrows():
            tz = ZoneInfo(str(row["tz_name"]))
            start_loc = row["start_utc"].tz_convert(tz)
            end_loc = row["end_utc"].tz_convert(tz)

            start_naive = pd.Timestamp.combine(start_loc.tz_localize(None).date(), origin_time)

            naive_starts = pd.date_range(
                start=start_naive,
                end=end_loc.tz_localize(None) + timedelta(days=2),
                freq=bin_size,
                inclusive="left",
                tz=None,
            )
            naive_starts = naive_starts[
                (naive_starts + step > start_loc.tz_localize(None)) 
                & (naive_starts < end_loc.tz_localize(None))]
            if len(naive_starts) == 0:
                continue

            try:
                loc0 = naive_starts.tz_localize(tz, ambiguous=False, nonexistent=amb["nonexistent"])
            except Exception:
                loc0 = naive_starts.tz_localize(tz, ambiguous=False, nonexistent="NaT")
            try:
                loc1 = naive_starts.tz_localize(tz, ambiguous=True, nonexistent=amb["nonexistent"])
            except Exception:
                loc1 = naive_starts.tz_localize(tz, ambiguous=True, nonexistent="NaT")

            starts_localized = (
                pd.concat([pd.Series(loc0), pd.Series(loc1)], ignore_index=True).dropna().sort_values().reset_index(drop=True)
            )

            starts_utc = starts_localized.dt.tz_convert("UTC")
            dfseg = (
                pd.DataFrame({"start_localized": starts_localized, "start_utc": starts_utc})
                .drop_duplicates(subset=["start_utc"])
                .sort_values("start_utc")
                .reset_index(drop=True)
            )

            if dfseg.empty:
                continue

            end_localized = dfseg["start_localized"].shift(-1)
            end_localized.iloc[-1] = end_loc
            end_utc = end_localized.dt.tz_convert("UTC")

            dfb = pd.DataFrame(
                {
                    "user_id": uid,
                    "bin_start": dfseg["start_utc"].to_numpy(),
                    "bin_end": end_utc.to_numpy(),
                }
            )
            dfb = dfb[dfb["bin_end"] > dfb["bin_start"]]
            if dfb.empty:
                continue

            dfb["bin_seconds"] = (dfb["bin_end"] - dfb["bin_start"]).dt.total_seconds()
            dfb["bin_index"] = bin_index_from_start(dfb["bin_start"])
            
            pieces.append(dfb)

    bins = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["user_id", "bin_start", "bin_end", "bin_seconds", "bin_index"])

    if bins.empty:
        return bins

    bins = bins.sort_values(["user_id", "bin_start", "bin_end"]).drop_duplicates(subset=["user_id", "bin_start"])

    if (bins["bin_end"] <= bins["bin_start"]).any():
        raise ValueError("Non-positive bin width in local grid")

    return bins

# -----------------------------------------------------------------------------
# Window Validation + Diagnostics
# -----------------------------------------------------------------------------
WINDOW_REQ = {"user_id", "window_id", "window_start", "window_end", "window_seconds"}


def validate_window_table(windows: pd.DataFrame, window_kind: str = "grid", strict_nonoverlap: bool = True) -> None:
    """Validate window schema/invariants."""
    missing = WINDOW_REQ - set(windows.columns)
    if missing:
        raise ValueError(f"windows missing required columns: {sorted(missing)}")

    w = windows
    if w.empty:
        return

    if w[["window_start", "window_end"]].isna().any().any():
        raise ValueError("windows contain NaT in window_start/window_end")
    if (w["window_end"] <= w["window_start"]).any():
        raise ValueError("windows contain non-positive widths")
    if w.duplicated(subset=["user_id", "window_id"]).any():
        raise ValueError("Duplicate (user_id, window_id) in windows")

    if window_kind == "grid" and strict_nonoverlap:
        overlaps = count_window_overlaps(w)
        if overlaps > 0:
            raise ValueError("Overlapping windows found for grid windows")


def count_window_overlaps(windows: pd.DataFrame) -> int:
    """Count overlapping adjacent intervals per user (conservative check)."""
    if windows.empty:
        return 0
    w = windows.sort_values(["user_id", "window_start", "window_end"])
    over = 0
    for uid, g in w.groupby("user_id"):
        s = g["window_start"].astype("int64").to_numpy()
        e = g["window_end"].astype("int64").to_numpy()
        if len(s) <= 1:
            continue
        over += int(np.sum(s[1:] < e[:-1]))
    return over


def compute_window_overlap_depth_stats(windows: pd.DataFrame) -> Dict[str, Any]:
    """Compute overlap depth diagnostics for potentially overlapping windows."""
    if windows.empty:
        return {"users": 0, "windows": 0, "users_with_overlap": 0, "max_depth": 0}

    users_with_overlap = 0
    max_depth = 0
    for uid, g in windows.sort_values(["user_id", "window_start", "window_end"]).groupby("user_id"):
        starts = g["window_start"].astype("int64").to_numpy()
        ends   = g["window_end"].astype("int64").to_numpy()
        events = np.concatenate([np.stack([starts, np.ones_like(starts)], axis=1),
                                np.stack([ends, -np.ones_like(ends)], axis=1)], axis=0)
        events = events[np.lexsort((events[:, 1], events[:, 0]))]
        depth = 0
        local_max = 0
        for _, delta in events:
            depth += int(delta)
            local_max = max(local_max, depth)
        if local_max > 1:
            users_with_overlap += 1
        max_depth = max(max_depth, local_max)

    return {
        "users": int(windows["user_id"].nunique()),
        "windows": int(len(windows)),
        "users_with_overlap": int(users_with_overlap),
        "max_depth": int(max_depth),
    }


def compute_schedule_coverage(windows: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach schedule_covered_seconds and schedule_covered_fraction to windows (coalesced schedule)."""
    if windows.empty:
        return windows.assign(schedule_covered_seconds=np.nan, schedule_covered_fraction=np.nan)

    w = windows.copy()
    w["schedule_covered_seconds"] = 0.0

    sch = schedule if schedule is not None else pd.DataFrame(columns=["user_id", "start_utc", "end_utc"])
    sch = sch.sort_values(["user_id", "start_utc"]).copy()

    def _coalesce_intervals(df):
        if df.empty: return []
        s = df.sort_values("start_utc")
        merged = []
        cur_s, cur_e = s["start_utc"].iloc[0], s["end_utc"].iloc[0]
        for st, en in zip(s["start_utc"].iloc[1:], s["end_utc"].iloc[1:]):
            if st <= cur_e:
                cur_e = max(cur_e, en)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = st, en
        merged.append((cur_s, cur_e))
        # return as int64 ns for vector math
        return [(a.value, b.value) for a, b in merged]

    for uid, wg in w.groupby("user_id"):
        sg = sch.loc[sch["user_id"] == uid, ["start_utc","end_utc"]]
        unions = _coalesce_intervals(sg)
        if not unions:
            continue

        ws = wg["window_start"].astype("int64").to_numpy()
        we = wg["window_end"].astype("int64").to_numpy()
        covered = np.zeros(len(wg), dtype="float64")

        for a_ns, b_ns in unions:
            start = np.maximum(ws, a_ns)
            end   = np.minimum(we, b_ns)
            sec = (end - start) / 1e9  # ns → s
            sec = np.clip(sec, 0.0, None)
            covered += sec.astype("float64")

        w.loc[wg.index, "schedule_covered_seconds"] = covered

    # Coverage cannot exceed the window itself
    w["schedule_covered_seconds"] = np.minimum(w["schedule_covered_seconds"], w["window_seconds"])
    w["schedule_covered_fraction"] = w["schedule_covered_seconds"] / w["window_seconds"].replace({0: np.nan})
    return w


# -----------------------------------------------------------------------------
# Local-Time Labeling
# -----------------------------------------------------------------------------
def add_local_time_labels(
    windows: pd.DataFrame,
    schedule: pd.DataFrame,
    report: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Robust local time labeling for UTC windows using per-user schedule.

    The projection preserves every window column, assigns timezone by maximum
    schedule overlap, and uses deterministic tie-breaking. Offset and DST
    calculations support pandas versions without vectorized ``utcoffset``.
    """

    # -------------------------------
    # 0) Empty / None fast-path
    # -------------------------------
    if windows is None or windows.empty:
        base = windows.copy() if windows is not None else pd.DataFrame()
        return base.assign(
            window_start_local=pd.NaT,
            window_end_local=pd.NaT,
            local_date=pd.NaT,
            anchor_local_date=pd.NaT,
            local_hour=pd.Series(dtype="Int64"),
            minute_of_day=pd.Series(dtype="Int64"),
            local_minute_index=pd.Series(dtype="Int64"),
            tz_name=pd.Series(dtype="string"),
            is_dst=pd.Series(dtype="boolean"),
            fold=pd.Series(dtype="Int64"),
            utc_offset_seconds=pd.Series(dtype="float"),
            local_weekday=pd.Series(dtype="Int64"),
            iso_week=pd.Series(dtype="Int64"),
        )

    w = windows.copy()

    # -------------------------------
    # 1) Required columns
    # -------------------------------
    required = ["user_id", "window_start", "window_end"]
    missing = [c for c in required if c not in w.columns]
    if missing:
        # Can't compute without these
        return w.assign(
            window_start_local=pd.NaT,
            window_end_local=pd.NaT,
            local_date=pd.NaT,
            anchor_local_date=pd.NaT,
            local_hour=pd.Series(pd.NA, index=w.index, dtype="Int64"),
            minute_of_day=pd.Series(pd.NA, index=w.index, dtype="Int64"),
            local_minute_index=pd.Series(pd.NA, index=w.index, dtype="Int64"),
            tz_name=pd.Series(pd.NA, index=w.index, dtype="string"),
            is_dst=pd.Series(pd.NA, index=w.index, dtype="boolean"),
            fold=pd.Series(pd.NA, index=w.index, dtype="Int64"),
            utc_offset_seconds=pd.Series(np.nan, index=w.index, dtype="float"),
            local_weekday=pd.Series(pd.NA, index=w.index, dtype="Int64"),
            iso_week=pd.Series(pd.NA, index=w.index, dtype="Int64"),
        )

    # stable row id
    w["_ord"] = np.arange(len(w), dtype=np.int64)

    # Normalize user_id to numeric key contract.
    w["user_id"] = _normalize_user_id_series(w["user_id"])
    w = w.dropna(subset=["user_id"]).copy()
    if not w.empty:
        w["user_id"] = w["user_id"].astype("int64")

    # Normalize timestamps to tz-aware UTC
    w["window_start"] = pd.to_datetime(w["window_start"], utc=True, errors="coerce")
    w["window_end"] = pd.to_datetime(w["window_end"], utc=True, errors="coerce")

    if "window_kind" not in w.columns:
        w["window_kind"] = "grid"

    # -------------------------------
    # 2) Normalize schedule
    # -------------------------------
    sch_valid = False
    sch = None

    if schedule is not None and not schedule.empty:
        sch = schedule.copy()
        needed = ["user_id", "start_utc", "end_utc", "tz_name"]
        if all(c in sch.columns for c in needed):
            sch["user_id"] = _normalize_user_id_series(sch["user_id"])

            sch["start_utc"] = pd.to_datetime(sch["start_utc"], utc=True, errors="coerce")
            sch["end_utc"] = pd.to_datetime(sch["end_utc"], utc=True, errors="coerce")
            sch["tz_name"] = sch["tz_name"].astype("string")

            sch = sch.dropna(subset=["user_id", "start_utc", "end_utc", "tz_name"])
            if not sch.empty:
                sch["user_id"] = sch["user_id"].astype("int64")
            sch = sch[sch["end_utc"] > sch["start_utc"]].copy()

            if not sch.empty:
                sch = sch.sort_values(["user_id", "start_utc"], kind="mergesort")
                sch_valid = True

    # -------------------------------
    # 3) Per-user majority-overlap mapping
    # -------------------------------
    # Mapping is overlap-majority based regardless of schedule overlap policy.
    if sch_valid:
        sch_groups = {uid: g for uid, g in sch.groupby("user_id", sort=False)}
        merged_frags = []

        for uid, u_wins in w.groupby("user_id", sort=False):
            u_sch = sch_groups.get(uid)
            if u_sch is None or u_sch.empty:
                continue

            u_wins2 = u_wins.dropna(subset=["window_start", "window_end"]).copy()
            u_wins2 = u_wins2[u_wins2["window_end"] > u_wins2["window_start"]]
            if u_wins2.empty:
                continue

            u_sch2 = u_sch.dropna(subset=["start_utc", "end_utc", "tz_name"]).copy()
            u_sch2 = u_sch2[u_sch2["end_utc"] > u_sch2["start_utc"]].sort_values("start_utc", kind="mergesort")
            if u_sch2.empty:
                continue

            sched_starts_ns = u_sch2["start_utc"].astype("int64").to_numpy()
            sched_ends_ns = u_sch2["end_utc"].astype("int64").to_numpy()
            sched_tz = u_sch2["tz_name"].astype(str).to_numpy()
            sched_start_ts = u_sch2["start_utc"].tolist()
            sched_end_ts = u_sch2["end_utc"].tolist()

            win_ord = u_wins2["_ord"].to_numpy(dtype=np.int64)
            win_starts_ns = u_wins2["window_start"].astype("int64").to_numpy()
            win_ends_ns = u_wins2["window_end"].astype("int64").to_numpy()

            out_rows: List[Tuple[int, str, pd.Timestamp, pd.Timestamp]] = []
            for ord_i, ws_ns, we_ns in zip(win_ord, win_starts_ns, win_ends_ns):
                # Candidate schedule rows satisfy start_utc < window_end.
                right = int(np.searchsorted(sched_starts_ns, we_ns, side="left"))
                if right <= 0:
                    continue

                # Additional half-open overlap condition: end_utc > window_start.
                candidate_idx = np.nonzero(sched_ends_ns[:right] > ws_ns)[0]
                if candidate_idx.size == 0:
                    continue

                overlap_start_ns = np.maximum(ws_ns, sched_starts_ns[candidate_idx])
                overlap_end_ns = np.minimum(we_ns, sched_ends_ns[candidate_idx])
                overlap_ns = overlap_end_ns - overlap_start_ns
                positive = overlap_ns > 0
                if not np.any(positive):
                    continue

                candidate_idx = candidate_idx[positive]
                overlap_ns = overlap_ns[positive]
                overlap_start_ns = overlap_start_ns[positive]

                # Tie-break order:
                # 1) max overlap duration
                # 2) earliest overlap start
                # 3) earliest schedule start
                # 4) lexicographic tz_name
                rank = np.lexsort(
                    (
                        sched_tz[candidate_idx],
                        sched_starts_ns[candidate_idx],
                        overlap_start_ns,
                        -overlap_ns,
                    )
                )
                best_idx = int(candidate_idx[int(rank[0])])
                out_rows.append(
                    (
                        int(ord_i),
                        str(sched_tz[best_idx]),
                        sched_start_ts[best_idx],
                        sched_end_ts[best_idx],
                    )
                )

            if out_rows:
                merged_frags.append(
                    pd.DataFrame(out_rows, columns=["_ord", "tz_name", "start_utc_sched", "end_utc_sched"])
                )

        full_map = pd.concat(merged_frags, ignore_index=True) if merged_frags else pd.DataFrame(
            columns=["_ord", "tz_name", "start_utc_sched", "end_utc_sched"]
        )
    else:
        full_map = pd.DataFrame(columns=["_ord", "tz_name", "start_utc_sched", "end_utc_sched"])

    merged = w.merge(full_map, on="_ord", how="left")

    # -------------------------------
    # Helpers: offset seconds + DST (pandas-version compatible)
    # -------------------------------
    def _offset_seconds_from_strftime(tz_aware_series: pd.Series) -> pd.Series:
        """
        Vectorized UTC offset seconds from %z like +0530/-0700 (works without .dt.utcoffset()).
        Returns float Series with NaN where unavailable.
        """
        z = tz_aware_series.dt.strftime("%z")  # object; NaT -> NaN
        z = z.where(z.notna(), np.nan)

        # Expect 5-char strings like +HHMM / -HHMM
        valid = z.notna() & (z.str.len() == 5)
        out = pd.Series(np.nan, index=tz_aware_series.index, dtype="float")

        if not valid.any():
            return out

        zz = z[valid]
        sign = zz.str.slice(0, 1).map({"+": 1.0, "-": -1.0})
        hh = pd.to_numeric(zz.str.slice(1, 3), errors="coerce")
        mm = pd.to_numeric(zz.str.slice(3, 5), errors="coerce")

        out.loc[valid] = sign * (hh * 3600.0 + mm * 60.0)
        return out

    def _is_dst_series(tz_aware_series: pd.Series) -> pd.Series:
        """
        Compute DST flag as boolean dtype. Uses vectorized .dt.dst() if available,
        otherwise falls back to elementwise datetime.dst().
        """
        dtp = tz_aware_series.dt
        if hasattr(dtp, "dst"):
            td = dtp.dst()  # timedelta-like, may contain NaT
            # td can be dtype timedelta64[ns], so compare to 0 safely
            return (td.fillna(pd.Timedelta(0)) > pd.Timedelta(0)).astype("boolean")

        # Fallback (slower, but compatible)
        def _one(x):
            if pd.isna(x):
                return pd.NA
            d = x.dst()
            return (d is not None) and (d != pd.Timedelta(0))

        return tz_aware_series.map(_one).astype("boolean")

    # -------------------------------
    # 5) Compute local labels
    # -------------------------------
    merged["window_start_local"] = pd.NaT
    merged["window_end_local"] = pd.NaT
    merged["is_dst"] = pd.Series(pd.NA, dtype="boolean", index=merged.index)
    merged["utc_offset_seconds"] = np.nan

    tz_fail_count = 0
    if merged["tz_name"].notna().any():
        merged["tz_name"] = merged["tz_name"].astype("string")

        for tz in merged["tz_name"].dropna().unique():
            tz_str = str(tz)
            idx = merged.index[merged["tz_name"] == tz]
            if len(idx) == 0:
                continue

            try:
                ws_loc_aw = merged.loc[idx, "window_start"].dt.tz_convert(tz_str)
                we_loc_aw = merged.loc[idx, "window_end"].dt.tz_convert(tz_str)
            except Exception:
                _logger.warning(
                    "tz_convert failed for timezone '%s' (%d rows); local labels will be null",
                    tz_str, len(idx),
                )
                tz_fail_count += len(idx)
                merged.loc[idx, "tz_name"] = pd.NA
                continue

            local_starts = merged["window_start_local"].copy()
            local_ends = merged["window_end_local"].copy()
            local_starts.loc[idx] = ws_loc_aw.dt.tz_localize(None).to_numpy()
            local_ends.loc[idx] = we_loc_aw.dt.tz_localize(None).to_numpy()
            merged["window_start_local"] = local_starts
            merged["window_end_local"] = local_ends

            # ✅ compatible offset computation
            offsets = merged["utc_offset_seconds"].copy()
            offsets.loc[idx] = _offset_seconds_from_strftime(ws_loc_aw).astype("float").values
            merged["utc_offset_seconds"] = offsets

            # ✅ compatible DST computation
            dst_flags = merged["is_dst"].copy()
            dst_flags.loc[idx] = _is_dst_series(ws_loc_aw).values
            merged["is_dst"] = dst_flags

    if report is not None and tz_fail_count:
        report.setdefault("local_time_labels", {})["tz_conversion_failures"] = tz_fail_count

    # Derived fields
    wsl = pd.to_datetime(merged["window_start_local"], errors="coerce")
    wel = pd.to_datetime(merged["window_end_local"], errors="coerce")

    merged["local_date"] = wsl.dt.date
    merged["anchor_local_date"] = wel.dt.date
    merged["local_hour"] = wsl.dt.hour.astype("Int64")
    merged["minute_of_day"] = (wsl.dt.hour * 60 + wsl.dt.minute).astype("Int64")
    merged["local_weekday"] = wsl.dt.dayofweek.astype("Int64")
    merged["iso_week"] = wsl.dt.isocalendar().week.astype("Int64")

    # fold (duplicate local time within user+tz)
    merged["fold"] = pd.Series(pd.NA, dtype="Int64", index=merged.index)
    mask_fold = merged["tz_name"].notna() & merged["window_start_local"].notna()
    if mask_fold.any():
        tmp = merged.loc[mask_fold, ["user_id", "tz_name", "window_start_local", "_ord"]].copy()
        tmp = tmp.sort_values(["user_id", "tz_name", "window_start_local", "_ord"], kind="mergesort")
        fold_vals = (
            tmp.groupby(["user_id", "tz_name", "window_start_local"], dropna=False)
               .cumcount()
               .clip(upper=1)
               .astype("Int64")
        )
        fold = merged["fold"].copy()
        fold.loc[tmp.index] = fold_vals.values
        merged["fold"] = fold

    # local_minute_index (sequence within local day)
    merged["local_minute_index"] = pd.Series(pd.NA, dtype="Int64", index=merged.index)
    mask_lmi = merged["tz_name"].notna() & merged["local_date"].notna() & merged["window_start_local"].notna()
    if mask_lmi.any():
        tmp = merged.loc[mask_lmi, ["user_id", "tz_name", "local_date", "window_start_local", "_ord"]].copy()
        tmp = tmp.sort_values(["user_id", "tz_name", "local_date", "window_start_local", "_ord"], kind="mergesort")
        lmi_vals = (
            tmp.groupby(["user_id", "tz_name", "local_date"], dropna=False)
               .cumcount()
               .astype("Int64")
        )
        minute_index = merged["local_minute_index"].copy()
        minute_index.loc[tmp.index] = lmi_vals.values
        merged["local_minute_index"] = minute_index

    # -------------------------------
    # 6) Return all window metadata plus the local projection
    # -------------------------------
    internal = {"_ord", "start_utc_sched", "end_utc_sched"}
    original = [c for c in windows.columns if c in merged.columns and c not in internal]
    labels = [
        "window_start_local", "window_end_local", "local_date", "anchor_local_date",
        "local_hour", "minute_of_day", "local_minute_index",
        "tz_name", "is_dst", "fold", "utc_offset_seconds", "local_weekday", "iso_week",
    ]
    keep_existing = original + [c for c in labels if c in merged.columns and c not in original]
    return merged[keep_existing].copy()

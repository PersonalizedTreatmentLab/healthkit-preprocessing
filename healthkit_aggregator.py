"""
HealthKit aggregation pipeline for converting raw HealthKit samples into time-windowed feature matrices.

This module defines `HealthKitAggregator`, which:
- normalizes timestamp inputs (UTC or local+timezone) & builds bin (time grid) or assessment-anchored windows,
- assigns point/interval/categorical observations to windows and computes per-window aggregates,
- derives wear metrics from HR coverage, builds presence flags, and applies configurable zero-fill gates,
- produces a wide per-window feature table plus auxiliary artifacts (windows, wear, schedule) and a report,
- optionally applies quality control (drop/mask) and supports hierarchical rollups via LayerSpec.

Primary entrypoints:
- run_aggregation(df, schedule=None, assessments=None, duration_before=None) -> (wide, report, aux)
- apply_quality_control(wide, aux) -> (wide_qc, report_qc)

-------
Misc. Ideas
- Daily sleep features mirrors Samir's. Note: AwakeUnspecified labeled ASLEEP, durations aren't split - any segment that spans the 3 pm boundary is fully counted on the start day, first()/last() within the 3 pm day will happily pick a mid-day nap for bedrestOffset or sleepOffset, EnvironmentalAudioExposure should use appropriate leq methods instead of simple median, source/wear gating isn't connected.
- AggregationPlan: Easier way to compute features at different levels? What if I want cardio, vitals, audio to be granular → hourly → daily, but sleep to be granular → daily?
    - While not currently used, bin windows with tz_mode "local", bin_origin 15:00, bin_size "1D" parallel Samir's nightly bins.
- Aggregation relies on time behavior. Ideally, I'd have something like op_registry apply to all aggregations.
- Note: Many window/time columns less reliable if doing hierarchical aggregation. Decision rules may not reflect preferences in all cases.
- Note: Sleep / environmental audio event count can't be summed to find the number of observations. Original samples are split across bin boundaries and can be recounted in each bin they're in.
    - Unable to get exact rollups for means because counts refer to fragments rather than observations.
- What are all the possible sleepAnalysis values? Could sleep events better be aggregated?
- Currently uses source for some things, derived from `source_device_model` column. Check possible values and distributions over the full dataset. Will this work as intended?
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Any, Iterable, Callable
import os
import numpy as np, pandas as pd
from healthkit_aggregator_specs import HKAggConfig, LayerSpec, build_default_op_registry
import healthkit_aggregator_time as hk_time
from zoneinfo import ZoneInfo
from healthkit_hrv import HRVComputer, compute_hrv_features_for_windows
import logging
import time

# ============================================================================
# HealthKitAggregator class
# ============================================================================
class HealthKitAggregator:
    """
    Public API:
      run_aggregation(df, schedule=None, assessments=None, duration_before=None) → (wide, report, aux)
      apply_quality_control(wide, aux) → (wide_qc, report_qc)

      aggregate_layer(child, spec, prev_layer_name=None)
      run_hierarchical_aggregation(wide_l0, specs, apply_qc_during=False, return_final_only=False)
      apply_layer_quality_control(layers_by_name)

      compute_daily_sleep_features
      apply_sleep_quality_control
    """

    # ------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------
    def __init__(self, config: Optional[HKAggConfig] = None, logger: Optional[logging.Logger] = None):
        """Initialize, validate config, cache zero-fill eligibility, register ops, and set bin anchor."""
        self.cfg = config or HKAggConfig(); self.cfg.validate()
        self.logger = logger or logging.getLogger(__name__)

        # Caches/registries
        self.zero_fill_eligible: Dict[str, set] = {t: set(meta.get("zero_fill_stats", set())) for t, meta in self.cfg.type_registry.items()}
        self.op_registry = build_default_op_registry()

        # Bin anchoring (UTC epoch + origin)
        self._epoch_utc = pd.Timestamp("1970-01-01 00:00:00", tz="UTC")
        self._bin_seconds_nominal = int(pd.to_timedelta(self.cfg.bin_size).total_seconds())
        self._origin_offset = pd.to_timedelta(self.cfg.bin_origin)
        self._anchor_utc = self._epoch_utc + self._origin_offset

        # Central report container (reset per pipeline run).
        self.report: Dict[str, Any] = self._new_report()

        # Zero-fill plan
        self._zero_fill_plan: List[Tuple[str, Tuple[str, ...], Tuple[str, Dict[str, Any]], float]] = []
        for tname, meta in self.cfg.type_registry.items():
            eligible = tuple(sorted(meta.get("zero_fill_stats", set())))
            if not eligible:
                continue
            gate_spec = meta.get("zero_fill_gate", ("expr_false", {}))
            gate_spec = self._normalize_zero_fill_gate(gate_spec)
            self._zero_fill_plan.append((tname, eligible, gate_spec, 0.0))

        # HRV computer (if enabled)
        self.hrv_computer = None
        if self.cfg.compute_hrv:
            self.hrv_computer = HRVComputer(
                min_samples=self.cfg.hrv_min_samples,
                time_domain=self.cfg.hrv_time_domain,
                freq_domain=self.cfg.hrv_freq_domain,
                freq_min_samples=self.cfg.hrv_freq_min_samples,
                interpolation_fs=self.cfg.hrv_interpolation_fs,
            )

    def _new_report(self) -> Dict[str, Any]:
        """Return a fresh report container for one aggregation run."""
        return {
            "input": {},
            "bins": {
                "bin_seconds_nominal": self._bin_seconds_nominal,
                "bin_origin_utc": str(self._anchor_utc),
            },
            "wear": {},
            "fills": {},
            "qc": {},
            "timing": {},
            "config_snapshot": asdict(self.cfg),
        }

    # ------------------------------------------------------------------------
    # Input Normalization
    # ------------------------------------------------------------------------
    @staticmethod
    def _normalize_user_id_numeric(values: Any) -> Tuple[pd.Series, pd.Series]:
        """
        Coerce user ids to nullable integer values.

        Returns:
          normalized_user_id: pandas Int64 series
          invalid_mask: True where user_id is missing/non-finite/non-integer-like
        """
        idx = getattr(values, "index", None)
        ser = values.copy() if isinstance(values, pd.Series) else pd.Series(values, index=idx)
        numeric = pd.to_numeric(ser, errors="coerce")
        finite = numeric.notna() & np.isfinite(numeric)
        integer_like = finite & (np.floor(numeric) == numeric)
        normalized = pd.Series(pd.array(np.where(integer_like, numeric, np.nan), dtype="Int64"), index=ser.index)
        invalid = ~integer_like
        return normalized, invalid

    def validate_and_normalize_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure required columns exist and datetimes are parsable.
        Accepts either UTC inputs ('utc_start','utc_end') or local inputs ('local_start','local_end' + 'timezone').
        """
        required_base = ["user_id","type","value"]; missing_base = [c for c in required_base if c not in df.columns]
        if missing_base: raise ValueError(f"Missing required columns: {missing_base}")
        df = df.copy()
        uid_norm, bad_uid = self._normalize_user_id_numeric(df["user_id"])
        dropped_invalid_uid = int(bad_uid.sum())
        if dropped_invalid_uid:
            df = df.loc[~bad_uid].copy()
            uid_norm = uid_norm.loc[df.index]
        if not df.empty:
            df["user_id"] = uid_norm.astype("int64")
        else:
            df["user_id"] = uid_norm

        # Ensure numeric types have numeric values (important when reloading data from csvs - as they'll default object/string due to categorical type values)
        num_types = [t for t, m in self.cfg.type_registry.items()
                    if m["time_behavior"] in ("point", "interval_total", "interval_level")]
        m = df["type"].isin(num_types)
        df.loc[m, "value"] = pd.to_numeric(df.loc[m, "value"], errors="coerce")


        has_utc = {"utc_start","utc_end"}.issubset(df.columns)
        has_local = {"local_start","local_end"}.issubset(df.columns)

        if has_utc:
            df["utc_start"] = pd.to_datetime(df["utc_start"], utc=True, format="mixed").dt.tz_convert("UTC")
            df["utc_end"]   = pd.to_datetime(df["utc_end"],   utc=True, format="mixed").dt.tz_convert("UTC")
        elif has_local:
            tzcol = "timezone" if "timezone" in df.columns else None
            if tzcol is None: raise ValueError("Provide 'timezone' column for local inputs or pass 'utc_start/utc_end'.")
            df["local_start"] = pd.to_datetime(df["local_start"], errors="coerce")
            df["local_end"]   = pd.to_datetime(df["local_end"], errors="coerce")
            if df["local_start"].isna().any() or df["local_end"].isna().any(): raise ValueError("local_start/local_end contain non-parsable datetimes")
            if (df["local_end"] <= df["local_start"]).any(): raise ValueError("non-positive intervals in input")

            utc_s = pd.Series(index=df.index, dtype="datetime64[ns, UTC]"); utc_e = pd.Series(index=df.index, dtype="datetime64[ns, UTC]")
            amb_count_start = amb_count_end = nonex_count_start = nonex_count_end = 0
            for tz, sub in df.groupby(df[tzcol].astype(str)):
                tzinfo = ZoneInfo(str(tz)); s = sub["local_start"]; e = sub["local_end"]
                s_nonex_mask = s.dt.tz_localize(tzinfo, ambiguous="infer", nonexistent="NaT").isna()
                e_nonex_mask = e.dt.tz_localize(tzinfo, ambiguous="infer", nonexistent="NaT").isna()
                s_ambig_mask = s.dt.tz_localize(tzinfo, ambiguous="NaT", nonexistent="shift_forward").isna()
                e_ambig_mask = e.dt.tz_localize(tzinfo, ambiguous="NaT", nonexistent="shift_forward").isna()
                amb_count_start += int(s_ambig_mask.sum()); amb_count_end += int(e_ambig_mask.sum())
                nonex_count_start += int(s_nonex_mask.sum()); nonex_count_end += int(e_nonex_mask.sum())

                if self.cfg.tz_policy == "strict":
                    if s_ambig_mask.any() or e_ambig_mask.any() or s_nonex_mask.any() or e_nonex_mask.any():
                        raise ValueError("Ambiguous or nonexistent local inputs encountered under strict policy.")
                    s_loc = s.dt.tz_localize(tzinfo, ambiguous="NaT", nonexistent="NaT")
                    e_loc = e.dt.tz_localize(tzinfo, ambiguous="NaT", nonexistent="NaT")
                else:
                    s_loc = s.dt.tz_localize(tzinfo, ambiguous="infer", nonexistent="shift_forward")
                    e_loc = e.dt.tz_localize(tzinfo, ambiguous="infer", nonexistent="shift_forward")

                utc_s.loc[sub.index] = s_loc.dt.tz_convert("UTC"); utc_e.loc[sub.index] = e_loc.dt.tz_convert("UTC")

            df["utc_start"] = utc_s; df["utc_end"] = utc_e
            self.report["input"].update({
                "ambiguous_local_inferred_start": int(amb_count_start),
                "ambiguous_local_inferred_end": int(amb_count_end),
                "nonexistent_shifted_start": int(nonex_count_start),
                "nonexistent_shifted_end": int(nonex_count_end),
            })
        else:
            raise ValueError("Provide either ('utc_start','utc_end') or ('local_start','local_end').")

        if (df["utc_start"] < pd.Timestamp("1970-01-01", tz="UTC")).any(): raise ValueError("utc_start before 1970-01-01 UTC is not supported")
        if (df["utc_end"] < df["utc_start"]).any(): raise ValueError("non-positive intervals in input")
        self.report["input"]["mode"] = self.cfg.tz_mode_l0
        self.report["input"]["dropped_invalid_user_id_non_numeric"] = dropped_invalid_uid
        return df

    # ------------------------------------------------------------------------
    # Aggregation Window Construction
    # ------------------------------------------------------------------------
    def _bin_index_from_start(self, bin_start_utc: pd.Series) -> pd.Series:
        """Integer bin index relative to anchor."""
        step_ns = int(pd.to_timedelta(self._bin_seconds_nominal, unit="s").value)
        anchor_ns = int(self._anchor_utc.value)
        return ((bin_start_utc.astype("int64") - anchor_ns) // step_ns).astype("int64")
        
    def build_bin_windows(self, df: pd.DataFrame, schedule: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Construct per-user bin grid; add local time labels once."""
        windows, sch = hk_time.build_bin_windows(
            df=df,
            schedule=schedule,
            tz_mode_l0=self.cfg.tz_mode_l0,
            tz_policy=self.cfg.tz_policy,
            bin_size=self.cfg.bin_size,
            bin_origin=self.cfg.bin_origin,
            anchor_utc=self._anchor_utc,
            bin_seconds_nominal=self._bin_seconds_nominal,
            bin_index_from_start=self._bin_index_from_start,
            report=self.report,
            schedule_point_policy=self.cfg.schedule_point_policy,
            schedule_point_epsilon=self.cfg.schedule_point_epsilon,
            schedule_invalid_tz_policy=self.cfg.schedule_invalid_tz_policy,
            schedule_default_tz="UTC",
        )
        return windows, sch

    def build_assessment_anchored_windows(self, assessments: pd.DataFrame, schedule: Optional[pd.DataFrame], duration_before: Optional[Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Construct per-user assessment anchored windows; attach local projection labels once."""
        dur = duration_before if duration_before is not None else self.cfg.assessment_duration_before
        windows, sch = hk_time.build_assessment_anchored_windows(
            assessments=assessments,
            schedule=schedule,
            tz_policy=self.cfg.tz_policy,
            duration_before=dur,
            report=self.report,
        )
        return windows, sch

    # ------------------------------------------------------------------------
    # Assign Observations to Windows
    # ------------------------------------------------------------------------
    @staticmethod
    def _compute_overlap_seconds(a_start: pd.Series, a_end: pd.Series, b_start: pd.Series, b_end: pd.Series) -> pd.Series:
        """Pairwise overlap (seconds) between sets of intervals A and B (defensive dtype coercion)."""
        a_start = pd.to_datetime(a_start, utc=True, errors="coerce")
        a_end   = pd.to_datetime(a_end,   utc=True, errors="coerce")
        b_start = pd.to_datetime(b_start, utc=True, errors="coerce")
        b_end   = pd.to_datetime(b_end,   utc=True, errors="coerce")

        start = pd.concat([a_start, b_start], axis=1).max(axis=1)
        end   = pd.concat([a_end,   b_end],   axis=1).min(axis=1)

        # Ensure timedelta dtype even if some rows are NaT
        td = end - start
        sec = td / np.timedelta64(1, "s")
        return pd.Series(sec, index=start.index).clip(lower=0).fillna(0.0)

    def _assign_intervals_to_windows(self, df_int: pd.DataFrame, windows: pd.DataFrame, start_col: str = "utc_start", end_col: str = "utc_end") -> pd.DataFrame:
        """Expand interval rows into window-aligned fragments."""
        if df_int.empty or windows.empty:
            return pd.DataFrame(columns=["user_id","window_id","window_start","window_end","window_seconds","duration","value",start_col,end_col])

        kind = str(windows["window_kind"].iloc[0]) if "window_kind" in windows.columns and len(windows) else "grid"
        if kind == "grid":
            # Sorted two-pointer sweep per user for O(n + m) on non-overlapping grids
            dfi = df_int.sort_values(["user_id", start_col, end_col]).copy()
            wns = windows.sort_values(["user_id", "window_start", "window_end"]).copy()
            out_records: List[Dict[str, Any]] = []
            users = np.intersect1d(dfi["user_id"].unique(), wns["user_id"].unique())

            for uid in users:
                u_int = dfi.loc[dfi["user_id"] == uid]
                u_win = wns.loc[wns["user_id"] == uid]
                if u_int.empty or u_win.empty:
                    continue

                i_start = u_int[start_col].to_numpy()
                i_end = u_int[end_col].to_numpy()
                i_val = u_int["value"].to_numpy()
                extra_cols = [c for c in u_int.columns if c not in {"user_id", start_col, end_col, "value"}]
                extras = {c: u_int[c].to_numpy() for c in extra_cols} if extra_cols else {}

                w_start = u_win["window_start"].to_numpy()
                w_end = u_win["window_end"].to_numpy()
                w_sec = u_win["window_seconds"].to_numpy()
                w_id = u_win["window_id"].to_numpy()

                i = j = 0; nI, nW = len(u_int), len(u_win)
                while i < nI and j < nW:
                    start = i_start[i] if i_start[i] > w_start[j] else w_start[j]
                    end = i_end[i] if i_end[i] < w_end[j] else w_end[j]
                    if start < end:
                        dur = (end - start) / np.timedelta64(1, "s")
                        rec = {"user_id": uid, "window_id": w_id[j], "window_start": w_start[j], "window_end": w_end[j], "window_seconds": float(w_sec[j]),
                               "duration": float(dur), "value": i_val[i], start_col: i_start[i], end_col: i_end[i]}
                        for c in extras: rec[c] = extras[c][i]
                        out_records.append(rec)
                    if i_end[i] <= w_end[j]: i += 1
                    else: j += 1

            if not out_records:
                return pd.DataFrame(columns=["user_id","window_id","window_start","window_end","window_seconds","duration","value",start_col,end_col])
            return pd.DataFrame.from_records(out_records)

        # Overlap join for overlapping windows
        dfi = df_int.sort_values(["user_id", start_col, end_col]).copy()
        wns = windows.sort_values(["user_id", "window_start", "window_end"]).copy()
        out_records: List[Dict[str, Any]] = []
        users = np.intersect1d(dfi["user_id"].unique(), wns["user_id"].unique())

        for uid in users:
            u_int = dfi.loc[dfi["user_id"] == uid]
            u_win = wns.loc[wns["user_id"] == uid]
            if u_int.empty or u_win.empty:
                continue

            w_start = u_win["window_start"].to_numpy()
            w_end = u_win["window_end"].to_numpy()
            w_sec = u_win["window_seconds"].to_numpy()
            w_id = u_win["window_id"].to_numpy()

            i_start = u_int[start_col].to_numpy()
            i_end = u_int[end_col].to_numpy()
            i_val = u_int["value"].to_numpy()
            extra_cols = [c for c in u_int.columns if c not in {"user_id", start_col, end_col, "value"}]
            extras = {c: u_int[c].to_numpy() for c in extra_cols} if extra_cols else {}

            for i in range(len(u_int)):
                a_s = i_start[i]
                a_e = i_end[i]
                if pd.isna(a_s) or pd.isna(a_e) or a_e <= a_s:
                    continue
                k = int(np.searchsorted(w_start, a_e, side="left"))
                if k <= 0:
                    continue
                idxs = np.flatnonzero(w_end[:k] > a_s)
                if idxs.size == 0:
                    continue
                for j in idxs:
                    start = a_s if a_s > w_start[j] else w_start[j]
                    end = a_e if a_e < w_end[j] else w_end[j]
                    if start < end:
                        dur = (end - start) / np.timedelta64(1, "s")
                        rec = {
                            "user_id": uid,
                            "window_id": w_id[j],
                            "window_start": w_start[j],
                            "window_end": w_end[j],
                            "window_seconds": float(w_sec[j]),
                            "duration": float(dur),
                            "value": i_val[i],
                            start_col: a_s,
                            end_col: a_e,
                        }
                        for c in extra_cols:
                            rec[c] = extras[c][i]
                        out_records.append(rec)

        if not out_records:
            return pd.DataFrame(columns=["user_id","window_id","window_start","window_end","window_seconds","duration","value",start_col,end_col])
        return pd.DataFrame.from_records(out_records)

    def _assign_points_to_windows(self, df_pts: pd.DataFrame, windows: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        """Assign point rows to windows; replicate rows if windows overlap."""
        if df_pts.empty or windows.empty:
            return pd.DataFrame(columns=list(df_pts.columns) + ["window_id", "window_start", "window_end", "window_seconds"])

        df_pts = df_pts.copy()
        windows = windows.copy()

        pts_uid, pts_bad = self._normalize_user_id_numeric(df_pts["user_id"])
        win_uid, win_bad = self._normalize_user_id_numeric(windows["user_id"])
        if pts_bad.any():
            df_pts = df_pts.loc[~pts_bad].copy()
            pts_uid = pts_uid.loc[df_pts.index]
        if win_bad.any():
            windows = windows.loc[~win_bad].copy()
            win_uid = win_uid.loc[windows.index]
        if df_pts.empty or windows.empty:
            return pd.DataFrame(columns=list(df_pts.columns) + ["window_id", "window_start", "window_end", "window_seconds"])

        df_pts["user_id"] = pts_uid.astype("int64")
        windows["user_id"] = win_uid.astype("int64")
        if df_pts["user_id"].isna().any() or windows["user_id"].isna().any():
            raise ValueError("Failed to normalize user_id to numeric keys before point-window assignment")

        kind = str(windows["window_kind"].iloc[0]) if "window_kind" in windows.columns and len(windows) else "grid"
        pts = pd.to_datetime(df_pts[ts_col], utc=True).dt.tz_convert("UTC")

        win_sec = pd.to_numeric(windows.get("window_seconds"), errors="coerce")
        has_non_nominal_grid_bins = (
            bool(win_sec.notna().any())
            and bool(
                (~np.isclose(
                    win_sec.to_numpy(dtype="float64"),
                    float(self._bin_seconds_nominal),
                    rtol=0.0,
                    atol=1e-6,
                )).any()
            )
        )
        use_containment_grid = (
            kind == "grid"
            and str(self.cfg.tz_mode_l0) == "local"
            and has_non_nominal_grid_bins
        )

        if use_containment_grid:
            frames: List[pd.DataFrame] = []
            for uid, sub in df_pts.assign(_pt=pts).groupby("user_id"):
                wnu = windows.loc[
                    windows["user_id"] == uid,
                    ["window_id", "window_start", "window_end", "window_seconds"],
                ].sort_values(["window_start", "window_end"])
                if wnu.empty:
                    continue

                p = sub.sort_values("_pt").copy()
                p = p[p["_pt"].notna()].copy()
                if p.empty:
                    continue

                p_ns = p["_pt"].astype("int64").to_numpy()
                w_start_ns = wnu["window_start"].astype("int64").to_numpy()
                w_end_ns = wnu["window_end"].astype("int64").to_numpy()
                w_sec = wnu["window_seconds"].to_numpy()
                w_id = wnu["window_id"].to_numpy()

                idx = np.searchsorted(w_start_ns, p_ns, side="right") - 1
                can_match = idx >= 0
                if not np.any(can_match):
                    continue

                point_pos = np.flatnonzero(can_match)
                cand_win = idx[can_match]
                contains = p_ns[can_match] < w_end_ns[cand_win]
                if not np.any(contains):
                    continue

                matched_point_pos = point_pos[contains]
                matched_win_pos = cand_win[contains]
                part = p.iloc[matched_point_pos].copy()
                part["window_id"] = w_id[matched_win_pos]
                part["window_start"] = wnu["window_start"].to_numpy()[matched_win_pos]
                part["window_end"] = wnu["window_end"].to_numpy()[matched_win_pos]
                part["window_seconds"] = w_sec[matched_win_pos].astype("float64")
                frames.append(part)

            if not frames:
                return pd.DataFrame(columns=list(df_pts.columns) + ["window_id", "window_start", "window_end", "window_seconds"])
            return pd.concat(frames, ignore_index=True).drop(columns=["_pt"], errors="ignore")

        if kind == "grid":
            tmp = df_pts.copy()
            tmp["_pt"] = pts
            tmp["window_id"] = self._bin_index_from_start(tmp["_pt"]).astype("int64")
            tmp = tmp.merge(
                windows[["user_id", "window_id", "window_start", "window_end", "window_seconds"]],
                on=["user_id", "window_id"],
                how="inner",
            )
            return tmp.drop(columns=["_pt"], errors="ignore")

        frames: List[pd.DataFrame] = []
        for uid, sub in df_pts.assign(_pt=pts).groupby("user_id"):
            wnu = windows.loc[
                windows["user_id"] == uid,
                ["window_id", "window_start", "window_end", "window_seconds"],
            ].sort_values(["window_start", "window_end"])
            if wnu.empty:
                continue

            p = sub.sort_values("_pt").copy()
            p = p[p["_pt"].notna()].copy()
            if p.empty:
                continue

            w_start = wnu["window_start"].to_numpy()
            w_end = wnu["window_end"].to_numpy()
            w_sec = wnu["window_seconds"].to_numpy()
            w_id = wnu["window_id"].to_numpy()

            base_cols = [c for c in p.columns if c != "_pt"]
            base_arrays = {c: p[c].to_numpy() for c in base_cols}
            p_ts = p["_pt"].to_numpy()

            active: List[int] = []
            j = 0
            recs: List[Dict[str, Any]] = []

            for i, t in enumerate(p_ts):
                while j < len(wnu) and w_start[j] <= t:
                    active.append(j)
                    j += 1
                if active:
                    active = [k for k in active if w_end[k] > t]
                if not active:
                    continue
                for k in active:
                    rec = {c: base_arrays[c][i] for c in base_cols}
                    rec.update(
                        {
                            "window_id": w_id[k],
                            "window_start": w_start[k],
                            "window_end": w_end[k],
                            "window_seconds": float(w_sec[k]),
                        }
                    )
                    recs.append(rec)

            if recs:
                frames.append(pd.DataFrame.from_records(recs))

        if not frames:
            return pd.DataFrame(columns=list(df_pts.columns) + ["window_id", "window_start", "window_end", "window_seconds"])
        return pd.concat(frames, ignore_index=True)

    def assign_observations_to_windows(self, df: pd.DataFrame, windows: pd.DataFrame, types: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Route raw rows per type into window-aligned intermediate frames.
        - Interval types → expanded fragments per window.
        - Point types → assigned to windows by kind (grid or anchored).
        - Uses per-user parallelism for interval expansion (grid windows).
        """

        out: Dict[str, pd.DataFrame] = {}
        tnames = list(self.cfg.type_registry.keys()) if types is None else list(types)

        from concurrent.futures import ThreadPoolExecutor, as_completed
        cpu_count = os.cpu_count() or 1
        configured_cap = self.cfg.max_workers
        try:
            worker_cap = int(cpu_count if configured_cap is None else configured_cap)
        except (TypeError, ValueError):
            worker_cap = 1
        worker_cap = max(1, worker_cap)
        n_users = windows["user_id"].nunique() if not windows.empty else 0
        max_workers = 1 if n_users <= 1 else min(worker_cap, int(n_users))

        start_col = "utc_start"
        end_col = "utc_end"
        kind = str(windows["window_kind"].iloc[0]) if "window_kind" in windows.columns and len(windows) else "grid"
        if windows.empty:
            input_counts = df["type"].value_counts().to_dict() if "type" in df.columns else {}
            for tname in tnames:
                out[tname] = pd.DataFrame(columns=["user_id", "window_id", "window_start", "window_end", "window_seconds"])
            split_factors = {
                t: {
                    "input_rows": int(input_counts.get(t, 0)),
                    "child_rows": 0,
                    "expansion_factor": np.nan,
                }
                for t in tnames
            }
            self.report["input"]["split_expansion"] = split_factors
            return out

        df_work = df.copy()
        if "_src_bucket" not in df_work.columns:
            if "source_device_model" in df_work.columns:
                df_work["_src_bucket"] = df_work["source_device_model"].map(self._classify_source_device_buckets)

        type_mask = df_work["type"].isin(tnames)
        df_filtered = df_work.loc[type_mask].copy()
        input_counts = df_filtered.groupby("type", sort=False).size().to_dict() if not df_filtered.empty else {}
        df_by_type = {t: g.copy() for t, g in df_filtered.groupby("type", sort=False)}

        win_cols = ["user_id", "window_id", "window_start", "window_end", "window_seconds", "window_kind"]
        windows_by_user = {
            uid: g.loc[:, win_cols].copy()
            for uid, g in windows.groupby("user_id", sort=False)
        }
        users_with_windows = set(windows_by_user.keys())

        for tname in tnames:
            meta = self.cfg.type_registry[tname]
            time_behavior = meta["time_behavior"]
            dft = df_by_type.get(tname, pd.DataFrame())
            if dft.empty:
                out[tname] = pd.DataFrame(columns=["user_id", "window_id", "window_start", "window_end", "window_seconds"])
                continue

            if time_behavior in ("interval_total", "interval_level", "categorical_interval"):
                base_cols = ["user_id", start_col, end_col, "value"]
                if "_src_bucket" in dft.columns:
                    base_cols.append("_src_bucket")
                type_user_groups = {
                    uid: g.loc[:, base_cols].copy()
                    for uid, g in dft.groupby("user_id", sort=False)
                }
                users = [uid for uid in type_user_groups.keys() if uid in users_with_windows]

                def _process_user(uid: Any) -> pd.DataFrame:
                    return self._assign_intervals_to_windows(
                        type_user_groups[uid],
                        windows_by_user[uid],
                        start_col=start_col,
                        end_col=end_col,
                    )

                frames: List[pd.DataFrame] = []
                if max_workers > 1 and len(users) > 1 and kind == "grid":
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        for fut in as_completed({ex.submit(_process_user, u): u for u in users}):
                            frames.append(fut.result())
                else:
                    for u in users:
                        frames.append(_process_user(u))

                exp = (
                    pd.concat(frames, ignore_index=True)
                    if frames
                    else pd.DataFrame(
                        columns=["user_id", "window_id", "window_start", "window_end", "window_seconds", "duration", "value", start_col, end_col]
                    )
                )
                if exp.empty:
                    out[tname] = exp
                    continue

                if time_behavior == "interval_total":
                    interval_seconds = (exp[end_col] - exp[start_col]).dt.total_seconds()
                    with np.errstate(invalid="ignore", divide="ignore"):
                        frac = exp["duration"] / interval_seconds.replace(0, np.nan)
                    exp["value"] = exp["value"] * frac
                    keep_cols = ["user_id", "window_id", "window_start", "window_end", "window_seconds", "value", "duration", start_col, end_col]
                    if "_src_bucket" in exp.columns:
                        keep_cols.append("_src_bucket")
                    out[tname] = exp[keep_cols]
                elif time_behavior == "interval_level":
                    exp["value_x_seconds"] = exp["value"] * exp["duration"]
                    keep_cols = [
                        "user_id",
                        "window_id",
                        "window_start",
                        "window_end",
                        "window_seconds",
                        "value_x_seconds",
                        "duration",
                        "value",
                        start_col,
                        end_col,
                    ]
                    if "_src_bucket" in exp.columns:
                        keep_cols.append("_src_bucket")
                    out[tname] = exp[keep_cols]
                else:
                    exp["label"] = exp["value"].astype(str)
                    keep_cols = ["user_id", "window_id", "window_start", "window_end", "window_seconds", "label", "duration", start_col, end_col]
                    if "_src_bucket" in exp.columns:
                        keep_cols.append("_src_bucket")
                    out[tname] = exp[keep_cols]

            elif time_behavior == "point":
                dft = dft.copy()
                dft["_src_row_id"] = np.arange(len(dft))
                dft["point_ts_utc"] = pd.to_datetime(dft.get("utc_start"), utc=True).dt.tz_convert("UTC")
                dft = self._assign_points_to_windows(dft, windows, ts_col="point_ts_utc")

                if not dft.empty and tname == self.cfg.hr_type:
                    cov = self.cfg.hr_coverage_window_seconds
                    pts = dft["point_ts_utc"]
                    coverage_end = pts + pd.to_timedelta(cov, unit="s")
                    cov_seconds = self._compute_overlap_seconds(pts, coverage_end, dft["window_start"], dft["window_end"])
                    dft["coverage_seconds"] = cov_seconds.clip(upper=dft["window_seconds"])

                keep_cols = ["user_id", "window_id", "window_start", "window_end", "window_seconds", "value"]
                if tname == self.cfg.hr_type:
                    keep_cols.append("coverage_seconds")
                if "_src_bucket" in dft.columns:
                    keep_cols.append("_src_bucket")
                keep_cols.append("_src_row_id")
                out[tname] = dft[keep_cols]
            else:
                raise NotImplementedError(f"Unknown time_behavior for type {tname}: {time_behavior}")

        split_factors = {}
        for t in tnames:
            input_rows = int(input_counts.get(t, 0))
            out_df = out.get(t, pd.DataFrame())
            child_rows = int(len(out_df))
            factors: Dict[str, Any] = {
                "input_rows": input_rows,
                "child_rows": child_rows,
                "expansion_factor": (child_rows / input_rows) if input_rows else np.nan,
            }
            meta = self.cfg.type_registry.get(t, {})
            if meta.get("time_behavior") == "point" and "_src_row_id" in out_df.columns:
                assigned_unique = int(out_df["_src_row_id"].nunique()) if not out_df.empty else 0
                factors["unassigned_rows"] = max(0, input_rows - assigned_unique)
                out[t] = out_df.drop(columns=["_src_row_id"], errors="ignore")
            split_factors[t] = factors
        self.report["input"]["split_expansion"] = split_factors
        self.logger.info(
            "Window assignment: %s",
            ", ".join(f"{t}: {f['child_rows']}/{f['input_rows']}" for t, f in split_factors.items() if f["input_rows"] > 0),
        )
        return out

    # ------------------------------------------------------------------------
    # Per-Window Aggregation
    # ------------------------------------------------------------------------
    def register_operation(self, key: str, fn: Callable[[pd.core.groupby.generic.DataFrameGroupBy, Dict[str, Any]], pd.Series]) -> None:
        """Register a new operation callable(groupby, inputs) -> Series.
        See healthkit_aggregator_specs.py for operations available by default"""
        self.op_registry[key] = fn

    def aggregate_type_within_windows(self, tname: str, intermediate: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce per-type intermediate fragments into per-window aggregates.
        Returns one row per (user_id, window_id, window_start, window_end, window_seconds).
        """
        meta = self.cfg.type_registry[tname]; time_behavior = meta["time_behavior"]; requested = set(meta.get("aggregates", ()))
        cols_key = ["user_id","window_id","window_start","window_end","window_seconds"]
        if intermediate.empty: return pd.DataFrame(columns=cols_key)
        g = intermediate.groupby(cols_key, as_index=False, sort=False)
        g_idx = intermediate.groupby(cols_key, sort=False)

        if time_behavior == "interval_total":
            out = intermediate[cols_key].drop_duplicates().set_index(cols_key)
            need_sum = ("sum" in requested) or ("rate_per_min" in requested)
            need_dur = ("duration" in requested) or ("rate_per_min" in requested)
            if need_sum:
                out[f"{tname}__sum"] = g_idx["value"].sum().reindex(out.index)
            if need_dur:
                out[f"{tname}__duration"] = g_idx["duration"].sum().reindex(out.index)
            if "rate_per_min" in requested:
                out[f"{tname}__rate_per_min"] = out[f"{tname}__sum"] / (out[f"{tname}__duration"] / 60.0).replace({0: np.nan})
            if "sum" not in requested and f"{tname}__sum" in out.columns:
                out = out.drop(columns=[f"{tname}__sum"])
            if "duration" not in requested and f"{tname}__duration" in out.columns:
                out = out.drop(columns=[f"{tname}__duration"])
            if "sum_of_squares" in requested:
                v2 = intermediate.assign(v2=intermediate["value"] ** 2)
                out[f"{tname}__sum_of_squares"] = v2.groupby(cols_key, sort=False)["v2"].sum().reindex(out.index)
            return out.reset_index()

        if time_behavior == "interval_level":
            base = intermediate[cols_key].drop_duplicates().set_index(cols_key)
            pieces: List[pd.DataFrame] = [base]
            if "time_weighted_mean" in requested or "duration" in requested:
                twm = pd.concat(
                    [
                        g_idx["value_x_seconds"].sum().rename("_vxs"),
                        g_idx["duration"].sum().rename("_dur"),
                    ],
                    axis=1,
                )
                if "time_weighted_mean" in requested:
                    twm[f"{tname}__time_weighted_mean"] = twm["_vxs"] / twm["_dur"].replace({0: np.nan})
                    pieces.append(twm[[f"{tname}__time_weighted_mean"]])
                if "duration" in requested:
                    pieces.append(twm[["_dur"]].rename(columns={"_dur": f"{tname}__duration"}))
            if "max" in requested:
                pieces.append(g_idx["value"].max().rename(f"{tname}__max").to_frame())

            special = meta.get("special_aggregates", {})
            level_special = {
                agg_name: agg_spec
                for agg_name, agg_spec in special.items()
                if agg_name in requested and isinstance(agg_spec, dict)
            }
            if level_special:
                def _weighted_quantile(x, w, q):
                    if len(x) == 0: return np.nan
                    order = np.argsort(x); xs, ws = np.array(x)[order], np.array(w)[order]
                    cw = np.cumsum(ws); 
                    return np.nan if cw[-1] == 0 else np.interp(q * cw[-1], cw, xs)
                def _agg_block(block: pd.DataFrame) -> pd.Series:
                    vals = block["value"].to_numpy(); w = block["duration"].to_numpy(); out_s = {}
                    for agg_name, agg_spec in level_special.items():
                        op = str(agg_spec.get("op", "")).strip().lower()
                        if op == "weighted_quantile":
                            q = float(agg_spec.get("q", 0.95))
                            out_s[f"{tname}__{agg_name}"] = _weighted_quantile(vals, w, q)
                        elif op == "time_above":
                            thr = float(agg_spec.get("threshold", 85.0))
                            out_s[f"{tname}__{agg_name}"] = float(w[(vals > thr)].sum()) if len(vals) else 0.0
                    return pd.Series(out_s)
                tmp = intermediate.groupby(cols_key, sort=False).apply(_agg_block)
                if len(tmp.columns) > 0:
                    pieces.append(tmp)

            if "weighted_sum" in requested:
                pieces.append(g_idx["value_x_seconds"].sum().rename(f"{tname}__weighted_sum").to_frame())
            if "weighted_sum_of_squares" in requested:
                tmp = intermediate.assign(v2=intermediate["value"] ** 2 * intermediate["duration"])
                pieces.append(tmp.groupby(cols_key, sort=False)["v2"].sum().rename(f"{tname}__weighted_sum_of_squares").to_frame())

            if "leq" in requested or "energy_seconds" in requested:
                tmp = intermediate.assign(_energy_seconds=(10.0 ** (intermediate["value"] / 10.0)) * intermediate["duration"])
                es = tmp.groupby(cols_key, sort=False)["_energy_seconds"].sum().rename(f"{tname}__energy_seconds").to_frame()
                pieces.append(es)
                if "leq" in requested:
                    leq = pd.concat([es, g_idx["duration"].sum().rename("_dur")], axis=1)
                    leq[f"{tname}__leq"] = 10.0 * np.log10(leq[f"{tname}__energy_seconds"] / leq["_dur"].replace({0: np.nan}))
                    pieces.append(leq[[f"{tname}__leq"]])

            if "median" in requested:
                pieces.append(g_idx["value"].median().rename(f"{tname}__median").to_frame())

            if "median_duration" in requested:
                pieces.append(g_idx["duration"].median().rename(f"{tname}__median_duration").to_frame())

            if "count" in requested:
                pieces.append(g_idx["value"].count().rename(f"{tname}__count").to_frame())

            out = pd.concat(pieces, axis=1)
            out = out.loc[:, ~out.columns.duplicated()]
            return out.reset_index()

        if time_behavior == "point":
            aggregates = requested
            base = intermediate[cols_key].drop_duplicates().set_index(cols_key)
            agg_map = {}
            if "mean" in aggregates:   agg_map[f"{tname}__mean"] = ("value","mean")
            if "median" in aggregates: agg_map[f"{tname}__median"] = ("value","median")
            if "std" in aggregates:    agg_map[f"{tname}__std"] = ("value", (lambda s: np.nan if len(s) <= 1 else s.std(ddof=1)))
            if "max" in aggregates:    agg_map[f"{tname}__max"] = ("value","max")
            if "min" in aggregates:    agg_map[f"{tname}__min"] = ("value","min")
            if "count" in aggregates:  agg_map[f"{tname}__count"] = ("value","count")
            if "sum" in aggregates:    agg_map[f"{tname}__sum"] = ("value","sum")
            if "sum_of_squares" in aggregates: agg_map[f"{tname}__sum_of_squares"] = ("value", lambda s: (s ** 2).sum())

            agg_df = base
            if agg_map:
                agg_df = g_idx.agg(**agg_map)

            special = meta.get("special_aggregates", {})
            for agg_name, agg_spec in special.items():
                if agg_name not in aggregates or not isinstance(agg_spec, dict):
                    continue
                op = str(agg_spec.get("op", "")).strip().lower()
                out_col = f"{tname}__{agg_name}"
                if op == "quantile":
                    q = float(agg_spec.get("q", 0.5))
                    tmp = g_idx["value"].quantile(q).rename(out_col)
                elif op == "pct_below":
                    thr = float(agg_spec.get("threshold", 0.0))
                    tmp = g_idx["value"].apply(lambda s: (s < thr).mean() * 100.0 if len(s) else np.nan).rename(out_col)
                else:
                    continue
                agg_df[out_col] = tmp.reindex(agg_df.index)

            num_cols = [c for c in agg_df.columns if c.startswith(f"{tname}__") and not c.endswith("__count")]
            for c in num_cols: agg_df[c] = pd.to_numeric(agg_df[c], errors="coerce").astype("float64")
            return agg_df.reset_index()

        if time_behavior == "categorical_interval":
            requested = set(requested)
            if tname == "SleepAnalysis":
                label_map = {"in_bed": "InBed", "asleep": "Asleep", "awake": "Awake", "sleep_unknown": "CategoryValueUnknown"}
            elif tname == "EnvironmentalAudioExposureEvent":
                label_map = {"momentary_limit": "MOMENTARY_LIMIT"}
            else:
                return intermediate[cols_key].drop_duplicates()

            start_col = "utc_start" if "utc_start" in intermediate.columns else ("local_start" if "local_start" in intermediate.columns else None)
            end_col   = "utc_end"   if "utc_end"   in intermediate.columns   else ("local_end"   if "local_end"   in intermediate.columns   else None)

            def _agg_block(block: pd.DataFrame) -> pd.Series:
                out = {}
                for suf, raw in label_map.items():
                    sel = block["label"].astype(str) == raw; durs = block.loc[sel,"duration"].astype(float)
                    if f"sum_label_{suf}" in requested:  out[f"{tname}__sum_label_{suf}"] = float(durs.sum()) if len(durs) else np.nan
                    if f"mean_label_{suf}" in requested: out[f"{tname}__mean_label_{suf}"] = float(durs.mean()) if len(durs) else np.nan
                    if f"count_label_{suf}" in requested: out[f"{tname}__count_label_{suf}"] = float(len(durs)) if len(durs) else np.nan
                if start_col and end_col:
                    m_start = (block[start_col] <= block["window_start"]) & (block[end_col] > block["window_start"])
                    m_end   = (block[start_col] < block["window_end"]) & (block[end_col] >= block["window_end"])
                else:
                    m_start = pd.Series(False, index=block.index); m_end = pd.Series(False, index=block.index)

                start_label = block.loc[m_start,"label"].astype(str); end_label = block.loc[m_end,"label"].astype(str)
                start_val = start_label.iloc[0] if len(start_label) else None
                end_val = end_label.iloc[0] if len(end_label) else None
                for suf, raw in label_map.items():
                    if f"start_label_{suf}" in requested:
                        out[f"{tname}__start_label_{suf}"] = 1.0 if (start_val == raw) else (0.0 if start_val is not None else np.nan)
                    if f"end_label_{suf}" in requested:
                        out[f"{tname}__end_label_{suf}"] = 1.0 if (end_val == raw) else (0.0 if end_val is not None else np.nan)
                return pd.Series(out)

            return g.apply(_agg_block)

        raise NotImplementedError(f"Unknown time_behavior: {time_behavior}")

    def assemble_feature_matrix(self, windows: pd.DataFrame, per_type_agg: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all per-type aggregates onto the window grid (outer-left joins)."""
        keys = ["user_id","window_id","window_start","window_end","window_seconds"]
        wide = windows[keys + (["bin_index"] if "bin_index" in windows.columns else [])].copy()
        wide_idx = wide.set_index(keys)
        non_empty_counts = {}
        for tname, df_t in per_type_agg.items():
            if df_t is None or df_t.empty:
                non_empty_counts[tname] = 0
                continue
            wide_idx = wide_idx.join(df_t.set_index(keys), how="left")
            feat_cols = [c for c in df_t.columns if c not in keys]
            non_empty_counts[tname] = int(df_t.dropna(how="all", subset=feat_cols).shape[0])

        self.report["bins"]["non_empty_bin_counts_by_type"] = non_empty_counts
        return wide_idx.reset_index()

    # ------------------------------------------------------------------------
    # Compute Wear Metrics
    # ------------------------------------------------------------------------
    def derive_wear_from_hr_coverage(self, intermediate_by_type: Dict[str, pd.DataFrame], windows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Derive Window-level wear (seconds/fraction/watch_on), day-level validity, participant-level overall fraction from HeartRate coverage."""
        hr_meta = self.cfg.type_registry.get(self.cfg.hr_type, {})
        if not hr_meta or hr_meta.get("time_behavior") != "point": raise ValueError("hr_type must be a registered POINT type")

        hr_df = intermediate_by_type.get(self.cfg.hr_type, pd.DataFrame())
        keys = ["user_id","window_id","window_start","window_end","window_seconds"]

        if hr_df.empty or "coverage_seconds" not in hr_df.columns:
            empty = windows[keys].copy()
            empty["wear_seconds"] = 0.0
            empty["wear_fraction"] = 0.0
            if self.cfg.bin_frac_min is not None:
                empty["watch_on"] = pd.Series(False, index=empty.index, dtype="boolean")
            else:
                empty["watch_on"] = pd.Series(pd.NA, index=empty.index, dtype="boolean")
            day = self._compute_daily_wear_metrics(empty, windows)
            part = self._compute_participant_wear_metrics(empty)
            return empty, day, part


        g = hr_df.groupby(keys, as_index=False)
        win_wear = g["coverage_seconds"].sum().rename(columns={"coverage_seconds":"wear_seconds"})
        win_wear["wear_seconds"]   = win_wear[["wear_seconds","window_seconds"]].min(axis=1)
        win_wear["wear_fraction"]  = win_wear["wear_seconds"] / win_wear["window_seconds"].replace({0: np.nan})
        win_wear["watch_on"] = (win_wear["wear_fraction"] >= self.cfg.bin_frac_min) if self.cfg.bin_frac_min is not None else np.nan

        win_wear = windows.merge(win_wear, on=keys, how="left")
        win_wear["wear_seconds"]  = win_wear["wear_seconds"].fillna(0.0)
        win_wear["wear_fraction"] = win_wear["wear_fraction"].fillna(0.0)
        if self.cfg.bin_frac_min is not None: 
            win_wear["watch_on"] = (win_wear["wear_fraction"] >= self.cfg.bin_frac_min).astype("boolean")
            win_wear["watch_on"] = win_wear["watch_on"].fillna(False)
        else:
            win_wear["watch_on"] = pd.Series(pd.NA, index=win_wear.index, dtype="boolean")

        day_wear = self._compute_daily_wear_metrics(win_wear, windows)
        participant_wear = self._compute_participant_wear_metrics(win_wear)
        self.report["wear"]["bin_watch_on_rate"] = float(win_wear["watch_on"].mean()) if len(win_wear) else np.nan
        self.report["wear"]["day_valid_rate"] = float(day_wear["day_valid"].mean()) if len(day_wear) else np.nan
        self.report["wear"]["participant_overall_frac_mean"] = float(participant_wear["overall_frac"].mean()) if len(participant_wear) else np.nan
        return win_wear, day_wear, participant_wear

    def _compute_daily_wear_metrics(self, win_wear: pd.DataFrame, windows_with_local: pd.DataFrame) -> pd.DataFrame:
        """Collapse window wear to daily wear metrics per user using local lived day lengths."""

        # -----------------------------
        # Helpers
        # -----------------------------
        def _to_utc_dt(s: pd.Series) -> pd.Series:
            """Coerce to tz-aware UTC datetime64[ns, UTC]."""
            return pd.to_datetime(s, utc=True, errors="coerce")

        def _safe_localize_midnight(days_naive: pd.DatetimeIndex, tzinfo: ZoneInfo, amb: dict) -> pd.DatetimeIndex:
            """
            Localize naive midnights to tz with fallback behaviors to avoid crashes
            if ambiguous='infer' fails or tz has weird transitions at midnight.
            """
            try:
                return days_naive.tz_localize(
                    tzinfo,
                    ambiguous=amb["ambiguous"],
                    nonexistent=amb["nonexistent"],
                )
            except Exception:
                # Fallback 1: mark ambiguous as NaT, keep nonexistent policy
                try:
                    return days_naive.tz_localize(
                        tzinfo,
                        ambiguous="NaT",
                        nonexistent=amb["nonexistent"],
                    )
                except Exception:
                    # Fallback 2: last resort: ambiguous=False and shift_forward for nonexistent
                    return days_naive.tz_localize(
                        tzinfo,
                        ambiguous=False,
                        nonexistent="shift_forward",
                    )

        # -----------------------------
        # Case A: anchored windows (use anchor_local_date)
        # -----------------------------
        if (
            windows_with_local is not None
            and "anchor_local_date" in windows_with_local.columns
            and (windows_with_local.get("window_kind", "grid") == "anchored").any()
        ):
            if "anchor_local_date" in win_wear.columns:
                base = win_wear.copy()
            else:
                base = win_wear.merge(
                    windows_with_local[["user_id", "window_id", "anchor_local_date"]],
                    on=["user_id", "window_id"],
                    how="left",
                )

            # Ensure expected types
            if "window_start" in base.columns:
                base["window_start"] = _to_utc_dt(base["window_start"])
            if "window_end" in base.columns:
                base["window_end"] = _to_utc_dt(base["window_end"])

            base["_day"] = pd.to_datetime(base["anchor_local_date"], errors="coerce")
            g = base.groupby(["user_id", "_day"], as_index=False)

            day_seconds = g["wear_seconds"].sum().rename(columns={"wear_seconds": "wear_seconds"})
            denom = g["window_seconds"].sum().rename(columns={"window_seconds": "day_seconds_total"})

            out = day_seconds.merge(denom, on=["user_id", "_day"], how="left")
            out["pct_day_worn"] = out["wear_seconds"] / out["day_seconds_total"].replace({0: np.nan})
            out["day_valid"] = (out["pct_day_worn"] >= self.cfg.day_frac_min) if self.cfg.day_frac_min is not None else np.nan
            out = out.rename(columns={"_day": "anchor_local_date"})
            return out

        # -----------------------------
        # Case B: grid windows with local labeling (preferred)
        # -----------------------------
        if windows_with_local is not None and "local_date" in windows_with_local.columns:
            # Ensure base has local labeling + window metadata without creating _x/_y duplicates
            needed = {"local_date", "tz_name", "window_start", "window_end", "window_seconds"}
            if needed.issubset(set(win_wear.columns)):
                base = win_wear.copy()
            else:
                base = win_wear.merge(
                    windows_with_local[
                        ["user_id", "window_id", "local_date", "tz_name", "window_start", "window_end", "window_seconds"]
                    ],
                    on=["user_id", "window_id"],
                    how="left",
                    suffixes=("", "_win"),
                ).copy()

                # Canonicalize columns if merge created suffixed variants
                for col in ["local_date", "tz_name", "window_start", "window_end", "window_seconds"]:
                    col_win = f"{col}_win"
                    if col not in base.columns and col_win in base.columns:
                        base[col] = base[col_win]
                    elif col in base.columns and col_win in base.columns:
                        base[col] = base[col].where(base[col].notna(), base[col_win])

            # ---- Coerce datetime columns BEFORE any overlap math ----
            base["window_start"] = _to_utc_dt(base["window_start"])
            base["window_end"] = _to_utc_dt(base["window_end"])

            # local_date in your codebase is typically python date; normalize to midnight datetime
            base["_day"] = pd.to_datetime(base["local_date"], errors="coerce").dt.normalize()

            # tz labels
            if "tz_name" in base.columns:
                base["tz_name"] = base["tz_name"].astype("string")

            # --- (1) Pick dominant timezone per (user_id, _day) by total window_seconds
            # Include all user-days even if tz_name missing
            all_days = base[["user_id", "_day"]].drop_duplicates()

            tz_weight = (
                base.dropna(subset=["tz_name"])
                    .groupby(["user_id", "_day", "tz_name"], as_index=False)["window_seconds"]
                    .sum()
                    .sort_values(["user_id", "_day", "window_seconds"], ascending=[True, True, False])
            )
            dominant = (
                tz_weight.drop_duplicates(["user_id", "_day"])
                        .rename(columns={"tz_name": "dom_tz"})[["user_id", "_day", "dom_tz"]]
            )

            days = all_days.merge(dominant, on=["user_id", "_day"], how="left")
            days = days.rename(columns={"dom_tz": "tz_name"}).copy()

            # Pre-allocate with correct tz-aware dtype (avoid object columns)
            days["start_utc"] = pd.Series(pd.NaT, index=days.index, dtype="datetime64[ns, UTC]")
            days["end_utc"] = pd.Series(pd.NaT, index=days.index, dtype="datetime64[ns, UTC]")
            days["day_seconds_expected"] = np.nan

            amb = hk_time.get_tz_localize_kwargs(self.cfg.tz_policy)

            # --- (2) Build civil-day bounds in UTC for each tz
            for tz in days["tz_name"].dropna().astype(str).unique():
                idx = days["tz_name"].astype(str) == tz
                tzinfo = ZoneInfo(tz)

                # Build naive midnights as DatetimeIndex (not Series!)
                dvals = pd.to_datetime(days.loc[idx, "_day"], errors="coerce")
                start_naive = pd.DatetimeIndex(dvals).normalize()

                start_local = _safe_localize_midnight(start_naive, tzinfo, amb)
                end_local = start_local + pd.Timedelta(days=1)

                start_utc = start_local.tz_convert("UTC")
                end_utc = end_local.tz_convert("UTC")

                # Assign as Series aligned to days' index -> preserves datetime64[ns, UTC] dtype
                days.loc[idx, "start_utc"] = pd.Series(start_utc, index=days.index[idx], dtype="datetime64[ns, UTC]")
                days.loc[idx, "end_utc"] = pd.Series(end_utc, index=days.index[idx], dtype="datetime64[ns, UTC]")
                days.loc[idx, "day_seconds_expected"] = (end_utc - start_utc) / np.timedelta64(1, "s")

            # Fallback for missing tz_name: treat as UTC civil day
            na_idx = days["tz_name"].isna()
            if na_idx.any():
                dvals = pd.to_datetime(days.loc[na_idx, "_day"], errors="coerce")
                start_utc = pd.DatetimeIndex(dvals).tz_localize("UTC")
                end_utc = start_utc + pd.Timedelta(days=1)

                days.loc[na_idx, "start_utc"] = pd.Series(start_utc, index=days.index[na_idx], dtype="datetime64[ns, UTC]")
                days.loc[na_idx, "end_utc"] = pd.Series(end_utc, index=days.index[na_idx], dtype="datetime64[ns, UTC]")
                days.loc[na_idx, "day_seconds_expected"] = 86400.0

            # Final coercion safety net (if anything slipped to object)
            days["start_utc"] = _to_utc_dt(days["start_utc"])
            days["end_utc"] = _to_utc_dt(days["end_utc"])

            # --- (3) Numerator: wear within the selected civil-day bounds only
            tmp = base.merge(
                days[["user_id", "_day", "start_utc", "end_utc", "day_seconds_expected"]],
                on=["user_id", "_day"],
                how="left",
            )

            # Coerce merged bounds again (merge can upcast to object if inputs were mixed elsewhere)
            tmp["start_utc"] = _to_utc_dt(tmp["start_utc"])
            tmp["end_utc"] = _to_utc_dt(tmp["end_utc"])
            tmp["window_start"] = _to_utc_dt(tmp["window_start"])
            tmp["window_end"] = _to_utc_dt(tmp["window_end"])

            overlap = self._compute_overlap_seconds(
                tmp["window_start"], tmp["window_end"],
                tmp["start_utc"], tmp["end_utc"]
            )

            # Assume wear uniformly spread across the window
            wear_rate = (tmp["wear_seconds"] / tmp["window_seconds"].replace({0: np.nan})).fillna(0.0)
            tmp["wear_seconds_in_day"] = overlap * wear_rate

            day_wear = (
                tmp.groupby(["user_id", "_day"], as_index=False)["wear_seconds_in_day"]
                .sum()
                .rename(columns={"wear_seconds_in_day": "wear_seconds"})
            )

            out = day_wear.merge(days[["user_id", "_day", "day_seconds_expected"]], on=["user_id", "_day"], how="left")
            out["pct_day_worn"] = out["wear_seconds"] / out["day_seconds_expected"].replace({0: np.nan})
            out["day_valid"] = (out["pct_day_worn"] >= self.cfg.day_frac_min) if self.cfg.day_frac_min is not None else np.nan
            out = out.rename(columns={"_day": "local_date"})
            return out

        # -----------------------------
        # Case C: fallback UTC-day wear
        # -----------------------------
        tmp = win_wear.copy()
        tmp["window_start"] = _to_utc_dt(tmp["window_start"])
        tmp["day"] = tmp["window_start"].dt.floor("D")

        g = tmp.groupby(["user_id", "day"], as_index=False)
        day_seconds = g["wear_seconds"].sum().rename(columns={"wear_seconds": "wear_seconds"})
        day_seconds["pct_day_worn"] = day_seconds["wear_seconds"] / (24 * 3600.0)
        day_seconds["day_valid"] = (day_seconds["pct_day_worn"] >= self.cfg.day_frac_min) if self.cfg.day_frac_min is not None else np.nan
        return day_seconds

    def _compute_participant_wear_metrics(self, win_wear: pd.DataFrame) -> pd.DataFrame:
        """Summarize wear at participant level (overall fraction, validity flag)."""
        g = win_wear.groupby(["user_id"], as_index=False)
        part = g.agg(wear_seconds=("wear_seconds","sum"), window_seconds=("window_seconds","sum"))
        part["overall_frac"] = part["wear_seconds"] / part["window_seconds"].replace({0: np.nan})
        part["participant_valid"] = (part["overall_frac"] >= self.cfg.overall_frac_min) if self.cfg.overall_frac_min is not None else np.nan
        return part

    def add_wear_columns(self, wide: pd.DataFrame, win_wear: pd.DataFrame) -> pd.DataFrame:
        """Attach wear columns (seconds/fraction/watch_on) to the wide frame."""
        keys = ["user_id","window_id","window_start","window_end"]
        cols = keys + ["wear_seconds","wear_fraction","watch_on"]
        return wide.merge(win_wear[cols], on=keys, how="left")

    def _infer_schedule_from_input_or_raise(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Infer timezone schedule from input rows.
        For local mode: fail fast with actionable guidance.
        For utc mode: behavior depends on cfg.utc_schedule_inference_policy.
        """
        policy = self.cfg.utc_schedule_inference_policy
        sch_rep = self.report.setdefault("schedule", {})
        sch_rep["inference_mode"] = "input_auto"
        sch_rep["inference_attempted"] = False
        sch_rep["inference_fallback_used"] = False
        sch_rep["inference_error"] = None

        if self.cfg.tz_mode_l0 == "utc" and policy == "skip":
            sch_rep["inference_mode"] = "skipped_utc_policy"
            return None

        if self.cfg.tz_mode_l0 == "local":
            if "timezone" not in df.columns:
                raise ValueError(
                    "tz_mode_l0='local' requires timezone signal when schedule is not provided. "
                    "Provide a timezone schedule or include a non-empty 'timezone' column with IANA names "
                    "(e.g., 'America/New_York')."
                )
            tz_series = df["timezone"].astype("string")
            has_tz_signal = tz_series.notna() & tz_series.str.strip().ne("")
            if not has_tz_signal.any():
                raise ValueError(
                    "tz_mode_l0='local' requires timezone signal when schedule is not provided. "
                    "Found no non-empty timezone values in input. "
                    "Provide a schedule or populate 'timezone' with IANA timezone names "
                    "(e.g., 'America/New_York')."
                )

        sch_rep["inference_attempted"] = True
        try:
            return hk_time.build_schedule_from_input(
                df,
                user_col="user_id",
                start_col="utc_start",
                end_col="utc_end",
                tz_col="timezone",
                default_tz="UTC",
                fill_gaps="ffill",
                point_policy=self.cfg.schedule_point_policy,
                point_epsilon=self.cfg.schedule_point_epsilon,
                invalid_tz_policy=self.cfg.schedule_invalid_tz_policy,
                report=self.report,
            )
        except Exception as exc:
            sch_rep["inference_error"] = str(exc)
            schedule_diag = self.report.get("schedule", {})
            diag_parts = []
            for key in (
                "source_rows_total",
                "source_rows_epsilon_promoted",
                "source_rows_dropped_invalid",
                "invalid_tz_rows",
            ):
                if key in schedule_diag:
                    diag_parts.append(f"{key}={schedule_diag[key]}")
            diag_txt = f" diagnostics: {', '.join(diag_parts)}." if diag_parts else ""

            if self.cfg.tz_mode_l0 == "utc" and policy == "infer_fallback":
                sch_rep["inference_fallback_used"] = True
                self.logger.warning(
                    "Schedule inference failed (%s); falling back to no schedule.%s",
                    exc, diag_txt,
                )
                return None

            raise ValueError(
                "Failed to infer timezone schedule from input. "
                "Expected columns: 'timezone', 'utc_start', 'utc_end'. "
                "Ensure timezone values are valid IANA names (e.g., 'America/New_York'). "
                f"Details: {exc}.{diag_txt}"
            ) from exc

    def _apply_output_timezone_policy(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Shape output timestamp columns according to cfg.output_timezone."""
        if wide is None or wide.empty:
            return wide

        mode = self.cfg.output_timezone
        out = wide.copy()

        local_cols = [
            "window_start_local", "window_end_local",
            "local_date", "anchor_local_date", "local_hour", "minute_of_day",
            "local_minute_index", "tz_name", "is_dst", "fold", "utc_offset_seconds",
            "local_weekday", "iso_week",
        ]

        if mode == "both":
            return out

        if mode == "UTC":
            drop_cols = [c for c in local_cols if c in out.columns]
            if drop_cols:
                out = out.drop(columns=drop_cols)
            return out

        if mode == "local":
            required_local = ["window_start_local", "window_end_local"]
            missing = [c for c in required_local if c not in out.columns]
            if missing:
                raise ValueError(
                    f"output_timezone='local' requires local projection columns, missing: {missing}. "
                    "Use output_timezone='both' or ensure local labels are available."
                )

            out["window_start_utc"] = out["window_start"]
            out["window_end_utc"] = out["window_end"]
            out["window_start"] = pd.to_datetime(out["window_start_local"], errors="coerce")
            out["window_end"] = pd.to_datetime(out["window_end_local"], errors="coerce")
            return out

        return out

    def _sanitize_output_schema(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Clean known merge artifacts while preserving data columns."""
        if wide is None or wide.empty:
            return wide

        out = wide.copy()

        # Defensive coalesce for legacy merge artifacts.
        if "bin_index_x" in out.columns or "bin_index_y" in out.columns:
            if "bin_index" not in out.columns:
                if "bin_index_x" in out.columns and "bin_index_y" in out.columns:
                    out["bin_index"] = out["bin_index_x"].where(out["bin_index_x"].notna(), out["bin_index_y"])
                elif "bin_index_x" in out.columns:
                    out["bin_index"] = out["bin_index_x"]
                else:
                    out["bin_index"] = out["bin_index_y"]
            drop_bin_artifacts = [c for c in ("bin_index_x", "bin_index_y") if c in out.columns]
            if drop_bin_artifacts:
                out = out.drop(columns=drop_bin_artifacts)

        # Drop truly empty index artifact only.
        if "index" in out.columns and out["index"].isna().all():
            out = out.drop(columns=["index"])

        return out
    
    # ------------------------------------------------------------------------
    # Compute Source/Type Presence
    # ------------------------------------------------------------------------
    def _classify_source_device_buckets(self, s: Any) -> str:
        """Heuristic device source bucketing into {'watch','phone','other'} for presence gates."""
        if pd.isna(s): return "other"
        s = str(s).lower()
        if "watch" in s: return "watch"
        if "phone" in s or "iphone" in s: return "phone"
        return "other"

    def build_window_presence_flags(self, intermediate_by_type: Dict[str, pd.DataFrame], windows: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Build boolean presence indicators by type and by source bucket per window."""
        keys = ["user_id", "window_id", "window_start", "window_end", "window_seconds"]
        if windows.empty:
            base = windows[keys].copy()
            return (
                {t: pd.DataFrame(columns=["user_id", "window_id", "present"]) for t in self.cfg.type_registry.keys()},
                base.assign(
                    presence_any=False,
                    presence_non_watch=False,
                    source_active__watch=False,
                    source_active__phone=False,
                    source_active__other=False,
                ),
            )

        presence_by_type: Dict[str, pd.DataFrame] = {}
        pieces: List[pd.DataFrame] = []

        for tname in self.cfg.type_registry.keys():
            inter = intermediate_by_type.get(tname, pd.DataFrame())
            if inter is None or inter.empty:
                presence_by_type[tname] = pd.DataFrame(columns=["user_id", "window_id", "present"])
                continue

            presence_by_type[tname] = inter[["user_id", "window_id"]].drop_duplicates().assign(present=True)

            src = (
                inter["_src_bucket"].map(self._classify_source_device_buckets)
                if "_src_bucket" in inter.columns
                else pd.Series("other", index=inter.index)
            )
            src_flags = inter[keys].copy()
            src_flags["_src_bucket"] = src.fillna("other")
            src_flags = (
                src_flags.groupby(keys + ["_src_bucket"])
                .size()
                .reset_index(name="size")
                .pivot_table(
                    index=keys,
                    columns="_src_bucket",
                    values="size",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .reset_index()
            )
            pieces.append(src_flags)

        if pieces:
            tmp = pd.concat(pieces, ignore_index=True, sort=False)
            for col in ["watch", "phone", "other"]:
                if col not in tmp.columns:
                    tmp[col] = 0.0
            tmp[["watch", "phone", "other"]] = tmp[["watch", "phone", "other"]].fillna(0.0).astype("float64")
            src_all = tmp.groupby(keys, as_index=False)[["watch", "phone", "other"]].sum()
            src_all = windows[keys].merge(src_all, on=keys, how="left")
        else:
            src_all = windows[keys].copy()
            src_all["watch"] = 0.0
            src_all["phone"] = 0.0
            src_all["other"] = 0.0

        for col in ["watch", "phone", "other"]:
            if col not in src_all.columns:
                src_all[col] = 0.0
        src_all[["watch", "phone", "other"]] = src_all[["watch", "phone", "other"]].fillna(0.0)

        src_all["source_active__watch"] = src_all["watch"] > 0
        src_all["source_active__phone"] = src_all["phone"] > 0
        src_all["source_active__other"] = src_all["other"] > 0
        src_all["presence_any"] = (
            src_all["source_active__watch"] | src_all["source_active__phone"] | src_all["source_active__other"]
        ).astype(bool)
        src_all["presence_non_watch"] = (src_all["source_active__phone"] | src_all["source_active__other"]).astype(bool)

        for c in ["source_active__watch", "source_active__phone", "source_active__other"]:
            src_all[c] = src_all[c].astype(bool)

        return presence_by_type, src_all[
            [
                "user_id",
                "window_id",
                "window_start",
                "window_end",
                "window_seconds",
                "presence_any",
                "presence_non_watch",
                "source_active__watch",
                "source_active__phone",
                "source_active__other",
            ]
        ]

    # ------------------------------------------------------------------------
    # Zero-Fill Policy
    # ------------------------------------------------------------------------
    def _normalize_zero_fill_gate(self, gate: Any) -> Tuple[str, Dict[str, Any]]:
        """Normalize zero-fill gate spec to (op, params)."""
        if isinstance(gate, tuple) and len(gate) == 2 and isinstance(gate[0], str) and isinstance(gate[1], dict):
            return gate[0], dict(gate[1])

        if isinstance(gate, str):
            s = gate.strip()
            if s == "" or s == "expr_false":
                return "expr_false", {}
            if s == "watch_on":
                return "watch_on", {}
            if s.startswith("watch_on:"):
                parts = s.split(":", 1)[1]
                kv = dict(kv.split("=") for kv in parts.split(",") if "=" in kv)
                return "watch_on", {"min_fraction": float(kv.get("min_fraction", 0.0))}
            if s.startswith("source_present:"):
                return "source_present", {"source": s.split(":", 1)[1].strip().lower()}
            if s == "any_source_present":
                return "any_source_present", {}
            if s == "any_non_watch_present":
                return "any_non_watch_present", {}
            if s.startswith("presence_types:"):
                types = [t.strip() for t in s.split(":", 1)[1].split(",") if t.strip()]
                return "presence_types", {"types": types}
            return "expr_false", {}

        return "expr_false", {}

    def _compute_zero_fill_gate_mask(self, ctx: pd.DataFrame, gate: Tuple[str, Dict[str, Any]], presence_maps: Dict[str, set]) -> pd.Series:
        """Return a boolean Series mask for gate (op, params) evaluated over ctx."""
        if ctx.empty:
            return pd.Series(False, index=ctx.index)

        op, params = gate if isinstance(gate, tuple) else ("expr_false", {})
        if op == "expr_false":
            return pd.Series(False, index=ctx.index)

        if op == "watch_on":
            thr = float(params.get("min_fraction", 0.0))
            return ctx["watch_on"].fillna(False) & (ctx["wear_fraction"].fillna(0.0) >= thr)

        if op == "source_present":
            src = str(params.get("source", "")).lower()
            col = f"source_active__{src}"
            return ctx[col].fillna(False) if col in ctx.columns else pd.Series(False, index=ctx.index)

        if op == "any_source_present":
            return (ctx["source_active__watch"].fillna(False)
                    | ctx["source_active__phone"].fillna(False)
                    | ctx["source_active__other"].fillna(False))

        if op == "any_non_watch_present":
            return ctx["presence_non_watch"].fillna(False)

        if op == "presence_types":
            union_pairs: set = set()
            for t in params.get("types", []):
                union_pairs |= set(presence_maps.get(t, set()))
            if not union_pairs:
                return pd.Series(False, index=ctx.index)
            mindex = pd.MultiIndex.from_arrays([ctx["user_id"], ctx["window_id"]])
            return pd.Series(mindex.isin(list(union_pairs)), index=ctx.index)

        return pd.Series(False, index=ctx.index)

    def apply_zero_fill_policy(self, wide: pd.DataFrame, presence_frame: pd.DataFrame, presence_by_type: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Zero-fill selected stats under configurable gate conditions.
        Gates provide policy; zero-fill eligibility is per-type/per-stat (registry).
        """
        if wide.empty: return wide
        wide = wide.copy()
        key_cols = ["user_id", "window_id"]
        wide_idx = wide.set_index(key_cols, drop=False)
        ctx = presence_frame.drop_duplicates(key_cols).set_index(key_cols, drop=False).reindex(wide_idx.index)
        ctx["wear_seconds"] = wide_idx["wear_seconds"].to_numpy()
        ctx["wear_fraction"] = wide_idx["wear_fraction"].to_numpy()
        ctx["watch_on"] = wide_idx["watch_on"].to_numpy()
        presence_maps: Dict[str, set] = {t: (set() if pres.empty else set(zip(pres["user_id"], pres["window_id"])))
                                        for t, pres in presence_by_type.items()}

        for c in ["presence_any", "presence_non_watch", "source_active__watch", "source_active__phone", "source_active__other"]:
            if c in ctx.columns:
                wide[c] = ctx[c].to_numpy()
            else:
                wide[c] = False

        fill_counts: Dict[str, int] = {}

        for tname, eligible_stats, gate, fill_value in self._zero_fill_plan:
            cols = [f"{tname}__{s}" for s in eligible_stats if f"{tname}__{s}" in wide.columns]
            if not cols:
                continue
            gate_mask_ctx = self._compute_zero_fill_gate_mask(ctx, gate, presence_maps)
            m_gate = pd.Series(np.asarray(gate_mask_ctx, dtype=bool), index=ctx.index).reindex(wide_idx.index).fillna(False).astype(bool).to_numpy()
            if not bool(np.any(m_gate)):
                for c in cols:
                    fill_counts[c] = 0
                continue

            na_before = wide.loc[m_gate, cols].isna().sum()
            wide.loc[m_gate, cols] = wide.loc[m_gate, cols].fillna(fill_value)
            for c, n in na_before.items():
                fill_counts[c] = int(n)

        self.report["fills"]["counts_by_column"] = fill_counts
        return wide

    # ------------------------------------------------------------------------
    # Build Aggregation Report
    # ------------------------------------------------------------------------
    def build_aggregation_report(self, windows: pd.DataFrame, win_wear: pd.DataFrame, day_wear: pd.DataFrame, participant_wear: pd.DataFrame) -> Dict[str, Any]:
        """Assemble a summary report of counts and QC validity stats."""
        rep = deepcopy(self.report)
        rep["bins"]["counts_per_user"] = windows.groupby("user_id").size().to_dict() if not windows.empty else {}
        rep["wear"]["bin_seconds_worn_total"] = float(win_wear["wear_seconds"].sum()) if len(win_wear) else 0.0
        rep["wear"]["day_valid_counts"] = day_wear["day_valid"].value_counts(dropna=False).to_dict() if len(day_wear) else {}
        rep["wear"]["participant_valid_counts"] = participant_wear["participant_valid"].value_counts(dropna=False).to_dict() if len(participant_wear) else {}
        return rep

    # ------------------------------------------------------------------------
    # QC Policy & Output Validation
    # ------------------------------------------------------------------------
    def _mask_selected_columns(self, df: pd.DataFrame, mask: pd.Series, selector: Dict[str, Any]) -> pd.DataFrame:
        """Apply NaN masking to selected columns (based on device class or types)."""
        if not mask.any(): return df
        df = df.copy()
        if selector.get("all", False):
            keep = {"user_id","window_id","window_start","window_end","window_seconds","wear_seconds","wear_fraction","watch_on"}
            cols = [c for c in df.columns if c not in keep]; df.loc[mask, cols] = np.nan; return df

        types_to_mask: List[str] = []
        if "device_class" in selector:
            classes = set(selector.get("device_class") or [])
            for t, meta in self.cfg.type_registry.items():
                if meta.get("device_class") in classes: types_to_mask.append(t)
        if "types" in selector: types_to_mask.extend(selector.get("types") or [])
        types_to_mask = list(dict.fromkeys(types_to_mask))

        if types_to_mask:
            cols = [c for t in types_to_mask for c in df.columns if c.startswith(f"{t}__")]
            if cols: df.loc[mask, cols] = np.nan
        return df

    def apply_quality_control_policy(self, wide: pd.DataFrame, day_wear: pd.DataFrame, participant_wear: pd.DataFrame) -> pd.DataFrame:
        """
        Apply QC actions ('drop' or 'mask') at participant/day/window levels.
        Controlled via cfg.enable_* flags and thresholds in cfg.
        """
        wide = wide.copy()
        dropped = {"bins": 0, "days": 0, "participants": 0}
        masked = {"bins": 0, "days": 0, "participants": 0}
        action = self.cfg.qc_action

        # Drop participants with watch-wear below threshold
        if self.cfg.enable_participant_qc and self.cfg.overall_frac_min is not None:
            bad_users = set(participant_wear.loc[~participant_wear["participant_valid"], "user_id"])
            if action == "drop":
                dropped["participants"] = int(wide["user_id"].isin(bad_users).sum()); wide = wide.loc[~wide["user_id"].isin(bad_users)]
            else:
                m = wide["user_id"].isin(bad_users); sel = self.cfg.qc_mask_selectors.get("participant", {"all": True})
                masked["participants"] = int(m.sum())
                wide = self._mask_selected_columns(wide, m, sel)

        # Drop days with watch-wear below threshold
        if self.cfg.enable_day_qc and self.cfg.day_frac_min is not None and not wide.empty:
            day_key = "anchor_local_date" if "anchor_local_date" in wide.columns and (wide.get("window_kind", "grid") == "anchored").any() else ("local_date" if "local_date" in wide.columns else None)
            if day_key:
                wide["_day"] = pd.to_datetime(wide[day_key])
                day_map = day_wear.rename(columns={day_key:"_day"}) if day_key in day_wear.columns else day_wear.rename(columns={"local_date":"_day"})
            else:
                wide["_day"] = wide["window_start"].dt.floor("D"); day_map = day_wear.rename(columns={"day":"_day"}) if "day" in day_wear.columns else day_wear
            wide = wide.merge(day_map[["user_id","_day","day_valid"]], on=["user_id","_day"], how="left")
            m_bad_day = ~wide["day_valid"].fillna(False)
            if action == "drop":
                dropped["days"] = int(m_bad_day.sum()); wide = wide.loc[~m_bad_day].drop(columns=["day_valid","_day"])
            else:
                masked["days"] = int(m_bad_day.sum())
                sel = self.cfg.qc_mask_selectors.get("day", {"all": True})
                wide = self._mask_selected_columns(wide, m_bad_day, sel).drop(columns=["day_valid","_day"])

        # Drop bins with watch-wear below threshold
        if self.cfg.enable_bin_qc and self.cfg.bin_frac_min is not None and not wide.empty:
            m_bad_bin = ~wide["watch_on"].fillna(False)
            if action == "drop": dropped["bins"] = int(m_bad_bin.sum()); wide = wide.loc[~m_bad_bin]
            else:
                masked["bins"] = int(m_bad_bin.sum())
                sel = self.cfg.qc_mask_selectors.get("bin", {"device_class": ["watch_exclusive","either"]})
                wide = self._mask_selected_columns(wide, m_bad_bin, sel)

        # Remove empty rows
        pruned = 0
        if self.cfg.prune_empty_rows and not wide.empty:
            keep = {"user_id","window_id","window_start","window_end","window_seconds","wear_seconds","wear_fraction","watch_on"}
            data_cols = [c for c in wide.columns if c not in keep]
            all_nan = wide[data_cols].isna().all(axis=1) if data_cols else pd.Series(False, index=wide.index)
            pruned = int(all_nan.sum())
            wide = wide.loc[~all_nan].copy()

        self.report["qc"]["dropped_counts"] = dropped
        self.report["qc"]["masked_counts"] = masked
        self.report["qc"]["pruned_empty_rows"] = pruned
        return wide

    def validate_feature_matrix(self, wide: pd.DataFrame) -> pd.DataFrame:
        """Validate final wide frame schema/invariants."""
        required = ["user_id","window_id","window_start","window_end","window_seconds","wear_seconds","wear_fraction","watch_on"]
        missing = [c for c in required if c not in wide.columns]
        if missing: raise ValueError(f"Output missing required columns: {missing}")
        if wide.duplicated(subset=["user_id","window_id"]).any(): raise ValueError("Duplicate (user_id, window_id) in output")
        if (wide["window_end"] <= wide["window_start"]).any(): raise ValueError("Found non-positive-width windows")
        return wide

    # ------------------------------------------------------------------------
    # Primary Pipeline API
    # ------------------------------------------------------------------------
    def run_aggregation(self, df: pd.DataFrame, schedule: Optional[pd.DataFrame] = None, assessments: Optional[pd.DataFrame] = None, duration_before: Optional[Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, pd.DataFrame]]:
        """
        Run the end-to-end aggregation pipeline (without QC masking/dropping).

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned long-form observations from HealthKitCleaner.
        schedule : pd.DataFrame, optional
            Per-user timezone schedule. Inferred from df if not provided.
        assessments : pd.DataFrame, optional
            Assessment timestamps for anchored windowing. If None, fixed-bin
            windows are built instead.
        duration_before : any, optional
            Look-back duration passed to assessment-anchored window builder.

        Returns
        -------
        wide : pd.DataFrame
            One row per (user_id, window_id) with aggregated feature columns.
        report : dict
            Counts, wear metrics, fill counts, timing, and config snapshot.
        aux : dict
            Supporting artifacts keyed by: "windows", "win_wear", "day_wear",
            "participant_wear", "presence", "schedule".
        """

        # --- 1. Initialize / validate input ----------------------------------
        # Reset report so each call is stateless with respect to prior runs.
        self.report = self._new_report()
        df = self.validate_and_normalize_input(df)
        self.logger.info("Aggregation: %d input rows", len(df))

        # --- 2. Build time windows -------------------------------------------
        # Infer timezone schedule from the data when the caller doesn't supply one.
        if schedule is None:
            schedule = self._infer_schedule_from_input_or_raise(df)

        # Choose fixed-bin or assessment-anchored windowing.
        if assessments is None:
            windows, sch = self.build_bin_windows(df, schedule)
        else:
            windows, sch = self.build_assessment_anchored_windows(assessments, schedule, duration_before=duration_before)

        # Guard: no windows means nothing to aggregate.
        if windows.empty:
            self.logger.warning("No windows generated; returning empty output")
            wide = pd.DataFrame(columns=["user_id","window_id","window_start","window_end","window_seconds","wear_seconds","wear_fraction","watch_on","bin_index"])
            rep = self.build_aggregation_report(windows, wide, pd.DataFrame(), pd.DataFrame())
            aux = {"windows": windows, "win_wear": pd.DataFrame(), "day_wear": pd.DataFrame(), "participant_wear": pd.DataFrame(), "presence": pd.DataFrame(), "schedule": sch}
            return wide, rep, aux

        self.logger.info("Built %d windows for %d users", len(windows), windows["user_id"].nunique())

        # --- 3. Assign observations → windows --------------------------------
        # Each observation is mapped to the window(s) it overlaps, per type.
        # inter_full is a dict {type_name -> intermediate DataFrame}.
        all_types = list(self.cfg.type_registry.keys())
        inter_full = self.assign_observations_to_windows(df, windows, types=all_types)
        self.logger.info("Assigned observations for %d types", len(all_types))

        # --- 4. Derive wear metrics ------------------------------------------
        # Wear is estimated from continuous heart-rate coverage only.
        hr_only = {self.cfg.hr_type: inter_full.get(self.cfg.hr_type, pd.DataFrame())}
        win_wear, day_wear, participant_wear = self.derive_wear_from_hr_coverage(hr_only, windows)

        # --- 5. Build presence flags -----------------------------------------
        # presence_by_type: {type -> DataFrame of (user_id, window_id) that had data}
        # presence_frame:   one row per window with boolean presence_any / presence_non_watch
        presence_by_type, presence_frame = self.build_window_presence_flags(inter_full, windows)

        # --- 6. Early-prune inactive windows ---------------------------------
        # Drop windows that are unlikely to yield useful features before the
        # expensive per-type aggregation step.
        mode = (self.cfg.early_prune_mode or "wear_fraction")

        if mode == "none":
            # Keep every window unconditionally.
            windows_active = windows.copy()
            self.report["bins"]["early_pruned_bins"] = 0

        elif mode == "wear_seconds_gt0":
            # Keep windows where the watch was worn for at least 1 second.
            windows_active = win_wear.loc[win_wear["wear_seconds"] > 0, ["user_id","window_id","window_start","window_end","window_seconds","window_kind"]]
            self.report["bins"]["early_pruned_bins"] = int(len(windows) - len(windows_active))

        elif mode == "any_data":
            # Keep windows that contain at least one observation of any type.
            windows_active = presence_frame.loc[presence_frame["presence_any"], ["user_id","window_id","window_start","window_end","window_seconds"]]
            windows_active = windows_active.merge(windows[["user_id","window_id","window_kind"]], on=["user_id","window_id"], how="left")
            # If early_pruning_types is set, further narrow to windows that have
            # data for at least one of those specific types.
            if self.cfg.early_pruning_types:
                mask_df = None
                for t in self.cfg.early_pruning_types:
                    p = presence_by_type.get(t, pd.DataFrame(columns=["user_id","window_id"]))
                    if p.empty:
                        continue
                    part = p.merge(windows[["user_id","window_id","window_start","window_end","window_seconds","window_kind"]], on=["user_id","window_id"], how="inner")
                    mask_df = part if mask_df is None else pd.concat([mask_df, part], ignore_index=True)
                if mask_df is not None and not mask_df.empty:
                    windows_active = mask_df[["user_id","window_id","window_start","window_end","window_seconds","window_kind"]].drop_duplicates()
            self.report["bins"]["early_pruned_bins"] = int(len(windows) - len(windows_active))

        elif mode == "any_non_watch_data":
            # Keep windows with phone/manual data (i.e. not exclusively watch data).
            windows_active = presence_frame.loc[presence_frame["presence_non_watch"], ["user_id","window_id","window_start","window_end","window_seconds"]]
            windows_active = windows_active.merge(windows[["user_id","window_id","window_kind"]], on=["user_id","window_id"], how="left")
            self.report["bins"]["early_pruned_bins"] = int(len(windows) - len(windows_active))

        else:
            # wear_fraction (default): keep windows that pass the bin-QC wear threshold.
            if self.cfg.enable_bin_qc and (self.cfg.bin_frac_min is not None):
                windows_active = win_wear.loc[win_wear["watch_on"].fillna(False), ["user_id","window_id","window_start","window_end","window_seconds","window_kind"]]
                self.report["bins"]["early_pruned_bins"] = int(len(windows) - len(windows_active))
            else:
                windows_active = windows.copy()
                self.report["bins"]["early_pruned_bins"] = 0

        self.logger.info("Early pruning: %d/%d windows active", len(windows_active), len(windows))

        # --- 7. Filter intermediate data to active windows ------------------
        # Drop observations that belong to pruned windows (except heart rate,
        # which is always kept because it underpins wear metrics and HRV).
        active_pairs = set(zip(windows_active["user_id"], windows_active["window_id"]))
        inter: Dict[str, pd.DataFrame] = {}
        for t in all_types:
            cached = inter_full.get(t, pd.DataFrame())
            if t == self.cfg.hr_type or cached.empty:
                inter[t] = cached
                continue
            pair_idx = pd.MultiIndex.from_arrays([cached["user_id"], cached["window_id"]])
            inter[t] = cached.loc[pair_idx.isin(active_pairs)].copy()

        # --- 8. Track split-expansion in report ------------------------------
        # Record how many input rows each type expanded to after window
        # assignment (one observation can appear in multiple overlapping windows).
        remaining_types = [t for t in all_types if t != self.cfg.hr_type]
        prev_factors = self.report.get("input", {}).get("split_expansion", {})
        split_factors = {}
        for t in remaining_types:
            input_rows = int((df["type"] == t).sum())
            child_rows = int(len(inter.get(t, pd.DataFrame())))
            factors: Dict[str, Any] = {
                "input_rows": input_rows,
                "child_rows": child_rows,
                "expansion_factor": (child_rows / input_rows) if input_rows else np.nan,
            }
            if "unassigned_rows" in prev_factors.get(t, {}):
                factors["unassigned_rows"] = prev_factors[t]["unassigned_rows"]
            split_factors[t] = factors
        self.report["input"]["split_expansion"] = split_factors

        # --- 9. Aggregate each type within windows ---------------------------
        per_type_agg = {t: self.aggregate_type_within_windows(t, inter[t]) for t in all_types}

        # --- 10. Assemble wide feature matrix + attach wear columns ----------
        wide = self.assemble_feature_matrix(windows, per_type_agg)
        wide = self.add_wear_columns(wide, win_wear)

        # --- 11. (Optional) HRV features ------------------------------------
        if self.cfg.compute_hrv:
            t0_hrv = time.time()
            hr_type = self.cfg.hr_type

            if hr_type in inter and len(inter[hr_type]) > 0:
                try:
                    self.logger.info("Computing HRV features from %s...", hr_type)
                    hrv_features = compute_hrv_features_for_windows(
                        windows_df=windows,
                        hr_intermediate=inter[hr_type],
                        hrv_computer=self.hrv_computer
                    )

                    if len(hrv_features) > 0:
                        hrv_cols = [c for c in hrv_features.columns if c.startswith('HRV__')]
                        n_hrv_cols = len(hrv_cols)

                        wide = wide.merge(
                            hrv_features,
                            on=['user_id', 'window_id', 'window_start', 'window_end'],
                            how='left'
                        )

                        elapsed_hrv = time.time() - t0_hrv
                        n_windows_with_hrv = int(wide['HRV__nn_count'].notna().sum()) if 'HRV__nn_count' in wide.columns else 0
                        self.logger.info("Added %d HRV features to %d windows (%.2fs)", n_hrv_cols, n_windows_with_hrv, elapsed_hrv)

                        self.report['hrv'] = {
                            'enabled': True,
                            'time_domain': self.cfg.hrv_time_domain,
                            'freq_domain': self.cfg.hrv_freq_domain,
                            'n_features': n_hrv_cols,
                            'n_windows_total': len(wide),
                            'n_windows_with_hrv': n_windows_with_hrv,
                            'coverage_pct': round(100.0 * n_windows_with_hrv / len(wide), 2) if len(wide) > 0 else 0.0,
                            'min_samples_threshold': self.cfg.hrv_min_samples,
                            'freq_min_samples_threshold': self.cfg.hrv_freq_min_samples,
                            'interpolation_fs': self.cfg.hrv_interpolation_fs,
                            'elapsed_seconds': round(elapsed_hrv, 3),
                        }
                    else:
                        self.logger.warning("No HRV features computed (insufficient %s data)", hr_type)
                        self.report['hrv'] = {
                            'enabled': True,
                            'n_features': 0,
                            'reason': 'insufficient_data',
                            'elapsed_seconds': round(time.time() - t0_hrv, 3),
                        }

                except Exception as e:
                    elapsed_hrv = time.time() - t0_hrv
                    self.logger.warning("HRV computation failed: %s", e)
                    self.report['hrv'] = {
                        'enabled': True,
                        'error': str(e),
                        'elapsed_seconds': round(elapsed_hrv, 3),
                    }
            else:
                self.logger.warning("No %s data available for HRV computation", hr_type)
                self.report['hrv'] = {
                    'enabled': True,
                    'n_features': 0,
                    'reason': f'no_{hr_type}_data',
                }
        else:
            self.report['hrv'] = {'enabled': False}

        # --- 12. Attach local-time metadata columns -------------------------
        # If windows carry local-time fields (start/end, date, DST, etc.),
        # forward them into the wide frame so downstream users have full context.
        if "window_start_local" in windows.columns:
            merge_keys = ["user_id", "window_id", "window_start", "window_end", "window_seconds"]
            proj_candidates = [
                "window_kind",
                "window_start_local", "window_end_local", "local_date", "anchor_local_date", "local_hour",
                "minute_of_day", "local_minute_index", "tz_name", "is_dst", "fold", "utc_offset_seconds",
                "local_weekday", "iso_week",
            ]
            proj_cols = [c for c in proj_candidates if c in windows.columns and c not in wide.columns]
            if proj_cols:
                wide = wide.merge(windows[merge_keys + proj_cols], on=merge_keys, how="left")

        # --- 13. (Optional) Zero-fill ----------------------------------------
        if self.cfg.apply_zero_fill:
            wide = self.apply_zero_fill_policy(wide, presence_frame, presence_by_type)

        # --- 14. Finalize schema and return ----------------------------------
        wide = self._sanitize_output_schema(wide)
        wide = self._apply_output_timezone_policy(wide)

        rep = self.build_aggregation_report(windows, win_wear, day_wear, participant_wear)

        self.logger.info("Aggregation complete: %d output rows", len(wide))
        return wide.sort_values(["user_id","window_start"]).reset_index(drop=True), rep, {
            "windows": windows, "win_wear": win_wear, "day_wear": day_wear, "participant_wear": participant_wear, "presence": presence_frame, "schedule": sch
        }

    def apply_quality_control(self, wide: pd.DataFrame, aux: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply QC dropping/masking to a wide frame using auxiliary wear artifacts."""
        day_wear = aux.get("day_wear", pd.DataFrame()); participant_wear = aux.get("participant_wear", pd.DataFrame())
        wide_qc = self.apply_quality_control_policy(wide, day_wear, participant_wear)
        wide_qc = self.validate_feature_matrix(wide_qc)
        rep_qc = self.build_aggregation_report(aux.get("windows", pd.DataFrame()), aux.get("win_wear", pd.DataFrame()), day_wear, participant_wear)
        rep_qc["qc"]["post_qc_rows"] = int(len(wide_qc))
        return wide_qc, rep_qc

    # ------------------------------------------------------------------------
    # Hierarchical Aggregation
    # ------------------------------------------------------------------------
    def _add_layer_window_columns(self, child: pd.DataFrame, spec: LayerSpec) -> pd.DataFrame:
        """Add window_* columns by snapping child windows to the layer window frequency."""
        df = child.copy()
        if spec.tz_mode == "local" and "window_start_local" in df.columns and "tz_name" in df.columns:
            offset = pd.to_timedelta(spec.origin)
            df["window_start_local"] = (df["window_start_local"] - offset).dt.floor(spec.freq) + offset
            freq_off = pd.tseries.frequencies.to_offset(spec.freq)
            df["window_end_local"] = (pd.DatetimeIndex(df["window_start_local"]) + freq_off).to_series(index=df.index)
            amb = hk_time.get_tz_localize_kwargs(self.cfg.tz_policy)
            ws = pd.Series(index=df.index, dtype="datetime64[ns, UTC]")
            we = pd.Series(index=df.index, dtype="datetime64[ns, UTC]")
            tz_series = df["tz_name"]
            # drop rows with missing tz_name from iteration
            for tz in tz_series.dropna().astype(str).unique():
                idx = df.index[tz_series.astype(str) == tz]
                tzinfo = ZoneInfo(tz)
                wsl = pd.DatetimeIndex(df.loc[idx,"window_start_local"]).tz_localize(
                    tzinfo, ambiguous=amb["ambiguous"], nonexistent=amb["nonexistent"]
                )
                wel = pd.DatetimeIndex(df.loc[idx,"window_end_local"]).tz_localize(
                    tzinfo, ambiguous=amb["ambiguous"], nonexistent=amb["nonexistent"]
                )
                ws.loc[idx] = wsl.tz_convert("UTC"); we.loc[idx] = wel.tz_convert("UTC")
            # rows with NA tz_name will remain NA in ws/we
            df["layer_window_start"] = ws; df["layer_window_end"] = we
            df["layer_window_seconds"] = (df["layer_window_end"] - df["layer_window_start"]).dt.total_seconds()
            return df
        offset = pd.to_timedelta(spec.origin)
        freq_off = pd.tseries.frequencies.to_offset(spec.freq)

        df["layer_window_start"] = (df["window_start"] - offset).dt.floor(spec.freq) + offset
        df["layer_window_end"] = df["layer_window_start"] + freq_off
        df["layer_window_seconds"] = (df["layer_window_end"] - df["layer_window_start"]).dt.total_seconds()
        return df

    def _assert_layer_divisible_by_child_windows(self, child: pd.DataFrame, spec: LayerSpec) -> float:
        """Ensure the layer window length is an integer multiple of child window size (UTC mode)."""
        if spec.tz_mode == "local": return float(self._bin_seconds_nominal)
        uniq = np.unique(child["window_seconds"].astype(float))
        if len(uniq) != 1: raise ValueError("Non-constant child window_seconds not supported for hierarchical aggregation.")
        child_sec = float(uniq[0]); win_sec = float(pd.to_timedelta(pd.tseries.frequencies.to_offset(spec.freq)).total_seconds())
        if (win_sec / child_sec) % 1 != 0: raise ValueError(f"Non-divisible layer: {spec.freq} is not an integer multiple of child window size.")
        return child_sec

    def _compute_completeness_metrics(
        self,
        dfw: pd.DataFrame,
        spec: LayerSpec,
        child_win_seconds: float,
        prev_layer_name: Optional[str],
    ) -> pd.DataFrame:
        """Compute observed/expected child window counts and seconds within each window + wear_seconds if available."""
        keys = ["user_id", "layer_window_start", "layer_window_end", "layer_window_seconds"]
        name = spec.name

        # groupby should stay as_index=False because we merge downstream
        g = dfw.groupby(keys, as_index=False)

        # Observed counts / seconds
        comp = g.agg(
            child_bins_observed=("window_start", "count"),
            seconds_observed=("window_seconds", "sum"),
        )

        # Expected values
        if spec.tz_mode == "local":
            comp["child_bins_expected"] = np.nan
            comp["seconds_expected"] = (comp["layer_window_end"] - comp["layer_window_start"]).dt.total_seconds()
        else:
            # For UTC layers, expected bins is constant (layer_window_seconds / child_win_seconds)
            if len(dfw) == 0:
                exp_bins = np.nan
            else:
                # This is safe even if layer_window_seconds varies a little; we treat it as constant per layer spec.
                exp_bins = float(dfw["layer_window_seconds"].iloc[0]) / float(child_win_seconds) if child_win_seconds else np.nan

            comp["child_bins_expected"] = int(round(exp_bins)) if not np.isnan(exp_bins) else np.nan
            comp["seconds_expected"] = comp["child_bins_expected"] * float(child_win_seconds)

        # Wear seconds observed (robust against as_index=False)
        if "wear_seconds" in dfw.columns:
            wear = g.agg(wear_seconds_observed=("wear_seconds", "sum"))[keys + ["wear_seconds_observed"]]
            comp = comp.merge(wear, on=keys, how="left")
        elif prev_layer_name and f"{prev_layer_name}__wear_seconds_observed" in dfw.columns:
            col = f"{prev_layer_name}__wear_seconds_observed"
            wear = g.agg(wear_seconds_observed=(col, "sum"))[keys + ["wear_seconds_observed"]]
            comp = comp.merge(wear, on=keys, how="left")
        else:
            comp["wear_seconds_observed"] = np.nan

        # Rename with layer prefix
        comp = comp.rename(
            columns={
                "child_bins_observed": f"{name}__child_bins_observed",
                "child_bins_expected": f"{name}__child_bins_expected",
                "seconds_observed": f"{name}__seconds_observed",
                "seconds_expected": f"{name}__seconds_expected",
                "wear_seconds_observed": f"{name}__wear_seconds_observed",
            }
        )

        # Fractions
        comp[f"{name}__child_bins_fraction"] = (
            comp[f"{name}__child_bins_observed"] / comp[f"{name}__child_bins_expected"].replace({0: np.nan})
        )
        comp[f"{name}__seconds_fraction"] = (
            comp[f"{name}__seconds_observed"] / comp[f"{name}__seconds_expected"].replace({0: np.nan})
        )
        comp[f"{name}__wear_seconds_fraction"] = (
            comp[f"{name}__wear_seconds_observed"] / comp[f"{name}__seconds_expected"].replace({0: np.nan})
        )

        comp[f"{name}__seconds_observed_cumulative"] = comp[f"{name}__seconds_observed"]
        return comp

    def _apply_layer_rules(self, dfw: pd.DataFrame, spec: LayerSpec) -> pd.DataFrame:
        keys = ["user_id","layer_window_start","layer_window_end","layer_window_seconds"]
        g = dfw.groupby(keys)  # as_index=True so ops return Series

        results = []
        for r in spec.rules:
            if r.op not in self.op_registry:
                raise KeyError(f"Operator '{r.op}' not registered.")

        for r in spec.rules:
            missing_inputs = [v for v in r.inputs.values() if isinstance(v, str) and v not in dfw.columns]
            if missing_inputs and (r.strict if r.strict is not None else spec.strict):
                raise KeyError(f"Layer '{spec.name}' rule '{r.out}' missing inputs: {missing_inputs}")
            if missing_inputs:
                continue

            res = self.op_registry[r.op](g, r.inputs)

            if isinstance(res, pd.Series):
                res = res.rename(r.out).reset_index()
            elif isinstance(res, pd.DataFrame):
                # keep exactly one value column (drop group key columns if present)
                value_cols = [c for c in res.columns if c not in keys]
                if len(value_cols) != 1:
                    raise ValueError(f"Operator '{r.op}' for rule '{r.out}' must return 1 value column; got {len(value_cols)}.")
                res = res[value_cols[0]].rename(r.out).reset_index()
            else:
                raise TypeError(f"Operator '{r.op}' must return a Series or 1-column DataFrame.")

            results.append(res)

        if not results:
            return dfw[keys].drop_duplicates()

        out = results[0]
        for r_ in results[1:]:
            out = out.merge(r_, on=keys, how="outer")
        return out

    def aggregate_layer(self, child: pd.DataFrame, spec: LayerSpec, prev_layer_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Aggregate a child layer into a coarser window per LayerSpec.
        Steps: windowize → assert divisibility → early prune → completeness → rules → optional layer-level QC.
        """
        if child.empty:
            return pd.DataFrame(columns=["user_id","layer_window_start","layer_window_end","layer_window_seconds"]), {"layer": spec.name, "rows": 0}

        dfw = self._add_layer_window_columns(child, spec); child_sec = self._assert_layer_divisible_by_child_windows(child, spec)

        lcfg = self.cfg.qc_by_layer.get(spec.name, {}).get("early_prune", {}); mode = lcfg.get("mode","none")
        keys = ["user_id","layer_window_start","layer_window_end","layer_window_seconds"]
        if mode in {"seconds_fraction","wear_seconds_gt0"}:
            comp_tmp = self._compute_completeness_metrics(dfw, spec, child_sec, prev_layer_name)
            if mode == "seconds_fraction":
                thr = float(lcfg.get("threshold", 0.0)); keep_df = comp_tmp[keys].copy(); keep_df["keep"] = comp_tmp[f"{spec.name}__seconds_fraction"] >= thr
            else:
                keep_df = comp_tmp[keys].copy(); keep_df["keep"] = comp_tmp[f"{spec.name}__wear_seconds_observed"] > 0
            dfw = dfw.merge(keep_df, on=keys, how="left"); dfw = dfw[dfw["keep"].fillna(False)].drop(columns=["keep"])

        comp = self._compute_completeness_metrics(dfw, spec, child_sec, prev_layer_name)
        rules_out = self._apply_layer_rules(dfw, spec)
        base = rules_out.merge(comp, on=keys, how="left").sort_values(["user_id","layer_window_start"]).reset_index(drop=True)

        # attach selected window label columns
        desired = [
            "user_id","window_id","window_start","window_end","window_seconds","bin_index","window_kind",
            "window_start_local","window_end_local","local_date","anchor_local_date","local_hour",
            "minute_of_day","local_minute_index","tz_name","is_dst","fold","utc_offset_seconds","local_weekday","iso_week",
        ]

        # representatives from child rows per layer window
        rep_cols = {"window_id":"min", "bin_index":"min", "window_kind":"first"}
        if "window_id" not in dfw.columns: rep_cols.pop("window_id", None)
        if "bin_index" not in dfw.columns: rep_cols.pop("bin_index", None)
        if "window_kind" not in dfw.columns: rep_cols.pop("window_kind", None)

        reps = []
        if rep_cols:
            agg_rep = dfw.groupby(keys).agg(**{k: (k, v) for k, v in rep_cols.items()}).reset_index()
            reps.append(agg_rep)

        if reps:
            labels = reps[0]
            for r_ in reps[1:]:
                labels = labels.merge(r_, on=keys, how="outer")
            base = base.merge(labels, on=keys, how="left")

        # local projection for layer windows using the same overlap-based rules as L0.
        if {"user_id", "window_start", "window_end", "tz_name"}.issubset(dfw.columns):
            tz_schedule = (
                dfw[["user_id", "window_start", "window_end", "tz_name"]]
                .rename(columns={"window_start": "start_utc", "window_end": "end_utc"})
                .copy()
            )

            base["_proj_ord"] = np.arange(len(base), dtype=np.int64)
            proj_windows = base[keys + ["_proj_ord"]].copy()
            proj_windows["window_id"] = proj_windows["_proj_ord"]
            proj_windows["window_start"] = proj_windows["layer_window_start"]
            proj_windows["window_end"] = proj_windows["layer_window_end"]
            proj_windows["window_seconds"] = proj_windows["layer_window_seconds"]
            proj_windows["window_kind"] = "grid"

            projected = hk_time.add_local_time_labels(
                proj_windows[
                    [
                        "user_id",
                        "window_id",
                        "window_start",
                        "window_end",
                        "window_seconds",
                        "window_kind",
                    ]
                ],
                tz_schedule,
                report=None,
            ).rename(columns={"window_id": "_proj_ord"})

            proj_cols = [
                "tz_name",
                "window_start_local",
                "window_end_local",
                "local_date",
                "anchor_local_date",
                "local_hour",
                "minute_of_day",
                "local_minute_index",
                "is_dst",
                "fold",
                "utc_offset_seconds",
                "local_weekday",
                "iso_week",
            ]
            drop_existing = [c for c in proj_cols if c in base.columns]
            if drop_existing:
                base = base.drop(columns=drop_existing)

            keep_proj = [c for c in ["_proj_ord"] + proj_cols if c in projected.columns]
            base = base.merge(projected[keep_proj], on="_proj_ord", how="left")
            base = base.drop(columns=["_proj_ord"])

        # keep the requested identifiers if present among desired list
        keep_extra = [c for c in desired if c in base.columns and c not in keys]
        layer_report = {"layer": spec.name, "rows": int(len(base)), "divisible": (spec.tz_mode != "local"), "child_bin_seconds": child_sec}
        return base[keys + keep_extra + [c for c in base.columns if c not in keys + keep_extra]].reset_index(drop=True), layer_report

    def run_hierarchical_aggregation(self, wide_l0: pd.DataFrame, specs: List[LayerSpec], apply_qc_during: bool = False, return_final_only: bool = False) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Execute multiple hierarchical layers sequentially.
        Returns (layers_by_name, report). If return_final_only=True, only last layer returned.
        """
        layers: Dict[str, pd.DataFrame] = {}; reports: Dict[str, Any] = {}
        child, prev_name = wide_l0, None
        for i, spec in enumerate(specs, start=1):
            layer_df, layer_rep = self.aggregate_layer(child, spec, prev_layer_name=prev_name)
            if apply_qc_during: pass  # explicit no-op for parity
            if return_final_only:
                if i == len(specs): layers = {spec.name: layer_df}
            else:
                layers[spec.name] = layer_df
            reports[spec.name] = layer_rep
            child = layer_df.rename(columns={"layer_window_start":"window_start","layer_window_end":"window_end","layer_window_seconds":"window_seconds"}).copy()
            prev_name = spec.name
        return layers, {"layers": reports}

    def apply_layer_quality_control(self, layers_by_name: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """Pass-through placeholder for per-layer QC post-processing."""
        out = {}; rep = {"layer_qc": {}}
        for lname, frame in layers_by_name.items():
            out[lname] = frame; rep["layer_qc"][lname] = {"rows": int(len(frame))}
        return out, rep

    # ------------------------------------------------------------------------
    # Extra: Daily Sleep Features
    # ------------------------------------------------------------------------
    def compute_daily_sleep_features(self, hk_data: pd.DataFrame, qc: bool = True) -> pd.DataFrame:
        """
        Compute per-day sleep features from cleaned/normalized HealthKit data.
        
        Assumptions (already cleaned):
        - SleepAnalysis labels are normalized (e.g., 'InBed','Asleep','Awake', 'AsleepCore', 'AsleepREM', 'AsleepDeep')
        - Overlaps have been resolved for SleepAnalysis (categorical) rows
        - HeartRate / HeartRateVariabilitySDNN / EnvironmentalAudioExposure values are numeric
        - local_start/local_end exist (naive local datetimes), and user_id/type/value columns present

        Returns
        -------
        pd.DataFrame with one row per (user_id, date).
        """
        if hk_data is None or hk_data.empty:
            return pd.DataFrame(columns=["user_id", "date"])

        # Different labels from watch/phone, labels for awake only occur if device thinks you're in bed (sleep data doesn't cover all times)
        IN_BED_CATEGORIES = [
            "InBed",
            "Asleep",
            "AsleepUnspecified",
            "CategoryValueUnknown",
            "Awake",
            "AwakeUnspecified",
            "AsleepCore",
            "AsleepDeep",
            "AsleepREM",
        ]
        ASLEEP_CATEGORIES = [
            "Asleep",
            "AsleepUnspecified",
            "AwakeUnspecified", # Based on Samir looking at the data. It didn't show up often, but it mapped to time within sleep periods, often overlapping with asleep label.
            "CategoryValueUnknown",
            "AsleepCore",
            "AsleepDeep",
            "AsleepREM",
        ]
        AWAKE_CATEGORIES = {"Awake", "AwakeUnspecified"}

        sleep_data = []
        for uid, data in hk_data.groupby("user_id", sort=False):
            # slice types we need
            sleep_events = (
                data.loc[data["type"] == "SleepAnalysis",
                        ["local_start", "local_end", "value", "user_id", "type"]]
                .copy()
            )
            if sleep_events.empty:
                continue

            # label cleanup
            sleep_events["value"] = (
                sleep_events["value"].astype(str)
                    .str.replace("HKCategoryValueSleepAnalysis", "", regex=False)
            )
            sleep_events["local_start"] = pd.to_datetime(sleep_events["local_start"], errors="coerce")
            sleep_events["local_end"]   = pd.to_datetime(sleep_events["local_end"],   errors="coerce")
            sleep_events = sleep_events.dropna(subset=["local_start","local_end"]).drop_duplicates().sort_values("local_start")
            if sleep_events.empty:
                continue

            sleep_events["duration"] = sleep_events["local_end"] - sleep_events["local_start"]

            heart_rate_samples   = data.loc[data["type"] == "HeartRate", ["local_start","value"]].copy()
            hrv_sdnn_samples     = data.loc[data["type"] == "HeartRateVariabilitySDNN", ["local_start","value"]].copy()
            audio_exposure_samples = data.loc[data["type"] == "EnvironmentalAudioExposure", ["local_start","value"]].copy()

            # coerce numeric
            for df_ in (heart_rate_samples, hrv_sdnn_samples, audio_exposure_samples):
                if not df_.empty:
                    df_["value"] = pd.to_numeric(df_["value"], errors="coerce")
                    df_.dropna(subset=["value","local_start"], inplace=True)
                    df_["local_start"] = pd.to_datetime(df_["local_start"], errors="coerce")
                    df_.dropna(subset=["local_start"], inplace=True)

            # day-bucketing anchor: 3pm local day anchor relative to first observed local_start
            day_anchor_local = pd.to_datetime(sleep_events["local_start"].min()).replace(
                hour=15, minute=0, second=0, microsecond=0
            )

            # handy masks
            in_bed = sleep_events["value"].isin(IN_BED_CATEGORIES)
            asleep = sleep_events["value"].isin(ASLEEP_CATEGORIES)
            awake  = sleep_events["value"].isin(AWAKE_CATEGORIES)

            # --- per-day aggregations (resample by local_start with origin=day_anchor_local) ---
            bedrest_onset = (
                sleep_events.loc[in_bed]
                    .resample("1D", origin=day_anchor_local, on="local_start")["local_start"]
                    .first()
                    .rename("bedrest_onset")
            )
            bedrest_offset = (
                sleep_events.loc[in_bed]
                    .resample("1D", origin=day_anchor_local, on="local_start")["local_end"]
                    .last()
                    .rename("bedrest_offset")
            )
            bedrest_duration = (
                sleep_events.loc[in_bed]
                    .resample("1D", origin=day_anchor_local, on="local_start")["duration"]
                    .sum()
                    .rename("bedrest_duration")
            )

            sleep_onset = (
                sleep_events.loc[asleep]
                    .resample("1D", origin=day_anchor_local, on="local_start")["local_start"]
                    .first()
                    .rename("sleep_onset")
            )
            sleep_offset = (
                sleep_events.loc[asleep]
                    .resample("1D", origin=day_anchor_local, on="local_start")["local_end"]
                    .last()
                    .rename("sleep_offset")
            )
            sleep_duration = (
                sleep_events.loc[asleep]
                    .resample("1D", origin=day_anchor_local, on="local_start")["duration"]
                    .sum()
                    .rename("sleep_duration")
            )

            # first Awake period (to later compute WASO = total awake - first awake)
            first_awake_segment = (
                sleep_events.loc[awake]
                    .resample("1D", origin=day_anchor_local, on="local_start")[["duration","local_start"]]
                    .first()
            )
            # restrict first awake that falls between bedrest onset and sleep onset
            first_awake_segment = pd.concat([first_awake_segment, bedrest_onset, sleep_onset], axis=1)
            fa_mask = (first_awake_segment["local_start"] > first_awake_segment["bedrest_onset"]) & \
                    (first_awake_segment["local_start"] < first_awake_segment["sleep_onset"])
            first_awake_duration = first_awake_segment.loc[fa_mask, "duration"].rename("first_awake_duration")

            awake_duration = (
                sleep_events.loc[awake]
                    .resample("1D", origin=day_anchor_local, on="local_start")["duration"]
                    .sum()
                    .rename("awake_duration")
            )

            # sleep category statistics
            sleep_category_agg = (
                sleep_events
                    .groupby(
                        ["value", pd.Grouper(key="local_start", freq="1D", origin=day_anchor_local)]
                    )["duration"]
                    .agg(sum="sum", mean="mean", count="count")
                    .reset_index())

            if not sleep_category_agg.empty:
                sleep_category_agg = sleep_category_agg.pivot_table(
                    index="local_start",
                    columns="value",
                    values=["sum", "mean", "count"],
                    aggfunc="first",
                )
                sleep_category_agg = sleep_category_agg.swaplevel(0, 1, axis=1).sort_index(axis=1)
                sleep_category_agg.columns = [
                    "sleep_" + "_".join([str(col[0]), str(col[1])]).strip("_")
                    for col in sleep_category_agg.columns
                ]
                duration_cols = [c for c in sleep_category_agg.columns if c.endswith("sum") or c.endswith("mean")]
                for c in duration_cols:
                    sleep_category_agg[c] = pd.to_timedelta(sleep_category_agg[c], errors="coerce") / pd.Timedelta("1h")
            else:
                sleep_category_agg = pd.DataFrame()

            # assemble
            sleep_agg = pd.concat(
                [bedrest_onset, bedrest_offset, bedrest_duration,
                sleep_onset, sleep_offset, sleep_duration,
                first_awake_duration, awake_duration],
                axis=1,
            )
            if not sleep_category_agg.empty:
                sleep_agg = sleep_agg.join(sleep_category_agg, how="left")
            sleep_agg = sleep_agg.copy()
            sleep_agg["first_awake_duration"] = sleep_agg["first_awake_duration"].fillna(pd.Timedelta(0))

            # guardrail cleanup: keep bedrest bounds wrt sleep bounds when present
            m = (sleep_agg["bedrest_onset"].notna() & sleep_agg["sleep_onset"].notna() &
                (sleep_agg["bedrest_onset"] > sleep_agg["sleep_onset"]))
            sleep_agg.loc[m, "bedrest_onset"] = sleep_agg.loc[m, "sleep_onset"]

            m = (sleep_agg["bedrest_offset"].notna() & sleep_agg["sleep_offset"].notna() &
                (sleep_agg["bedrest_offset"] < sleep_agg["sleep_offset"]))
            sleep_agg.loc[m, "bedrest_offset"] = sleep_agg.loc[m, "sleep_offset"]

            # contextual medians: HR/HRV during sleep; noise during bedrest
            sleep_hr, sleep_hrv, bedrest_noise = [], [], []
            for _, r in sleep_agg.iterrows():
                # sleep intervals may be missing for some days
                s_on, s_off = r.get("sleep_onset"), r.get("sleep_offset")
                b_on, b_off = r.get("bedrest_onset"), r.get("bedrest_offset")

                if not heart_rate_samples.empty and pd.notna(s_on) and pd.notna(s_off):
                    sleep_hr.append(heart_rate_samples[(heart_rate_samples["local_start"] >= s_on) & (heart_rate_samples["local_start"] <= s_off)]["value"].median())
                else:
                    sleep_hr.append(np.nan)

                if not hrv_sdnn_samples.empty and pd.notna(s_on) and pd.notna(s_off):
                    sleep_hrv.append(hrv_sdnn_samples[(hrv_sdnn_samples["local_start"] >= s_on) & (hrv_sdnn_samples["local_start"] <= s_off)]["value"].median())
                else:
                    sleep_hrv.append(np.nan)

                if not audio_exposure_samples.empty and pd.notna(b_on) and pd.notna(b_off):
                    bedrest_noise.append(audio_exposure_samples[(audio_exposure_samples["local_start"] >= b_on) & (audio_exposure_samples["local_start"] <= b_off)]["value"].median())
                else:
                    bedrest_noise.append(np.nan)

            sleep_agg["sleep_hr_median"] = sleep_hr
            sleep_agg["sleep_hrv_median"] = sleep_hrv
            sleep_agg["bedrest_audio_median"] = bedrest_noise

            sleep_agg["sleep_onset_latency"] = (
                (sleep_agg["sleep_onset"] - sleep_agg["bedrest_onset"]).to_numpy() / pd.Timedelta("1h")
            )
            wake_after_sleep_onset_td = sleep_agg["awake_duration"] - sleep_agg["first_awake_duration"]

            # convert timedeltas to hours
            to_hours = lambda td: (td.to_numpy() / pd.Timedelta("1h")) if isinstance(td, pd.Series) else np.nan
            sleep_agg["bedrest_duration"] = to_hours(sleep_agg["bedrest_duration"])
            sleep_agg["sleep_duration"] = to_hours(sleep_agg["sleep_duration"])
            sleep_agg["awake_duration"] = to_hours(sleep_agg["awake_duration"])
            sleep_agg["wake_after_sleep_onset"] = wake_after_sleep_onset_td.to_numpy() / pd.Timedelta("1h")

            # efficiency
            with np.errstate(divide="ignore", invalid="ignore"):
                sleep_agg["efficiency"] = sleep_agg["sleep_duration"] / np.where(sleep_agg["bedrest_duration"] == 0, np.nan, sleep_agg["bedrest_duration"])
            sleep_agg.loc[sleep_agg["efficiency"] > 1, "efficiency"] = 1
            sleep_agg.loc[sleep_agg["efficiency"] == 0, "efficiency"] = np.nan

            # clock-hour features relative to prior midnight (offset +15h because of 3pm anchor)
            hours_offset = 15
            idx = sleep_agg.index.tz_localize(None) if isinstance(sleep_agg.index, pd.DatetimeIndex) else pd.to_datetime(sleep_agg.index)
            def _hours(col):
                return hours_offset + ((sleep_agg[col] - idx) / pd.Timedelta("1h")).to_numpy()
            for col, out in [("bedrest_onset","bedrest_onset_hours"),
                            ("bedrest_offset","bedrest_offset_hours"),
                            ("sleep_onset","sleep_onset_hours"),
                            ("sleep_offset","sleep_offset_hours")]:
                if col in sleep_agg.columns:
                    sleep_agg[out] = _hours(col)

            sleep_agg = sleep_agg.drop(
                columns=[
                    "bedrest_onset",
                    "bedrest_offset",
                    "sleep_onset",
                    "sleep_offset",
                    "first_awake_duration",
                ],
                errors="ignore",
            )

            # drop timestamp cols; derive "date" as day after anchor (3pm)
            sleep_agg = sleep_agg.reset_index().rename(columns={"index": "local_start"})
            sleep_agg["date"] = (sleep_agg["local_start"] + pd.Timedelta("1D")).dt.date
            sleep_agg = sleep_agg.drop(columns=["local_start"])
            sleep_agg["user_id"] = uid

            # rename with sleep_ prefix and _day suffix (without double-prefixing)
            rename = {}
            for c in sleep_agg.columns:
                if c in ("user_id", "date"):
                    rename[c] = c
                elif c.startswith("sleep_"):
                    # stage columns + "sleep_duration", "sleep_onset_latency", etc.
                    rename[c] = f"{c}_day"
                else:
                    rename[c] = f"sleep_{c}_day"

            sleep_agg = sleep_agg.rename(columns=rename)
            sleep_data.append(sleep_agg)

        if not sleep_data:
            return pd.DataFrame(columns=["user_id", "date"])

        out = pd.concat(sleep_data, ignore_index=True)

        if qc:
            out = self.apply_sleep_quality_control(out)
        return out

    def apply_sleep_quality_control(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return data

        d = data.copy()

        # efficiency (clip to 1, set 0 to nan)
        if "sleep_efficiency_day" in d.columns:
            d.loc[d["sleep_efficiency_day"] == 0, "sleep_efficiency_day"] = np.nan
            d.loc[d["sleep_efficiency_day"] > 1, "sleep_efficiency_day"] = 1.0

        # durations (set 0 to nan)
        if "sleep_duration_day" in d.columns:
            d.loc[d["sleep_duration_day"] == 0, "sleep_duration_day"] = np.nan

        if "sleep_bedrest_duration_day" in d.columns:
            d.loc[d["sleep_bedrest_duration_day"] == 0, "sleep_bedrest_duration_day"] = np.nan

        # stage sums (hours; set > 20 to nan)
        if "sleep_Awake_sum_day" in d.columns:
            d.loc[d["sleep_Awake_sum_day"] > 20, "sleep_Awake_sum_day"] = np.nan

        return d

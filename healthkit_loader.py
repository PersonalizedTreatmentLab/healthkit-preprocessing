from __future__ import annotations
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
from zoneinfo import ZoneInfo
from tqdm.auto import tqdm
from healthkit_loader_specs import HKLoadConfig

#* While handling of travel works well, DST fall-back is flawed, often assuming points lie in the second repeat of the hour.
# ===================================================
# Class: HealthKitLoader
# ===================================================
class HealthKitLoader:
    """
    Loads HealthKit CSV data and normalizes columns.
    """

    DST_SAMPLE_LIMIT = 5

    def __init__(self, config: Optional[HKLoadConfig] = None, logger: Optional[logging.Logger] = None):
        self.cfg = config or HKLoadConfig()
        self.cfg.validate()
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _init_file_report(path: Path, file_type: str) -> Dict[str, Any]:
        """Create a stable report container for one loaded file."""
        return {
            "__global__": {
                "path": str(path),
                "file_type": file_type,
                "rows_in_raw": 0,
                "rows_out": 0,
                "rows_dropped_total": 0,
                "dropped_by_reason": {},
            },
            "timezone": {
                "missing_column_rows": 0,
                "missing_value_rows": 0,
                "invalid_value_rows": 0,
                "filled_default_rows": 0,
                "dropped_rows": 0,
                "invalid_values_sample": [],
            },
            "dst": {
                "fallback_ambiguous_false_rows": 0,
                "nonexistent_detected_rows": 0,
                "nonexistent_resolved_policy": None,
                "ambiguous_infer_fail_rows": 0,
                "fallback_used": False,
                "fallback_samples": [],
            },
            "user_id": {
                "rows_in": 0,
                "rows_dropped_non_numeric": 0,
                "sample_values": [],
            },
        }

    @staticmethod
    def _ensure_report_section(
        report: Optional[Dict[str, Any]],
        key: str,
        defaults: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Create and seed a report section only when report capture is enabled."""
        if report is None:
            return None
        section = report.setdefault(key, {})
        for k, v in defaults.items():
            if k not in section:
                if isinstance(v, list):
                    section[k] = []
                elif isinstance(v, dict):
                    section[k] = {}
                else:
                    section[k] = v
        return section

    @staticmethod
    def _normalize_sample_value(value: Any) -> Any:
        """Convert values to JSON-serializable primitives for report samples."""
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value

    @staticmethod
    def _extend_unique_samples(target: list[Any], values: list[Any], limit: int) -> None:
        """Append values into target while preserving order and sample cap."""
        if limit <= 0:
            return
        for value in values:
            if len(target) >= limit:
                break
            if value not in target:
                target.append(value)

    @staticmethod
    def _bump_drop_reason(report: Dict[str, Any], reason: str, amount: int) -> None:
        """Increment dropped-row reason counters in the global report section."""
        if amount <= 0:
            return
        global_rep = report.setdefault("__global__", {})
        dropped = global_rep.setdefault("dropped_by_reason", {})
        dropped[reason] = int(dropped.get(reason, 0)) + int(amount)

    @staticmethod
    def _is_valid_timezone_name(tz_name: str) -> bool:
        """Validate IANA timezone name."""
        try:
            ZoneInfo(str(tz_name))
            return True
        except Exception:
            return False

    def _normalize_timezone_column(
        self,
        data: pd.DataFrame,
        *,
        tz_col: str = "timezone",
        context: str = "input",
        report: Optional[Dict[str, Any]] = None,
        report_key: str = "timezone",
    ) -> pd.DataFrame:
        """Normalize timezone labels and handle missing/invalid values consistently."""
        policy = self.cfg.timezone_missing_policy
        default_tz = self.cfg.default_tz
        out = data.copy()
        rep = self._ensure_report_section(
            report,
            report_key,
            {
                "missing_column_rows": 0,
                "missing_value_rows": 0,
                "invalid_value_rows": 0,
                "filled_default_rows": 0,
                "dropped_rows": 0,
                "invalid_values_sample": [],
            },
        )

        if tz_col not in out.columns:
            missing_col_rows = int(len(out))
            if rep is not None:
                rep["missing_column_rows"] = int(rep["missing_column_rows"]) + missing_col_rows
            if policy == "error":
                raise ValueError(f"[{context}] Missing {tz_col} column in {missing_col_rows} rows")
            elif policy == "fill_default":
                out[tz_col] = default_tz
                self.logger.warning("[%s] Missing %s column; filled all rows with %s", context, tz_col, default_tz)
            else:
                out[tz_col] = pd.Series(pd.NA, index=out.index, dtype="string")
                self.logger.warning("[%s] Missing %s column; rows with missing timezone will be dropped", context, tz_col)

        tz_series = out[tz_col].astype("string").str.strip()
        missing_mask = tz_series.isna() | tz_series.eq("")
        missing_count = int(missing_mask.sum())
        if rep is not None:
            rep["missing_value_rows"] = int(rep["missing_value_rows"]) + missing_count

        invalid_tzs: set[str] = set()
        for tz in tz_series[~missing_mask].dropna().unique():
            tz_str = str(tz)
            if not self._is_valid_timezone_name(tz_str):
                invalid_tzs.add(tz_str)
        invalid_mask = tz_series.isin(list(invalid_tzs)) if invalid_tzs else pd.Series(False, index=out.index, dtype=bool)
        invalid_count = int(invalid_mask.sum())
        if rep is not None:
            rep["invalid_value_rows"] = int(rep["invalid_value_rows"]) + invalid_count
            invalid_samples = [self._normalize_sample_value(x) for x in list(invalid_tzs)[: self.DST_SAMPLE_LIMIT]]
            self._extend_unique_samples(rep["invalid_values_sample"], invalid_samples, self.DST_SAMPLE_LIMIT)
        bad_mask = missing_mask | invalid_mask

        if bad_mask.any():
            affected = int(bad_mask.sum())
            if policy == "error":
                raise ValueError(
                    f"[{context}] {affected} rows with missing/invalid timezone values"
                )
            elif policy == "fill_default":
                out.loc[bad_mask, tz_col] = default_tz
                if rep is not None:
                    rep["filled_default_rows"] = int(rep["filled_default_rows"]) + affected
                self.logger.warning(
                    "[%s] Missing/invalid %s values found; filled %s rows with %s",
                    context,
                    tz_col,
                    affected,
                    default_tz,
                )
            else:
                out = out.loc[~bad_mask].copy()
                if rep is not None:
                    rep["dropped_rows"] = int(rep["dropped_rows"]) + affected
                self.logger.warning(
                    "[%s] Missing/invalid %s values found; dropped %s rows",
                    context,
                    tz_col,
                    affected,
                )
        out[tz_col] = out[tz_col].astype("string").str.strip()
        return out

    def _normalize_user_id_column(
        self,
        data: pd.DataFrame,
        *,
        context: str,
        report: Optional[Dict[str, Any]] = None,
        report_key: str = "user_id",
    ) -> pd.DataFrame:
        """Coerce user_id to integer ids; drop invalid rows and log counts."""
        out = data.copy()
        rep = self._ensure_report_section(
            report,
            report_key,
            {
                "rows_in": 0,
                "rows_dropped_non_numeric": 0,
                "sample_values": [],
            },
        )
        if rep is not None:
            rep["rows_in"] = int(len(out))

        if "user_id" not in out.columns:
            return out

        uid_num = pd.to_numeric(out["user_id"], errors="coerce")
        finite = uid_num.notna() & np.isfinite(uid_num)
        integer_like = finite & (np.floor(uid_num) == uid_num)
        invalid = ~integer_like
        dropped = int(invalid.sum())
        if dropped:
            sample = out.loc[invalid, "user_id"].astype("string").dropna().head(5).tolist()
            if rep is not None:
                rep["rows_dropped_non_numeric"] = int(rep["rows_dropped_non_numeric"]) + dropped
                self._extend_unique_samples(rep["sample_values"], sample, self.DST_SAMPLE_LIMIT)
            self.logger.warning("[%s] Dropping %s rows with non-numeric user_id; sample=%s", context, dropped, sample)
            out = out.loc[~invalid].copy()
            uid_num = uid_num.loc[out.index]
        if not out.empty:
            out["user_id"] = uid_num.astype("int64")
        else:
            out["user_id"] = pd.Series(dtype="int64")
        return out

    # ========================
    # Public entry points
    # ========================
    def load_file(
            self,
            path: str | Path,
            *,
            return_report: bool = False,
        ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Load a single HealthKit CSV file and normalize columns.
        Assumes data already has required columns.
        Returns a dataframe with expected columns: user_id, local_start, local_end, timezone, type, utc_start, utc_end, value, ...
        If return_report=True, returns (dataframe, report).
        """
        # Resolve + validate input path
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")

        suffix = path.suffix.lower()
        report = self._init_file_report(path, file_type=suffix.lstrip("."))

        # Load raw data
        if suffix == ".csv":
            data = pd.read_csv(path, low_memory=False)
            for c in ("local_start", "local_end"):
                if c in data.columns:
                    data[c] = pd.to_datetime(data[c], errors="coerce")
        else:
            raise ValueError(f"Unknown file format: {path}")
        report["__global__"]["rows_in_raw"] = int(len(data))

        # Normalize "value" column
        if "body.quantity.value" in data.columns or "body.category.value" in data.columns:
            data = data.rename(columns={"body.quantity.value": "value"})
            if "body.category.value" in data.columns:
                cat = data["body.category.value"]
                data["value"] = cat.where(cat.notnull(), data.get("value"))
                data = data.drop(columns=["body.category.value"])

        # Normalize "type" strings (handle SHOUTY types + Sleep values)
        if "type" in data.columns and data["type"].astype(str).str.isupper().any():
            camel = lambda s: "".join(w.capitalize() for w in str(s).split("_"))
            data["type"] = data["type"].map(camel).str.replace("Sdnn", "SDNN", regex=False)
            sleep_mask = data["type"].str.contains("Sleep", na=False)
            if sleep_mask.any():
                data.loc[sleep_mask, "value"] = data.loc[sleep_mask, "value"].astype(str).map(camel)

        # Normalize timezone column name variants
        data = data.rename(columns={"HKTimezone": "timezone", "HKTimeZone": "timezone"})

        # ----------------------------------- START OF TZ HANDLING -----------------------------------
        # CSV-only: fill/validate timezone and compute tz-aware + UTC timestamps if missing
        if suffix == ".csv":
            # Determine whether any derived time columns need to be created
            needs_any = any(c not in data.columns for c in ("utc_start", "utc_end", "local_start_tz", "local_end_tz"))

            # Always normalize timezone labels, even if UTC columns already exist.
            data = self._normalize_timezone_column(
                data,
                tz_col="timezone",
                context=f"csv:{path.name}",
                report=report,
                report_key="timezone",
            )
            
            # Localize naive local times per timezone, and backfill UTC equivalents
            if needs_any and all(c in data.columns for c in ("local_start", "local_end", "timezone")):
                dst_rep = report["dst"]

                # Per-timezone localization + UTC conversion
                def _localize_group(group: pd.DataFrame, tz: str) -> pd.DataFrame:
                    group = group.copy()
                    group["timezone"] = tz

                    def _sample_rows(s_sorted: pd.Series, idx_values: pd.Index, label: str) -> None:
                        remaining = self.DST_SAMPLE_LIMIT - len(dst_rep["fallback_samples"])
                        if remaining <= 0 or len(idx_values) == 0:
                            return
                        for idx in idx_values[:remaining]:
                            local_value = s_sorted.loc[idx]
                            if isinstance(local_value, pd.Series):
                                local_value = local_value.iloc[0]
                            uid_value = None
                            if "user_id" in group.columns:
                                try:
                                    uid_series = group.loc[[idx], "user_id"]
                                    if not uid_series.empty:
                                        uid_value = uid_series.iloc[0]
                                except Exception:
                                    uid_value = None
                            dst_rep["fallback_samples"].append(
                                {
                                    "row_index": self._normalize_sample_value(idx),
                                    "user_id": self._normalize_sample_value(uid_value),
                                    "timezone": str(tz),
                                    "label": label,
                                    "local_time": self._normalize_sample_value(local_value),
                                }
                            )

                    # Localize one naive datetime series into tz, preserving original index
                    def _localize_series(s: pd.Series, label: str) -> pd.Series:
                        # Ensure naive input
                        if getattr(s.dt, "tz", None) is not None:
                            s = s.dt.tz_localize(None)
                        # Sort for better ambiguous="infer" behavior
                        s_sorted = s.sort_values()
                        non_null_mask = s_sorted.notna()

                        # Probe rows that require nonexistent-shift handling.
                        nonexistent_idx = pd.Index([])
                        if non_null_mask.any():
                            try:
                                nonexistent_probe = s_sorted.dt.tz_localize(
                                    tz,
                                    ambiguous=False,
                                    nonexistent="NaT",
                                )
                                nonexistent_mask = nonexistent_probe.isna() & non_null_mask
                                nonexistent_idx = nonexistent_probe[nonexistent_mask].index
                            except Exception:
                                nonexistent_idx = pd.Index([])
                        nonexistent_count = int(len(nonexistent_idx))
                        if nonexistent_count:
                            dst_rep["nonexistent_detected_rows"] = (
                                int(dst_rep["nonexistent_detected_rows"]) + nonexistent_count
                            )
                            dst_rep["nonexistent_resolved_policy"] = str(self.cfg.dst_nonexistent)
                            _sample_rows(s_sorted, nonexistent_idx, label)

                        # Probe rows that are DST-ambiguous under this timezone.
                        ambiguous_idx = pd.Index([])
                        if non_null_mask.any():
                            try:
                                ambiguous_probe = s_sorted.dt.tz_localize(
                                    tz,
                                    ambiguous="NaT",
                                    nonexistent=self.cfg.dst_nonexistent,
                                )
                                ambiguous_mask = ambiguous_probe.isna() & non_null_mask
                                ambiguous_idx = ambiguous_probe[ambiguous_mask].index
                            except Exception:
                                ambiguous_idx = pd.Index([])

                        try:
                            localized = s_sorted.dt.tz_localize(
                                tz,
                                ambiguous=self.cfg.dst_ambiguous,
                                nonexistent=self.cfg.dst_nonexistent,
                            )
                        except Exception:
                            # 1) Fallback: choose the later (second) fold for ambiguous times
                            fallback_idx = ambiguous_idx if len(ambiguous_idx) else s_sorted[non_null_mask].index
                            fallback_count = int(len(fallback_idx))
                            dst_rep["fallback_used"] = True
                            dst_rep["fallback_ambiguous_false_rows"] = (
                                int(dst_rep["fallback_ambiguous_false_rows"]) + fallback_count
                            )
                            dst_rep["ambiguous_infer_fail_rows"] = (
                                int(dst_rep["ambiguous_infer_fail_rows"]) + fallback_count
                            )
                            _sample_rows(s_sorted, fallback_idx, label)
                            try:
                                localized = s_sorted.dt.tz_localize(
                                    tz,
                                    ambiguous=False,  # later hour (standard time)
                                    nonexistent=self.cfg.dst_nonexistent,  # e.g., "shift_forward"
                                )
                            except Exception as e2:
                                # 2) Fallback failed too — collect diagnostics and raise
                                bad_idx = s.index
                                if self.cfg.dst_ambiguous == "infer":
                                    try:
                                        diag = s_sorted.dt.tz_localize(
                                            tz,
                                            ambiguous="NaT",
                                            nonexistent=self.cfg.dst_nonexistent,
                                        )
                                        amb = diag.isna() & s_sorted.notna()
                                        bad_idx = diag[amb].index
                                    except Exception:
                                        pass  # keep default bad_idx if probing fails

                                uid = None
                                if "user_id" in group.columns:
                                    try:
                                        uid = group["user_id"].dropna().iloc[0]
                                    except Exception:
                                        pass

                                msg = f"Failed to localize {label} for timezone={tz} at rows={list(bad_idx[:50])}"
                                if uid is not None:
                                    msg += f" user_id={uid}"
                                raise ValueError(msg) from e2
                        # Return in original row order
                        return localized.reindex(s.index)

                    # Create timezone-aware local timestamps (if missing)
                    if "local_start_tz" not in group.columns:
                        group["local_start_tz"] = _localize_series(group["local_start"], "local_start")
                    if "local_end_tz" not in group.columns:
                        group["local_end_tz"] = _localize_series(group["local_end"], "local_end")

                    # Parse/normalize tz-aware columns
                    ls_tz = pd.to_datetime(group["local_start_tz"], errors="coerce")
                    le_tz = pd.to_datetime(group["local_end_tz"], errors="coerce")

                    # Create UTC timestamps (if missing)
                    if "utc_start" not in group.columns:
                        group["utc_start"] = ls_tz.dt.tz_convert("UTC")
                    if "utc_end" not in group.columns:
                        group["utc_end"] = le_tz.dt.tz_convert("UTC")
                    
                    # Ensure naive local timestamps are present/normalized
                    group["local_start"] = ls_tz.dt.tz_localize(None)
                    group["local_end"] = le_tz.dt.tz_localize(None)
                    return group
                
                localized_parts = []
                for tz, group in data.groupby("timezone", sort=False):
                    localized_parts.append(_localize_group(group, str(tz)))
                if localized_parts:
                    data = pd.concat(localized_parts, axis=0, sort=False).sort_index()

                if bool(dst_rep.get("fallback_used")) or int(dst_rep.get("nonexistent_detected_rows", 0)) > 0:
                    self.logger.warning(
                        "[csv:%s] DST localization policy applied: fallback_ambiguous_false_rows=%s, "
                        "nonexistent_detected_rows=%s, ambiguous_infer_fail_rows=%s, samples=%s",
                        path.name,
                        int(dst_rep.get("fallback_ambiguous_false_rows", 0)),
                        int(dst_rep.get("nonexistent_detected_rows", 0)),
                        int(dst_rep.get("ambiguous_infer_fail_rows", 0)),
                        dst_rep.get("fallback_samples", []),
                    )
        # ----------------------------------- END OF TZ HANDLING -----------------------------------

        # Validate required columns exist
        expected = ["user_id", "local_start", "local_end", "timezone", "type", "utc_start", "utc_end"]
        missing = [c for c in expected if c not in data.columns]
        if missing:
            raise ValueError(f"File missing expected columns {missing}: {path}")

        data = self._normalize_user_id_column(
            data,
            context=f"file:{path.name}",
            report=report,
            report_key="user_id",
        )

        # Add stable device code column if needed
        CODE_MAP = {"Apple Watch": 2, "iPhone": 1, "other": 0}
        if "device.name" in data.columns and "device.name_id" not in data.columns:
            data["device.name_id"] = data["device.name"].map(CODE_MAP).astype("Int8")

        self._bump_drop_reason(report, "timezone_missing_or_invalid", int(report["timezone"]["dropped_rows"]))
        self._bump_drop_reason(report, "user_id_non_numeric", int(report["user_id"]["rows_dropped_non_numeric"]))

        report["__global__"]["rows_out"] = int(len(data))
        report["__global__"]["rows_dropped_total"] = max(
            0,
            int(report["__global__"]["rows_in_raw"]) - int(report["__global__"]["rows_out"]),
        )

        self.logger.info(
            "Loaded file %s type=%s rows_in=%s rows_out=%s dropped=%s tz_filled=%s tz_dropped=%s "
            "uid_dropped=%s dst_fallback=%s dst_nonexistent_detected=%s",
            path,
            report["__global__"]["file_type"],
            report["__global__"]["rows_in_raw"],
            report["__global__"]["rows_out"],
            report["__global__"]["rows_dropped_total"],
            int(report["timezone"]["filled_default_rows"]),
            int(report["timezone"]["dropped_rows"]),
            int(report["user_id"]["rows_dropped_non_numeric"]),
            int(report["dst"]["fallback_ambiguous_false_rows"]),
            int(report["dst"]["nonexistent_detected_rows"]),
        )

        if return_report:
            return data, report
        return data


    def load_participant(
            self,
            data_folder: str | Path,
            user_id: int | str,
            sensor_folders: Iterable[str] | Iterable[Path] | None = None,
            return_report: bool = False,
        ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Load all HealthKit CSVs for a single participant using the naming pattern: <sensor_folder>/<user_id>-<sensor_folder.name>.csv
        Returns concatenated dataframe.
        If return_report=True, returns (dataframe, report).
        """
        # Resolve and validate data folder
        data_folder = Path(data_folder).expanduser().resolve()
        if not data_folder.exists():
            raise FileNotFoundError(f"Folder not found: {data_folder}")

        # Determine which sensor folders to search
        sensor_paths = ([f for f in data_folder.iterdir() if f.is_dir()] if sensor_folders is None 
                        else [Path(f) for f in sensor_folders])

        participant_report: Dict[str, Any] = {
            "__global__": {
                "data_folder": str(data_folder),
                "user_id": self._normalize_sample_value(user_id),
                "files_scanned": int(len(sensor_paths)),
                "files_found": 0,
                "files_missing": 0,
                "rows_out": 0,
            },
            "files": [],
            "missing_files": [],
        }

        # Load all existing HealthKit CSVs for this participant
        uid_int = int(user_id)
        hk_data_list: list[pd.DataFrame] = []
        for p in sensor_paths:
            fp = p / f"{uid_int}-{p.name}.csv"
            if not fp.exists():
                participant_report["missing_files"].append(str(fp))
                continue
            if return_report:
                loaded, file_report = self.load_file(fp, return_report=True)
                hk_data_list.append(loaded)
                participant_report["files"].append(file_report)
            else:
                hk_data_list.append(self.load_file(fp))

        participant_report["__global__"]["files_found"] = int(len(hk_data_list))
        participant_report["__global__"]["files_missing"] = int(len(participant_report["missing_files"]))

        # Handle case where no data was found
        if not hk_data_list:
            self.logger.info(
                "Loaded participant user_id=%s from %s files_found=%s files_missing=%s rows_out=0",
                user_id,
                data_folder,
                participant_report["__global__"]["files_found"],
                participant_report["__global__"]["files_missing"],
            )
            out = pd.DataFrame()
            if return_report:
                return out, participant_report
            return out
        
        # Concatenate all sensor data into a single DataFrame
        out = pd.concat(hk_data_list, ignore_index=True)
        participant_report["__global__"]["rows_out"] = int(len(out))
        self.logger.info(
            "Loaded participant user_id=%s from %s files_found=%s files_missing=%s rows_out=%s",
            user_id,
            data_folder,
            participant_report["__global__"]["files_found"],
            participant_report["__global__"]["files_missing"],
            participant_report["__global__"]["rows_out"],
        )
        if return_report:
            return out, participant_report
        return out

    def load_bulk(
        self,
        root: str | Path,
        *,
        patterns: Sequence[str] = ("**/*.csv",),
        strict: bool = False,
        return_report: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Recursively load many HealthKit CSV files under `root`.
        - `strict=True`: raise on first file error; else log and continue.
        Returns a single concatenated dataframe.
        If return_report=True, returns (dataframe, report).
        """
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Root not found: {root}")

        # Collect files that match any of the provided patterns.
        files = sorted({fp for pat in patterns for fp in root.rglob(pat)})
        bulk_report: Dict[str, Any] = {
            "__global__": {
                "root": str(root),
                "patterns": [str(p) for p in patterns],
                "files_matched": int(len(files)),
                "files_loaded": 0,
                "files_failed": 0,
                "rows_out": 0,
            },
            "files": [],
            "failed_files": [],
        }

        # Load each file independently so one bad file doesn't break the batch
        frames: list[pd.DataFrame] = []
        for fp in tqdm(files, desc="Loading HealthKit files"):
            try:
                if return_report:
                    loaded, file_report = self.load_file(fp, return_report=True)
                    frames.append(loaded)
                    bulk_report["files"].append(file_report)
                else:
                    frames.append(self.load_file(fp))
            except Exception as e:  # noqa: BLE001
                if strict:
                    raise
                self.logger.warning("Failed to load %s: %s", fp, e)
                bulk_report["failed_files"].append(
                    {
                        "path": str(fp),
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                )

        bulk_report["__global__"]["files_loaded"] = int(len(frames))
        bulk_report["__global__"]["files_failed"] = int(len(bulk_report["failed_files"]))
        
        # Nothing matched or everything failed
        if not frames:
            out = pd.DataFrame()
            if files:
                self.logger.info(
                    "Bulk load under %s matched=%s loaded=0 failed=%s rows_out=0",
                    root,
                    bulk_report["__global__"]["files_matched"],
                    bulk_report["__global__"]["files_failed"],
                )
            else:
                self.logger.info("No files matched patterns %s under %s", patterns, root)
            if return_report:
                return out, bulk_report
            return out
        
        # Combine all per-file DataFrames into one
        out = pd.concat(frames, ignore_index=True)
        bulk_report["__global__"]["rows_out"] = int(len(out))
        self.logger.info(
            "Bulk load under %s matched=%s loaded=%s failed=%s rows_out=%s",
            root,
            bulk_report["__global__"]["files_matched"],
            bulk_report["__global__"]["files_loaded"],
            bulk_report["__global__"]["files_failed"],
            bulk_report["__global__"]["rows_out"],
        )
        if return_report:
            return out, bulk_report
        return out

"""
Universal CSV / Excel Ingestion, Smart Column Auto-Mapping & Data Quality Gates.

Supports:
- Single CSV or Excel (.xlsx / .xlsm) file (die-level OR wafer-summary level)
- Multi-sheet Excel workbooks (e.g. sort_die_results + lot_context_mes + wat_pcm + test_limits + final_test_burnin)
- Multiple CSV/Excel files dropped together
- Never touches ground_truth/ files during inference.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REF_DATASET_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "AX7_SOC_synthetic_chip_test_dataset"
)

KNOWN_PARAMETRICS = [
    "CONTACT_RES",
    "IDDQ",
    "LEAKAGE",
    "SCAN_FAIL_COUNT",
    "MBIST_FAIL",
    "PLL_LOCK",
    "VMIN",
    "FMAX",
    "RING_OSC",
    "ADC_INL",
]

DEFAULT_BIN_NAMES = {
    1: "PASS",
    30: "CONTACT",
    12: "IDDQ",
    21: "LEAKAGE",
    3: "SCAN",
    9: "MBIST",
    15: "PLL_LOCK",
    6: "RING_OSC",
    7: "VMIN",
    5: "FMAX",
    18: "ADC_INL",
}

COLUMN_ALIASES = {
    "lot_id": ["lot_id", "lot", "lotname", "lot_name", "lotid", "lot_no", "batch_id"],
    "wafer_id": ["wafer_id", "wafer", "wafer_no", "waferid", "slot", "wafer_num"],
    "die_x": ["die_x", "x", "x_coord", "diex", "col", "x_pos"],
    "die_y": ["die_y", "y", "y_coord", "diey", "row", "y_pos"],
    "pass": ["pass", "is_pass", "pass_fail", "passed", "result", "status", "die_pass"],
    "soft_bin": ["soft_bin", "sbin", "bin", "fail_bin", "bin_code", "softbin"],
    "hard_bin": ["hard_bin", "hbin", "hardbin"],
    "insertion": ["insertion", "pass_num", "retest_flag", "run_num"],
    "site": ["site", "probe_site", "test_site", "site_id"],
    "tester_id": ["tester_id", "tester", "ate_id", "equipment"],
    "probe_card_id": ["probe_card_id", "probe_card", "card_id", "pc_id"],
    "program_rev": ["program_rev", "prog_rev", "test_program"],
    "temp_c": ["temp_c", "temperature", "temp"],
    "test_time_ms": ["test_time_ms", "test_time", "time_ms", "duration_ms"],
    "ts": ["ts", "timestamp", "sort_ts", "test_ts", "sort_start_ts", "datetime"],
    "final_yield": ["final_yield", "yield", "yield_pct", "pass_rate", "first_pass_yield"],
    "gross_die": ["gross_die", "total_die", "die_count", "tested_die"],
    "good_die": ["good_die", "pass_die", "passed_die"],
}


def _normalize_col_name(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def auto_map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename recognizable column aliases to canonical schema while preserving parametric columns."""
    df = df.copy()
    existing = set(df.columns)
    norm_to_orig = {_normalize_col_name(c): c for c in df.columns}
    rename_map: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in existing:
            continue
        for alias in aliases:
            if alias in norm_to_orig:
                orig = norm_to_orig[alias]
                if orig not in rename_map:
                    rename_map[orig] = canonical
                    break

    # Also match canonical uppercase parametric columns case-insensitively
    for p in KNOWN_PARAMETRICS:
        if p not in existing and p.lower() in norm_to_orig:
            orig = norm_to_orig[p.lower()]
            if orig not in rename_map:
                rename_map[orig] = p

    if rename_map:
        df = df.rename(columns=rename_map)

    # Normalize pass column values if string/boolean
    if "pass" in df.columns:
        if df["pass"].dtype == object or str(df["pass"].dtype) == "bool":
            s = df["pass"].astype(str).str.strip().str.lower()
            mapped = s.map(
                {
                    "1": 1, "1.0": 1, "true": 1, "pass": 1, "p": 1, "good": 1, "yes": 1, "ok": 1,
                    "0": 0, "0.0": 0, "false": 0, "fail": 0, "f": 0, "bad": 0, "no": 0,
                }
            )
            df["pass"] = pd.to_numeric(mapped, errors="coerce").fillna(0).astype(int)
        else:
            df["pass"] = (pd.to_numeric(df["pass"], errors="coerce").fillna(0) > 0).astype(int)
    elif "soft_bin" in df.columns:
        # Derive pass from soft_bin == 1 if pass column omitted
        sb = pd.to_numeric(df["soft_bin"], errors="coerce")
        df["pass"] = (sb == 1).astype(int)

    # Ensure lot_id & wafer_id exist cleanly if omitted on a single-wafer upload
    if "lot_id" not in df.columns and ("die_x" in df.columns or "final_yield" in df.columns):
        df["lot_id"] = "UPLOADED_LOT"
    if "wafer_id" not in df.columns and "die_x" in df.columns:
        df["wafer_id"] = 1

    if "insertion" not in df.columns and "die_x" in df.columns:
        df["insertion"] = 1

    return df, {v: k for k, v in rename_map.items()}


def _classify_table(df: pd.DataFrame, name_hint: str = "") -> str:
    cols = set(df.columns)
    norm_cols = {_normalize_col_name(c) for c in cols}
    hint = name_hint.lower()

    if "defect_source" in norm_cols or "is_latent_outlier" in norm_cols or "pattern_labels" in norm_cols:
        return "ignore_ground_truth"
    if "die_x" in norm_cols and "die_y" in norm_cols and ("ft_pass" in norm_cols or "burnin_fail" in norm_cols):
        return "final_test"
    if ("x" in norm_cols and "y" in norm_cols or "die_x" in norm_cols and "die_y" in norm_cols) and (
        "pass" in norm_cols or "soft_bin" in norm_cols or "bin" in norm_cols or "result" in norm_cols
    ):
        return "sort_die"
    if "equipment_id" in norm_cols and "step" in norm_cols:
        return "mes"
    if "vt_n_mv" in norm_cols or "wat_site" in norm_cols or "idsat_n_ua_um" in norm_cols:
        return "wat"
    if "test_name" in norm_cols and ("lo_limit" in norm_cols or "hi_limit" in norm_cols):
        return "limits"
    if "soft_bin" in norm_cols and "name" in norm_cols and len(norm_cols) <= 5:
        return "bins"
    if "final_yield" in norm_cols or "first_pass_yield" in norm_cols or "yield" in norm_cols or "yield_pct" in norm_cols:
        return "wafer_summary"
    if "wafer_count" in norm_cols and "lot_id" in norm_cols:
        return "lots"

    # Fallback by filename / sheet name hint
    if "sort" in hint or "die" in hint:
        return "sort_die"
    if "mes" in hint or "context" in hint:
        return "mes"
    if "wat" in hint or "pcm" in hint:
        return "wat"
    if "limit" in hint:
        return "limits"
    if "final" in hint or "burn" in hint:
        return "final_test"
    if "wafer" in hint:
        return "wafer_summary"
    if "lot" in hint:
        return "lots"
    return "unknown"


def parse_uploaded_files(
    files: list[tuple[str, bytes]],
    match_reference_context: bool = False,
) -> dict[str, Any]:
    """
    Parse one or more uploaded CSV / Excel files into canonical yield tables.
    Returns structured dict with tables, column_mappings, and data quality report.
    """
    tables: dict[str, list[pd.DataFrame]] = {
        "sort_die": [],
        "wafer_summary": [],
        "mes": [],
        "wat": [],
        "limits": [],
        "bins": [],
        "final_test": [],
        "lots": [],
    }
    file_summaries = []
    warnings = []

    for fname, raw in files:
        ext = Path(fname).suffix.lower()
        if ext in (".xlsx", ".xlsm", ".xls"):
            xls = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
            for sheet in xls.sheet_names:
                df_sheet = pd.read_excel(xls, sheet_name=sheet)
                if df_sheet.empty:
                    continue
                role = _classify_table(df_sheet, sheet if len(xls.sheet_names) > 1 else fname)
                if role == "ignore_ground_truth":
                    warnings.append(f"Skipped ground-truth sheet '{sheet}' in {fname} (kept strictly out of model inputs).")
                    continue
                if role in tables:
                    tables[role].append(df_sheet)
                    file_summaries.append({"file": f"{fname} [{sheet}]", "role": role, "rows": len(df_sheet)})
                else:
                    warnings.append(f"Sheet '{sheet}' in {fname} could not be matched to a known test schema ({len(df_sheet)} rows).")
        elif ext in (".csv", ".tsv", ".txt"):
            sep = "\t" if ext == ".tsv" else ","
            df_csv = pd.read_csv(io.BytesIO(raw), sep=sep)
            if df_csv.empty:
                continue
            role = _classify_table(df_csv, fname)
            if role == "ignore_ground_truth":
                warnings.append(f"Skipped ground-truth file '{fname}' (kept strictly out of model inputs).")
                continue
            if role in tables:
                tables[role].append(df_csv)
                file_summaries.append({"file": fname, "role": role, "rows": len(df_csv)})
            else:
                # Last-resort check: if it has lot/yield or numeric parametric columns
                df_mapped, _ = auto_map_columns(df_csv)
                if "die_x" in df_mapped.columns and "die_y" in df_mapped.columns:
                    tables["sort_die"].append(df_csv)
                    file_summaries.append({"file": fname, "role": "sort_die", "rows": len(df_csv)})
                elif "final_yield" in df_mapped.columns:
                    tables["wafer_summary"].append(df_csv)
                    file_summaries.append({"file": fname, "role": "wafer_summary", "rows": len(df_csv)})
                else:
                    warnings.append(f"Unrecognized CSV columns in '{fname}': {list(df_csv.columns[:8])}")
        else:
            warnings.append(f"Unsupported file format '{fname}'. Please upload CSV or Excel (.xlsx).")

    merged: dict[str, pd.DataFrame | None] = {}
    col_mappings: dict[str, dict[str, str]] = {}
    for role, dfs in tables.items():
        if not dfs:
            merged[role] = None
            continue
        combined = pd.concat(dfs, ignore_index=True)
        if role in ("sort_die", "wafer_summary", "lots"):
            combined, cmap = auto_map_columns(combined)
            if cmap:
                col_mappings[role] = cmap
        merged[role] = combined

    # Optional enrichment with reference non-ground-truth context when user requests it
    # and lot_ids match AX7-SOC dataset
    enriched_sources = []
    if match_reference_context and REF_DATASET_DIR.exists():
        uploaded_lots: set[str] = set()
        for r in ("sort_die", "wafer_summary"):
            if merged[r] is not None and "lot_id" in merged[r].columns:
                uploaded_lots.update(merged[r]["lot_id"].astype(str).unique())

        if uploaded_lots:
            ref_files = {
                "mes": "lot_context_mes.csv",
                "wat": "wat_pcm.csv",
                "limits": "test_limits.csv",
                "bins": "bin_definitions.csv",
                "lots": "lots.csv",
                "final_test": "final_test_burnin.csv",
            }
            for role, csv_name in ref_files.items():
                p = REF_DATASET_DIR / csv_name
                if merged[role] is None and p.exists():
                    ref_df = pd.read_csv(p)
                    if "lot_id" in ref_df.columns:
                        ref_df = ref_df[ref_df["lot_id"].astype(str).isin(uploaded_lots)].copy()
                    if not ref_df.empty:
                        merged[role] = ref_df
                        enriched_sources.append(csv_name)

    return build_canonical_bundle(merged, file_summaries, col_mappings, warnings, enriched_sources)


def load_reference_dataset(max_lots: int | None = None) -> dict[str, Any]:
    """
    Load the AX7-SOC dataset from disk if available locally, or fall back cleanly
    to the bundled multi-sheet demo Excel workbook in sample_uploads/ for cloud deploys.
    """
    if not REF_DATASET_DIR.exists():
        demo_xlsx = Path(__file__).resolve().parent.parent.parent / "sample_uploads" / "AX7_Complete_Demo_8_Lots.xlsx"
        if demo_xlsx.exists():
            return parse_uploaded_files([(demo_xlsx.name, demo_xlsx.read_bytes())], match_reference_context=False)
        raise FileNotFoundError(f"Reference dataset directory not found: {REF_DATASET_DIR}")

    lots_df = pd.read_csv(REF_DATASET_DIR / "lots.csv")
    selected_lots = None
    if max_lots is not None and max_lots < len(lots_df):
        selected_lots = set(lots_df["lot_id"].head(max_lots).astype(str))
        lots_df = lots_df[lots_df["lot_id"].astype(str).isin(selected_lots)].copy()

    def _read_and_filter(fname: str) -> pd.DataFrame | None:
        p = REF_DATASET_DIR / fname
        if not p.exists():
            return None

        read_kwargs: dict[str, Any] = {}
        if fname == "sort_die_results.csv":
            read_kwargs = {
                "usecols": [
                    "lot_id", "wafer_id", "die_x", "die_y", "site", "insertion",
                    "tester_id", "probe_card_id", "program_rev", "soft_bin", "pass",
                    "test_time_ms", *KNOWN_PARAMETRICS,
                ],
                "dtype": {
                    "lot_id": "string",
                    "wafer_id": "int16",
                    "die_x": "int16",
                    "die_y": "int16",
                    "site": "int8",
                    "insertion": "int8",
                    "tester_id": "category",
                    "probe_card_id": "category",
                    "program_rev": "category",
                    "soft_bin": "int16",
                    "pass": "int8",
                    "test_time_ms": "float32",
                    **{k: "float32" for k in KNOWN_PARAMETRICS},
                },
                "chunksize": 100_000,
            }
        elif fname == "final_test_burnin.csv":
            read_kwargs = {
                "usecols": ["lot_id", "wafer_id", "die_x", "die_y", "burnin_fail"],
                "dtype": {
                    "lot_id": "string",
                    "wafer_id": "int16",
                    "die_x": "int16",
                    "die_y": "int16",
                    "burnin_fail": "int8",
                },
                "chunksize": 100_000,
            }

        if "chunksize" in read_kwargs:
            chunks = []
            for chunk in pd.read_csv(p, **read_kwargs):
                if selected_lots is not None and "lot_id" in chunk.columns:
                    chunk = chunk[chunk["lot_id"].astype(str).isin(selected_lots)]
                if not chunk.empty:
                    chunks.append(chunk)
            if not chunks:
                return pd.DataFrame()
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(p)
            if selected_lots is not None and "lot_id" in df.columns:
                df = df[df["lot_id"].astype(str).isin(selected_lots)].copy()
            for col in df.select_dtypes(include=["float64"]).columns:
                df[col] = df[col].astype("float32")
            for col in df.select_dtypes(include=["int64"]).columns:
                df[col] = df[col].astype("int32")
        return df

    merged = {
        "sort_die": _read_and_filter("sort_die_results.csv"),
        "wafer_summary": _read_and_filter("wafer_summary.csv"),
        "mes": _read_and_filter("lot_context_mes.csv"),
        "wat": _read_and_filter("wat_pcm.csv"),
        "limits": _read_and_filter("test_limits.csv"),
        "bins": _read_and_filter("bin_definitions.csv"),
        "final_test": _read_and_filter("final_test_burnin.csv"),
        "lots": lots_df,
    }
    file_summaries = [
        {"file": "AX7_SOC Built-in Dataset", "role": k, "rows": len(v)}
        for k, v in merged.items()
        if v is not None
    ]
    return build_canonical_bundle(merged, file_summaries, {}, [], [])


def build_canonical_bundle(
    merged: dict[str, pd.DataFrame | None],
    file_summaries: list[dict[str, Any]],
    col_mappings: dict[str, dict[str, str]],
    warnings: list[str],
    enriched_sources: list[str],
) -> dict[str, Any]:
    """Run Data Quality Gates and derive canonical first-pass & final-disposition views."""
    sort_df = merged.get("sort_die")
    ws_df = merged.get("wafer_summary")
    limits_df = merged.get("limits")
    bins_df = merged.get("bins")

    if sort_df is None and ws_df is None:
        raise ValueError(
            "Could not find wafer sort die results or wafer yield summary in the uploaded file(s). "
            "Please upload a CSV or Excel file containing wafer/die test columns (e.g., lot_id, wafer_id, die_x, die_y, pass)."
        )

    # Bin name dictionary
    bin_names = dict(DEFAULT_BIN_NAMES)
    if bins_df is not None and "soft_bin" in bins_df.columns and "name" in bins_df.columns:
        for r in bins_df.itertuples():
            try:
                bin_names[int(r.soft_bin)] = str(r.name)
            except Exception:
                pass
    if limits_df is not None and "soft_bin" in limits_df.columns and "soft_bin_name" in limits_df.columns:
        for r in limits_df.itertuples():
            if pd.notna(r.soft_bin) and pd.notna(r.soft_bin_name):
                bin_names[int(r.soft_bin)] = str(r.soft_bin_name)

    quality_report: dict[str, Any] = {
        "files_loaded": file_summaries,
        "column_mappings": col_mappings,
        "enriched_context_files": enriched_sources,
        "warnings": warnings,
        "invalid_rows_dropped": 0,
        "retest_rows_detected": 0,
        "completeness_alerts": [],
    }

    first_pass_df = None
    final_disp_df = None
    parametric_cols: list[str] = []

    if sort_df is not None:
        sort_df["lot_id"] = sort_df["lot_id"].astype(str)
        sort_df["wafer_id"] = pd.to_numeric(sort_df["wafer_id"], errors="coerce").fillna(1).astype("int32")
        sort_df["insertion"] = pd.to_numeric(sort_df["insertion"], errors="coerce").fillna(1).astype("int8")

        if "die_x" in sort_df.columns and "die_y" in sort_df.columns:
            sort_df["die_x"] = pd.to_numeric(sort_df["die_x"], errors="coerce")
            sort_df["die_y"] = pd.to_numeric(sort_df["die_y"], errors="coerce")
            bad = sort_df["die_x"].isna() | sort_df["die_y"].isna()
            if bad.any():
                quality_report["invalid_rows_dropped"] = int(bad.sum())
                sort_df = sort_df[~bad].copy()
            sort_df["die_x"] = sort_df["die_x"].astype("int16")
            sort_df["die_y"] = sort_df["die_y"].astype("int16")

        if "soft_bin" not in sort_df.columns:
            sort_df["soft_bin"] = np.where(sort_df["pass"] == 1, 1, 99).astype("int16")
        else:
            sort_df["soft_bin"] = pd.to_numeric(sort_df["soft_bin"], errors="coerce").fillna(1).astype("int16")

        has_external_ts = (
            (merged.get("lots") is not None and "sort_start_ts" in merged["lots"].columns)
            or (ws_df is not None and "sort_start_ts" in ws_df.columns)
        )
        if not has_external_ts and "ts" in sort_df.columns:
            sort_df["_ts_parsed"] = pd.to_datetime(sort_df["ts"], errors="coerce")
        else:
            sort_df["_ts_parsed"] = pd.NaT

        # Identify parametric numeric columns present in sort_df
        excluded = {
            "lot_id", "wafer_id", "die_x", "die_y", "site", "stage", "insertion",
            "tester_id", "probe_card_id", "program_rev", "temp_c", "hard_bin",
            "soft_bin", "pass", "test_time_ms", "ts", "_ts_parsed",
        }
        for c in sort_df.columns:
            if c in excluded:
                continue
            if pd.api.types.is_numeric_dtype(sort_df[c]):
                parametric_cols.append(c)

        # Completeness check per probe site
        if "site" in sort_df.columns:
            site_counts = sort_df[sort_df["insertion"] == 1]["site"].value_counts()
            if len(site_counts) > 1:
                min_ratio = float(site_counts.min() / site_counts.max())
                if min_ratio < 0.75:
                    quality_report["completeness_alerts"].append(
                        f"Probe site sample imbalance detected (min/max site ratio = {min_ratio:.2f})."
                    )

        first_pass_df = sort_df[sort_df["insertion"] == 1].copy()
        if first_pass_df.empty:
            first_pass_df = sort_df.copy()

        retest_count = int((sort_df["insertion"] > 1).sum())
        quality_report["retest_rows_detected"] = retest_count

        # Final disposition view: highest insertion supersedes insertion 1 per die
        if "die_x" in sort_df.columns and "die_y" in sort_df.columns:
            final_disp_df = (
                sort_df.sort_values("insertion")
                .drop_duplicates(["lot_id", "wafer_id", "die_x", "die_y"], keep="last")
                .copy()
            )
        else:
            final_disp_df = sort_df.copy()

        # Build / reconcile wafer_summary
        ws_calc = (
            final_disp_df.groupby(["lot_id", "wafer_id"], as_index=False)
            .agg(gross_die=("pass", "size"), good_die=("pass", "sum"))
        )
        fp_calc = (
            first_pass_df.groupby(["lot_id", "wafer_id"], as_index=False)["pass"]
            .mean()
            .rename(columns={"pass": "first_pass_yield"})
        )
        ws_calc = ws_calc.merge(fp_calc, on=["lot_id", "wafer_id"], how="left")
        ws_calc["final_yield"] = (ws_calc["good_die"] / ws_calc["gross_die"]).round(4)
        ws_calc["first_pass_yield"] = ws_calc["first_pass_yield"].round(4)

        # Attach sort_start_ts per lot
        if merged.get("lots") is not None and "sort_start_ts" in merged["lots"].columns:
            lot_ts = merged["lots"][["lot_id", "sort_start_ts"]].copy()
            lot_ts["lot_id"] = lot_ts["lot_id"].astype(str)
            ws_calc = ws_calc.merge(lot_ts, on="lot_id", how="left")
        elif ws_df is not None and "sort_start_ts" in ws_df.columns:
            ws_ts = ws_df[["lot_id", "wafer_id", "sort_start_ts"]].copy()
            ws_ts["lot_id"] = ws_ts["lot_id"].astype(str)
            ws_calc = ws_calc.merge(ws_ts, on=["lot_id", "wafer_id"], how="left")
        elif sort_df["_ts_parsed"].notna().any():
            lot_ts = (
                first_pass_df.groupby("lot_id", as_index=False)["_ts_parsed"]
                .min()
                .rename(columns={"_ts_parsed": "sort_start_ts"})
            )
            ws_calc = ws_calc.merge(lot_ts, on="lot_id", how="left")
        else:
            ws_calc["sort_start_ts"] = pd.NaT

        ws_df = ws_calc
    else:
        # Wafer-summary only mode
        ws_df = ws_df.copy()
        ws_df["lot_id"] = ws_df["lot_id"].astype(str)
        if "wafer_id" not in ws_df.columns:
            ws_df["wafer_id"] = ws_df.groupby("lot_id").cumcount() + 1
        ws_df["final_yield"] = pd.to_numeric(ws_df["final_yield"], errors="coerce")
        if ws_df["final_yield"].max() > 1.5:
            ws_df["final_yield"] = ws_df["final_yield"] / 100.0
        if "first_pass_yield" not in ws_df.columns:
            ws_df["first_pass_yield"] = ws_df["final_yield"]
        if "gross_die" not in ws_df.columns:
            ws_df["gross_die"] = 500
        if "good_die" not in ws_df.columns:
            ws_df["good_die"] = (ws_df["final_yield"] * ws_df["gross_die"]).round().astype(int)

    quality_report["total_lots"] = int(ws_df["lot_id"].nunique())
    quality_report["total_wafers"] = int(len(ws_df))
    quality_report["total_sort_rows"] = int(len(sort_df)) if sort_df is not None else 0
    quality_report["mode"] = "full_die" if sort_df is not None else "wafer_summary"

    return {
        "sort_die": sort_df,
        "first_pass": first_pass_df,
        "final_disp": final_disp_df,
        "wafer_summary": ws_df,
        "mes": merged.get("mes"),
        "wat": merged.get("wat"),
        "limits": limits_df,
        "bins": bin_names,
        "final_test": merged.get("final_test"),
        "lots": merged.get("lots"),
        "parametric_cols": parametric_cols,
        "quality_report": quality_report,
    }

"""
End-to-End Yield Advisor Pipeline Orchestrator

Coordinates data ingestion, all 6 ML engines, Advisor scoring, and fast
in-memory session state for interactive wafer Grad-CAM & Q&A drill-downs.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.engines.advisor import answer_question, build_recommendations
from app.engines.excursion import analyze_excursions
from app.engines.outlier_screen import analyze_outliers
from app.engines.root_cause import analyze_root_causes
from app.engines.test_optimizer import analyze_test_time
from app.engines.wafer_pattern import WaferPatternPredictor
from app.engines.yield_forecast import forecast_lot_yield


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class SessionStore:
    """Bounded in-memory session cache (keeps last 6 uploaded analyses)."""

    def __init__(self, max_sessions: int = 6):
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max = max_sessions

    def put(self, session_id: str, payload: dict[str, Any]) -> None:
        if session_id in self._store:
            self._store.move_to_end(session_id)
        self._store[session_id] = payload
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    def get(self, session_id: str) -> dict[str, Any] | None:
        if session_id in self._store:
            self._store.move_to_end(session_id)
            return self._store[session_id]
        return None


SESSIONS = SessionStore(max_sessions=6)
_WAFER_PREDICTOR: WaferPatternPredictor | None = None


def get_wafer_predictor() -> WaferPatternPredictor:
    global _WAFER_PREDICTOR
    if _WAFER_PREDICTOR is None:
        _WAFER_PREDICTOR = WaferPatternPredictor(MODEL_DIR)
    return _WAFER_PREDICTOR


def run_analysis_pipeline(bundle: dict[str, Any]) -> dict[str, Any]:
    """Run all Yield Advisor ML engines on an ingested dataset bundle."""
    t_start = time.perf_counter()

    sort_df = bundle["sort_die"]
    fp_df = bundle["first_pass"]
    final_df = bundle["final_disp"]
    ws_df = bundle["wafer_summary"]
    mes_df = bundle["mes"]
    wat_df = bundle["wat"]
    limits_df = bundle["limits"]
    bin_names = bundle["bins"]
    ft_df = bundle["final_test"]
    parametric_cols = bundle["parametric_cols"]
    quality_report = bundle["quality_report"]

    predictor = get_wafer_predictor()

    # 1. Wafer Map Pattern AI (PyTorch CNN + 2D Periodicity)
    wafer_patterns_df = pd.DataFrame()
    wafer_cache: dict[tuple[str, Any], dict[str, Any]] = {}
    if fp_df is not None and not fp_df.empty and "die_x" in fp_df.columns and "die_y" in fp_df.columns:
        wafer_patterns_df, wafer_cache = predictor.predict_batch(fp_df)

    # 2. Excursion Detector (EWMA / CUSUM / Change-point / Isolation Forest)
    excursion_out = analyze_excursions(ws_df, final_df, bin_names, parametric_cols, wafer_patterns_df)

    # 3. Root Cause Engine (LightGBM + SHAP + Commonality + Probe-vs-Process)
    root_cause_out = analyze_root_causes(ws_df, sort_df, mes_df, wat_df, wafer_patterns_df, bin_names)

    # 4. Adaptive Limits & Hidden Risk Outlier Screening (DPAT / NNR / GDBN / IsoForest)
    outlier_out = analyze_outliers(fp_df, final_df, ft_df, parametric_cols)
    wafer_outlier_lookup = outlier_out.pop("wafer_outlier_lookup", {})

    # 5. Test Time Optimizer (Redundancy / Rule-of-Three / Reordering)
    test_opt_out = analyze_test_time(fp_df, limits_df, parametric_cols)

    # 6. Pre-Sort Yield Forecasting (Quantile LightGBM)
    forecast_out = forecast_lot_yield(ws_df, mes_df, wat_df)

    # Compute executive KPIs & Soft-Bin Pareto
    total_gross = int(ws_df["gross_die"].sum())
    total_good = int(ws_df["good_die"].sum())
    final_yield_pct = round(float(ws_df["final_yield"].mean()) * 100.0, 2)
    first_pass_pct = round(float(ws_df["first_pass_yield"].mean()) * 100.0, 2)
    retest_recovered_chips = 0
    if quality_report["retest_rows_detected"] > 0 and sort_df is not None:
        retest_recovered_chips = int((sort_df[sort_df["insertion"] > 1]["pass"] == 1).sum())

    pareto = []
    if final_df is not None and not final_df.empty and "soft_bin" in final_df.columns:
        fails = final_df[final_df["pass"] == 0]
        if not fails.empty:
            counts = fails["soft_bin"].value_counts()
            total_fails = int(counts.sum())
            cum = 0
            for sb, cnt in counts.head(10).items():
                cum += int(cnt)
                bname = bin_names.get(int(sb), f"SB{int(sb)}")
                pareto.append(
                    {
                        "soft_bin": int(sb),
                        "name": bname,
                        "fail_count": int(cnt),
                        "share_pct": round(int(cnt) / max(1, total_fails) * 100.0, 1),
                        "cum_pct": round(cum / max(1, total_fails) * 100.0, 1),
                    }
                )

    # Pattern breakdown counts
    pattern_counts: dict[str, int] = {}
    if not wafer_patterns_df.empty:
        pattern_counts = wafer_patterns_df["primary_pattern"].value_counts().to_dict()
        pattern_counts = {str(k): int(v) for k, v in pattern_counts.items()}

    wafer_patterns_summary = {
        "available": not wafer_patterns_df.empty,
        "counts": pattern_counts,
        "non_clean_wafers": sum(v for k, v in pattern_counts.items() if k != "Clean"),
    }

    summary = {
        "total_lots": int(ws_df["lot_id"].nunique()),
        "total_wafers": int(len(ws_df)),
        "total_chips": total_gross,
        "good_chips": total_good,
        "final_yield_pct": final_yield_pct,
        "first_pass_yield_pct": first_pass_pct,
        "retest_recovered_chips": retest_recovered_chips,
        "active_alerts_count": len(excursion_out.get("excursions", [])) + len(root_cause_out.get("probe_findings", [])),
        "flagged_risk_pct": outlier_out.get("flagged_pct", 0.0) if outlier_out.get("available") else None,
        "flagged_risk_count": outlier_out.get("total_flagged", 0) if outlier_out.get("available") else 0,
        "test_time_saved_pct": test_opt_out.get("pct_time_reduction", 0.0) if test_opt_out.get("available") else None,
        "test_time_saved_ms": test_opt_out.get("total_saved_ms", 0.0) if test_opt_out.get("available") else 0.0,
    }

    # 7. Ranked Action Recommendations
    actions = build_recommendations(
        summary,
        wafer_patterns_summary,
        excursion_out,
        root_cause_out,
        outlier_out,
        test_opt_out,
    )

    # Compact wafer table for browsing & filtering
    ws_view = ws_df.copy()
    ws_view["lot_id"] = ws_view["lot_id"].astype(str)
    ws_view["wafer_id"] = ws_view["wafer_id"].astype(int)
    if not wafer_patterns_df.empty:
        wp_cols = [
            "lot_id", "wafer_id", "primary_pattern", "pattern_labels",
            "pattern_confidence", "plain_pattern_name", "est_pattern_lost_dies",
        ]
        ws_view = ws_view.merge(wafer_patterns_df[wp_cols], on=["lot_id", "wafer_id"], how="left")
    else:
        ws_view["primary_pattern"] = "N/A"
        ws_view["pattern_labels"] = "N/A"
        ws_view["pattern_confidence"] = 1.0
        ws_view["plain_pattern_name"] = "Summary Only"
        ws_view["est_pattern_lost_dies"] = 0

    wafer_table = []
    for r in ws_view.itertuples():
        n_outliers = len(wafer_outlier_lookup.get((str(r.lot_id), int(r.wafer_id)), []))
        wafer_table.append(
            {
                "lot_id": str(r.lot_id),
                "wafer_id": int(r.wafer_id),
                "gross_die": int(r.gross_die),
                "good_die": int(r.good_die),
                "first_pass_pct": round(float(r.first_pass_yield) * 100.0, 1),
                "final_yield_pct": round(float(r.final_yield) * 100.0, 1),
                "primary_pattern": str(r.primary_pattern),
                "pattern_labels": str(r.pattern_labels),
                "pattern_confidence": round(float(r.pattern_confidence), 2),
                "plain_pattern_name": str(r.plain_pattern_name),
                "est_pattern_lost_dies": int(r.est_pattern_lost_dies) if pd.notna(r.est_pattern_lost_dies) else 0,
                "outlier_count": n_outliers,
            }
        )

    elapsed_sec = round(time.perf_counter() - t_start, 2)

    # Model transparency summary
    cnn_meta = predictor.metrics
    models_used = [
        {
            "engine": "Wafer Defect Pattern AI",
            "model": "PyTorch Multi-Label ResNet-CNN + 2D Stepper Field Periodicity",
            "status": "Active" if wafer_patterns_summary["available"] else "Skipped (summary-only upload)",
            "metric": f"Hold-out Macro F1: {cnn_meta.get('holdout_macro_f1', 'N/A')}" if cnn_meta else "Trained",
        },
        {
            "engine": "Early-Warning Trend Detector",
            "model": "EWMA + CUSUM + Change-Point Segmentation + Isolation Forest",
            "status": "Active",
            "metric": f"{len(excursion_out.get('excursions', []))} lot shifts, {len(excursion_out.get('anomalous_wafers', []))} outlier wafers flagged",
        },
        {
            "engine": "Root Cause Attribution",
            "model": "LightGBM Regressor + SHAP TreeExplainer + Fisher/Mann-Whitney FDR",
            "status": "Active",
            "metric": f"Model R2 = {root_cause_out.get('lgbm_r2', 'N/A')}" if root_cause_out.get("lgbm_r2") is not None else "Statistical Commonality",
        },
        {
            "engine": "Hidden Risk Outlier Screener",
            "model": "Nearest-Neighbor Residual (NNR) + DPAT + GDBN + Isolation Forest",
            "status": "Active" if outlier_out.get("available") else "Skipped (needs die data)",
            "metric": (
                f"Catches {outlier_out['burnin_validation']['burnin_recall_pct']}% burn-in fails"
                if outlier_out.get("burnin_validation")
                else f"{outlier_out.get('total_flagged', 0):,} risky passing chips flagged"
            ),
        },
        {
            "engine": "Test Time Optimizer",
            "model": "Unique-Fail Redundancy + Pearson Correlation + Rule-of-Three Bound",
            "status": "Active" if test_opt_out.get("available") else "Skipped (needs die parametrics)",
            "metric": f"{test_opt_out.get('pct_time_reduction', 0)}% time cut identified",
        },
        {
            "engine": "Pre-Sort Yield Forecaster",
            "model": "Quantile Gradient Boosting (LightGBM P10 / P50 / P90)",
            "status": "Active" if forecast_out.get("available") else "Skipped (needs WAT or multi-lot routing)",
            "metric": f"Out-of-sample MAE = {forecast_out.get('mae_yield_pts', 'N/A')} pts" if forecast_out.get("available") else "N/A",
        },
    ]

    session_id = uuid.uuid4().hex[:12]

    # Store raw lightweight lookup structures in SessionStore for fast wafer/Q&A requests
    fp_indexed = None
    if fp_df is not None and not fp_df.empty:
        fp_indexed = {k: g for k, g in fp_df.groupby(["lot_id", "wafer_id"], sort=False)}

    wp_lookup = {}
    if not wafer_patterns_df.empty:
        wp_lookup = {
            (str(r.lot_id), int(r.wafer_id)): r._asdict()
            for r in wafer_patterns_df.itertuples(index=False)
        }

    public_payload = {
        "session_id": session_id,
        "runtime_seconds": elapsed_sec,
        "summary": summary,
        "quality_report": quality_report,
        "actions": actions,
        "pareto": pareto,
        "wafer_patterns_summary": wafer_patterns_summary,
        "wafer_table": wafer_table,
        "excursion": excursion_out,
        "root_cause": root_cause_out,
        "outliers": outlier_out,
        "test_optimizer": test_opt_out,
        "yield_forecast": forecast_out,
        "models_used": models_used,
    }

    SESSIONS.put(
        session_id,
        {
            **public_payload,
            "_fp_by_wafer": fp_indexed,
            "_wafer_cache": wafer_cache,
            "_wp_lookup": wp_lookup,
            "_wafer_outliers": wafer_outlier_lookup,
            "_bin_names": bin_names,
            "_parametric_cols": parametric_cols,
        },
    )

    return public_payload


def get_wafer_map_detail(session_id: str, lot_id: str, wafer_id: int) -> dict[str, Any]:
    """Return full die-level wafer map, Grad-CAM attention, and outlier overlay for one wafer."""
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise KeyError("Session expired or not found. Please re-upload your dataset.")

    fp_by_wafer = sess.get("_fp_by_wafer")
    if fp_by_wafer is None or (lot_id, wafer_id) not in fp_by_wafer:
        raise KeyError(f"Die-level map not available for {lot_id} wafer {wafer_id}.")

    grp = fp_by_wafer[(lot_id, wafer_id)]
    dx = grp["die_x"].to_numpy(dtype=int)
    dy = grp["die_y"].to_numpy(dtype=int)

    predictor = get_wafer_predictor()
    wcache = sess["_wafer_cache"].get((lot_id, wafer_id))
    gradcam = (
        predictor.explain_wafer_gradcam(wcache, dx, dy)
        if wcache is not None
        else np.zeros(len(dx), dtype=float)
    )

    outliers_list = sess["_wafer_outliers"].get((lot_id, wafer_id), [])
    outlier_map = {(o["x"], o["y"]): o["reason"] for o in outliers_list}
    bin_names = sess["_bin_names"]
    p_cols = sess["_parametric_cols"]

    pass_arr = grp["pass"].to_numpy(dtype=int)
    sb_arr = grp["soft_bin"].to_numpy(dtype=int) if "soft_bin" in grp.columns else np.where(pass_arr == 1, 1, 99)
    site_arr = grp["site"].to_numpy(dtype=int) if "site" in grp.columns else np.ones(len(dx), dtype=int)
    param_arrays = {
        pc: grp[pc].to_numpy(dtype=float)
        for pc in p_cols[:6]
        if pc in grp.columns
    }

    dies = []
    for idx in range(len(dx)):
        x_val = int(dx[idx])
        y_val = int(dy[idx])
        p_flag = int(pass_arr[idx])
        sb_val = int(sb_arr[idx])
        params = {}
        for pc, arr in param_arrays.items():
            v = arr[idx]
            if np.isfinite(v):
                params[pc] = round(float(v), 3)

        dies.append(
            {
                "x": x_val,
                "y": y_val,
                "pass": p_flag,
                "soft_bin": sb_val,
                "bin_name": bin_names.get(sb_val, f"SB{sb_val}"),
                "site": int(site_arr[idx]),
                "gradcam": round(float(gradcam[idx]), 3),
                "outlier_reason": outlier_map.get((x_val, y_val)),
                "params": params,
            }
        )

    wp_info = sess["_wp_lookup"].get((lot_id, wafer_id), {})
    return {
        "lot_id": lot_id,
        "wafer_id": wafer_id,
        "bounds": {
            "xmin": int(dx.min()),
            "xmax": int(dx.max()),
            "ymin": int(dy.min()),
            "ymax": int(dy.max()),
        },
        "pattern": {
            "primary_pattern": wp_info.get("primary_pattern", "Clean"),
            "plain_pattern_name": wp_info.get("plain_pattern_name", "Normal"),
            "pattern_confidence": wp_info.get("pattern_confidence", 1.0),
            "cause_hint": wp_info.get("cause_hint", ""),
            "first_check": wp_info.get("first_check", ""),
            "est_pattern_lost_dies": wp_info.get("est_pattern_lost_dies", 0),
            "reticle_detail": wcache.get("reticle_detail") if wcache else None,
            "class_probs": wp_info.get("class_probs", {}),
        },
        "dies": dies,
    }


def ask_session_advisor(session_id: str, question: str) -> dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise KeyError("Session expired or not found.")
    return answer_question(question, sess)


def update_action_status(session_id: str, action_id: str, status: str) -> dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise KeyError("Session expired or not found.")
    for act in sess["actions"]:
        if act["id"] == action_id:
            act["status"] = status
            return act
    raise KeyError(f"Action {action_id} not found.")

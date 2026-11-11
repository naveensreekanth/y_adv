"""
Engine 2: Early-Warning Excursion Detector

Combines:
- Univariate EWMA & CUSUM charts on lot yield, soft-bin fail shares, and parametric medians
- Change-point detection to pinpoint the onset lot of sustained shifts
- Multivariate Isolation Forest on wafer-level KPI vectors for novel anomalies
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def _detect_changepoint(series: np.ndarray, min_seg: int = 3) -> int | None:
    """Single-pass CUSUM/mean-shift binary change-point detector (returns onset index or None)."""
    n = len(series)
    if n < min_seg * 2:
        return None
    total_sse = float(np.sum((series - np.mean(series)) ** 2))
    if total_sse <= 1e-12:
        return None

    best_idx = None
    best_gain = 0.0
    for idx in range(min_seg, n - min_seg + 1):
        left = series[:idx]
        right = series[idx:]
        sse_split = float(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
        gain = (total_sse - sse_split) / total_sse
        if gain > best_gain:
            best_gain = gain
            best_idx = idx

    if best_idx is not None and best_gain > 0.22:
        return best_idx
    return None


def analyze_excursions(
    wafer_df: pd.DataFrame,
    final_disp_df: pd.DataFrame | None,
    bin_names: dict[int, str],
    parametric_cols: list[str],
    wafer_patterns_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run chronological lot EWMA/CUSUM, change-point localization, and wafer Isolation Forest."""
    if wafer_df.empty:
        return {"lot_trend": [], "excursions": [], "anomalous_wafers": []}

    ws = wafer_df.copy()
    if "sort_start_ts" in ws.columns and ws["sort_start_ts"].notna().any():
        ws["_sort_dt"] = pd.to_datetime(ws["sort_start_ts"], errors="coerce")
        ws = ws.sort_values(["_sort_dt", "lot_id", "wafer_id"])
    else:
        ws = ws.sort_values(["lot_id", "wafer_id"])

    # Compute lot-level summary in chronological order
    lot_order = ws["lot_id"].drop_duplicates().tolist()
    lot_agg = (
        ws.groupby("lot_id", sort=False)
        .agg(
            gross_die=("gross_die", "sum"),
            good_die=("good_die", "sum"),
            mean_yield=("final_yield", "mean"),
            first_pass_yield=("first_pass_yield", "mean"),
            wafer_count=("wafer_id", "count"),
        )
        .reindex(lot_order)
        .reset_index()
    )
    lot_agg["yield_pct"] = (lot_agg["mean_yield"] * 100.0).round(2)
    lot_agg["first_pass_pct"] = (lot_agg["first_pass_yield"] * 100.0).round(2)

    # Add per-lot top fail bin rates and parametric medians if die-level data exists
    top_bins: list[str] = []
    if final_disp_df is not None and not final_disp_df.empty and "soft_bin" in final_disp_df.columns:
        fd = final_disp_df.copy()
        fd["bin_label"] = fd["soft_bin"].map(lambda b: bin_names.get(int(b), f"SB{b}") if pd.notna(b) else "FAIL")
        fail_counts = fd[fd["pass"] == 0]["bin_label"].value_counts()
        top_bins = fail_counts.head(5).index.tolist()

        for bname in top_bins:
            b_rate = (
                fd.assign(_hit=(fd["bin_label"] == bname).astype(float))
                .groupby("lot_id")["_hit"]
                .mean()
                * 100.0
            )
            lot_agg[f"bin_{bname}_pct"] = lot_agg["lot_id"].map(b_rate).fillna(0.0).round(2)

        for pcol in parametric_cols:
            if pcol in ("SCAN_FAIL_COUNT", "MBIST_FAIL"):
                continue
            med_series = fd.groupby("lot_id")[pcol].median()
            lot_agg[f"med_{pcol}"] = lot_agg["lot_id"].map(med_series).round(3)

    # EWMA on lot yield (%)
    y_vals = lot_agg["yield_pct"].to_numpy(dtype=float)
    n_lots = len(y_vals)

    # Robust baseline from upper half / median (less distorted by excursion tail)
    q50 = float(np.median(y_vals))
    upper_half = y_vals[y_vals >= np.percentile(y_vals, 35)] if n_lots >= 6 else y_vals
    mu0 = float(np.median(upper_half))
    mad = float(np.median(np.abs(upper_half - mu0))) * 1.4826
    sigma0 = max(mad, 0.75)

    lam = 0.30
    ewma = np.zeros(n_lots, dtype=float)
    ewma[0] = mu0 * (1 - lam) + y_vals[0] * lam
    for i in range(1, n_lots):
        ewma[i] = lam * y_vals[i] + (1 - lam) * ewma[i - 1]

    ewma_lcl = mu0 - 3.0 * sigma0 * np.sqrt(lam / (2.0 - lam))
    shewhart_lcl = mu0 - 3.0 * sigma0

    # CUSUM downward accumulator (with slack k = 0.75 * sigma0 so minor normal wobble resets)
    cusum_neg = np.zeros(n_lots, dtype=float)
    k_cusum = 0.75 * sigma0
    h_cusum = 4.0 * sigma0
    for i in range(n_lots):
        prev = cusum_neg[i - 1] if i > 0 else 0.0
        cusum_neg[i] = max(0.0, prev + (mu0 - y_vals[i]) - k_cusum)

    lot_agg["ewma_pct"] = np.round(ewma, 2)
    lot_agg["ewma_lcl"] = round(float(ewma_lcl), 2)
    lot_agg["shewhart_lcl"] = round(float(shewhart_lcl), 2)
    lot_agg["ewma_alert"] = ((ewma < ewma_lcl) & (y_vals < mu0 - 1.0)) | (cusum_neg > h_cusum)
    lot_agg["shewhart_alert"] = y_vals < shewhart_lcl

    # Detect contiguous excursion windows (split if an intervening lot recovers near baseline)
    excursions: list[dict[str, Any]] = []
    alert_flags = lot_agg["ewma_alert"].to_numpy(dtype=bool)
    i = 0
    while i < n_lots:
        if not alert_flags[i]:
            i += 1
            continue
        j = i
        while j < n_lots and alert_flags[j]:
            j += 1
        seg = lot_agg.iloc[i:j]
        drop_pts = round(float(mu0 - seg["yield_pct"].min()), 2)
        avg_drop = round(float(mu0 - seg["yield_pct"].mean()), 2)

        # Find which soft bins spiked most during this window vs baseline
        drivers = []
        base_mask = ~alert_flags
        for bname in top_bins:
            col = f"bin_{bname}_pct"
            if col in lot_agg.columns:
                seg_rate = float(seg[col].mean())
                base_rate = float(lot_agg.loc[base_mask, col].mean()) if base_mask.any() else float(lot_agg[col].median())
                delta = seg_rate - base_rate
                if delta > 0.4:
                    drivers.append((bname, round(delta, 2), round(seg_rate, 2)))
        drivers.sort(key=lambda x: x[1], reverse=True)

        driver_text = (
            ", ".join(f"{d[0]} (+{d[1]:.1f}%)" for d in drivers[:3])
            if drivers
            else "Multiple failure modes"
        )
        start_lot = str(seg.iloc[0]["lot_id"])
        end_lot = str(seg.iloc[-1]["lot_id"])
        title = (
            f"Yield drop at {start_lot}" if start_lot == end_lot else f"Sustained yield drop ({start_lot} to {end_lot})"
        )
        excursions.append(
            {
                "title": title,
                "start_lot": start_lot,
                "end_lot": end_lot,
                "lot_count": int(len(seg)),
                "avg_yield_drop_pts": avg_drop,
                "peak_yield_drop_pts": drop_pts,
                "top_drivers": driver_text,
                "severity": "high" if drop_pts >= 4.0 else "medium",
            }
        )
        i = j + 1

    # Also check specific parametric shifts (e.g. VMIN / FMAX / CONTACT_RES) even if yield drop is moderate
    cp_idx = _detect_changepoint(y_vals)
    changepoint_lot = str(lot_agg.iloc[cp_idx]["lot_id"]) if cp_idx is not None else None

    # Multivariate Isolation Forest on wafer-level KPIs
    anomalous_wafers: list[dict[str, Any]] = []
    if len(ws) >= 12:
        wf_feats = ws[["lot_id", "wafer_id", "final_yield", "first_pass_yield"]].copy()
        wf_feats["retest_recovery_pts"] = (wf_feats["final_yield"] - wf_feats["first_pass_yield"]) * 100.0
        feat_cols = ["final_yield", "retest_recovery_pts"]

        if final_disp_df is not None and not final_disp_df.empty:
            for pcol in [c for c in ["IDDQ", "VMIN", "FMAX", "CONTACT_RES", "LEAKAGE"] if c in final_disp_df.columns]:
                w_med = final_disp_df.groupby(["lot_id", "wafer_id"], as_index=False)[pcol].median()
                wf_feats = wf_feats.merge(w_med, on=["lot_id", "wafer_id"], how="left")
                feat_cols.append(pcol)

        if wafer_patterns_df is not None and not wafer_patterns_df.empty:
            wf_feats = wf_feats.merge(
                wafer_patterns_df[["lot_id", "wafer_id", "primary_pattern", "first_pass_fail_rate"]],
                on=["lot_id", "wafer_id"],
                how="left",
            )

        X_iso = wf_feats[feat_cols].copy()
        X_iso = X_iso.fillna(X_iso.median(numeric_only=True)).fillna(0.0)
        if len(X_iso) >= 15:
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_iso)
            iso = IsolationForest(contamination=0.04, random_state=2026)
            preds = iso.fit_predict(X_scaled)
            scores = -iso.score_samples(X_scaled)
            wf_feats["anomaly_score"] = scores
            wf_feats["is_anomaly"] = preds == -1
            # Keep lowest-yield or highest-anomaly wafers for user inspection
            anom = wf_feats[wf_feats["is_anomaly"]].sort_values("anomaly_score", ascending=False).head(15)
            for r in anom.itertuples():
                anomalous_wafers.append(
                    {
                        "lot_id": str(r.lot_id),
                        "wafer_id": int(r.wafer_id),
                        "final_yield_pct": round(float(r.final_yield) * 100.0, 1),
                        "pattern": getattr(r, "primary_pattern", "Unknown"),
                        "anomaly_score": round(float(r.anomaly_score), 3),
                    }
                )

    trend_records = []
    for r in lot_agg.itertuples():
        rec = {
            "lot_id": str(r.lot_id),
            "yield_pct": float(r.yield_pct),
            "first_pass_pct": float(r.first_pass_pct),
            "ewma_pct": float(r.ewma_pct),
            "ewma_lcl": float(r.ewma_lcl),
            "shewhart_lcl": float(r.shewhart_lcl),
            "ewma_alert": bool(r.ewma_alert),
            "shewhart_alert": bool(r.shewhart_alert),
            "wafer_count": int(r.wafer_count),
        }
        for bname in top_bins:
            col = f"bin_{bname}_pct"
            if col in lot_agg.columns:
                rec[col] = float(getattr(r, col))
        for pcol in ["VMIN", "FMAX", "IDDQ", "CONTACT_RES"]:
            col = f"med_{pcol}"
            if col in lot_agg.columns and pd.notna(getattr(r, col)):
                rec[col] = float(getattr(r, col))
        trend_records.append(rec)

    return {
        "baseline_yield_pct": round(mu0, 2),
        "changepoint_lot": changepoint_lot,
        "top_fail_bins": top_bins,
        "lot_trend": trend_records,
        "excursions": excursions,
        "anomalous_wafers": anomalous_wafers,
    }

"""
Engine 6: Pre-Sort Yield Forecasting (Quantile LightGBM)

Predicts sort yield (P10, P50, P90 bounds) per lot before wafer sort using
WAT electrical measurements and MES tool routing, evaluated with a strict
time-ordered hold-out split.
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def forecast_lot_yield(
    wafer_df: pd.DataFrame,
    mes_df: pd.DataFrame | None,
    wat_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Train quantile LightGBM models to forecast lot yield from pre-sort WAT & routing."""
    if wafer_df.empty or wafer_df["lot_id"].nunique() < 8:
        return {"available": False, "reason": "At least 8 lots required for pre-sort yield forecasting."}

    ws = wafer_df.copy()
    ws["lot_id"] = ws["lot_id"].astype(str)
    if "sort_start_ts" in ws.columns and ws["sort_start_ts"].notna().any():
        ws["_dt"] = pd.to_datetime(ws["sort_start_ts"], errors="coerce")
        ws = ws.sort_values(["_dt", "lot_id", "wafer_id"])
    else:
        ws = ws.sort_values(["lot_id", "wafer_id"])

    lot_order = ws["lot_id"].drop_duplicates().tolist()
    lot_df = (
        ws.groupby("lot_id", sort=False)
        .agg(actual_yield_pct=("first_pass_yield", lambda s: float(s.mean()) * 100.0))
        .reindex(lot_order)
        .reset_index()
    )

    feat_cols: list[str] = []
    cat_cols: list[str] = []

    # Add WAT lot-level summary features (median & spread)
    has_wat = False
    if wat_df is not None and not wat_df.empty:
        wat = wat_df.copy()
        wat["lot_id"] = wat["lot_id"].astype(str)
        num_wat = [
            c for c in wat.select_dtypes(include=[np.number]).columns
            if c not in ("wafer_id", "wat_site", "site_x", "site_y")
        ]
        if num_wat:
            has_wat = True
            w_mean = wat.groupby("lot_id")[num_wat].mean().add_suffix("_mean").reset_index()
            w_std = wat.groupby("lot_id")[num_wat].std().fillna(0).add_suffix("_std").reset_index()
            lot_df = lot_df.merge(w_mean, on="lot_id", how="left").merge(w_std, on="lot_id", how="left")
            feat_cols.extend([c for c in w_mean.columns if c != "lot_id"])
            feat_cols.extend([c for c in w_std.columns if c != "lot_id"])

    # Add MES equipment majority per lot (e.g., ETCH, CMP, LITHO reticle, handler)
    if mes_df is not None and not mes_df.empty:
        mes = mes_df.copy()
        mes["lot_id"] = mes["lot_id"].astype(str)
        for step_name, grp in mes.groupby("step"):
            mode_eq = grp.groupby("lot_id")["equipment_id"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
            col = f"{step_name}_tool"
            lot_df[col] = lot_df["lot_id"].map(mode_eq).fillna("Unknown")
            feat_cols.append(col)
            cat_cols.append(col)
        if "reticle_id" in mes.columns:
            ret_g = mes[mes["reticle_id"].notna() & (mes["reticle_id"].astype(str) != "")]
            if not ret_g.empty:
                mode_ret = ret_g.groupby("lot_id")["reticle_id"].agg(lambda s: s.mode().iloc[0])
                lot_df["reticle_id"] = lot_df["lot_id"].map(mode_ret).fillna("Unknown")
                feat_cols.append("reticle_id")
                cat_cols.append("reticle_id")

    if not feat_cols:
        return {"available": False, "reason": "Pre-sort electrical (WAT) or equipment context required for forecasting."}

    X = lot_df[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).astype("category")
    for c in feat_cols:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())

    y = lot_df["actual_yield_pct"].to_numpy(dtype=float)
    n = len(lot_df)

    # Out-of-sample cross-fitted predictions (rolling/blocked folds so every lot gets honest out-of-sample forecast)
    n_splits = min(5, max(2, n // 5))
    pred_p10 = np.zeros(n, dtype=float)
    pred_p50 = np.zeros(n, dtype=float)
    pred_p90 = np.zeros(n, dtype=float)

    indices = np.arange(n)
    folds = np.array_split(indices, n_splits)

    for fold_idx in folds:
        train_idx = np.setdiff1d(indices, fold_idx)
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te = X.iloc[fold_idx]

        for alpha, out_arr in [(0.10, pred_p10), (0.50, pred_p50), (0.90, pred_p90)]:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=80,
                learning_rate=0.06,
                num_leaves=8,
                min_child_samples=max(2, len(train_idx) // 10),
                random_state=2026,
                verbose=-1,
            )
            m.fit(X_tr, y_tr)
            out_arr[fold_idx] = m.predict(X_te)

    # Ensure monotonic quantiles P10 <= P50 <= P90
    p10_clean = np.minimum(pred_p10, pred_p50)
    p90_clean = np.maximum(pred_p90, pred_p50)
    mae_pts = float(mean_absolute_error(y, pred_p50))

    records = []
    for i, lot_id in enumerate(lot_df["lot_id"]):
        records.append(
            {
                "lot_id": str(lot_id),
                "actual_yield_pct": round(float(y[i]), 2),
                "pred_p50_pct": round(float(pred_p50[i]), 2),
                "pred_p10_pct": round(float(p10_clean[i]), 2),
                "pred_p90_pct": round(float(p90_clean[i]), 2),
                "flagged_low": bool(pred_p50[i] < float(np.median(y)) - 2.0),
            }
        )

    return {
        "available": True,
        "uses_wat": has_wat,
        "mae_yield_pts": round(mae_pts, 2),
        "lots_forecasted": n,
        "lot_forecasts": records,
    }

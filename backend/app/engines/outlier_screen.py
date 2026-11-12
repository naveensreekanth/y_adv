"""
Engine 4: Adaptive Limits & Hidden Risk Outlier Screening

Applies four complementary quality screens to passing chips:
1. Dynamic Part Average Testing (DPAT) per wafer using robust median & IQR sigma
2. Nearest Neighbor Residual (NNR) removing spatial gradients (e.g. radial IDDQ trend)
3. Good Die in Bad Neighborhood (GDBN) flagging passing dies surrounded by >=5 failing neighbors
4. Multivariate Isolation / Mahalanobis score across correlated passing parametrics

Validates against burn-in failures when final_test / burn-in data is available.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def analyze_outliers(
    first_pass_df: pd.DataFrame | None,
    final_disp_df: pd.DataFrame | None,
    final_test_df: pd.DataFrame | None,
    parametric_cols: list[str],
) -> dict[str, Any]:
    """Screen passing dies for latent reliability risk and summarize yield-vs-quality trade-off."""
    if final_disp_df is None or final_disp_df.empty:
        return {"available": False, "reason": "Die-level parametric data required for outlier screening."}

    keep = [c for c in ["lot_id", "wafer_id", "die_x", "die_y", "IDDQ", "LEAKAGE", "VMIN", "FMAX", "PLL_LOCK", "CONTACT_RES"] if c in final_disp_df.columns]
    passing = final_disp_df.loc[final_disp_df["pass"] == 1, keep].copy()
    if len(passing) < 30:
        return {"available": False, "reason": "Too few passing dies for statistical screening."}

    has_coords = "die_x" in passing.columns and "die_y" in passing.columns
    has_iddq = "IDDQ" in passing.columns

    # ---------------- 1. DPAT (Dynamic Part Average Testing on IDDQ & LEAKAGE) ----------------
    passing["flag_dpat"] = False
    dpat_cols = [c for c in ["IDDQ", "LEAKAGE", "VMIN", "PLL_LOCK"] if c in passing.columns]
    dpat_example_wafer = None

    if has_iddq:
        med = passing.groupby(["lot_id", "wafer_id"])["IDDQ"].transform("median")
        q75 = passing.groupby(["lot_id", "wafer_id"])["IDDQ"].transform(lambda s: np.nanpercentile(s, 75))
        q25 = passing.groupby(["lot_id", "wafer_id"])["IDDQ"].transform(lambda s: np.nanpercentile(s, 25))
        rsig = np.maximum((q75 - q25) / 1.35, 0.8)
        passing["dpat_z_iddq"] = (passing["IDDQ"] - med) / rsig
        passing["flag_dpat"] = passing["dpat_z_iddq"].abs() > 5.0

    # ---------------- 2. NNR (Nearest Neighbor Residual on IDDQ) & 3. GDBN ----------------
    passing["flag_nnr"] = False
    passing["nnr_residual_iddq"] = 0.0
    passing["flag_gdbn"] = False
    passing["neighbor_fail_count"] = 0

    if has_coords:
        xmin = int(final_disp_df["die_x"].min())
        xmax = int(final_disp_df["die_x"].max())
        ymin = int(final_disp_df["die_y"].min())
        ymax = int(final_disp_df["die_y"].max())
        W = xmax - xmin + 1
        H = ymax - ymin + 1

        # Build lookup for first-pass fails (for GDBN)
        fp_src = first_pass_df if first_pass_df is not None else final_disp_df
        kernel_nb = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)

        # Pre-index passing dataframe by (lot_id, wafer_id)
        fp_groups = {k: g for k, g in fp_src.groupby(["lot_id", "wafer_id"], sort=False)}
        pass_indices_by_wafer = passing.groupby(["lot_id", "wafer_id"], sort=False).groups

        nnr_res_arr = np.zeros(len(passing), dtype=np.float32)
        nb_fail_arr = np.zeros(len(passing), dtype=np.int32)
        pos_in_passing = pd.Series(np.arange(len(passing)), index=passing.index)

        shifts = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

        for wkey, p_idx in pass_indices_by_wafer.items():
            sub_p = passing.loc[p_idx]
            p_pos = pos_in_passing.loc[p_idx].to_numpy()
            px = (sub_p["die_x"].to_numpy(dtype=int) - xmin)
            py = (sub_p["die_y"].to_numpy(dtype=int) - ymin)

            # GDBN: 3x3 neighbor fail count from first-pass fail grid
            if wkey in fp_groups:
                fg = fp_groups[wkey]
                fail_grid = np.zeros((H, W), dtype=np.int32)
                fx = fg["die_x"].to_numpy(dtype=int) - xmin
                fy = fg["die_y"].to_numpy(dtype=int) - ymin
                ff = (fg["pass"].to_numpy(dtype=int) == 0).astype(np.int32)
                fail_grid[fy, fx] = ff
                nb_grid = convolve(fail_grid, kernel_nb, mode="constant", cval=0)
                nb_fail_arr[p_pos] = nb_grid[py, px]

            # NNR: IDDQ minus median of valid passing neighbors
            if has_iddq:
                val_grid = np.full((H, W), np.nan, dtype=np.float32)
                iddq_vals = sub_p["IDDQ"].to_numpy(dtype=np.float32)
                val_grid[py, px] = iddq_vals

                # Stack 8 shifted neighbor values and take nanmedian
                nb_vals = np.empty((len(shifts), len(px)), dtype=np.float32)
                for si, (sx, sy) in enumerate(shifts):
                    nx = px + sx
                    ny = py + sy
                    inb = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
                    arr = np.full(len(px), np.nan, dtype=np.float32)
                    arr[inb] = val_grid[ny[inb], nx[inb]]
                    nb_vals[si] = arr

                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        nb_med = np.nanmedian(nb_vals, axis=0)
                res = np.where(np.isfinite(nb_med), iddq_vals - nb_med, 0.0)
                nnr_res_arr[p_pos] = res

        passing["neighbor_fail_count"] = nb_fail_arr
        passing["flag_gdbn"] = passing["neighbor_fail_count"] >= 5
        passing["nnr_residual_iddq"] = nnr_res_arr

        if has_iddq:
            # Global robust sigma of neighbor residuals
            res_med = float(np.median(nnr_res_arr))
            q75_r, q25_r = np.percentile(nnr_res_arr, [75, 25])
            res_sig = max((q75_r - q25_r) / 1.35, 1.0)
            # Flag sharp positive IDDQ spikes relative to spatial neighbors (+5 sigma, typically > ~6.5 uA)
            nnr_thresh = max(5.8, res_med + 4.8 * res_sig)
            passing["flag_nnr"] = passing["nnr_residual_iddq"] > nnr_thresh

    # ---------------- 4. Multivariate Outlier Screen (Isolation Forest on passing die parametrics) ----------------
    mv_cols = [c for c in ["IDDQ", "LEAKAGE", "VMIN", "FMAX", "PLL_LOCK", "CONTACT_RES"] if c in passing.columns]
    passing["flag_mv"] = False
    if len(mv_cols) >= 3 and len(passing) >= 200:
        # Use residualized IDDQ when available so radial gradient doesn't trigger false alarms
        X_mv = passing[mv_cols].copy()
        if "nnr_residual_iddq" in passing.columns and has_iddq:
            X_mv["IDDQ"] = passing["nnr_residual_iddq"]
        X_mv = X_mv.fillna(X_mv.median())
        # Fit on a representative subsample for speed if huge, predict on all
        scaler = RobustScaler()
        fit_slice = X_mv.sample(n=min(50000, len(X_mv)), random_state=2026)
        scaler.fit(fit_slice)
        iso = IsolationForest(contamination=0.004, random_state=2026, n_jobs=-1)
        iso.fit(scaler.transform(fit_slice))
        preds_mv = iso.predict(scaler.transform(X_mv))
        passing["flag_mv"] = preds_mv == -1

    # Combine recommended latent risk screen (NNR + DPAT + GDBN)
    passing["flag_any_risk"] = passing["flag_nnr"] | passing["flag_dpat"] | passing["flag_gdbn"]

    n_pass = len(passing)
    n_dpat = int(passing["flag_dpat"].sum())
    n_nnr = int(passing["flag_nnr"].sum())
    n_gdbn = int(passing["flag_gdbn"].sum())
    n_mv = int(passing["flag_mv"].sum())
    n_total_flagged = int(passing["flag_any_risk"].sum())

    # Burn-in validation if final_test_burnin data is attached
    burnin_validation = None
    if final_test_df is not None and not final_test_df.empty and "burnin_fail" in final_test_df.columns:
        join_cols = [c for c in ["lot_id", "wafer_id", "die_x", "die_y"] if c in final_test_df.columns]
        if len(join_cols) == 4:
            ft_sub = final_test_df[join_cols + ["burnin_fail"]].copy()
            ft_sub["lot_id"] = ft_sub["lot_id"].astype(str)
            merged = passing.merge(ft_sub, on=join_cols, how="inner")
            if len(merged) > 0 and merged["burnin_fail"].sum() > 0:
                total_bi = int(merged["burnin_fail"].sum())
                unflagged = merged[~merged["flag_any_risk"]]
                nnr_grp = merged[merged["flag_nnr"]]
                gdbn_grp = merged[merged["flag_gdbn"]]
                any_grp = merged[merged["flag_any_risk"]]

                burnin_validation = {
                    "matched_chips": int(len(merged)),
                    "total_burnin_fails": total_bi,
                    "baseline_fail_rate_pct": round(float(unflagged["burnin_fail"].mean()) * 100.0, 3) if len(unflagged) else 0.0,
                    "nnr_flagged_count": int(len(nnr_grp)),
                    "nnr_fail_rate_pct": round(float(nnr_grp["burnin_fail"].mean()) * 100.0, 2) if len(nnr_grp) else 0.0,
                    "gdbn_flagged_count": int(len(gdbn_grp)),
                    "gdbn_fail_rate_pct": round(float(gdbn_grp["burnin_fail"].mean()) * 100.0, 2) if len(gdbn_grp) else 0.0,
                    "combined_flagged_count": int(len(any_grp)),
                    "combined_caught_fails": int(any_grp["burnin_fail"].sum()),
                    "burnin_recall_pct": round(int(any_grp["burnin_fail"].sum()) / max(1, total_bi) * 100.0, 1),
                    "yield_cost_pct": round(len(any_grp) / max(1, len(merged)) * 100.0, 2),
                }

    # Histogram of IDDQ values for clean visual distribution chart (comparing normal vs NNR outliers)
    iddq_hist = []
    if has_iddq:
        vals = passing["IDDQ"].dropna().to_numpy()
        lo, hi = float(np.percentile(vals, 0.2)), float(np.percentile(vals, 99.9))
        bins = np.linspace(lo, max(hi, 65.0), 28)
        c_norm, edges = np.histogram(passing.loc[~passing["flag_nnr"], "IDDQ"].dropna(), bins=bins)
        c_out, _ = np.histogram(passing.loc[passing["flag_nnr"], "IDDQ"].dropna(), bins=bins)
        for idx in range(len(c_norm)):
            iddq_hist.append(
                {
                    "bin_center": round(float((edges[idx] + edges[idx + 1]) / 2.0), 1),
                    "normal_count": int(c_norm[idx]),
                    "outlier_count": int(c_out[idx]),
                }
            )

    # Compact per-wafer map of flagged coordinates for wafer inspector overlay
    flagged_sub = passing[passing["flag_any_risk"]][
        ["lot_id", "wafer_id", "die_x", "die_y", "flag_nnr", "flag_dpat", "flag_gdbn", "nnr_residual_iddq"]
    ].copy()

    wafer_outlier_lookup: dict[tuple[str, int], list[dict[str, Any]]] = {}
    if has_coords and not flagged_sub.empty:
        for (l_id, w_id), g in flagged_sub.groupby(["lot_id", "wafer_id"], sort=False):
            wafer_outlier_lookup[(str(l_id), int(w_id))] = [
                {
                    "x": int(r.die_x),
                    "y": int(r.die_y),
                    "reason": (
                        "Neighbor Spike (NNR)" if r.flag_nnr
                        else ("Surrounded by Fails (GDBN)" if r.flag_gdbn else "Wafer Outlier (DPAT)")
                    ),
                }
                for r in g.itertuples()
            ]

    return {
        "available": True,
        "passing_chips_analyzed": n_pass,
        "total_flagged": n_total_flagged,
        "flagged_pct": round(n_total_flagged / max(1, n_pass) * 100.0, 2),
        "screens": [
            {
                "key": "nnr",
                "name": "Neighbor Spike Screen (NNR)",
                "plain_desc": "Compares each passing chip's standby power (IDDQ) against its immediate neighbors on the wafer.",
                "flagged_count": n_nnr,
                "flagged_pct": round(n_nnr / max(1, n_pass) * 100.0, 2),
            },
            {
                "key": "dpat",
                "name": "Adaptive Wafer Limits (DPAT)",
                "plain_desc": "Flags passing chips that sit far outside their own wafer's normal distribution.",
                "flagged_count": n_dpat,
                "flagged_pct": round(n_dpat / max(1, n_pass) * 100.0, 2),
            },
            {
                "key": "gdbn",
                "name": "Bad-Neighborhood Risk (GDBN)",
                "plain_desc": "Flags passing chips surrounded by 5 or more failing neighbors on the wafer.",
                "flagged_count": n_gdbn,
                "flagged_pct": round(n_gdbn / max(1, n_pass) * 100.0, 2),
            },
            {
                "key": "mv",
                "name": "Multivariate Anomaly (Isolation Forest)",
                "plain_desc": "Finds subtle unusual combinations across multiple tests at once.",
                "flagged_count": n_mv,
                "flagged_pct": round(n_mv / max(1, n_pass) * 100.0, 2),
            },
        ],
        "burnin_validation": burnin_validation,
        "iddq_distribution": iddq_hist,
        "wafer_outlier_lookup": wafer_outlier_lookup,
    }

"""
Engine 3: Root Cause Engine (LightGBM + SHAP + Statistical Commonality + Probe-vs-Process)

Combines:
- LightGBM regression predicting wafer yield loss from MES equipment, chambers,
  reticle, handler, tester, probe card, and WAT electrical measurements
- SHAP TreeExplainer attribution across wafers to rank real drivers
- Classical Fisher Exact & Mann-Whitney U commonality tests with Benjamini-Hochberg FDR
- Probe-card site-to-site & retest-recovery analysis (distinguishes probe wear from fab silicon loss)
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr


def _bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for rank_rev in range(m - 1, -1, -1):
        idx = order[rank_rev]
        val = pvals[idx] * m / (rank_rev + 1)
        prev = min(prev, val)
        adj[idx] = min(1.0, prev)
    return adj.tolist()


def _pivot_mes_context(mes_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot wafer x process-step MES table to one row per (lot_id, wafer_id)."""
    mes = mes_df.copy()
    mes["lot_id"] = mes["lot_id"].astype(str)
    mes["wafer_id"] = pd.to_numeric(mes["wafer_id"], errors="coerce").fillna(1).astype(int)

    base = mes[["lot_id", "wafer_id"]].drop_duplicates().copy()

    for step_name, grp in mes.groupby("step"):
        col_eq = f"{step_name}_tool"
        sub = grp.drop_duplicates(["lot_id", "wafer_id"])[["lot_id", "wafer_id", "equipment_id"]].rename(
            columns={"equipment_id": col_eq}
        )
        base = base.merge(sub, on=["lot_id", "wafer_id"], how="left")

        if "chamber" in grp.columns and grp["chamber"].notna().any():
            non_empty = grp["chamber"].astype(str).str.strip().replace({"nan": "", "None": ""})
            if (non_empty != "").any():
                g_ch = grp.drop_duplicates(["lot_id", "wafer_id"]).copy()
                g_ch[f"{step_name}_tool_chamber"] = (
                    g_ch["equipment_id"].astype(str) + " / Ch-" + g_ch["chamber"].astype(str)
                )
                base = base.merge(
                    g_ch[["lot_id", "wafer_id", f"{step_name}_tool_chamber"]],
                    on=["lot_id", "wafer_id"],
                    how="left",
                )

        if "reticle_id" in grp.columns and grp["reticle_id"].notna().any():
            non_empty = grp["reticle_id"].astype(str).str.strip().replace({"nan": "", "None": ""})
            if (non_empty != "").any():
                g_ret = grp.drop_duplicates(["lot_id", "wafer_id"])[["lot_id", "wafer_id", "reticle_id"]]
                base = base.merge(g_ret, on=["lot_id", "wafer_id"], how="left")

        if "ts_in" in grp.columns:
            g_ts = grp.drop_duplicates(["lot_id", "wafer_id"])[["lot_id", "wafer_id", "ts_in"]].rename(
                columns={"ts_in": f"{step_name}_ts"}
            )
            base = base.merge(g_ts, on=["lot_id", "wafer_id"], how="left")

    return base


def _analyze_probe_and_retest(
    sort_df: pd.DataFrame | None,
    bin_names: dict[int, str],
) -> dict[str, Any]:
    """Compare probe-site yield & contact resistance and quantify retest recovery."""
    if sort_df is None or sort_df.empty:
        return {"site_summary": [], "retest_summary": [], "probe_findings": []}

    fp = sort_df[sort_df["insertion"] == 1].copy()
    rt = sort_df[sort_df["insertion"] > 1].copy()

    site_summary = []
    probe_findings = []

    if "site" in fp.columns and "probe_card_id" in fp.columns:
        grp = (
            fp.groupby(["probe_card_id", "site"], as_index=False)
            .agg(
                die_count=("pass", "size"),
                yield_pct=("pass", lambda s: round(float(s.mean()) * 100.0, 2)),
                contact_res_mean=("CONTACT_RES", "mean") if "CONTACT_RES" in fp.columns else ("pass", "size"),
            )
        )
        if "CONTACT_RES" not in fp.columns:
            grp["contact_res_mean"] = np.nan

        for r in grp.itertuples():
            site_summary.append(
                {
                    "probe_card_id": str(r.probe_card_id),
                    "site": int(r.site),
                    "die_count": int(r.die_count),
                    "yield_pct": float(r.yield_pct),
                    "contact_res_mean": round(float(r.contact_res_mean), 3) if pd.notna(r.contact_res_mean) else None,
                }
            )

        # Detect any (probe_card, site) significantly lagging sibling sites on the same card
        for pc, pc_grp in grp.groupby("probe_card_id"):
            if len(pc_grp) < 2:
                continue
            med_y = float(pc_grp["yield_pct"].median())
            worst = pc_grp.sort_values("yield_pct").iloc[0]
            gap = med_y - float(worst["yield_pct"])
            if gap >= 0.8:
                # Check trend over time / wafers if CONTACT_RES exists
                extra = ""
                if "CONTACT_RES" in fp.columns:
                    sub_s = fp[(fp["probe_card_id"] == pc) & (fp["site"] == worst["site"])]
                    other_s = fp[(fp["probe_card_id"] == pc) & (fp["site"] != worst["site"])]
                    c_diff = float(sub_s["CONTACT_RES"].median() - other_s["CONTACT_RES"].median())
                    extra = f"Contact resistance is +{c_diff:.2f} ohm higher than sibling sites. "

                # Check retest recovery for this probe card / site
                retest_msg = ""
                if not rt.empty:
                    rt_pc = rt[(rt["probe_card_id"] == pc) & (rt["site"] == worst["site"])]
                    if len(rt_pc) >= 10:
                        rec_rate = float(rt_pc["pass"].mean()) * 100.0
                        retest_msg = f"{rec_rate:.0f}% of these fails pass when retested."

                probe_findings.append(
                    {
                        "probe_card_id": str(pc),
                        "site": int(worst["site"]),
                        "yield_gap_pts": round(gap, 2),
                        "site_yield_pct": float(worst["yield_pct"]),
                        "other_sites_median_pct": round(med_y, 2),
                        "evidence": (extra + retest_msg).strip() or "Site yield lags sibling sites on same probe card.",
                    }
                )

    retest_summary = []
    if not rt.empty and "soft_bin" in fp.columns:
        # Match retest rows back to insertion 1 soft_bin
        join_keys = [c for c in ["lot_id", "wafer_id", "die_x", "die_y"] if c in fp.columns and c in rt.columns]
        if len(join_keys) == 4:
            fp_bins = fp[join_keys + ["soft_bin"]].rename(columns={"soft_bin": "orig_soft_bin"})
            rt_m = rt.merge(fp_bins, on=join_keys, how="left")
            for sb, g in rt_m.groupby("orig_soft_bin"):
                if pd.isna(sb):
                    continue
                n_rt = len(g)
                recovered = int((g["pass"] == 1).sum())
                retest_summary.append(
                    {
                        "soft_bin": int(sb),
                        "bin_name": bin_names.get(int(sb), f"SB{int(sb)}"),
                        "retested_dies": n_rt,
                        "recovered_dies": recovered,
                        "recovery_rate_pct": round(recovered / max(1, n_rt) * 100.0, 1),
                    }
                )

    return {
        "site_summary": site_summary,
        "retest_summary": retest_summary,
        "probe_findings": probe_findings,
    }


def analyze_root_causes(
    wafer_df: pd.DataFrame,
    sort_df: pd.DataFrame | None,
    mes_df: pd.DataFrame | None,
    wat_df: pd.DataFrame | None,
    wafer_patterns_df: pd.DataFrame | None,
    bin_names: dict[int, str],
) -> dict[str, Any]:
    """
    Build a unified wafer-level feature table, run LightGBM + SHAP, run Fisher/Mann-Whitney
    commonality tests (including temporal post-shift interactions), and analyze probe-card sites.
    """
    ws = wafer_df.copy()
    ws["lot_id"] = ws["lot_id"].astype(str)
    ws["wafer_id"] = pd.to_numeric(ws["wafer_id"], errors="coerce").fillna(1).astype(int)
    ws["yield_loss_pts"] = (1.0 - ws["first_pass_yield"]) * 100.0

    # Attach tester / probe_card from sort_df if available
    if sort_df is not None and not sort_df.empty:
        ctx_cols = [c for c in ["tester_id", "probe_card_id", "program_rev"] if c in sort_df.columns]
        if ctx_cols:
            w_ctx = sort_df.drop_duplicates(["lot_id", "wafer_id"])[["lot_id", "wafer_id"] + ctx_cols].copy()
            w_ctx["lot_id"] = w_ctx["lot_id"].astype(str)
            w_ctx["wafer_id"] = w_ctx["wafer_id"].astype(int)
            ws = ws.merge(w_ctx, on=["lot_id", "wafer_id"], how="left")

    # Merge MES context
    mes_ts_cols: list[str] = []
    if mes_df is not None and not mes_df.empty:
        mes_piv = _pivot_mes_context(mes_df)
        mes_ts_cols = [c for c in mes_piv.columns if c.endswith("_ts")]
        ws = ws.merge(mes_piv, on=["lot_id", "wafer_id"], how="left")

    # Merge WAT/PCM wafer averages
    wat_cols: list[str] = []
    if wat_df is not None and not wat_df.empty:
        wat = wat_df.copy()
        wat["lot_id"] = wat["lot_id"].astype(str)
        wat["wafer_id"] = pd.to_numeric(wat["wafer_id"], errors="coerce").fillna(1).astype(int)
        num_wat = [
            c for c in wat.select_dtypes(include=[np.number]).columns
            if c not in ("wafer_id", "wat_site", "site_x", "site_y")
        ]
        if num_wat:
            wat_agg = wat.groupby(["lot_id", "wafer_id"], as_index=False)[num_wat].mean()
            wat_cols = num_wat
            ws = ws.merge(wat_agg, on=["lot_id", "wafer_id"], how="left")

    # Merge Wafer Pattern AI predictions
    if wafer_patterns_df is not None and not wafer_patterns_df.empty:
        wp_sub = wafer_patterns_df[["lot_id", "wafer_id", "primary_pattern", "pattern_labels"]].copy()
        wp_sub["lot_id"] = wp_sub["lot_id"].astype(str)
        wp_sub["wafer_id"] = wp_sub["wafer_id"].astype(int)
        ws = ws.merge(wp_sub, on=["lot_id", "wafer_id"], how="left")
    else:
        ws["primary_pattern"] = "Clean"
        ws["pattern_labels"] = "Clean"

    # ---------------- 1. LightGBM + SHAP Model Attribution ----------------
    ignore_cols = {
        "lot_id", "wafer_id", "gross_die", "good_die", "first_pass_yield",
        "final_yield", "yield_loss_pts", "sort_start_ts", "primary_pattern", "pattern_labels",
    } | set(mes_ts_cols)

    feature_cols = [c for c in ws.columns if c not in ignore_cols]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(ws[c])]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(ws[c])]

    shap_drivers: list[dict[str, Any]] = []
    model_r2 = None

    if len(ws) >= 20 and len(feature_cols) >= 1:
        X_lgb = ws[feature_cols].copy()
        for c in cat_cols:
            X_lgb[c] = X_lgb[c].astype(str).fillna("Unknown").astype("category")
        for c in num_cols:
            X_lgb[c] = pd.to_numeric(X_lgb[c], errors="coerce").fillna(X_lgb[c].median())

        # Add relative process time feature if any timestamp exists so post-PM / drift shifts are learnable
        if "sort_start_ts" in ws.columns and ws["sort_start_ts"].notna().any():
            dt = pd.to_datetime(ws["sort_start_ts"], errors="coerce")
            if dt.notna().any():
                X_lgb["timeline_day"] = (dt - dt.min()).dt.total_seconds() / 86400.0
                num_cols.append("timeline_day")

        y_loss = ws["yield_loss_pts"].to_numpy(dtype=float)
        model = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.06,
            num_leaves=15,
            min_child_samples=max(5, len(ws) // 60),
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=2026,
            verbose=-1,
        )
        model.fit(X_lgb, y_loss)
        preds = model.predict(X_lgb)
        ss_res = float(np.sum((y_loss - preds) ** 2))
        ss_tot = float(np.sum((y_loss - np.mean(y_loss)) ** 2))
        model_r2 = round(1.0 - ss_res / ss_tot, 3) if ss_tot > 1e-9 else 0.0

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_lgb)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        total_shap = float(mean_abs_shap.sum()) or 1.0
        for idx, col in enumerate(X_lgb.columns):
            imp_pts = float(mean_abs_shap[idx])
            if imp_pts <= 0.01:
                continue
            # Describe what value caused the highest positive yield loss SHAP
            col_shap = shap_vals[:, idx]
            top_detail = ""
            if col in cat_cols:
                tmp = pd.DataFrame({"val": ws[col].astype(str), "shap": col_shap, "loss": y_loss})
                grp_s = tmp.groupby("val", as_index=False).agg(mean_shap=("shap", "mean"), n=("loss", "size"), mean_loss=("loss", "mean"))
                grp_s = grp_s[grp_s["n"] >= 3].sort_values("mean_shap", ascending=False)
                if not grp_s.empty:
                    top_val = grp_s.iloc[0]
                    top_detail = f"Worst group: {top_val['val']} (avg yield {100.0 - top_val['mean_loss']:.1f}% across {int(top_val['n'])} wafers)"
            else:
                corr, _ = spearmanr(X_lgb[col], col_shap)
                direction = "higher" if (pd.notna(corr) and corr > 0) else "lower"
                top_detail = f"{direction.capitalize()} values correlate with higher yield loss"

            pretty_name = (
                col.replace("_tool_chamber", " Tool & Chamber")
                .replace("_tool", " Equipment")
                .replace("reticle_id", "Lithography Reticle")
                .replace("probe_card_id", "Probe Card")
                .replace("tester_id", "Tester")
                .replace("vt_n_mV", "WAT Transistor Threshold (Vt_n)")
                .replace("idsat_n_uA_um", "WAT Drive Current (Idsat)")
                .replace("timeline_day", "Time / Process Window")
            )
            shap_drivers.append(
                {
                    "feature": col,
                    "display_name": pretty_name,
                    "mean_abs_shap_pts": round(imp_pts, 3),
                    "share_pct": round(imp_pts / total_shap * 100.0, 1),
                    "detail": top_detail,
                }
            )
        shap_drivers.sort(key=lambda d: d["mean_abs_shap_pts"], reverse=True)

    # ---------------- 2. Statistical Commonality Tests (Equipment, Chamber, Reticle, Time-Split) ----------------
    # Identify affected wafers: either low yield (< median - 2*MAD) OR specific defect pattern
    med_y = float(ws["first_pass_yield"].median())
    mad_y = max(0.015, float(np.median(np.abs(ws["first_pass_yield"] - med_y))) * 1.4826)
    low_yield_flag = ws["first_pass_yield"] < (med_y - 2.2 * mad_y)

    commonality_candidates = []
    pvals_raw = []

    # Candidate categorical columns to test
    test_cats = [c for c in cat_cols if c != "program_rev"]
    for c in test_cats:
        vals = ws[c].astype(str)
        for val in vals.unique():
            if val in ("nan", "None", "Unknown", ""):
                continue
            in_grp = vals == val
            n_in = int(in_grp.sum())
            n_out = int((~in_grp).sum())
            if n_in < 3 or n_out < 5:
                continue

            # Test A: Continuous yield loss shift (Mann-Whitney U)
            loss_in = ws.loc[in_grp, "yield_loss_pts"]
            loss_out = ws.loc[~in_grp, "yield_loss_pts"]
            excess_loss = float(loss_in.mean() - loss_out.mean())
            if excess_loss > 0.6:
                _, p_mw = mannwhitneyu(loss_in, loss_out, alternative="greater")
            else:
                p_mw = 1.0

            # Test B: Specific defect pattern enrichment (Fisher Exact)
            best_pat = None
            best_pat_p = 1.0
            best_pat_rate = 0.0
            best_pat_base = 0.0
            for pat in ["Edge-Ring", "Center", "Scratch", "Loc", "Reticle-Repeat"]:
                has_pat = ws["pattern_labels"].astype(str).str.contains(pat, regex=False)
                a = int((in_grp & has_pat).sum())
                b = int((in_grp & ~has_pat).sum())
                c_cnt = int((~in_grp & has_pat).sum())
                d_cnt = int((~in_grp & ~has_pat).sum())
                if a >= 3 and (a / n_in) > (c_cnt / max(1, n_out)) + 0.08:
                    _, p_fe = fisher_exact([[a, b], [c_cnt, d_cnt]], alternative="greater")
                    if p_fe < best_pat_p:
                        best_pat_p = float(p_fe)
                        best_pat = pat
                        best_pat_rate = a / n_in
                        best_pat_base = c_cnt / max(1, n_out)

            # Test C: Temporal onset check inside a group WITH PEER-IN-SAME-TIME-WINDOW CONTROL
            # Prevents time confounding (e.g. a normal tool running late in the timeline when Vt shifted)
            post_shift_info = None
            best_p = min(float(p_mw), float(best_pat_p))
            ts_col = None
            for cand_ts in [f"{c.split('_')[0]}_ts", "sort_start_ts"]:
                if cand_ts in ws.columns and ws[cand_ts].notna().any():
                    ts_col = cand_ts
                    break

            if ts_col is not None and n_in >= 10:
                sub_df = ws.loc[in_grp].copy()
                sub_df["_dt"] = pd.to_datetime(sub_df[ts_col], errors="coerce")
                all_dt = pd.to_datetime(ws[ts_col], errors="coerce")
                sub_df = sub_df.dropna(subset=["_dt"]).sort_values("_dt")
                if len(sub_df) >= 10:
                    best_post_p = 1.0
                    for split_idx in range(3, len(sub_df) - 2):
                        post = sub_df.iloc[split_idx:]
                        pre = sub_df.iloc[:split_idx]
                        if len(post) < 3:
                            continue
                        post_excess = float(post["yield_loss_pts"].mean() - pre["yield_loss_pts"].mean())
                        if post_excess > 6.0:
                            cutoff_dt = post["_dt"].iloc[0]
                            # Causal guardrail: compare against PEER tools at the same process step AFTER cutoff_dt
                            peer_post = ws.loc[(~in_grp) & (all_dt >= cutoff_dt), "yield_loss_pts"]
                            if len(peer_post) >= 5:
                                peer_excess = float(post["yield_loss_pts"].mean() - peer_post.mean())
                                if peer_excess < 4.0:
                                    continue
                                _, p_peer = mannwhitneyu(post["yield_loss_pts"], peer_post, alternative="greater")
                                if p_peer >= 0.005:
                                    continue
                            _, p_post = mannwhitneyu(post["yield_loss_pts"], pre["yield_loss_pts"], alternative="greater")
                            if p_post < best_post_p:
                                best_post_p = float(p_post)
                                post_date = str(cutoff_dt.strftime("%b %d"))
                                post_pats = post["primary_pattern"].value_counts()
                                dom_pat = str(post_pats.index[0]) if not post_pats.empty else "Defects"
                                post_shift_info = {
                                    "date": post_date,
                                    "wafers": int(len(post)),
                                    "yield_pct": round(float(100.0 - post["yield_loss_pts"].mean()), 1),
                                    "pre_yield_pct": round(float(100.0 - pre["yield_loss_pts"].mean()), 1),
                                    "dominant_pattern": dom_pat,
                                    "p_value": float(p_post),
                                }

            if best_p < 0.01 or (post_shift_info is not None and post_shift_info["p_value"] < 0.005):
                if post_shift_info is not None and post_shift_info["p_value"] < best_p:
                    eff_p = post_shift_info["p_value"]
                    loss_est = round(
                        (post_shift_info["pre_yield_pct"] - post_shift_info["yield_pct"])
                        * post_shift_info["wafers"] / max(1, len(ws)),
                        2,
                    )
                    ev = (
                        f"Yield dropped to {post_shift_info['yield_pct']}% (vs {post_shift_info['pre_yield_pct']}% earlier & peer tools) "
                        f"across {post_shift_info['wafers']} wafers after {post_shift_info['date']} "
                        f"({post_shift_info['dominant_pattern']} signature)."
                    )
                    wafers_aff = post_shift_info["wafers"]
                    factor_label = f"{val} (after {post_shift_info['date']})"
                    tag = post_shift_info["dominant_pattern"]
                    explained_mask = in_grp & (pd.to_datetime(ws[ts_col], errors="coerce") >= pd.to_datetime(sub_df[sub_df["_dt"] >= pd.to_datetime(post_shift_info["date"] + " 2026", errors="coerce")]["_dt"].min()) if False else in_grp)
                    # Build exact mask for post-date wafers
                    cutoff_exact = sub_df.loc[sub_df["_dt"].dt.strftime("%b %d") == post_shift_info["date"], "_dt"].min()
                    explained_mask = in_grp & (all_dt >= cutoff_exact)
                elif best_pat is not None and best_pat_p <= p_mw:
                    eff_p = best_pat_p
                    loss_est = round(max(0.1, excess_loss * n_in / max(1, len(ws))), 2)
                    ev = (
                        f"{best_pat} defect rate is {best_pat_rate*100:.0f}% vs {best_pat_base*100:.0f}% on other tools "
                        f"({n_in} wafers checked)."
                    )
                    wafers_aff = int((in_grp & ws["pattern_labels"].astype(str).str.contains(best_pat, regex=False)).sum())
                    factor_label = val
                    tag = best_pat
                    explained_mask = in_grp & ws["pattern_labels"].astype(str).str.contains(best_pat, regex=False)
                else:
                    eff_p = float(p_mw)
                    loss_est = round(max(0.1, excess_loss * n_in / max(1, len(ws))), 2)
                    ev = (
                        f"Average yield is {100.0 - float(loss_in.mean()):.1f}% vs {100.0 - float(loss_out.mean()):.1f}% elsewhere "
                        f"across {n_in} wafers."
                    )
                    wafers_aff = n_in
                    factor_label = val
                    tag = "Yield Loss"
                    explained_mask = in_grp

                pvals_raw.append(eff_p)
                commonality_candidates.append(
                    {
                        "column": c,
                        "factor": factor_label,
                        "raw_value": val,
                        "signature": tag,
                        "wafers_affected": wafers_aff,
                        "est_dataset_yield_impact_pts": loss_est,
                        "evidence": ev,
                        "p_value": eff_p,
                        "_mask": explained_mask.to_numpy(dtype=bool),
                    }
                )

    # Apply Benjamini-Hochberg FDR + Confounder Residual Deduplication
    ranked_causes: list[dict[str, Any]] = []
    if commonality_candidates:
        adj_p = _bh_fdr(pvals_raw)
        valid_cands = []
        for cand, ap in zip(commonality_candidates, adj_p):
            if ap < 0.01:
                cand["adj_p_value"] = float(f"{ap:.2e}")
                cand["confidence"] = "High" if ap < 1e-4 else "Medium"
                valid_cands.append(cand)

        # Rank strongest statistical signals first (log10 p-value significance + impact)
        valid_cands.sort(key=lambda r: (-np.log10(max(r["p_value"], 1e-300)), r["est_dataset_yield_impact_pts"]), reverse=True)

        # Greedy residual deduplication: if a stronger cause already covers >=70% of the same affected wafers
        # (e.g. RT-118B explains all Reticle-Repeat wafers so an accidental CMP tool on those same lots is suppressed),
        # suppress the confounded secondary shadow!
        accepted = []
        for rc in valid_cands:
            raw_v = rc["raw_value"]
            if "_tool" in rc["column"] and "chamber" not in rc["column"]:
                if any(other["raw_value"].startswith(raw_v + " /") for other in valid_cands):
                    continue
            m = rc["_mask"]
            m_sum = max(1, int(m.sum()))
            shadowed = False
            for prev in accepted:
                overlap = int((m & prev["_mask"]).sum())
                if overlap / m_sum >= 0.65:
                    shadowed = True
                    break
            if shadowed:
                continue
            accepted.append(rc)

        for rc in accepted:
            rc.pop("_mask", None)
        accepted.sort(key=lambda r: (r["est_dataset_yield_impact_pts"], -r["p_value"]), reverse=True)
        ranked_causes = accepted

    # Add continuous WAT parametric correlations (e.g., Vt_n shift correlating with yield / VMIN)
    wat_findings = []
    if wat_cols:
        for wcol in wat_cols:
            valid = ws[wcol].notna() & ws["yield_loss_pts"].notna()
            if valid.sum() >= 15:
                corr, p_c = spearmanr(ws.loc[valid, wcol], ws.loc[valid, "yield_loss_pts"])
                if pd.notna(corr) and abs(corr) >= 0.30 and p_c < 0.005:
                    # Also quantify shift magnitude in low-yield vs clean wafers
                    med_norm = float(ws.loc[~low_yield_flag, wcol].median()) if (~low_yield_flag).any() else float(ws[wcol].median())
                    med_low = float(ws.loc[low_yield_flag, wcol].median()) if low_yield_flag.any() else med_norm
                    wat_findings.append(
                        {
                            "parameter": wcol,
                            "spearman_r": round(float(corr), 3),
                            "p_value": float(f"{p_c:.2e}"),
                            "baseline_median": round(med_norm, 2),
                            "low_yield_median": round(med_low, 2),
                            "delta": round(med_low - med_norm, 2),
                        }
                    )

    probe_info = _analyze_probe_and_retest(sort_df, bin_names)

    return {
        "lgbm_r2": model_r2,
        "shap_feature_importance": shap_drivers[:8],
        "ranked_causes": ranked_causes[:8],
        "wat_correlations": wat_findings,
        "probe_site_summary": probe_info["site_summary"],
        "probe_findings": probe_info["probe_findings"],
        "retest_summary": probe_info["retest_summary"],
    }

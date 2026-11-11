"""
Engine 5: Test Time Optimizer

Analyzes per-test unique fails, parametric redundancy (Pearson r & R^2),
Rule-of-Three DPPM escape risk bounds, and stop-on-fail test reordering
to safely reduce tester time per chip.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2


# Default AX7-SOC test program metadata if test_limits.csv is not uploaded
DEFAULT_TEST_METADATA = [
    ("T1000", "CONTACT_RES", 18.0, 30),
    ("T2100", "IDDQ", 42.0, 12),
    ("T2200", "LEAKAGE", 35.0, 21),
    ("T3000", "SCAN_FAIL_COUNT", 310.0, 3),
    ("T3500", "MBIST_FAIL", 260.0, 9),
    ("T4000", "PLL_LOCK", 28.0, 15),
    ("T5000", "VMIN", 190.0, 7),
    ("T5100", "FMAX", 230.0, 5),
    ("T4120", "RING_OSC", 22.0, 6),
    ("T6000", "ADC_INL", 95.0, 18),
]


def analyze_test_time(
    first_pass_df: pd.DataFrame | None,
    limits_df: pd.DataFrame | None,
    parametric_cols: list[str],
) -> dict[str, Any]:
    """Identify redundant/low-coverage tests and quantify safe test time reductions."""
    if first_pass_df is None or first_pass_df.empty or not parametric_cols:
        return {"available": False, "reason": "Die-level parametric test results required for test-time optimization."}

    fp = first_pass_df
    n_total = len(fp)

    # Assemble test metadata (execution order, test_time_ms, soft_bin)
    meta_rows = []
    if limits_df is not None and not limits_df.empty and "test_name" in limits_df.columns:
        for idx, r in limits_df.iterrows():
            tname = str(r["test_name"])
            if tname in fp.columns:
                meta_rows.append(
                    {
                        "order": idx,
                        "test_name": tname,
                        "test_num": str(getattr(r, "test_num", f"T{idx+1}")),
                        "test_time_ms": float(getattr(r, "test_time_ms", 50.0)),
                        "soft_bin": int(r["soft_bin"]) if pd.notna(getattr(r, "soft_bin", np.nan)) else None,
                    }
                )
    else:
        for idx, (tnum, tname, tms, sbin) in enumerate(DEFAULT_TEST_METADATA):
            if tname in fp.columns:
                meta_rows.append(
                    {
                        "order": idx,
                        "test_name": tname,
                        "test_num": tnum,
                        "test_time_ms": tms,
                        "soft_bin": sbin,
                    }
                )
        # Include any extra custom numeric parametric columns
        known_names = {m["test_name"] for m in meta_rows}
        for pcol in parametric_cols:
            if pcol not in known_names:
                meta_rows.append(
                    {
                        "order": len(meta_rows),
                        "test_name": pcol,
                        "test_num": pcol,
                        "test_time_ms": 50.0,
                        "soft_bin": None,
                    }
                )

    if not meta_rows:
        return {"available": False, "reason": "No recognizable test items found."}

    # Compute pairwise Pearson correlations among numeric parametric columns
    num_df = fp[[m["test_name"] for m in meta_rows]].select_dtypes(include=[np.number])
    corr_mat = num_df.corr() if len(num_df.columns) >= 2 else pd.DataFrame()

    total_full_flow_ms = sum(m["test_time_ms"] for m in meta_rows)
    mean_exec_ms = float(fp["test_time_ms"].mean()) if "test_time_ms" in fp.columns else total_full_flow_ms

    # Under stop-on-fail, a die's soft_bin identifies the exact test that uniquely stopped it
    sb_counts = fp[fp["pass"] == 0]["soft_bin"].value_counts().to_dict() if "soft_bin" in fp.columns else {}

    test_items = []
    candidates_for_sampling = []
    saved_ms_sum = 0.0

    for m in meta_rows:
        tname = m["test_name"]
        executed_n = int(fp[tname].notna().sum())
        sbin = m["soft_bin"]
        unique_fails = int(sb_counts.get(sbin, 0)) if sbin is not None else 0

        # 95% Poisson upper bound on escape rate (generalized rule-of-three: 0.5 * chi2.ppf(0.95, 2*(k+1)) / n)
        upper_95_count = 0.5 * float(chi2.ppf(0.95, 2 * (unique_fails + 1)))
        escape_bound_dppm = round(upper_95_count / max(1, executed_n) * 1e6, 1)

        # Find highest absolute correlation with another earlier/stronger test
        best_corr_name = None
        best_corr_val = 0.0
        if tname in corr_mat.columns:
            for other in corr_mat.columns:
                if other == tname:
                    continue
                val = corr_mat.loc[tname, other]
                if pd.notna(val) and abs(float(val)) > abs(best_corr_val):
                    best_corr_val = float(val)
                    best_corr_name = str(other)

        fail_rate_pct = unique_fails / max(1, executed_n) * 100.0
        # Candidate if near-zero unique fails OR highly redundant (|r| > 0.96 and very low unique fails)
        is_zero_fail_candidate = (unique_fails <= max(2, int(0.00002 * executed_n))) and (escape_bound_dppm <= 50.0 or unique_fails == 0)
        is_redundant_candidate = (abs(best_corr_val) >= 0.96) and (unique_fails <= max(25, int(0.0001 * executed_n)))
        is_candidate = is_zero_fail_candidate or is_redundant_candidate

        reason = "Essential active screen"
        if is_redundant_candidate and best_corr_name:
            reason = f"Tracks {best_corr_name} closely (r = {best_corr_val:.3f}); catches only {unique_fails} unique fails"
        elif is_zero_fail_candidate:
            reason = f"Caught {unique_fails} unique fails across {executed_n:,} chips tested"

        if is_candidate:
            # 95% sampling reduction saves 95% of this test's executed time
            exec_share = executed_n / max(1, n_total)
            ms_saved = m["test_time_ms"] * exec_share * 0.95
            saved_ms_sum += ms_saved
            candidates_for_sampling.append(
                {
                    "test_name": tname,
                    "test_num": m["test_num"],
                    "test_time_ms": m["test_time_ms"],
                    "unique_fails": unique_fails,
                    "correlated_with": best_corr_name if abs(best_corr_val) >= 0.90 else None,
                    "correlation_r": round(best_corr_val, 3) if abs(best_corr_val) >= 0.90 else None,
                    "escape_bound_dppm": escape_bound_dppm,
                    "ms_saved_per_chip": round(ms_saved, 1),
                    "reason": reason,
                }
            )

        test_items.append(
            {
                "test_num": m["test_num"],
                "test_name": tname,
                "test_time_ms": m["test_time_ms"],
                "executed_count": executed_n,
                "unique_fails": unique_fails,
                "fail_rate_pct": round(fail_rate_pct, 3),
                "fail_per_sec": round((unique_fails / max(1, executed_n)) / max(0.001, m["test_time_ms"] / 1000.0), 4),
                "best_corr_test": best_corr_name,
                "best_corr_r": round(best_corr_val, 3),
                "escape_bound_dppm": escape_bound_dppm,
                "candidate_for_sampling": is_candidate,
                "reason": reason,
            }
        )

    # Simulate stop-on-fail reordering benefit (place fast/high-fail-density tests earlier, keeping CONTACT_RES first)
    reorder_saved_ms = 0.0
    if "soft_bin" in fp.columns:
        # Approximate average executed ms before vs after sorting non-contact tests by (unique_fails / test_time_ms)
        current_avg = mean_exec_ms
        # Calculate expected execution time if failing dies hit earlier
        # Sort tests: CONTACT_RES first (safety), then descending fail_per_sec
        contact_items = [t for t in test_items if t["test_name"] == "CONTACT_RES"]
        other_items = sorted(
            [t for t in test_items if t["test_name"] != "CONTACT_RES"],
            key=lambda t: t["fail_per_sec"],
            reverse=True,
        )
        reordered = contact_items + other_items
        # Probability die stops at each test (approx using unique fail share)
        cum_alive = 1.0
        exp_reordered_ms = 0.0
        for t in reordered:
            exp_reordered_ms += cum_alive * t["test_time_ms"]
            p_fail_here = t["unique_fails"] / max(1, n_total)
            cum_alive = max(0.0, cum_alive - p_fail_here)
        if exp_reordered_ms < current_avg:
            reorder_saved_ms = max(0.0, current_avg - exp_reordered_ms)

    total_saved_ms = saved_ms_sum + reorder_saved_ms
    pct_reduction = round(total_saved_ms / max(1.0, mean_exec_ms) * 100.0, 1)

    return {
        "available": True,
        "chips_analyzed": n_total,
        "full_flow_ms": round(total_full_flow_ms, 1),
        "current_avg_ms": round(mean_exec_ms, 1),
        "sampling_saved_ms": round(saved_ms_sum, 1),
        "reordering_saved_ms": round(reorder_saved_ms, 1),
        "total_saved_ms": round(total_saved_ms, 1),
        "pct_time_reduction": pct_reduction,
        "test_items": test_items,
        "candidates": candidates_for_sampling,
        "recommended_order": [t["test_name"] for t in (contact_items + other_items)] if "soft_bin" in fp.columns else [],
    }

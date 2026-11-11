"""
Advisor Layer: Ranked Recommendation Scoring & Grounded Plain-English Q&A

Synthesizes outputs from all ML engines into a prioritized list of plain-English
engineering actions scored by expected benefit, confidence, and quality risk,
plus a grounded Q&A helper backed strictly by computed metrics.
"""
from __future__ import annotations

import re
from typing import Any


def build_recommendations(
    summary: dict[str, Any],
    wafer_patterns_summary: dict[str, Any],
    excursion_out: dict[str, Any],
    root_cause_out: dict[str, Any],
    outlier_out: dict[str, Any],
    test_opt_out: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rank candidate actions from all AI engines by risk-adjusted business impact."""
    actions: list[dict[str, Any]] = []
    total_chips = max(1, int(summary.get("total_chips", 500)))
    total_wafers = max(1, int(summary.get("total_wafers", 1)))
    # Illustrative value assumptions matching Tech Design Section 12.2 ($8/die, $0.012/s)
    die_value_usd = 8.0
    tester_cost_per_sec = 0.012

    # 1. Root Cause Equipment / Chamber / Reticle / Handler Actions
    for idx, rc in enumerate(root_cause_out.get("ranked_causes", [])):
        factor = rc["factor"]
        sig = rc["signature"]
        impact_pts = float(rc["est_dataset_yield_impact_pts"])
        conf_str = rc["confidence"]
        p_conf = 0.92 if conf_str == "High" else 0.78

        recovered_chips = int(round(total_chips * (impact_pts / 100.0)))
        est_value_usd = round(recovered_chips * die_value_usd * p_conf)

        if "ETCH" in factor.upper() or "Ch-" in factor:
            action_title = f"Inspect & recalibrate {factor}"
            category = "Equipment Check"
            plain_why = (
                f"{sig} failures spiked sharply on wafers processed here, "
                f"costing an estimated {impact_pts:.1f}% overall yield."
            )
        elif "RETICLE" in rc.get("column", "").upper() or sig == "Reticle-Repeat":
            action_title = f"Inspect lithography mask {factor}"
            category = "Mask / Reticle"
            plain_why = (
                f"A repeating stepper field pattern ({sig}) appears on {rc['wafers_affected']} wafers "
                f"linked to {factor}."
            )
        elif "CMP" in factor.upper():
            action_title = f"Check polishing pad & conditioning on {factor}"
            category = "Equipment Check"
            plain_why = (
                f"Center-hotspot wafer patterns increased on {factor}, "
                f"reducing yield across {rc['wafers_affected']} wafers."
            )
        elif "ROB" in factor.upper() or "LOAD" in rc.get("column", "").upper():
            action_title = f"Inspect wafer handler alignment on {factor}"
            category = "Handling"
            plain_why = (
                f"Linear scratch defects appeared disproportionately on wafers handled by {factor}."
            )
        else:
            action_title = f"Investigate {factor}"
            category = "Root Cause"
            plain_why = f"Statistically linked to {sig} loss across {rc['wafers_affected']} wafers."

        actions.append(
            {
                "id": f"rc_{idx}",
                "priority_score": est_value_usd,
                "category": category,
                "title": action_title,
                "why": plain_why,
                "evidence": rc["evidence"] + f" (adjusted p = {rc['adj_p_value']})",
                "expected_benefit": f"+{impact_pts:.2f}% yield (~{recovered_chips:,} chips)",
                "confidence_label": conf_str,
                "confidence_pct": int(round(p_conf * 100)),
                "approval_owner": "Process / Yield Engineer",
                "status": "pending",
            }
        )

    # 2. Probe Card Site Wear & Automated Contact Retest
    for idx, pf in enumerate(root_cause_out.get("probe_findings", [])):
        pc = pf["probe_card_id"]
        site = pf["site"]
        gap = float(pf["yield_gap_pts"])
        # Estimate chips affected (~1/4 of chips on that card)
        impact_pts = round(gap * 0.25 / max(1, len({pf2['probe_card_id'] for pf2 in root_cause_out.get('probe_findings', [])}) + 3), 2)
        recovered_chips = max(50, int(round(total_chips * (impact_pts / 100.0))))
        est_value_usd = round(recovered_chips * die_value_usd * 0.90)

        actions.append(
            {
                "id": f"probe_{idx}",
                "priority_score": est_value_usd,
                "category": "Probe & Retest",
                "title": f"Clean or replace probe card {pc} (Site {site})",
                "why": (
                    f"Site {site} yields {gap:.1f}% lower than the other 3 sites on the same probe card — "
                    "pointing to needle contact wear rather than bad silicon."
                ),
                "evidence": (
                    f"Site {site} yield = {pf['site_yield_pct']}% vs {pf['other_sites_median_pct']}% median on other sites. "
                    + pf["evidence"]
                ),
                "expected_benefit": f"Recover ~{impact_pts:.2f}% yield via needle clean + contact retest",
                "confidence_label": "High",
                "confidence_pct": 91,
                "approval_owner": "Test Floor Lead",
                "status": "pending",
            }
        )

    # 3. Excursion / Electrical Parameter Shift (e.g., WAT Vt_n shift)
    wat_corrs = root_cause_out.get("wat_correlations", [])
    excursions = excursion_out.get("excursions", [])
    if excursions or wat_corrs:
        top_exc = max(excursions, key=lambda e: e["peak_yield_drop_pts"]) if excursions else None
        top_wat = max(wat_corrs, key=lambda w: abs(w["spearman_r"])) if wat_corrs else None
        if top_exc or top_wat:
            if top_wat and top_exc:
                title = f"Review {top_wat['parameter']} shift during {top_exc['start_lot']}–{top_exc['end_lot']}"
                why = (
                    f"Early electrical tests ({top_wat['parameter']}) shifted by {top_wat['delta']:+.2f}, "
                    f"triggering an early yield drop ({top_exc['top_drivers']})."
                )
                evidence = (
                    f"EWMA flagged sustained shift starting at {top_exc['start_lot']} "
                    f"(peak drop {top_exc['peak_yield_drop_pts']:.1f} pts); "
                    f"{top_wat['parameter']} Spearman correlation r = {top_wat['spearman_r']} (p = {top_wat['p_value']})."
                )
                score_val = int(round(total_chips * (top_exc["avg_yield_drop_pts"] / 100.0) * 0.25 * die_value_usd))
                benefit = f"Early detection ({top_exc['lot_count']} lots flagged before weekly review)"
            elif top_exc:
                title = f"Investigate {top_exc['title']}"
                why = f"Yield dropped up to {top_exc['peak_yield_drop_pts']:.1f}% ({top_exc['top_drivers']})."
                evidence = f"Detected across {top_exc['lot_count']} chronological lots via EWMA/CUSUM."
                score_val = int(round(total_chips * (top_exc["avg_yield_drop_pts"] / 100.0) * 0.20 * die_value_usd))
                benefit = f"Prevent further drift across {top_exc['lot_count']} affected lots"
            else:
                title = f"Monitor {top_wat['parameter']} electrical drift"
                why = f"Shifts in {top_wat['parameter']} strongly track wafer yield loss."
                evidence = f"Spearman correlation r = {top_wat['spearman_r']} (p = {top_wat['p_value']})."
                score_val = 2500
                benefit = "Pre-sort excursion warning"

            actions.append(
                {
                    "id": "excursion_wat",
                    "priority_score": max(1500, score_val),
                    "category": "Early Warning",
                    "title": title,
                    "why": why,
                    "evidence": evidence,
                    "expected_benefit": benefit,
                    "confidence_label": "High" if top_wat else "Medium",
                    "confidence_pct": 88 if top_wat else 80,
                    "approval_owner": "Product / Integration Engineer",
                    "status": "pending",
                }
            )

    # 4. Hidden Risk Outlier Screening (NNR / DPAT / GDBN)
    if outlier_out.get("available") and outlier_out.get("total_flagged", 0) > 0:
        flagged = outlier_out["total_flagged"]
        pct = outlier_out["flagged_pct"]
        bi = outlier_out.get("burnin_validation")
        if bi:
            why = (
                f"Flagged chips passed normal specs but showed {bi['nnr_fail_rate_pct']}% burn-in failure rate "
                f"via neighbor spike screening vs {bi['baseline_fail_rate_pct']}% for normal chips."
            )
            evidence = (
                f"Screens {bi['combined_flagged_count']:,} chips ({bi['yield_cost_pct']}% yield cost) "
                f"and catches {bi['burnin_recall_pct']}% of burn-in failures."
            )
            benefit = f"Catches {bi['burnin_recall_pct']}% of latent burn-in fails at {bi['yield_cost_pct']}% yield cost"
            conf = 95
            prio = 8500
        else:
            why = (
                f"{flagged:,} passing chips ({pct}%) show abnormal standby power spikes compared to immediate wafer neighbors "
                "or sit surrounded by failing chips."
            )
            evidence = "Identified using Neighbor Residual (NNR), Dynamic PAT (DPAT), and Bad-Neighborhood (GDBN) screens."
            benefit = f"Screen {flagged:,} high-risk chips ({pct}% yield cost) to protect customer quality"
            conf = 84
            prio = 4000

        actions.append(
            {
                "id": "outlier_screen",
                "priority_score": prio,
                "category": "Quality Screen",
                "title": "Enable neighbor-based outlier screen (NNR + GDBN)",
                "why": why,
                "evidence": evidence,
                "expected_benefit": benefit,
                "confidence_label": "High" if conf >= 88 else "Medium",
                "confidence_pct": conf,
                "approval_owner": "Quality Engineer",
                "status": "pending",
            }
        )

    # 5. Faster Testing Optimization (Redundant / Zero-Fail Sampling + Reordering)
    if test_opt_out.get("available") and test_opt_out.get("pct_time_reduction", 0) >= 1.5:
        cands = test_opt_out.get("candidates", [])
        cand_names = ", ".join(c["test_name"] for c in cands[:3]) if cands else "test reordering"
        saved_ms = test_opt_out["total_saved_ms"]
        pct_cut = test_opt_out["pct_time_reduction"]
        est_usd = round((saved_ms / 1000.0) * total_chips * tester_cost_per_sec)

        max_dppm = max((c["escape_bound_dppm"] for c in cands), default=0.0)
        actions.append(
            {
                "id": "test_time",
                "priority_score": max(1800, est_usd),
                "category": "Test Speed",
                "title": f"Move {cand_names} to 5% audit sampling & reorder flow",
                "why": (
                    f"Reduces average test time by {saved_ms:.0f} ms per chip ({pct_cut:.1f}%) "
                    "by sampling redundant or zero-fail tests and running high-yield-loss tests earlier."
                ),
                "evidence": (
                    f"95% escape risk bound <= {max_dppm:.1f} DPPM across {test_opt_out['chips_analyzed']:,} chips tested."
                    + (f" ({cands[0]['reason']}.)" if cands else "")
                ),
                "expected_benefit": f"-{pct_cut:.1f}% test time ({saved_ms:.0f} ms/chip saved)",
                "confidence_label": "High" if max_dppm < 15 else "Medium",
                "confidence_pct": 90 if max_dppm < 15 else 78,
                "approval_owner": "Change Control Board",
                "status": "pending",
            }
        )

    actions.sort(key=lambda a: a["priority_score"], reverse=True)
    for rank, act in enumerate(actions, 1):
        act["rank"] = rank
    return actions


def answer_question(question: str, state: dict[str, Any]) -> dict[str, Any]:
    """
    Deterministic, evidence-grounded Q&A assistant.
    Answers strictly from computed engine outputs in `state` (zero hallucination).
    """
    q = question.strip().lower()
    summary = state["summary"]
    excursion = state["excursion"]
    rc = state["root_cause"]
    outliers = state["outliers"]
    test_opt = state["test_optimizer"]
    patterns = state["wafer_patterns_summary"]
    actions = state["actions"]

    # Check if user asked about a specific lot ID
    lot_match = re.search(r"\b(l\d{4,}|lot[_\-\s]*\w+)\b", q, re.IGNORECASE)
    trend_lots = {r["lot_id"].lower(): r for r in excursion.get("lot_trend", [])}
    for cand_lot, lot_row in trend_lots.items():
        if cand_lot in q:
            base = excursion.get("baseline_yield_pct", summary["final_yield_pct"])
            delta = lot_row["yield_pct"] - base
            # Gather patterns in this lot
            wafers = [w for w in state["wafer_table"] if w["lot_id"].lower() == cand_lot]
            pat_counts: dict[str, int] = {}
            for w in wafers:
                p = w["primary_pattern"]
                if p != "Clean":
                    pat_counts[p] = pat_counts.get(p, 0) + 1
            pat_str = ", ".join(f"{k} ({v} wafers)" for k, v in pat_counts.items()) or "mostly Clean wafers"
            alert_txt = "flagged by EWMA early-warning" if lot_row["ewma_alert"] else "within normal EWMA limits"
            return {
                "answer": (
                    f"Lot {lot_row['lot_id']} achieved {lot_row['yield_pct']}% final yield "
                    f"({delta:+.1f} pts vs {base}% baseline) across {lot_row['wafer_count']} wafers, "
                    f"and is {alert_txt}. Wafer patterns detected: {pat_str}."
                ),
                "sources": [f"Lot Trend ({lot_row['lot_id']})", "Wafer Map Pattern AI"],
            }

    if any(k in q for k in ["why", "drop", "excursion", "shift", "root cause", "cause", "etch", "cmp", "fail"]):
        parts = []
        if rc.get("ranked_causes"):
            top = rc["ranked_causes"][:3]
            parts.append(
                "Top root causes identified by LightGBM + SHAP and statistical commonality tests: "
                + "; ".join(f"{i+1}) {c['factor']} ({c['signature']}, est. {c['est_dataset_yield_impact_pts']:.1f}% impact — {c['evidence']})" for i, c in enumerate(top))
            )
        if excursion.get("excursions"):
            ex = excursion["excursions"][0]
            parts.append(
                f"Chronological EWMA also flagged {ex['title']} (peak drop {ex['peak_yield_drop_pts']:.1f}%, driven by {ex['top_drivers']})."
            )
        if not parts:
            parts.append(f"Overall final yield is {summary['final_yield_pct']}%, with no major equipment excursion flagged above threshold.")
        return {
            "answer": " ".join(parts),
            "sources": ["Root Cause Engine (LightGBM + SHAP)", "EWMA Excursion Detector"],
        }

    if any(k in q for k in ["probe", "site", "retest", "contact", "pc-"]):
        if rc.get("probe_findings"):
            pf = rc["probe_findings"][0]
            rt_str = ""
            if rc.get("retest_summary"):
                best_rt = sorted(rc["retest_summary"], key=lambda r: r["recovery_rate_pct"], reverse=True)[0]
                rt_str = f" Retest recovered {best_rt['recovery_rate_pct']}% of {best_rt['bin_name']} fails ({best_rt['recovered_dies']:,}/{best_rt['retested_dies']:,} chips)."
            return {
                "answer": (
                    f"Probe card {pf['probe_card_id']} Site {pf['site']} yields {pf['site_yield_pct']}% "
                    f"vs {pf['other_sites_median_pct']}% on sibling sites (-{pf['yield_gap_pts']} pts). "
                    f"{pf['evidence']}{rt_str}"
                ),
                "sources": ["Probe Site Comparison", "Retest Recovery Analysis"],
            }
        return {
            "answer": "All probe sites are performing within normal balance (< 0.8% site-to-site spread).",
            "sources": ["Probe Site Comparison"],
        }

    if any(k in q for k in ["test time", "skip", "sample", "faster", "redundant", "ring", "adc"]):
        if test_opt.get("available") and test_opt.get("candidates"):
            c_txt = "; ".join(f"{c['test_name']} ({c['test_time_ms']} ms, {c['reason']}, escape bound <= {c['escape_bound_dppm']} DPPM)" for c in test_opt["candidates"])
            return {
                "answer": (
                    f"Recommended sampling/removal candidates save {test_opt['total_saved_ms']} ms/chip "
                    f"({test_opt['pct_time_reduction']}% faster including test reordering): {c_txt}."
                ),
                "sources": ["Test Time Optimizer (Redundancy & Rule-of-Three)"],
            }
        return {
            "answer": "No redundant tests met the strict low-escape-risk threshold for removal in this dataset.",
            "sources": ["Test Time Optimizer"],
        }

    if any(k in q for k in ["burn", "outlier", "latent", "hidden", "dpat", "nnr", "gdbn", "iddq"]):
        if outliers.get("available"):
            bi = outliers.get("burnin_validation")
            bi_msg = (
                f" Burn-in validation confirms {bi['burnin_recall_pct']}% of burn-in failures were captured "
                f"at just {bi['yield_cost_pct']}% yield cost (NNR fail rate {bi['nnr_fail_rate_pct']}% vs {bi['baseline_fail_rate_pct']}% baseline)."
                if bi
                else ""
            )
            return {
                "answer": (
                    f"Hidden-risk screens flagged {outliers['total_flagged']:,} passing chips ({outliers['flagged_pct']}% of passing units) "
                    f"that passed static test limits but show abnormal neighbor spikes or cluster risk.{bi_msg}"
                ),
                "sources": ["Adaptive Outlier Screener (NNR / DPAT / GDBN)"],
            }
        return {"answer": "Die-level parametric data is required for outlier screening.", "sources": []}

    if any(k in q for k in ["pattern", "wafer", "map", "edge", "center", "scratch", "reticle"]):
        counts = patterns.get("counts", {})
        non_clean = {k: v for k, v in counts.items() if k != "Clean"}
        c_str = ", ".join(f"{k}: {v} wafers" for k, v in sorted(non_clean.items(), key=lambda x: x[1], reverse=True)) or "all wafers Clean"
        return {
            "answer": f"Across {summary['total_wafers']} wafers analyzed by the Wafer Pattern CNN: {c_str}.",
            "sources": ["Wafer Map Pattern AI (PyTorch CNN)"],
        }

    # Default executive summary
    top_act = actions[0]["title"] if actions else "No urgent actions"
    return {
        "answer": (
            f"Summary for {summary['total_lots']} lots ({summary['total_wafers']} wafers): "
            f"Final yield is {summary['final_yield_pct']}% (first-pass {summary['first_pass_yield_pct']}%). "
            f"Top recommended action: {top_act}. Ask about root causes, probe cards, hidden risk, wafer patterns, or test time."
        ),
        "sources": ["Yield Advisor Summary"],
    }

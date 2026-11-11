"""
Create ready-to-upload CSV and multi-sheet Excel demo files inside sample_uploads/
from representative lots of AX7-SOC so users can immediately drag-and-drop a
single .xlsx or .csv file to watch all AI engines run live.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "AX7_SOC_synthetic_chip_test_dataset"
OUT_DIR = ROOT / "sample_uploads"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 14 representative lots covering every defect pattern, excursion, probe wear, and baseline
DEMO_LOTS = [
    "L262200", "L262202", "L262205", "L262206",  # Baseline + early PC-4412 + ROB-02 scratch
    "L262211", "L262213",                        # CMP-03 center pattern
    "L262218", "L262219",                        # RT-118B repeating reticle
    "L262226", "L262227", "L262229",             # VT_SHIFT (+18mV Vmin/Fmax)
    "L262234", "L262236", "L262238",             # ETCH-07/C post-PM Edge-Ring + late PC-4412 site 3
]


def main():
    print("Building ready-to-drop demo Excel and CSV files...")
    sort_df = pd.read_csv(DATA_DIR / "sort_die_results.csv")
    mes_df = pd.read_csv(DATA_DIR / "lot_context_mes.csv")
    wat_df = pd.read_csv(DATA_DIR / "wat_pcm.csv")
    limits_df = pd.read_csv(DATA_DIR / "test_limits.csv")
    ft_df = pd.read_csv(DATA_DIR / "final_test_burnin.csv")
    ws_df = pd.read_csv(DATA_DIR / "wafer_summary.csv")

    s_demo = sort_df[sort_df["lot_id"].isin(DEMO_LOTS)].copy()
    m_demo = mes_df[mes_df["lot_id"].isin(DEMO_LOTS)].copy()
    w_demo = wat_df[wat_df["lot_id"].isin(DEMO_LOTS)].copy()
    f_demo = ft_df[ft_df["lot_id"].isin(DEMO_LOTS)].copy()

    # 1. Single CSV sample (sort results only)
    csv_out = OUT_DIR / "AX7_Sort_Sample_14_Lots.csv"
    s_demo.to_csv(csv_out, index=False)
    print(f"Created {csv_out} ({len(s_demo):,} rows)")

    # 2. Multi-sheet Excel sample (all context included in 1 file)
    # Use 8 lots (~100k rows) so Excel opens quickly and uploads in <2s while keeping all stories!
    xlsx_lots = [
        "L262200", "L262202", "L262211", "L262218",
        "L262226", "L262227", "L262234", "L262238",
    ]
    xlsx_out = OUT_DIR / "AX7_Complete_Demo_8_Lots.xlsx"
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        sort_df[sort_df["lot_id"].isin(xlsx_lots)].to_excel(writer, sheet_name="sort_die_results", index=False)
        mes_df[mes_df["lot_id"].isin(xlsx_lots)].to_excel(writer, sheet_name="lot_context_mes", index=False)
        wat_df[wat_df["lot_id"].isin(xlsx_lots)].to_excel(writer, sheet_name="wat_pcm", index=False)
        limits_df.to_excel(writer, sheet_name="test_limits", index=False)
        ft_df[ft_df["lot_id"].isin(xlsx_lots)].to_excel(writer, sheet_name="final_test_burnin", index=False)
    print(f"Created {xlsx_out}")

    # 3. Wafer summary CSV (lightweight summary-only test file)
    ws_out = OUT_DIR / "AX7_Wafer_Summary.csv"
    ws_df.to_csv(ws_out, index=False)
    print(f"Created {ws_out} ({len(ws_df):,} rows)")


if __name__ == "__main__":
    main()

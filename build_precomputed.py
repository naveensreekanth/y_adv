"""
Precompute the full AX7-SOC ML analysis outputs into compact gzipped JSON files
so the Vercel deployment runs with ultra-low latency (<50ms) and stays well
under Vercel's serverless bundle size limit.
"""
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.ingest import load_reference_dataset
from app.pipeline import get_wafer_map_detail, run_analysis_pipeline


def main():
    t0 = time.perf_counter()
    print("Loading full AX7-SOC dataset...")
    bundle = load_reference_dataset(max_lots=None)
    print("Running all 6 ML engines (PyTorch ResNet-CNN + Grad-CAM, LightGBM + SHAP, EWMA, NNR/DPAT/GDBN, Optimizer, Forecast)...")
    payload = run_analysis_pipeline(bundle)
    orig_sid = payload["session_id"]
    payload["session_id"] = "ax7-soc-preloaded"

    out_dir = BACKEND / "precomputed"
    wafers_dir = out_dir / "wafers"
    wafers_dir.mkdir(parents=True, exist_ok=True)

    preloaded_path = out_dir / "preloaded.json.gz"
    with gzip.open(preloaded_path, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Saved {preloaded_path} ({preloaded_path.stat().st_size / 1024:.1f} KB)")

    # Group wafer details by lot_id for fast lazy loading per lot
    by_lot: dict[str, dict[str, dict]] = {}
    for row in payload["wafer_table"]:
        lot_id = row["lot_id"]
        wafer_id = int(row["wafer_id"])
        detail = get_wafer_map_detail(orig_sid, lot_id, wafer_id)
        by_lot.setdefault(lot_id, {})[str(wafer_id)] = detail

    total_wafer_kb = 0.0
    for lot_id, wmap in by_lot.items():
        wpath = wafers_dir / f"{lot_id}.json.gz"
        with gzip.open(wpath, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(wmap, f, separators=(",", ":"))
        total_wafer_kb += wpath.stat().st_size / 1024.0

    print(
        f"Saved {len(by_lot)} lot wafer bundles ({sum(len(v) for v in by_lot.values())} wafers, "
        f"{total_wafer_kb / 1024:.2f} MB total) in {time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()

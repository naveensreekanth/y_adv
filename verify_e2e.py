"""End-to-end verification tests for all upload modes and ML engines."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app  # noqa: E402

client = TestClient(app)
SAMPLES = ROOT / "sample_uploads"


def test_health_and_frontend():
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    print("[OK] /api/health:", r.json())

    r_html = client.get("/")
    assert r_html.status_code == 200
    assert "Yield Advisor" in r_html.text
    print("[OK] Frontend index.html served cleanly")


def test_full_builtin_sample():
    r = client.post("/api/load-sample", data={"scope": "full"})
    assert r.status_code == 200, r.text
    res = r.json()
    s = res["summary"]
    print(f"[OK] Full 40-lot dataset analyzed in {res['runtime_seconds']}s: {s['total_lots']} lots, {s['total_wafers']} wafers, yield={s['final_yield_pct']}%")
    assert s["total_lots"] == 40
    assert s["total_wafers"] == 1000
    assert len(res["actions"]) >= 4

    print("  Top Actions:")
    for a in res["actions"]:
        print(f"    {a['rank']}. {a['title']} ({a['expected_benefit']})")

    # Test Wafer Map + Grad-CAM drill-down on an Edge-Ring wafer (L262238) and Reticle wafer (L262218)
    edge_wafer = next((w for w in res["wafer_table"] if w["primary_pattern"] == "Edge-Ring"), None)
    assert edge_wafer is not None, "Expected Edge-Ring wafer to be detected"
    rw = client.get(f"/api/session/{res['session_id']}/wafer/{edge_wafer['lot_id']}/{edge_wafer['wafer_id']}")
    assert rw.status_code == 200, rw.text
    wdata = rw.json()
    assert len(wdata["dies"]) == 500
    assert any(d["gradcam"] > 0.5 for d in wdata["dies"])
    print(f"  [OK] Wafer Grad-CAM drilldown on {edge_wafer['lot_id']} W{edge_wafer['wafer_id']} ({edge_wafer['primary_pattern']})")

    # Test grounded Q&A
    rq = client.post(f"/api/session/{res['session_id']}/ask", json={"question": "Which probe card has a contact issue?"})
    assert rq.status_code == 200, rq.text
    print("  [OK] Grounded Q&A:", rq.json()["answer"])

    # Test action approval + report export
    ra = client.post(f"/api/session/{res['session_id']}/action/{res['actions'][0]['id']}", json={"status": "accepted"})
    assert ra.status_code == 200
    re_rep = client.get(f"/api/session/{res['session_id']}/export")
    assert re_rep.status_code == 200 and "[ACCEPTED]" in re_rep.text
    print("  [OK] Action approval & Markdown report export verified")


def test_excel_upload():
    xlsx_path = SAMPLES / "AX7_Complete_Demo_8_Lots.xlsx"
    with open(xlsx_path, "rb") as f:
        r = client.post(
            "/api/analyze",
            files={"files": (xlsx_path.name, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"match_reference_context": "false"},
        )
    assert r.status_code == 200, r.text
    res = r.json()
    print(f"[OK] Multi-sheet Excel upload ({xlsx_path.name}) in {res['runtime_seconds']}s -> {res['summary']['total_lots']} lots, {len(res['actions'])} actions")
    assert res["summary"]["total_lots"] == 8
    assert res["outliers"]["burnin_validation"] is not None
    assert res["yield_forecast"]["available"] is True


def test_csv_sort_only_upload():
    csv_path = SAMPLES / "AX7_Sort_Sample_14_Lots.csv"
    with open(csv_path, "rb") as f:
        r = client.post(
            "/api/analyze",
            files={"files": (csv_path.name, f, "text/csv")},
            data={"match_reference_context": "false"},
        )
    assert r.status_code == 200, r.text
    res = r.json()
    print(f"[OK] Standalone CSV upload ({csv_path.name}, no MES/WAT) in {res['runtime_seconds']}s -> {res['summary']['total_wafers']} wafers, {len(res['actions'])} actions")
    assert res["summary"]["total_lots"] == 14


def test_wafer_summary_csv_upload():
    ws_path = SAMPLES / "AX7_Wafer_Summary.csv"
    with open(ws_path, "rb") as f:
        r = client.post(
            "/api/analyze",
            files={"files": (ws_path.name, f, "text/csv")},
            data={"match_reference_context": "false"},
        )
    assert r.status_code == 200, r.text
    res = r.json()
    print(f"[OK] Wafer Summary CSV upload in {res['runtime_seconds']}s -> {res['summary']['total_wafers']} wafers")
    assert res["quality_report"]["mode"] == "wafer_summary"


def test_custom_aliased_csv_upload():
    # Simulate a completely custom user CSV with renamed columns (Lot, Wafer, X, Y, Result)
    df = pd.DataFrame(
        [
            {"Lot": "MYLOT_1", "Wafer": 1, "X": x, "Y": y, "Result": "FAIL" if (x - 5) ** 2 + (y - 5) ** 2 > 18 else "PASS", "IDDQ": 52.0 + (0.5 if x == 5 else 0.0)}
            for x in range(12)
            for y in range(12)
            if (x - 5.5) ** 2 + (y - 5.5) ** 2 <= 36
        ]
    )
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    r = client.post(
        "/api/analyze",
        files={"files": ("custom_wafer_test.csv", buf, "text/csv")},
        data={"match_reference_context": "false"},
    )
    assert r.status_code == 200, r.text
    res = r.json()
    print(f"[OK] Custom aliased CSV upload -> mapped columns: {res['quality_report']['column_mappings']}")
    assert res["summary"]["total_lots"] == 1


if __name__ == "__main__":
    test_health_and_frontend()
    test_full_builtin_sample()
    test_excel_upload()
    test_csv_sort_only_upload()
    test_wafer_summary_csv_upload()
    test_custom_aliased_csv_upload()
    print("\nALL END-TO-END TESTS PASSED!")

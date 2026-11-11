"""End-to-end verification tests for preloaded Yield Advisor app and ML engines."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app  # noqa: E402


def run_all():
    with TestClient(app) as client:
        r_html = client.get("/")
        assert r_html.status_code == 200
        assert "Yield Advisor" in r_html.text
        print("[OK] Frontend index.html served cleanly")

        r_pre = client.get("/api/preloaded")
        assert r_pre.status_code == 200, r_pre.text
        res = r_pre.json()
        s = res["summary"]
        print(
            f"[OK] Preloaded dataset ready in {res['runtime_seconds']}s: "
            f"{s['total_lots']} lots, {s['total_wafers']} wafers, yield={s['final_yield_pct']}%"
        )
        assert s["total_lots"] >= 8
        assert len(res["actions"]) >= 4

        edge_wafer = next((w for w in res["wafer_table"] if w["primary_pattern"] == "Edge-Ring"), None)
        assert edge_wafer is not None, "Expected Edge-Ring wafer to be detected"
        rw = client.get(f"/api/session/{res['session_id']}/wafer/{edge_wafer['lot_id']}/{edge_wafer['wafer_id']}")
        assert rw.status_code == 200, rw.text
        wdata = rw.json()
        assert len(wdata["dies"]) == 500
        assert any(d["gradcam"] > 0.5 for d in wdata["dies"])
        print(f"[OK] Wafer Grad-CAM drilldown on {edge_wafer['lot_id']} W{edge_wafer['wafer_id']}")

        rq = client.post(f"/api/session/{res['session_id']}/ask", json={"question": "Which probe card has a contact issue?"})
        assert rq.status_code == 200, rq.text
        print("[OK] Grounded Q&A:", rq.json()["answer"])

        ra = client.post(f"/api/session/{res['session_id']}/action/{res['actions'][0]['id']}", json={"status": "accepted"})
        assert ra.status_code == 200
        re_rep = client.get(f"/api/session/{res['session_id']}/export")
        assert re_rep.status_code == 200 and "[ACCEPTED]" in re_rep.text
        print("[OK] Action approval & Markdown report export verified")

        rh = client.get("/api/health")
        assert rh.status_code == 200 and rh.json()["ready"] is True
        print("[OK] /api/health reports ready=True:", rh.json())

    print("\nALL PRELOADED TESTS PASSED!")


if __name__ == "__main__":
    run_all()

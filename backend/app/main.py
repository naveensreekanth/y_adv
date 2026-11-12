"""
FastAPI Backend for the AI-Optimized Yield Advisor.

Serves the precomputed full AX7-SOC ML analysis (1,000 wafers, 40 lots, 500,000 dies)
with sub-50ms latency in serverless/cloud deployments, or dynamically runs the ML
pipeline if precomputed artifacts are absent.
"""
from __future__ import annotations

import gzip
import json
import sys
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.engines.advisor import answer_question

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PRECOMPUTED_DIR = Path(__file__).resolve().parent.parent / "precomputed"
PRELOADED_GZ = PRECOMPUTED_DIR / "preloaded.json.gz"
WAFERS_DIR = PRECOMPUTED_DIR / "wafers"
SAMPLES_DIR = ROOT_DIR / "sample_uploads"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

_PRELOAD_LOCK = threading.Lock()
_PRELOADED_RESULT: dict[str, Any] | None = None
_PRELOAD_ERROR: str | None = None
_PRELOAD_STARTED = False
_LOT_WAFER_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX_LOT_CACHE = 4


def _load_precomputed() -> dict[str, Any]:
    global _PRELOADED_RESULT, _PRELOAD_ERROR
    with _PRELOAD_LOCK:
        if _PRELOADED_RESULT is not None:
            return _PRELOADED_RESULT
        try:
            with gzip.open(PRELOADED_GZ, "rt", encoding="utf-8") as f:
                _PRELOADED_RESULT = json.load(f)
            _PRELOAD_ERROR = None
            return _PRELOADED_RESULT
        except Exception as exc:
            _PRELOAD_ERROR = str(exc)
            raise


def _run_preload() -> dict[str, Any]:
    if PRELOADED_GZ.exists():
        return _load_precomputed()

    global _PRELOADED_RESULT, _PRELOAD_ERROR
    with _PRELOAD_LOCK:
        if _PRELOADED_RESULT is not None:
            return _PRELOADED_RESULT
        try:
            from app.ingest import load_reference_dataset
            from app.pipeline import run_analysis_pipeline

            bundle = load_reference_dataset(max_lots=None)
            _PRELOADED_RESULT = run_analysis_pipeline(bundle)
            _PRELOAD_ERROR = None
            return _PRELOADED_RESULT
        except Exception as exc:
            _PRELOAD_ERROR = str(exc)
            raise


def _warmup_worker():
    try:
        _run_preload()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _PRELOAD_STARTED
    if PRELOADED_GZ.exists():
        _load_precomputed()
    elif not _PRELOAD_STARTED:
        _PRELOAD_STARTED = True
        threading.Thread(target=_warmup_worker, daemon=True).start()
    yield


app = FastAPI(title="AI-Optimized Yield Advisor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class ActionStatusRequest(BaseModel):
    status: str


def _get_session_state(session_id: str) -> dict[str, Any] | None:
    if PRELOADED_GZ.exists():
        data = _load_precomputed()
        if session_id == data.get("session_id") or session_id == "ax7-soc-preloaded":
            return data
        return None
    from app.pipeline import SESSIONS

    return SESSIONS.get(session_id)


@app.get("/api/health")
def health():
    if PRELOADED_GZ.exists() and _PRELOADED_RESULT is None:
        _load_precomputed()
    return {
        "status": "ok",
        "ready": _PRELOADED_RESULT is not None,
        "mode": "precomputed" if PRELOADED_GZ.exists() else "dynamic",
        "has_full_dataset": True,
        "error": _PRELOAD_ERROR,
    }


@app.get("/api/preloaded")
def get_preloaded():
    if _PRELOADED_RESULT is not None:
        return _PRELOADED_RESULT
    if _PRELOAD_ERROR is not None:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {_PRELOAD_ERROR}")
    try:
        return _run_preload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {exc}") from exc


@app.get("/api/session/{session_id}/wafer/{lot_id}/{wafer_id}")
def wafer_detail(session_id: str, lot_id: str, wafer_id: int):
    if PRELOADED_GZ.exists():
        safe_lot = Path(lot_id).name
        if safe_lot in _LOT_WAFER_CACHE:
            _LOT_WAFER_CACHE.move_to_end(safe_lot)
            lot_map = _LOT_WAFER_CACHE[safe_lot]
        else:
            wpath = WAFERS_DIR / f"{safe_lot}.json.gz"
            if not wpath.exists():
                raise HTTPException(status_code=404, detail=f"Die-level map not available for {lot_id} wafer {wafer_id}.")
            with gzip.open(wpath, "rt", encoding="utf-8") as f:
                lot_map = json.load(f)
            _LOT_WAFER_CACHE[safe_lot] = lot_map
            while len(_LOT_WAFER_CACHE) > _MAX_LOT_CACHE:
                _LOT_WAFER_CACHE.popitem(last=False)
        detail = lot_map.get(str(wafer_id))
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Die-level map not available for {lot_id} wafer {wafer_id}.")
        return detail

    from app.pipeline import get_wafer_map_detail

    try:
        return get_wafer_map_detail(session_id, lot_id, wafer_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/session/{session_id}/ask")
def ask_advisor(session_id: str, body: AskRequest):
    sess = _get_session_state(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
    return answer_question(body.question, sess)


@app.post("/api/session/{session_id}/action/{action_id}")
def set_action_status(session_id: str, action_id: str, body: ActionStatusRequest):
    if body.status not in ("pending", "accepted", "dismissed"):
        raise HTTPException(status_code=400, detail="Status must be pending, accepted, or dismissed.")
    sess = _get_session_state(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
    for act in sess.get("actions", []):
        if act["id"] == action_id:
            act["status"] = body.status
            return act
    raise HTTPException(status_code=404, detail=f"Action {action_id} not found.")


@app.get("/api/session/{session_id}/export")
def export_report(session_id: str):
    sess = _get_session_state(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    s = sess["summary"]
    lines = [
        "# AI-Optimized Yield Advisor — Summary Report",
        "",
        f"- **Lots analyzed:** {s['total_lots']}",
        f"- **Wafers analyzed:** {s['total_wafers']}",
        f"- **Chips tested:** {s['total_chips']:,}",
        f"- **Final Yield:** {s['final_yield_pct']}% (First-pass: {s['first_pass_yield_pct']}%)",
        f"- **Chips recovered by retest:** {s['retest_recovered_chips']:,}",
        "",
        "## Prioritized Action Plan",
        "",
    ]
    for a in sess["actions"]:
        lines.extend(
            [
                f"### {a['rank']}. {a['title']} [{a['status'].upper()}]",
                f"- **Expected benefit:** {a['expected_benefit']}",
                f"- **Confidence:** {a['confidence_label']} ({a['confidence_pct']}%)",
                f"- **Why:** {a['why']}",
                f"- **Evidence:** {a['evidence']}",
                f"- **Owner:** {a['approval_owner']}",
                "",
            ]
        )
    content = "\n".join(lines)
    return PlainTextResponse(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=yield_advisor_report.md"},
    )


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    @app.get("/{full_path:path}")
    def serve_spa(full_path: str = ""):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Frontend build not found")

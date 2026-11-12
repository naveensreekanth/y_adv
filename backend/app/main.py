"""
FastAPI Backend for the AI-Optimized Yield Advisor.

Preloads the AX7-SOC chip-test dataset and runs the end-to-end ML pipeline
automatically at startup, serving interactive wafer Grad-CAM drill-downs,
grounded Q&A, and the minimalist React frontend.
"""
from __future__ import annotations

import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingest import REF_DATASET_DIR, load_reference_dataset, parse_uploaded_files
from app.pipeline import (
    SESSIONS,
    ask_session_advisor,
    get_wafer_map_detail,
    run_analysis_pipeline,
    update_action_status,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = ROOT_DIR / "sample_uploads"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

_PRELOAD_LOCK = threading.Lock()
_PRELOADED_RESULT: dict[str, Any] | None = None
_PRELOAD_ERROR: str | None = None
_PRELOAD_STARTED = False


def _run_preload() -> dict[str, Any]:
    global _PRELOADED_RESULT, _PRELOAD_ERROR
    with _PRELOAD_LOCK:
        if _PRELOADED_RESULT is not None:
            return _PRELOADED_RESULT
        try:
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
    if not _PRELOAD_STARTED:
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


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "ready": _PRELOADED_RESULT is not None,
        "has_full_dataset": REF_DATASET_DIR.exists(),
        "samples_available": sorted([p.name for p in SAMPLES_DIR.glob("*") if p.is_file()]) if SAMPLES_DIR.exists() else [],
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


@app.post("/api/analyze")
async def analyze_upload(
    files: List[UploadFile] = File(...),
    match_reference_context: bool = Form(False),
):
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one CSV or Excel file.")

    parsed_inputs: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        if content:
            parsed_inputs.append((f.filename or "upload.csv", content))

    if not parsed_inputs:
        raise HTTPException(status_code=400, detail="Uploaded file(s) are empty.")

    try:
        bundle = parse_uploaded_files(parsed_inputs, match_reference_context=match_reference_context)
        return run_analysis_pipeline(bundle)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.get("/api/samples/{filename}")
def download_sample(filename: str):
    if not SAMPLES_DIR.exists():
        raise HTTPException(status_code=404, detail="Samples directory not found.")
    target = (SAMPLES_DIR / filename).resolve()
    if target.parent != SAMPLES_DIR.resolve() or not target.exists():
        raise HTTPException(status_code=404, detail=f"Sample file '{filename}' not found.")
    return FileResponse(path=str(target), filename=target.name)


@app.get("/api/session/{session_id}/wafer/{lot_id}/{wafer_id}")
def wafer_detail(session_id: str, lot_id: str, wafer_id: int):
    try:
        return get_wafer_map_detail(session_id, lot_id, wafer_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/session/{session_id}/ask")
def ask_advisor(session_id: str, body: AskRequest):
    try:
        return ask_session_advisor(session_id, body.question)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/session/{session_id}/action/{action_id}")
def set_action_status(session_id: str, action_id: str, body: ActionStatusRequest):
    if body.status not in ("pending", "accepted", "dismissed"):
        raise HTTPException(status_code=400, detail="Status must be pending, accepted, or dismissed.")
    try:
        return update_action_status(session_id, action_id, body.status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/session/{session_id}/export")
def export_report(session_id: str):
    sess = SESSIONS.get(session_id)
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

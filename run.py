"""
One-command launcher for the AI-Optimized Yield Advisor.
Starts the FastAPI backend (which also serves the built React UI) and opens the browser.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

import uvicorn  # noqa: E402


def _open_browser():
    time.sleep(1.3)
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    print("Starting AI-Optimized Yield Advisor on http://127.0.0.1:8000 ...")
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, log_level="info")

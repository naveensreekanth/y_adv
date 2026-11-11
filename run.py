"""
One-command launcher for the AI-Optimized Yield Advisor.
Automatically picks an available port (preferring 8080 / 8000 / 8001+), starts the
FastAPI backend serving the built React UI, and opens the browser once ready.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

import uvicorn  # noqa: E402


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) != 0


def _pick_port(host: str, preferred: int | None = None) -> int:
    candidates = ([preferred] if preferred else []) + [8080, 8001, 8002, 8003, 8500, 8000]
    for port in candidates:
        if _is_port_free(host, port):
            return port
    # Fall back to any ephemeral free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _open_browser_when_ready(url: str):
    health_url = f"{url}/api/health"
    for _ in range(30):
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(0.3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    host = "127.0.0.1"
    env_port = int(os.environ["PORT"]) if os.environ.get("PORT") else None
    port = _pick_port(host, args.port or env_port)
    url = f"http://{host}:{port}"

    print(f"Starting AI-Optimized Yield Advisor on {url} ...")
    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    uvicorn.run("app.main:app", host=host, port=port, log_level="info")

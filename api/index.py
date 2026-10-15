"""Vercel entry point: the same FastAPI app as `uvicorn backend.app.main:app`, serving /api/*.
The static frontend (frontend/dist) is served by Vercel's CDN; see vercel.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "app"))
from main import app  # noqa: E402,F401

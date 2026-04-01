"""Entrypoint for deploying the Stanford Dogs image demo on Streamlit Community Cloud."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENT_ROOT = REPO_ROOT / "assignments" / "assignment-1"
sys.path.insert(0, str(ASSIGNMENT_ROOT))

from app.streamlit_app import run_app


run_app()

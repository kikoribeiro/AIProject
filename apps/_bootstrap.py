"""Bootstrap module so the Streamlit app can be run without setting PYTHONPATH.

When running:
  streamlit run apps/confusion_matrix_app.py

the working directory is the repository root, but Python does not automatically
add ./src to sys.path. This helper adds it.

This keeps the project beginner-friendly (no environment variables required).
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"

    if src_path.is_dir():
        src_str = str(src_path)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

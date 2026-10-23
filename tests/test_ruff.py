"""
tests/test_ruff.py — Static analysis: all modules must be free of F821 (undefined names).

This test runs ruff --select F821 over core/, api/, and sources/ to catch missing
`logger` definitions and other undefined-name bugs before they surface at runtime.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_no_undefined_names_f821():
    """ruff --select F821 must exit 0 over core/, api/, sources/."""
    ruff = Path(sys.executable).parent / "ruff"
    result = subprocess.run(
        [str(ruff), "check", "--select", "F821", "core/", "api/", "sources/"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"ruff F821 check found undefined names:\n{result.stdout}\n{result.stderr}"
    )

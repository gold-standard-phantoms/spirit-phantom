"""Tests for lazy CLI dependency loading."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_module_import_avoids_heavy_dependencies() -> None:
    """Importing ``spirit_phantom.cli`` should not load ITK or thermometry deps."""
    script = """
import sys
import spirit_phantom.cli  # noqa: F401

heavy = {"itk", "matplotlib", "nibabel", "spirit_phantom.core.multi_echo_thermometry"}
loaded = sorted(name for name in heavy if name in sys.modules)
print(",".join(loaded))
"""
    completed = subprocess.run(
        args=[sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""

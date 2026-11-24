"""Tests for shared CLI progress and log-capture helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from spirit_phantom.utils.progress import (
    capture_logs_to_file,
    print_cli_stage,
    print_cli_stage_done,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_print_cli_stage_uses_given_total(capsys: pytest.CaptureFixture[str]) -> None:
    """Stage banners should report the workflow's own stage count."""
    started_at = print_cli_stage(
        stage=2, total=3, label="Measuring slice thickness", enabled=True
    )
    print_cli_stage_done(
        stage=2,
        total=3,
        label="Measuring slice thickness",
        started_at=started_at,
        enabled=True,
    )
    print_cli_stage(stage=3, total=3, label="Hidden", enabled=False)

    captured = capsys.readouterr().out
    assert "[2/3] Measuring slice thickness..." in captured
    assert "[2/3] Measuring slice thickness done (" in captured
    assert "Hidden" not in captured


def test_capture_logs_to_file_writes_debug_and_restores_logger(
    tmp_path: Path,
) -> None:
    """Debug records go to the file and the logger is restored afterwards."""
    library_logger = logging.getLogger("spirit_phantom.test_progress")
    parent_logger = logging.getLogger("spirit_phantom")
    previous_level = parent_logger.level
    previous_handlers = list(parent_logger.handlers)
    log_path = tmp_path / "analysis.log"

    with capture_logs_to_file(log_path=log_path):
        library_logger.debug("slice profile detail")
    library_logger.debug("after capture")

    contents = log_path.read_text(encoding="utf-8")
    assert "slice profile detail" in contents
    assert "after capture" not in contents
    assert parent_logger.level == previous_level
    assert parent_logger.handlers == previous_handlers

"""CLI progress and log-capture helpers shared by registration and analyses.

Numbered stage banners (``[1/4] Rigid registration...``) keep the CLI output
consistent across workflows, while :func:`capture_logs_to_file` routes detailed
library diagnostics to a log file instead of the terminal.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SECONDS_PER_MINUTE = 60
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def format_duration(*, seconds: float) -> str:
    """Format a duration for CLI progress messages.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Human-readable duration such as ``12.3s`` or ``2m 05s``.
    """
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    minutes = int(seconds // _SECONDS_PER_MINUTE)
    remaining_seconds = round(seconds - (minutes * _SECONDS_PER_MINUTE))
    if remaining_seconds == _SECONDS_PER_MINUTE:
        minutes += 1
        remaining_seconds = 0
    return f"{minutes}m {remaining_seconds:02d}s"


def print_cli_stage(*, stage: int, total: int, label: str, enabled: bool) -> float:
    """Print a numbered stage banner when CLI progress is enabled.

    Args:
        stage: One-based stage index.
        total: Total number of stages in the workflow.
        label: Short description of the stage.
        enabled: Whether CLI progress messages should be printed.

    Returns:
        ``time.perf_counter()`` value at the start of the stage.
    """
    if enabled:
        print(f"[{stage}/{total}] {label}...")  # noqa: T201
    return time.perf_counter()


def print_cli_stage_done(
    *, stage: int, total: int, label: str, started_at: float, enabled: bool
) -> None:
    """Print a numbered stage completion message with elapsed time.

    Args:
        stage: One-based stage index.
        total: Total number of stages in the workflow.
        label: Short description of the stage.
        started_at: ``time.perf_counter()`` value from stage start.
        enabled: Whether CLI progress messages should be printed.
    """
    if enabled:
        elapsed = format_duration(seconds=time.perf_counter() - started_at)
        print(f"[{stage}/{total}] {label} done ({elapsed})")  # noqa: T201


@contextmanager
def capture_logs_to_file(
    *,
    log_path: Path,
    logger_name: str = "spirit_phantom",
    level: int = logging.DEBUG,
) -> Iterator[Path]:
    """Write log records from ``logger_name`` to ``log_path`` for the block.

    The logger level is lowered to ``level`` for the duration of the block and
    restored afterwards. Records still propagate to any handlers configured by
    the caller (e.g. ``--verbose`` console output). When no handlers are
    configured, warnings are still echoed to stderr, matching Python's
    default behaviour.

    Args:
        log_path: Destination log file (overwritten).
        logger_name: Logger whose records (and descendants') are captured.
        level: Minimum level written to the log file.

    Yields:
        The log file path.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    target_logger = logging.getLogger(logger_name)
    previous_level = target_logger.level
    handlers: list[logging.Handler] = []
    if not target_logger.hasHandlers():
        warning_handler = logging.StreamHandler()
        warning_handler.setLevel(logging.WARNING)
        handlers.append(warning_handler)
    file_handler = logging.FileHandler(filename=log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handlers.append(file_handler)

    for handler in handlers:
        target_logger.addHandler(handler)
    target_logger.setLevel(level)
    try:
        yield log_path
    finally:
        for handler in handlers:
            target_logger.removeHandler(handler)
            handler.close()
        target_logger.setLevel(previous_level)

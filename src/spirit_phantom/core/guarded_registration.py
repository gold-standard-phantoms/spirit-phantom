"""Run `register_atlas` in an isolated child process.

`register_atlas` runs elastix in-process via the itk-elastix Python bindings
(`itk.elastix_registration_method`). Memory exhaustion during that call can
surface in two different ways, and this module handles both:

1. ITK's own allocator can fail cleanly. ITK defines `itk::MemoryAllocationError`,
   thrown from `itkImportImageContainer.hxx` when a single allocation request
   is refused outright (observed in practice for e.g. oversized whole-slide
   images, or under an address-space limit such as `RLIMIT_AS`). This
   crosses into Python as a normal, catchable exception -- but as a plain
   `RuntimeError`, not Python's built-in `MemoryError`, so it has to be
   recognised by message content (see `_is_itk_memory_allocation_failure`
   below) to be classified correctly.
2. The OS OOM killer can intervene instead. Linux overcommits memory by
   default: an allocation call typically succeeds immediately (address space
   reserved, no physical pages committed yet), and only fails later, when the
   process actually writes into those pages and the kernel discovers there's
   no physical RAM left. At that point the kernel sends `SIGKILL` directly --
   nothing in ITK, elastix, or Python is ever in the call stack for this, so
   no exception is ever thrown, and nothing inside the process can report it.

Running the registration in a child process means the parent survives
either way: it either receives a classified result over a pipe, or detects
that the child was killed and raises a clear exception itself.

If registration fails with a memory-related error, try a lower-resolution
SPIRIT atlas (for example ``vx0.5`` or ``vx1.0`` instead of the default
``vx0.25``) from
https://github.com/gold-standard-phantoms/public-data/tree/main/phantoms/SPIRIT/atlas ,
close other applications, or run on a machine with more RAM.

"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

    from spirit_phantom.core.registration import RegistrationResult

# Substrings observed in real itk::MemoryAllocationError messages as they
# cross into Python (via SimpleITK/itk-elastix bindings) as a plain
# RuntimeError. Matched case-sensitively against the exception's str().
_ITK_MEMORY_ERROR_MARKERS = (
    "MemoryAllocationError",
    "Failed to allocate memory",
)

_MEMORY_RECOVERY_ADVICE = (
    "Consider: using a lower-resolution SPIRIT atlas "
    "(for example vx0.5 or vx1.0 instead of the default vx0.25) from "
    "https://github.com/gold-standard-phantoms/public-data/tree/main/"
    "phantoms/SPIRIT/atlas , closing other applications, or running on a "
    "machine with more RAM."
)


def _is_itk_memory_allocation_failure(exc: Exception) -> bool:
    """Detect ITK MemoryAllocationError surfaced as a generic exception.

    Best-effort check for an ITK MemoryAllocationError that crossed into
    Python as a plain exception rather than the built-in MemoryError.

    Args:
        exc: The exception caught in the worker.

    Returns:
        True if the exception's message matches a known ITK memory-error
        marker.
    """
    message = str(exc)
    return any(marker in message for marker in _ITK_MEMORY_ERROR_MARKERS)


def _registration_worker(
    conn: Any,
    *,
    moving_image: Path,
    fixed_image: Path,
    output_directory: Path,
    phantom_inverted: bool,
    cli_user: bool,
) -> None:
    """Entry point run inside the isolated child process. Not called directly."""
    try:
        # Imported here, inside the child, so `--help` and other CLI paths
        # that never register anything don't pay the cost of importing ITK.
        from spirit_phantom.core.registration import register_atlas  # noqa: PLC0415

        # print()/stdout in the child is inherited from the parent process
        # (multiprocessing does not redirect it by default), so cli_user
        # progress messages still reach the terminal.
        result = register_atlas(
            moving_image=moving_image,
            fixed_image=fixed_image,
            output_directory=output_directory,
            cli_user=cli_user,
            phantom_inverted=phantom_inverted,
        )
        conn.send(("ok", result))
    except MemoryError as exc:
        # Python's own built-in MemoryError, if anything in the call stack
        # raises it directly.
        conn.send(("error", "memory", str(exc)))
    except Exception as exc:  # noqa: BLE001 - deliberately broad: must not crash silently
        if _is_itk_memory_allocation_failure(exc):
            # ITK's MemoryAllocationError crossed into Python as a
            # RuntimeError -- reclassify it as memory-related rather than
            # letting it fall into the generic exception bucket below.
            conn.send(("error", "memory", f"{type(exc).__name__}: {exc}"))
        else:
            conn.send(
                (
                    "error",
                    "exception",
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )
    finally:
        conn.close()


def run_registration_isolated(
    *,
    moving_image: Path,
    fixed_image: Path,
    output_directory: Path,
    phantom_inverted: bool = False,
    cli_user: bool = False,
) -> RegistrationResult:
    """Run `register_atlas` in a child process, with no memory limit applied.

    Surfaces a process being killed (by the OS OOM killer or otherwise), or
    an ITK memory allocation failure caught cleanly inside the child, as a
    catchable `MemoryError`/`RuntimeError` in the calling process, instead of
    silently taking down the interpreter. See module docstring for the two
    failure modes this handles, and their limitations.

    Args:
        moving_image: Path to the moving (atlas) image.
        fixed_image: Path to the fixed (scanner) image.
        output_directory: Directory for registration outputs.
        phantom_inverted: Whether to apply the initial 180-degree Y rotation.
        cli_user: Whether to print stage-by-stage progress to stdout (passed
            through to `register_atlas`; the child inherits the parent's
            stdout, so this is still visible in the terminal).

    Returns:
        The `RegistrationResult` produced by `register_atlas`.

    Raises:
        MemoryError: If the child process ran out of memory -- either an
            ITK/Python-level allocation failure caught and reclassified
            inside the child, or inferred from the child being killed by
            SIGKILL. Recovery options include using a lower-resolution
            SPIRIT atlas from
            https://github.com/gold-standard-phantoms/public-data/tree/main/phantoms/SPIRIT/atlas ,
            closing other applications, or running on a machine with more
            RAM.
        RuntimeError: If the child process failed for any other reason,
            including exiting abnormally without returning a result.
    """
    # "spawn" starts a fresh interpreter rather than copy-on-write forking
    # the parent's already-loaded ITK/numpy state, which avoids inheriting
    # unrelated memory pressure and possible fork-safety issues with open
    # library handles/threads.
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_registration_worker,
        kwargs={
            "conn": child_conn,
            "moving_image": moving_image,
            "fixed_image": fixed_image,
            "output_directory": output_directory,
            "phantom_inverted": phantom_inverted,
            "cli_user": cli_user,
        },
    )
    proc.start()
    child_conn.close()  # parent only needs its own end;

    result = None
    if parent_conn.poll(timeout=None):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = None
    proc.join()

    if result is None:
        # The child died without sending anything back -- it was killed by
        # a signal (typically SIGKILL from the OS OOM killer) rather than
        # exiting through its own try/except. A clean ITK allocation
        # failure would have been caught and sent as a result instead.
        if proc.exitcode == -9:  # noqa: PLR2004 - Unix SIGKILL (-signal)
            msg = (
                "elastix registration was killed by the operating system "
                "(SIGKILL), most likely due to insufficient available "
                f"memory. {_MEMORY_RECOVERY_ADVICE}"
            )
            raise MemoryError(msg)
        msg = (
            f"Registration worker process exited unexpectedly "
            f"(exit code {proc.exitcode}) without returning a result. This "
            "may indicate an out-of-memory condition or a low-level crash "
            f"in elastix. {_MEMORY_RECOVERY_ADVICE}"
        )
        raise RuntimeError(msg)

    status, *payload = result
    if status == "ok":
        return cast("RegistrationResult", payload[0])

    kind, message = payload
    if kind == "memory":
        msg = f"elastix ran out of memory: {message}. {_MEMORY_RECOVERY_ADVICE}"
        raise MemoryError(msg)
    msg = f"Registration failed in isolated worker process: {message}"
    raise RuntimeError(msg)

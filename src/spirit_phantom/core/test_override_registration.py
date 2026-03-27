"""Integration tests for registration parameter overrides.

This test depends on a specific scan being present in the configuration directory.
This is omitted from the repository as it's not cleared for public release.
spirit-phantom/src/spirit_phantom/core/configuration/SPIRIT_32ch_sess1_am_cold_9_FLASH_1p5mm_withGrappa_2_NoseInFeetDir_20251103083536_FLASH_1p5mm_withGrappa_2_NoseInFeetDir_ND.nii.gz
Replace with an alternative scan to utilise for testing.
"""

from __future__ import annotations

import logging
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from spirit_phantom import get_default_register_moving_image_path
from spirit_phantom.core.registration import register_atlas
from spirit_phantom.core.registration_optimizations import (
    AffineOverrides,
    BSplineOverrides,
    RegistrationOverrides,
    RigidOverrides,
)

logger = logging.getLogger(__name__)
_REAL_IMAGE_FILENAME = (
    "SPIRIT_32ch_sess1_am_cold_9_FLASH_1p5mm_withGrappa_2_"
    "NoseInFeetDir_20251103083536_FLASH_1p5mm_withGrappa_2_"
    "NoseInFeetDir_ND.nii.gz"
)

DEFAULT_MOVING_IMAGE_PATH = get_default_register_moving_image_path()


def _resolve_output_directory(
    *,
    output_directory: Path | None,
    run_label: str,
) -> Path:
    """Resolve an output directory, creating a persistent default when omitted.

    Args:
        output_directory: Optional caller-provided output directory.
        run_label: Label used in default persistent path naming.

    Returns:
        The resolved output directory path.
    """
    if output_directory is not None:
        return output_directory

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    default_root = Path.cwd() / "registration_override_outputs"
    resolved = default_root / f"{timestamp}_{run_label}"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_maximum_number_of_iterations(parameter_path: Path) -> int:
    """Read the MaximumNumberOfIterations value from an elastix parameter file.

    Args:
        parameter_path: Path to the elastix parameter file written by registration.

    Returns:
        The parsed MaximumNumberOfIterations value.
    """
    text = parameter_path.read_text(encoding="utf-8")
    match = re.search(
        r"\(MaximumNumberOfIterations\s+([0-9]+)\)",
        text,
    )
    if match is None:
        msg = (
            "MaximumNumberOfIterations was not found in "
            f"parameter file: {parameter_path}"
        )
        raise AssertionError(msg)
    return int(match.group(1))


def _run_registration_pair(
    *,
    fixed_image_path: Path,
    moving_image_path: Path,
    default_output_directory: Path,
    override_output_directory: Path,
) -> tuple[Path, Path]:
    """Run default and overridden registration and validate propagation.

    Args:
        fixed_image_path: Path to the image used as fixed input.
        moving_image_path: Path to the image used as moving input.
        default_output_directory: Output directory for default registration.
        override_output_directory: Output directory for override registration.

    Returns:
        Tuple of `(default_output_directory, override_output_directory)`.
    """
    default_result = register_atlas(
        moving_image=moving_image_path,
        fixed_image=fixed_image_path,
        output_directory=default_output_directory,
    )

    assert default_result.registered_image_path.exists()
    assert default_result.registration_transform_path.exists()
    assert (
        _read_maximum_number_of_iterations(default_result.rigid_parameters_path) == 500
    )
    assert (
        _read_maximum_number_of_iterations(default_result.affine_parameters_path)
        == 1000
    )
    assert (
        _read_maximum_number_of_iterations(default_result.bspline_parameters_path)
        == 1000
    )

    override_values = RegistrationOverrides(
        RigidOverrides=RigidOverrides(MaximumNumberOfIterations=60),
        AffineOverrides=AffineOverrides(MaximumNumberOfIterations=55),
        BSplineOverrides=BSplineOverrides(MaximumNumberOfIterations=50),
    )
    override_result = register_atlas(
        moving_image=moving_image_path,
        fixed_image=fixed_image_path,
        output_directory=override_output_directory,
        overrides=override_values,
    )

    assert override_result.registered_image_path.exists()
    assert override_result.registration_transform_path.exists()
    assert (
        _read_maximum_number_of_iterations(override_result.rigid_parameters_path) == 60
    )
    assert (
        _read_maximum_number_of_iterations(override_result.affine_parameters_path) == 55
    )
    assert (
        _read_maximum_number_of_iterations(override_result.bspline_parameters_path)
        == 50
    )
    return default_output_directory, override_output_directory


def test_registration_maximum_iterations_overrides_propagate() -> None:
    """Run default and override registration and verify stage parameter propagation."""
    fixed_image_path = Path(__file__).parent / "configuration" / _REAL_IMAGE_FILENAME
    if not fixed_image_path.exists():
        pytest.skip(f"Required integration image not found: {fixed_image_path}")

    moving_image_path = DEFAULT_MOVING_IMAGE_PATH

    with (
        tempfile.TemporaryDirectory() as default_dir_name,
        tempfile.TemporaryDirectory() as override_dir_name,
    ):
        _run_registration_pair(
            fixed_image_path=fixed_image_path,
            moving_image_path=moving_image_path,
            default_output_directory=Path(default_dir_name),
            override_output_directory=Path(override_dir_name),
        )


def main() -> int:
    """Run the override registration scenario with persistent output directories.

    Returns:
        Process exit code.
    """
    fixed_image_path = Path(__file__).parent / "configuration" / _REAL_IMAGE_FILENAME
    if not fixed_image_path.exists():
        logger.error("Required integration image not found: %s", fixed_image_path)
        return 1

    moving_image_path = DEFAULT_MOVING_IMAGE_PATH

    default_output_directory = _resolve_output_directory(
        output_directory=None,
        run_label="default",
    )
    override_output_directory = _resolve_output_directory(
        output_directory=None,
        run_label="override",
    )
    resolved_default, resolved_override = _run_registration_pair(
        fixed_image_path=fixed_image_path,
        moving_image_path=moving_image_path,
        default_output_directory=default_output_directory,
        override_output_directory=override_output_directory,
    )
    logger.info("Default registration output directory: %s", resolved_default)
    logger.info("Override registration output directory: %s", resolved_override)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())

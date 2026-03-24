"""Run registration cost sweeps for selected B-spline grid spacings.

Ensure that the fixed image and manual segmentation are available in the configuration directory."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from spirit_phantom import get_default_register_moving_image_path
from spirit_phantom.core.registration import compute_registration_cost
from spirit_phantom.core.registration_optimizations import (
    BSplineOverrides,
    RegistrationOverrides,
)

_FIXED_IMAGE_RELATIVE_PATH = Path(
    "spirit_phantom/core/configuration/"
    "SPIRIT_32ch_sess1_am_cold_9_FLASH_1p5mm_withGrappa_2_"
    "NoseInFeetDir_20251103083536_FLASH_1p5mm_withGrappa_2_"
    "NoseInFeetDir_ND.nii.gz"
)
_MANUAL_SEGMENTATION_RELATIVE_PATH = Path(
    "spirit_phantom/core/configuration/"
    "vials_nninteractive_SPIRIT_32ch_sess1_am_cold_9_FLASH_1p5mm_withGrappa_2_"
    "NoseInFeetDir_20251103083536_FLASH_1p5mm_withGrappa_2_"
    "NoseInFeetDir_ND.nii.gz"
)


def _build_default_output_directory() -> Path:
    """Create a timestamped default output directory.

    Returns:
        Path to the created output directory.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")  # noqa: UP017
    output_directory = Path.cwd() / "registered_data" / timestamp
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def _resolve_required_inputs() -> tuple[Path, Path]:
    """Resolve and validate fixed/manual image paths used for the sweep.

    Returns:
        Tuple of ``(fixed_image_path, manual_segmentation_path)``.

    Raises:
        FileNotFoundError: If either required fixed/manual image path is missing.
    """
    src_root = Path(__file__).resolve().parent
    fixed_image_path = src_root / _FIXED_IMAGE_RELATIVE_PATH
    manual_segmentation_path = src_root / _MANUAL_SEGMENTATION_RELATIVE_PATH

    if not fixed_image_path.exists():
        msg = f"Fixed image file not found: {fixed_image_path}"
        raise FileNotFoundError(msg)
    if not manual_segmentation_path.exists():
        msg = f"Manual segmentation file not found: {manual_segmentation_path}"
        raise FileNotFoundError(msg)

    return fixed_image_path, manual_segmentation_path


def main() -> None:
    """Run default and B-spline spacing sweeps and print summary scores."""
    moving_image_path = get_default_register_moving_image_path()
    fixed_image_path, manual_segmentation_path = _resolve_required_inputs()
    base_output_directory = _build_default_output_directory()

    scenarios: list[tuple[float | None, RegistrationOverrides | None]] = [
        (None, None),
        (
            5.0,
            RegistrationOverrides(
                BSplineOverrides=BSplineOverrides(
                    FinalGridSpacingInPhysicalUnits=5.0,
                )
            ),
        ),
        (
            7.0,
            RegistrationOverrides(
                BSplineOverrides=BSplineOverrides(
                    FinalGridSpacingInPhysicalUnits=7.0,
                )
            ),
        ),
        (
            30.0,
            RegistrationOverrides(
                BSplineOverrides=BSplineOverrides(
                    FinalGridSpacingInPhysicalUnits=30.0,
                )
            ),
        ),
    ]

    default_gdo: float | None = None
    overridden_gdo_by_spacing: dict[float, float] = {}
    failed_labels: dict[str, str] = {}

    for spacing_value, overrides in scenarios:
        run_label = (
            "default"
            if spacing_value is None
            else f"final_grid_spacing_{spacing_value:.1f}"
        )
        summary_label = (
            "Default"
            if spacing_value is None
            else f"FinalGridSpacingInPhysicalUnits={spacing_value:.1f}"
        )
        sys.stdout.write(f"Starting run: {summary_label}\n")
        run_output_directory = base_output_directory / run_label
        try:
            gdo_score = compute_registration_cost(
                moving_image=moving_image_path,
                fixed_image=fixed_image_path,
                output_directory=run_output_directory,
                manual_segmentation_image_path=manual_segmentation_path,
                cli_user=True,
                overrides=overrides,
            )
        except Exception as exc:  # noqa: BLE001
            failed_labels[summary_label] = str(exc)
            sys.stderr.write(
                f"Run failed for {summary_label}: {exc}\n"
            )
            continue

        if spacing_value is None:
            default_gdo = gdo_score
        else:
            overridden_gdo_by_spacing[spacing_value] = gdo_score

    if default_gdo is None:
        sys.stdout.write("Default: GDO = FAILED\n")
    else:
        sys.stdout.write(f"Default: GDO = {default_gdo:.6f}\n")

    for spacing_value in (5.0, 7.0, 30.0):
        label = f"FinalGridSpacingInPhysicalUnits={spacing_value:.1f}"
        if spacing_value in overridden_gdo_by_spacing:
            sys.stdout.write(
                f"{label}: GDP = {overridden_gdo_by_spacing[spacing_value]:.6f}\n"
            )
        else:
            sys.stdout.write(f"{label}: GDP = FAILED\n")

    if failed_labels:
        sys.stdout.write("\nFailures:\n")
        for label, error_message in failed_labels.items():
            sys.stdout.write(f"- {label}: {error_message}\n")


if __name__ == "__main__":
    main()

"""Test registration of two images.

Perform a registration of two simple images (rigid, affine, bspline).

"""

import tempfile
from pathlib import Path
from typing import NamedTuple

import itk
import numpy as np
import pytest

from spirit_phantom.core.registration import (
    RegistrationResult,
    _register,
    compute_registration_cost,
)

TOLERANCE_POINT_TRANSFORMATION = 2.0
DICE_SCORE_THRESHOLD = 0.9


# Named tuple for point transformation test setup
class PointTestSetup(NamedTuple):
    """Test setup data for point transformation tests."""

    fixed_image: itk.Image
    moving_image: itk.Image
    expected_fixed_center: list[float]
    expected_fixed_corner_1: list[float]
    expected_fixed_corner_2: list[float]
    points_path: Path


def image_generator(x1: int, x2: int, y1: int, y2: int, size: int = 100) -> itk.Image:
    """Generate a simple binary image with a rectangle.

    Args:
        x1: The x-coordinate of the top-left corner of the rectangle.
        x2: The x-coordinate of the bottom-right corner of the rectangle.
        y1: The y-coordinate of the top-left corner of the rectangle.
        y2: The y-coordinate of the bottom-right corner of the rectangle.
        size: The size of each side of the square image.

    Returns:
        A binary image with a rectangle.
    """
    image = np.zeros([size, size], np.float32)
    image[y1:y2, x1:x2] = 1
    return itk.image_view_from_array(image)


def test_registration() -> None:
    """Test registration and verify dice score > DICE_SCORE_THRESHOLD."""
    fixed_image = image_generator(25, 75, 25, 75)
    moving_image = image_generator(1, 51, 10, 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)
        _, composed_forward_transform, _ = _register(
            fixed_image, moving_image, save_path=save_path
        )

        # Forward: moving -> aligned with fixed
        registered_moving_image = itk.transformix_filter(
            moving_image, composed_forward_transform
        )

        # Compute Dice coefficients
        fixed_arr = itk.array_from_image(fixed_image)
        registered_moving_arr = itk.array_from_image(registered_moving_image)

        # Compute Dice score of the fixed image and the registered moving image.
        dice_moving_to_fixed = (
            2
            * np.sum(fixed_arr * registered_moving_arr)
            / (np.sum(fixed_arr) + np.sum(registered_moving_arr))
        )
        assert dice_moving_to_fixed > DICE_SCORE_THRESHOLD, (
            f"Moving to fixed registration failed with Dice score: {dice_moving_to_fixed}"
        )


def test_compute_registration_cost_writes_snapshot_and_returns_gdo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write Dice snapshot artefact and return computed GDO."""
    output_directory = tmp_path / "registration_output"
    registered_image_path = output_directory / "Bspline_Image.nii.gz"
    manual_segmentation_image_path = tmp_path / "manual_segmentation.nii.gz"
    moving_image_path = tmp_path / "moving.nii.gz"
    fixed_image_path = tmp_path / "fixed.nii.gz"

    for path in (manual_segmentation_image_path, moving_image_path, fixed_image_path):
        path.write_text("placeholder", encoding="utf-8")

    def _fake_register_atlas(**_: object) -> RegistrationResult:
        output_directory.mkdir(parents=True, exist_ok=True)
        registered_image_path.write_text("registered", encoding="utf-8")
        return RegistrationResult(
            rigid_image_path=output_directory / "Rigid_Image.nii.gz",
            affine_image_path=output_directory / "Affine_Image.nii.gz",
            bspline_image_path=registered_image_path,
            rigid_parameters_path=output_directory / "Rigid_Parameters_In.txt",
            affine_parameters_path=output_directory / "Affine_Parameters_In.txt",
            bspline_parameters_path=output_directory / "BSpline_Parameters_In.txt",
            rigid_transform_path=output_directory / "Rigid_Transform.txt",
            affine_transform_path=output_directory / "Affine_Transform.txt",
            bspline_transform_path=output_directory / "BSpline_Transform.txt",
            registered_image_path=registered_image_path,
            registration_transform_path=output_directory / "BSpline_Transform.txt",
        )

    monkeypatch.setattr(
        "spirit_phantom.core.registration.register_atlas",
        _fake_register_atlas,
    )

    fake_rows = [
        {
            "vial_id": "A",
            "manual_label": 1,
            "atlas_label": 17,
            "dice_score": 0.8,
            "manual_voxels": 100,
            "atlas_voxels": 100,
            "intersection_voxels": 80,
        }
    ]
    monkeypatch.setattr(
        "spirit_phantom.core.vials.generate_dice_score_table",
        lambda **_: fake_rows,
    )
    monkeypatch.setattr(
        "spirit_phantom.core.vials.compute_generalised_dice_overlap",
        lambda **_: 0.8,
    )

    gdo = compute_registration_cost(
        moving_image=moving_image_path,
        fixed_image=fixed_image_path,
        output_directory=output_directory,
        manual_segmentation_image_path=manual_segmentation_image_path,
    )

    assert gdo == pytest.approx(0.8)
    table_path = output_directory / "snapshot" / "dice_scores.txt"
    assert table_path.exists()
    table_content = table_path.read_text(encoding="utf-8")
    assert "Generalised Dice Overlap: 0.800000" in table_content

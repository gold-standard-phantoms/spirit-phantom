"""Test registration of two images.

Perform a registration of two simple images (rigid, affine, bspline).

"""

import tempfile
from pathlib import Path
from typing import NamedTuple

import itk
import numpy as np

from spirit_phantom.core.registration import _register

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

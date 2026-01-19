"""Test registration of two images.

Perform a registration of two simple images (rigid, affine, bspline).
Perform the inverse registration.
That the registration is correct and the logic is correct by registering images and points
from the moving image domain to the fixed image domain and back.

"""

import csv
import tempfile
from pathlib import Path
from typing import NamedTuple

import itk
import numpy as np

from spirit_phantom.core.registration import (
    inverse_register,
    register,
    transform_points_moving_to_fixed,
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
    points_csv_path: Path


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
        _, composed_forward_transform, _ = register(
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


def _setup_point_transformation_test(
    save_path: Path,
) -> PointTestSetup:
    """Set up test resources for point transformation testing.

    Args:
        save_path: Path to the temporary directory for test files.

    Returns:
        PointTestSetup named tuple containing:
        - fixed_image: The fixed (target) test image.
        - moving_image: The moving (source) test image.
        - expected_fixed_center: Expected center point in fixed image space.
        - expected_fixed_corner_1: Expected first corner point in fixed image space.
        - expected_fixed_corner_2: Expected second corner point in fixed image space.
        - points_csv_path: Path to the CSV file containing test points.
    """
    # Create test images
    fixed_image = image_generator(25, 75, 25, 75)
    moving_image = image_generator(1, 51, 10, 60)

    # Expected point mappings (from demo):
    # Moving center (26, 35) -> Fixed center (~50, 50)
    # Moving corners should map to fixed space accordingly
    moving_center = [26.0, 35.0]
    moving_corner1 = [1.0, 10.0]
    moving_corner2 = [51.0, 60.0]

    expected_fixed_center = [50.0, 50.0]  # Approximate, will verify with tolerance
    expected_fixed_corner_1 = [25.0, 25.0]
    expected_fixed_corner_2 = [75.0, 75.0]

    # Create temporary CSV file with test points
    points_csv_path = save_path / "test_points.csv"
    with points_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y"])  # Header
        writer.writerow(moving_center)
        writer.writerow(moving_corner1)
        writer.writerow(moving_corner2)

    return PointTestSetup(
        fixed_image=fixed_image,
        moving_image=moving_image,
        expected_fixed_center=expected_fixed_center,
        expected_fixed_corner_1=expected_fixed_corner_1,
        expected_fixed_corner_2=expected_fixed_corner_2,
        points_csv_path=points_csv_path,
    )


def test_inverse_registration() -> None:
    """Test inverse registration and verify dice score > DICE_SCORE_THRESHOLD."""
    # Create test images
    fixed_image = image_generator(25, 75, 25, 75)
    moving_image = image_generator(1, 51, 10, 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)

        # Perform registration
        _, _, _ = register(fixed_image, moving_image, save_path=save_path)

        # Get the forward transform file path (BSplineTransform.txt)
        forward_transform_file = save_path / "BSpline_Transform.txt"

        # Perform inverse registration
        composed_inverse_transform, _ = inverse_register(
            fixed_image, forward_transform_file, save_path=save_path
        )

        # Apply inverse transform to fixed image -> should match moving image
        inverse_created_moving_image = itk.transformix_filter(
            fixed_image, composed_inverse_transform
        )

        # Compute Dice coefficients
        moving_arr = itk.array_from_image(moving_image)
        inverse_created_moving_arr = itk.array_from_image(inverse_created_moving_image)

        dice_inverse = (
            2
            * np.sum(moving_arr * inverse_created_moving_arr)
            / (np.sum(moving_arr) + np.sum(inverse_created_moving_arr))
        )
        assert dice_inverse > DICE_SCORE_THRESHOLD, (
            f"Inverse registration failed with Dice score: {dice_inverse}"
        )


def test_transform_points_moving_to_fixed() -> None:
    """Test point transformation from moving to fixed image space."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir)

        # Set up test resources
        (
            fixed_image,
            moving_image,
            expected_fixed_center,
            expected_fixed_corner_1,
            expected_fixed_corner_2,
            points_csv_path,
        ) = _setup_point_transformation_test(save_path)

        # Perform registration
        _, _, _ = register(fixed_image, moving_image, save_path=save_path)

        # Get the forward transform file path (BSplineTransform.txt)
        forward_transform_file = save_path / "BSpline_Transform.txt"

        # Perform inverse registration
        _, _ = inverse_register(
            fixed_image, forward_transform_file, save_path=save_path
        )

        # Create subdirectory for CSV output
        csv_output_dir = save_path / "point_transforms"
        csv_output_dir.mkdir()

        # Transform points
        transformed_points = transform_points_moving_to_fixed(
            fixed_image,
            forward_transform_file,
            points_csv_path,
            save_path=csv_output_dir,
        )

        # Verify array results
        assert transformed_points.shape == (3, 2), (
            f"Expected shape (3, 2), got {transformed_points.shape}"
        )

        # Check transformed center point matches expected fixed space center
        transformed_center = transformed_points[0]
        center_error = np.linalg.norm(
            np.array(transformed_center) - np.array(expected_fixed_center)
        )
        assert center_error < TOLERANCE_POINT_TRANSFORMATION, (
            f"Center point transformation error too large: {center_error}. "
            f"Expected ~{expected_fixed_center}, got {transformed_center}"
        )

        # Check transformed corner points match expected fixed space corners
        transformed_corner_1 = transformed_points[1]
        transformed_corner_2 = transformed_points[2]
        corner_1_error = np.linalg.norm(
            np.array(transformed_corner_1) - np.array(expected_fixed_corner_1)
        )
        corner_2_error = np.linalg.norm(
            np.array(transformed_corner_2) - np.array(expected_fixed_corner_2)
        )
        assert corner_1_error < TOLERANCE_POINT_TRANSFORMATION, (
            f"Corner 1 point transformation error too large: {corner_1_error}. "
            f"Expected ~{expected_fixed_corner_1}, got {transformed_corner_1}"
        )
        assert corner_2_error < TOLERANCE_POINT_TRANSFORMATION, (
            f"Corner 2 point transformation error too large: {corner_2_error}. "
            f"Expected ~{expected_fixed_corner_2}, got {transformed_corner_2}"
        )

        # Verify all points are in reasonable fixed space range (0-100 for 100x100 image)
        assert np.all(transformed_points >= 0), "Some points are negative"
        assert np.all(transformed_points <= 100), "Some points exceed image bounds"

        # Verify CSV file was created and contains correct data
        # The CSV is saved to csv_output_dir (the save_path passed to transform_points_moving_to_fixed)
        output_csv_path = csv_output_dir / "transformed_points.csv"
        assert output_csv_path.exists(), "Transformed points CSV file was not created"

        # Read the saved CSV file
        csv_points = []
        csv_header = None
        with output_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            csv_header = next(reader)
            for row in reader:
                if row:
                    csv_points.append([float(x) for x in row])

        # Verify header matches input format
        assert csv_header == ["X", "Y"], (
            f"CSV header mismatch: expected ['X', 'Y'], got {csv_header}"
        )

        # Verify all points are present
        assert len(csv_points) == 3, f"Expected 3 points in CSV, got {len(csv_points)}"

        # Convert to numpy array for comparison
        csv_array = np.array(csv_points, dtype=np.float64)

        # Compare CSV values with array return values (should match exactly)
        assert np.allclose(csv_array, transformed_points, rtol=1e-10), (
            "CSV values do not match array return values"
        )

        # Compare CSV values with expected transformed coordinates
        # The center point should be close to expected fixed center
        csv_center = csv_array[0]
        csv_center_error = np.linalg.norm(
            np.array(csv_center) - np.array(expected_fixed_center)
        )
        assert csv_center_error < TOLERANCE_POINT_TRANSFORMATION, (
            f"CSV center point error too large: {csv_center_error}. "
            f"Expected ~{expected_fixed_center}, got {csv_center}"
        )
        csv_corner_1 = csv_array[1]
        csv_corner_2 = csv_array[2]
        csv_corner_1_error = np.linalg.norm(
            np.array(csv_corner_1) - np.array(expected_fixed_corner_1)
        )
        csv_corner_2_error = np.linalg.norm(
            np.array(csv_corner_2) - np.array(expected_fixed_corner_2)
        )
        assert csv_corner_1_error < TOLERANCE_POINT_TRANSFORMATION, (
            f"CSV corner 1 point error too large: {csv_corner_1_error}. "
            f"Expected ~{expected_fixed_corner_1}, got {csv_corner_1}"
        )
        assert csv_corner_2_error < TOLERANCE_POINT_TRANSFORMATION, (
            f"CSV corner 2 point error too large: {csv_corner_2_error}. "
            f"Expected ~{expected_fixed_corner_2}, got {csv_corner_2}"
        )

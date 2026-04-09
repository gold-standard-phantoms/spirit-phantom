"""Tests for initial registration transform helpers."""

from pathlib import Path

import itk
import numpy as np

from spirit_phantom.core.initial_transform import (
    image_geometric_centre,
    write_initial_flip_transform,
)


def _build_test_image() -> itk.Image:
    """Build a deterministic 3D test image with non-default geometry."""
    image_array = np.zeros((5, 7, 9), dtype=np.float32)
    image = itk.image_view_from_array(image_array)
    image.SetOrigin((10.0, -2.0, 3.0))
    image.SetSpacing((2.0, 4.0, 6.0))
    return image


def _read_transform_parameters_line(*, transform_path: Path) -> str:
    """Return the transform parameter line from a written transform file."""
    for line in transform_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("(TransformParameters"):
            return line
    msg = f"Transform parameters line not found in {transform_path}"
    raise AssertionError(msg)


def test_image_geometric_centre() -> None:
    """Image geometric centre should match the expected world coordinate."""
    image = _build_test_image()
    size = np.asarray(image.GetLargestPossibleRegion().GetSize(), dtype=float)
    expected_centre = (
        np.asarray(image.GetOrigin(), dtype=float)
        + np.asarray(
            image.GetSpacing(),
            dtype=float,
        )
        * (size - 1.0)
        / 2.0
    )

    measured_centre = image_geometric_centre(image=image)

    assert np.allclose(measured_centre, expected_centre)


def test_write_initial_flip_transform(tmp_path: Path) -> None:
    """Default transform file should be written with Euler flip content."""
    image = _build_test_image()
    transform_path = tmp_path / "initial_flip_transform.txt"

    written_path = write_initial_flip_transform(transform_path, image=image)
    content = written_path.read_text(encoding="utf-8")

    assert written_path.exists()
    assert written_path == transform_path
    assert content
    assert '(Transform "EulerTransform")' in content
    assert "(CenterOfRotationPoint " in content
    parameters_line = _read_transform_parameters_line(transform_path=written_path)
    assert parameters_line.startswith("(TransformParameters 0 3.14159")


def test_write_initial_flip_transform_custom_rotation(tmp_path: Path) -> None:
    """Custom Euler rotation should be reflected in transform parameters."""
    image = _build_test_image()
    transform_path = tmp_path / "custom_initial_flip_transform.txt"

    write_initial_flip_transform(
        transform_path,
        image=image,
        rotation=(0.1, 0.2, 0.3),
    )
    parameters_line = _read_transform_parameters_line(transform_path=transform_path)

    assert parameters_line == "(TransformParameters 0.1 0.2 0.3 0 0 0)"

"""Tests for fiducial transform error evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel
import numpy as np

from spirit_phantom.io import fiducial_evaluation
from spirit_phantom.io.fiducial_evaluation import evaluate_fiducial_transform_error
from spirit_phantom.io.points import save_points

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _write_segmentation(*, path: Path) -> None:
    """Create a small 3D labelled segmentation for centroid extraction."""
    data = np.zeros((5, 5, 5), dtype=np.float32)
    data[1, 1, 1] = 1.0
    data[3, 3, 3] = 2.0
    affine = np.diag([2.0, 3.0, 4.0, 1.0]).astype(np.float64)
    image = nibabel.Nifti1Image(dataobj=data, affine=affine)
    nibabel.save(img=image, filename=str(path))


def test_evaluate_fiducial_transform_error_returns_sse_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compute expected voxel/mm errors and return a detailed dataclass result."""
    segmentation_path = tmp_path / "seg.nii.gz"
    moving_image_path = tmp_path / "moving.nii.gz"
    transform_path = tmp_path / "transform.txt"
    reference_path = tmp_path / "reference_points.txt"
    output_directory = tmp_path / "outputs"

    _write_segmentation(path=segmentation_path)
    _write_segmentation(path=moving_image_path)
    transform_path.write_text("dummy", encoding="utf-8")

    save_points(
        points=[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        output_path=reference_path,
        point_type="index",
    )

    transformed = np.array([[2.0, 1.0, 1.5], [2.0, 4.0, 3.0]], dtype=np.float64)

    def _fake_transform_points_fixed_to_moving(
        *,
        moving_image_path: Path,
        registration_transform_path: Path,
        points_in_fixed_domain_path: Path,
        save_path: Path | None = None,
        log_to_console: bool = False,
    ) -> np.ndarray:
        del (
            moving_image_path,
            registration_transform_path,
            points_in_fixed_domain_path,
            save_path,
            log_to_console,
        )
        return transformed

    monkeypatch.setattr(
        fiducial_evaluation,
        "transform_points_fixed_to_moving",
        _fake_transform_points_fixed_to_moving,
    )

    sse_mm, result = evaluate_fiducial_transform_error(
        segmentation_path=segmentation_path,
        moving_image_path=moving_image_path,
        registration_transform_path=transform_path,
        reference_fiducials_path=reference_path,
        output_directory=output_directory,
    )

    expected_differences_vx = np.array([[1.0, 0.0, 0.5], [-1.0, 1.0, 0.0]])
    expected_differences_mm = expected_differences_vx * np.array([2.0, 3.0, 4.0])
    expected_sse_mm = float(np.sum(expected_differences_mm**2))

    assert np.isclose(sse_mm, expected_sse_mm)
    np.testing.assert_allclose(result.differences_vx, expected_differences_vx)
    np.testing.assert_allclose(result.differences_mm, expected_differences_mm)
    assert result.output_directory == output_directory
    assert (
        result.centres_points_path == output_directory / "centres_from_segmentation.txt"
    )

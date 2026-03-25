"""Utilities for evaluating fiducial transform accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import nibabel
import numpy as np
import numpy.typing as npt

from spirit_phantom.io.point_transforms import transform_points_fixed_to_moving
from spirit_phantom.io.points import load_points, save_points
from spirit_phantom.io.segmentations import compute_label_stats

PointType = Literal["point", "index"]
FloatArray = npt.NDArray[np.float64]
_EXPECTED_POINT_NDIM = 3
_POINT_ARRAY_NDIM = 2
_EXPECTED_AFFINE_SHAPE = (4, 4)


@dataclass(frozen=True)
class FiducialEvaluationResult:
    """Detailed fiducial registration error outputs."""

    segmentation_centres_vx: FloatArray
    transformed_centres_vx: FloatArray
    reference_centres_vx: FloatArray
    differences_vx: FloatArray
    differences_mm: FloatArray
    magnitudes_vx: FloatArray
    magnitudes_mm: FloatArray
    mean_squared_error_vx: float
    mean_squared_error_mm: float
    mean_absolute_error_vx: float
    mean_absolute_error_mm: float
    r2_mm: float
    aggregate_rms_magnitude_error_mm: float
    aggregate_mae_magnitude_error_mm: float
    centres_points_path: Path | None
    output_directory: Path | None


def _scale_vx_errors_to_mm(
    *, differences_vx: FloatArray, affine: FloatArray
) -> FloatArray:
    """Scale voxel-space error vectors to millimetres.

    Args:
        differences_vx: Array with shape ``(n_points, 3)`` of per-point error vectors.
        affine: NIfTI affine with shape ``(4, 4)``.

    Returns:
        Error vectors expressed in millimetres.
    """
    if (
        differences_vx.ndim != _POINT_ARRAY_NDIM
        or differences_vx.shape[1] != _EXPECTED_POINT_NDIM
    ):
        msg = (
            f"differences_vx must have shape (n_points, 3); got {differences_vx.shape}."
        )
        raise ValueError(msg)
    if affine.shape != _EXPECTED_AFFINE_SHAPE:
        msg = f"affine must have shape (4, 4); got {affine.shape}."
        raise ValueError(msg)

    linear = affine[:3, :3]
    return np.asarray(differences_vx @ linear.T, dtype=np.float64)


def _points_vx_to_mm(*, points_vx: FloatArray, affine: FloatArray) -> FloatArray:
    """Convert voxel index points to millimetres.

    Args:
        points_vx: Array with shape ``(n_points, 3)`` of voxel indices.
        affine: NIfTI affine with shape ``(4, 4)``.

    Returns:
        Point coordinates in millimetres.
    """
    if (
        points_vx.ndim != _POINT_ARRAY_NDIM
        or points_vx.shape[1] != _EXPECTED_POINT_NDIM
    ):
        msg = f"points_vx must have shape (n_points, 3); got {points_vx.shape}."
        raise ValueError(msg)
    if affine.shape != _EXPECTED_AFFINE_SHAPE:
        msg = f"affine must have shape (4, 4); got {affine.shape}."
        raise ValueError(msg)

    ones = np.ones((points_vx.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_vx, ones], axis=1)
    points_mm_h = (affine @ points_h.T).T
    return np.asarray(points_mm_h[:, :3], dtype=np.float64)


def _r2_score(*, true_vals: FloatArray, pred_vals: FloatArray) -> float:
    """Compute R^2 for two arrays after flattening.

    Args:
        true_vals: Ground-truth values.
        pred_vals: Predicted values.

    Returns:
        Coefficient of determination.
    """
    true_flat = np.asarray(true_vals, dtype=np.float64).ravel()
    pred_flat = np.asarray(pred_vals, dtype=np.float64).ravel()
    if true_flat.shape != pred_flat.shape:
        msg = (
            "true_vals and pred_vals must have the same shape; "
            f"got {true_flat.shape} vs {pred_flat.shape}."
        )
        raise ValueError(msg)

    ss_res = float(np.sum((true_flat - pred_flat) ** 2))
    ss_tot = float(np.sum((true_flat - float(np.mean(true_flat))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 1.0


def evaluate_fiducial_transform_error(
    segmentation_path: Path,
    moving_image_path: Path,
    registration_transform_path: Path,
    reference_fiducials_path: Path,
    *,
    output_directory: Path | None = None,
    segmentation_point_type: PointType = "index",
    reference_flip_affine: FloatArray | None = None,
    log_to_console: bool = False,
) -> tuple[float, FiducialEvaluationResult]:
    """Evaluate fiducial transform error from segmentation to reference points.

    Args:
        segmentation_path: Path to a labelled fiducial segmentation.
        moving_image_path: Path to the moving image used by transformix.
        registration_transform_path: Path to transformix transform parameters.
        reference_fiducials_path: Path to reference fiducial coordinates.
        output_directory: Optional output directory for intermediate point files.
        segmentation_point_type: Transformix point type for segmentation centroids.
        reference_flip_affine: Optional ``(3, 3)`` affine applied to reference points.
        log_to_console: Whether to enable transformix console logging.

    Returns:
        Tuple ``(sum_squared_error_mm, evaluation_result)``.
    """
    seg_img = nibabel.nifti1.load(str(segmentation_path))
    seg_data = np.asarray(seg_img.get_fdata(), dtype=np.float64)
    stats = compute_label_stats(seg_data=seg_data)
    if stats.shape[0] == 0:
        msg = "No non-zero labels found in segmentation."
        raise ValueError(msg)

    segmentation_centres_vx = np.asarray(stats[:, 1:4], dtype=np.float64)
    reference_centres_vx = np.asarray(load_points(points_path=reference_fiducials_path))
    reference_centres_vx = reference_centres_vx.astype(np.float64, copy=False)

    if reference_flip_affine is not None:
        flip = np.asarray(reference_flip_affine, dtype=np.float64)
        if flip.shape != (3, 3):
            msg = f"reference_flip_affine must have shape (3, 3); got {flip.shape}."
            raise ValueError(msg)
        reference_centres_vx = reference_centres_vx @ flip.T

    with TemporaryDirectory() as tmpdir:
        working_directory = output_directory or Path(tmpdir)
        working_directory.mkdir(parents=True, exist_ok=True)
        centres_points_path = working_directory / "centres_from_segmentation.txt"
        save_points(
            points=segmentation_centres_vx.tolist(),
            output_path=centres_points_path,
            point_type=segmentation_point_type,
        )
        transformed_centres_vx = transform_points_fixed_to_moving(
            moving_image_path=moving_image_path,
            registration_transform_path=registration_transform_path,
            points_in_fixed_domain_path=centres_points_path,
            save_path=working_directory,
            log_to_console=log_to_console,
        )

        transformed_centres_vx = np.asarray(transformed_centres_vx, dtype=np.float64)
        if transformed_centres_vx.shape != reference_centres_vx.shape:
            msg = (
                "Shape mismatch between transformed and reference centres: "
                f"{transformed_centres_vx.shape} vs {reference_centres_vx.shape}."
            )
            raise ValueError(msg)

        differences_vx = transformed_centres_vx - reference_centres_vx
        sum_squared_error_vx = float(np.sum(differences_vx**2))
        mean_squared_error_vx = sum_squared_error_vx / float(differences_vx.shape[0])
        mean_absolute_error_vx = float(np.mean(np.abs(differences_vx)))
        magnitudes_vx = np.linalg.norm(differences_vx, axis=1)

        moving_img = nibabel.nifti1.load(str(moving_image_path))
        moving_affine = np.asarray(moving_img.affine, dtype=np.float64)
        differences_mm = _scale_vx_errors_to_mm(
            differences_vx=differences_vx,
            affine=moving_affine,
        )
        sum_squared_error_mm = float(np.sum(differences_mm**2))
        mean_squared_error_mm = sum_squared_error_mm / float(differences_mm.shape[0])
        mean_absolute_error_mm = float(np.mean(np.abs(differences_mm)))
        magnitudes_mm = np.linalg.norm(differences_mm, axis=1)

        reference_mm = _points_vx_to_mm(
            points_vx=reference_centres_vx, affine=moving_affine
        )
        transformed_mm = _points_vx_to_mm(
            points_vx=transformed_centres_vx,
            affine=moving_affine,
        )
        r2_mm = _r2_score(true_vals=reference_mm, pred_vals=transformed_mm)

        result = FiducialEvaluationResult(
            segmentation_centres_vx=segmentation_centres_vx,
            transformed_centres_vx=transformed_centres_vx,
            reference_centres_vx=reference_centres_vx,
            differences_vx=differences_vx,
            differences_mm=differences_mm,
            magnitudes_vx=magnitudes_vx,
            magnitudes_mm=magnitudes_mm,
            mean_squared_error_vx=mean_squared_error_vx,
            mean_squared_error_mm=mean_squared_error_mm,
            mean_absolute_error_vx=mean_absolute_error_vx,
            mean_absolute_error_mm=mean_absolute_error_mm,
            r2_mm=r2_mm,
            aggregate_rms_magnitude_error_mm=float(np.sqrt(np.mean(magnitudes_mm**2))),
            aggregate_mae_magnitude_error_mm=float(np.mean(magnitudes_mm)),
            centres_points_path=centres_points_path if output_directory else None,
            output_directory=working_directory if output_directory else None,
        )
        return sum_squared_error_mm, result

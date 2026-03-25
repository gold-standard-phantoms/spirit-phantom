"""Utilities for transforming point sets with transformix."""

import shutil
import tempfile
from pathlib import Path

import itk
import numpy as np

from spirit_phantom.io.points import parse_transformix_output, save_points


def transform_points_fixed_to_moving(
    moving_image_path: Path,
    registration_transform_path: Path,
    points_in_fixed_domain_path: Path,
    save_path: Path | None = None,
    *,
    log_to_console: bool = False,
) -> np.ndarray:
    """Transform points from fixed image space to moving image space.

    Args:
        moving_image_path: Path to the moving image.
        registration_transform_path: Path to the registration transform.
        points_in_fixed_domain_path: Path to points in the fixed domain.
        save_path: Optional path to save transformed points artefacts.
        log_to_console: Whether to log transformix output to the console.

    Returns:
        Numpy array of transformed points with shape ``[n_points, n_dims]``.
    """
    moving_image_itk = itk.imread(str(moving_image_path), itk.F)

    param_obj = itk.ParameterObject.New()
    param_obj.ReadParameterFile(str(registration_transform_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        transformix_dir = Path(tmpdir) / "transformix_points"
        transformix_dir.mkdir(parents=True, exist_ok=True)
        transformix_filter = itk.TransformixFilter.New(moving_image_itk)
        transformix_filter.SetTransformParameterObject(param_obj)
        transformix_filter.SetLogToConsole(log_to_console)
        transformix_filter.SetFixedPointSetFileName(str(points_in_fixed_domain_path))
        transformix_filter.SetOutputDirectory(str(transformix_dir))
        transformix_filter.UpdateLargestPossibleRegion()

        output_file = transformix_dir / "outputpoints.txt"
        transformed_points = parse_transformix_output(
            output_path=output_file,
            point_type="point",
        )
        transformed_indices = parse_transformix_output(
            output_path=output_file,
            point_type="index",
        )
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(output_file, save_path / "transformix_output_points.txt")
            save_points(
                points=transformed_points,
                output_path=save_path / "transformed_points.txt",
            )
            save_points(
                points=transformed_indices,
                output_path=save_path / "transformed_indices.txt",
            )

    return np.array(transformed_points, dtype=np.float64)

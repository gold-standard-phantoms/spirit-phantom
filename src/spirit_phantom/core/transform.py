"""Module for performing transformations on phantom locations."""

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
) -> np.ndarray:
    """Transform points from fixed image space to moving image space.

    Args:
        moving_image_path: Path to the moving image.
        registration_transform_path: Path to the registration transform.
        points_in_fixed_domain_path: Path to the points in the fixed domain.
        save_path: Optional path to save the transformed points. If None, no file is created.

    Returns:
        Numpy array of transformed points with shape [n_points, n_dims].
        Dimensions are automatically detected from input file (2D or 3D).
    """
    moving_image_itk = itk.imread(str(moving_image_path), itk.F)

    # Load the composed forward transform produced by elastix.
    param_obj = itk.ParameterObject.New()
    param_obj.ReadParameterFile(str(registration_transform_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        transformix_dir = Path(tmpdir) / "transformix_points"
        transformix_dir.mkdir(parents=True, exist_ok=True)
        transformix_filter = itk.TransformixFilter.New(moving_image_itk)
        transformix_filter.SetTransformParameterObject(param_obj)
        transformix_filter.SetFixedPointSetFileName(str(points_in_fixed_domain_path))
        transformix_filter.SetOutputDirectory(str(transformix_dir))
        transformix_filter.Update()
        # Parse the transformix output file.
        output_file = transformix_dir / "outputpoints.txt"
        transformed_points = parse_transformix_output(output_file)
        if save_path is not None:
            # copy the transformix output to save path.
            shutil.copy(output_file, save_path / "transformix_output_points.txt")
            save_points(transformed_points, save_path / "transformed_points.txt")

    # Convert to numpy array
    return np.array(transformed_points, dtype=np.float64)

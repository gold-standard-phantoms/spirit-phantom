"""Module for performing elastix registration of phantom data.

Registration currently consists of an Euler (Rigid) Transform, followed by an Affine
Transform which is followed by a B-Splines transform. The default elastix transforms
loaded and are customized to the use case.

"""

import tempfile
from importlib import resources
from pathlib import Path

import itk
import numpy as np

from spirit_phantom.io.points import (
    load_points_from_csv,
    save_transformed_points_to_csv,
    write_points_to_transformix_format,
)

# Filename constants for output files
RIGID_PARAMETERS_IN_FILENAME = "Rigid_Parameters_In.txt"
AFFINE_PARAMETERS_IN_FILENAME = "Affine_Parameters_In.txt"
BSPLINE_PARAMETERS_IN_FILENAME = "BSpline_Parameters_In.txt"
RIGID_IMAGE_FILENAME = "Rigid_Image.nii.gz"
AFFINE_IMAGE_FILENAME = "Affine_Image.nii.gz"
BSPLINE_IMAGE_FILENAME = "Bspline_Image.nii.gz"
RIGID_TRANSFORM_FILENAME = "Rigid_Transform.txt"
AFFINE_TRANSFORM_FILENAME = "Affine_Transform.txt"
BSPLINE_TRANSFORM_FILENAME = "BSpline_Transform.txt"
INVERSE_TRANSFORM_FILENAME = "InverseTransform.txt"
TRANSFORMED_POINTS_FILENAME = "transformed_points.csv"


def _get_parameter_file_path(filename: str) -> Path:
    """Get the path to a registration parameters file.

    Args:
        filename: The name of the parameter file (e.g., "parameters_Rigid.txt").

    Returns:
        Path to the registration parameters file, either from package data
        or as a fallback from the source directory.

    Raises:
        FileNotFoundError: If the parameter file cannot be found in either
            the installed package or the source directory.
    """
    # Try to access from installed package using modern API
    try:
        config_package = resources.files("spirit_phantom.core.configuration")
        param_path = config_package / filename
        if param_path.is_file():
            return Path(str(param_path))
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    # Fallback to source location for development
    source_path = Path(__file__).parent / "configuration" / filename
    if not source_path.exists():
        msg = (
            f"Parameter file '{filename}' not found. "
            f"Checked package data and source location: {source_path}"
        )
        raise FileNotFoundError(msg)
    return source_path


# Load rigid parameters from file
RIGID_PARAM_FILE = _get_parameter_file_path("parameters_Rigid.txt")
RIGID_PARAM_OBJECT = itk.ParameterObject.New()
RIGID_PARAM_OBJECT.ReadParameterFile(str(RIGID_PARAM_FILE))

# Load affine parameters from file
AFFINE_PARAM_FILE = _get_parameter_file_path("parameters_Affine.txt")
AFFINE_PARAM_OBJECT = itk.ParameterObject.New()
AFFINE_PARAM_OBJECT.ReadParameterFile(str(AFFINE_PARAM_FILE))

# Load B-spline parameters from file
BSPLINE_PARAM_FILE = _get_parameter_file_path("parameters_B_Spline.txt")
BSPLINE_PARAM_OBJECT = itk.ParameterObject.New()
BSPLINE_PARAM_OBJECT.ReadParameterFile(str(BSPLINE_PARAM_FILE))

# Set up inverse transform as having an adjustment to the B-spline
INVERSE_PARAM_MAP = BSPLINE_PARAM_OBJECT.GetParameterMap(0)
INVERSE_PARAM_MAP["Metric"] = ["DisplacementMagnitudePenalty"]
INVERSE_PARAM_OBJECT = itk.ParameterObject.New()
INVERSE_PARAM_OBJECT.AddParameterMap(INVERSE_PARAM_MAP)


def _save_transform_to_file(
    transform_params: itk.ParameterObject,
    filename: Path,
    initial_transform_file: Path | None = None,
) -> Path:
    """Save transform parameters to a file, optionally with reference to initial transform.

    Args:
        transform_params: The transform parameters to save.
        filename: The path to save the transform parameters to.
        initial_transform_file: The path to the initial transform parameters file.

    Returns:
        The path to the saved transform parameters file.
    """
    transform_map = dict(transform_params.GetParameterMap(0))

    if initial_transform_file:
        transform_map["InitialTransformParameterFileName"] = (
            str(initial_transform_file),
        )
    else:
        transform_map["InitialTransformParameterFileName"] = ("NoInitialTransform",)

    param_obj = itk.ParameterObject.New()
    param_obj.AddParameterMap(transform_map)
    param_obj.WriteParameterFile(param_obj.GetParameterMap(0), str(filename))

    return filename


def _perform_rigid_registration(
    fixed_image: itk.Image,
    moving_image: itk.Image,
    save_path: Path,
) -> tuple[itk.Image, itk.ParameterObject, Path]:
    """Perform rigid registration stage.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        save_path: Path to directory for saving transform files.

    Returns:
        A tuple of (registered_image, transform, transform_file_path).
    """
    # Save the parameters used by elastix to perform the rigid transform
    RIGID_PARAM_OBJECT.WriteParameterFile(
        RIGID_PARAM_OBJECT.GetParameterMap(0),
        str(save_path / RIGID_PARAMETERS_IN_FILENAME),
    )
    rigid_image, rigid_transform = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=RIGID_PARAM_OBJECT,
        log_to_console=False,
    )

    # Save rigid registered atlas image to file
    rigid_image_file = save_path / RIGID_IMAGE_FILENAME
    itk.imwrite(rigid_image, str(rigid_image_file))

    # Save the rigid transform produced by elastix to file
    rigid_file = save_path / RIGID_TRANSFORM_FILENAME
    _save_transform_to_file(rigid_transform, rigid_file)

    return rigid_image, rigid_transform, rigid_file


def _perform_affine_registration(
    fixed_image: itk.Image,
    moving_image: itk.Image,
    initial_transform_file: Path,
    save_path: Path,
) -> tuple[itk.Image, itk.ParameterObject, Path]:
    """Perform affine registration stage.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        initial_transform_file: Path to the initial transform file (rigid).
        save_path: Path to directory for saving transform files.

    Returns:
        A tuple of (registered_image, transform, transform_file_path).
    """
    # Save the parameters used by elastix to perform the affine transform
    AFFINE_PARAM_OBJECT.WriteParameterFile(
        AFFINE_PARAM_OBJECT.GetParameterMap(0),
        str(save_path / AFFINE_PARAMETERS_IN_FILENAME),
    )

    affine_image, affine_transform = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=AFFINE_PARAM_OBJECT,
        initial_transform_parameter_file_name=str(initial_transform_file),
        log_to_console=False,
    )

    # Save affine-registered atlas image to file
    affine_image_file = save_path / AFFINE_IMAGE_FILENAME
    itk.imwrite(affine_image, str(affine_image_file))

    # Save the affine transform produced by elastix to file
    affine_file = save_path / AFFINE_TRANSFORM_FILENAME
    _save_transform_to_file(
        affine_transform, affine_file, initial_transform_file=initial_transform_file
    )

    return affine_image, affine_transform, affine_file


def _perform_bspline_registration(
    fixed_image: itk.Image,
    moving_image: itk.Image,
    initial_transform_file: Path,
    save_path: Path,
) -> tuple[itk.Image, itk.ParameterObject, Path]:
    """Perform B-spline registration stage.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        initial_transform_file: Path to the initial transform file (affine).
        save_path: Path to directory for saving transform files.

    Returns:
        A tuple of (registered_image, transform, transform_file_path).
    """
    # Save parameters used by elastix to perform the B-Spline transform
    BSPLINE_PARAM_OBJECT.WriteParameterFile(
        BSPLINE_PARAM_OBJECT.GetParameterMap(0),
        str(save_path / BSPLINE_PARAMETERS_IN_FILENAME),
    )

    bspline_image, bspline_transform = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=BSPLINE_PARAM_OBJECT,
        initial_transform_parameter_file_name=str(initial_transform_file),
        log_to_console=False,
    )

    # Save bspline-registered image to file
    bspline_image_file = save_path / BSPLINE_IMAGE_FILENAME
    itk.imwrite(bspline_image, str(bspline_image_file))

    # Save the B-spline transform produced by elastix to file
    bspline_file = save_path / BSPLINE_TRANSFORM_FILENAME
    _save_transform_to_file(
        bspline_transform, bspline_file, initial_transform_file=initial_transform_file
    )

    return bspline_image, bspline_transform, bspline_file


def _parse_transformix_output_points(
    output_file: Path, expected_count: int
) -> list[list[float]]:
    """Parse transformed points from transformix output file.

    Args:
        output_file: Path to the transformix outputpoints.txt file.
        expected_count: Expected number of points to parse.

    Returns:
        List of transformed point coordinates, each point is a list of floats.

    Raises:
        FileNotFoundError: If output_file does not exist.
        ValueError: If output file is malformed or parsing fails.
    """
    if not output_file.exists():
        msg = (
            f"Transformix output file not found: {output_file}. "
            "Transformix may have failed to process points."
        )
        raise FileNotFoundError(msg)

    transformed_points = []
    output_point_marker = "OutputPoint = ["

    with output_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        for line_num, line in enumerate(lines, start=1):
            if "OutputPoint" in line:
                # Parse line like: "Point   0   ; ... ; OutputPoint = [ 50.0 50.0 ]  ; ..."
                # or "OutputPoint = [ 50.0 50.0 0.0 ]" for 3D
                output_start = line.find(output_point_marker)
                if output_start == -1:
                    msg = (
                        f"Malformed transformix output at line {line_num}: "
                        f"found 'OutputPoint' but missing '{output_point_marker}'"
                    )
                    raise ValueError(msg)

                output_start += len(output_point_marker)
                output_end = line.find("]", output_start)
                if output_end == -1:
                    msg = (
                        f"Malformed transformix output at line {line_num}: "
                        "missing closing bracket ']' for OutputPoint"
                    )
                    raise ValueError(msg)

                coords_str = line[output_start:output_end].strip()
                if not coords_str:
                    msg = (
                        f"Malformed transformix output at line {line_num}: "
                        "OutputPoint coordinates are empty"
                    )
                    raise ValueError(msg)

                try:
                    coords = [float(x) for x in coords_str.split()]
                except ValueError as e:
                    msg = (
                        f"Failed to parse coordinates at line {line_num}: "
                        f"'{coords_str}'. Error: {e}"
                    )
                    raise ValueError(msg) from e

                if not coords:
                    msg = f"Empty coordinates at line {line_num} after parsing"
                    raise ValueError(msg)

                transformed_points.append(coords)

    if len(transformed_points) != expected_count:
        msg = (
            f"Expected {expected_count} transformed points, "
            f"got {len(transformed_points)}"
        )
        raise ValueError(msg)

    return transformed_points


def register(
    fixed_image: itk.Image,
    moving_image: itk.Image,
    save_path: Path,
) -> tuple[itk.Image, itk.ParameterObject, list[Path]]:
    """Register the moving image to the fixed image.

    This function performs a three-stage registration (rigid, affine, B-spline)
    and saves intermediate results and transform files to disk as side effects.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        save_path: Path to directory for saving transform files. Must be writable.

    Returns:
        A tuple of (registered_image, transform, list of paths).
        The list contains the transform files produced during the registration.

    Raises:
        OSError: If save_path cannot be created or is not writable.
        ValueError: If images have incompatible dimensions.
    """
    # Validate save_path can be created
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = save_path / ".write_test"
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        msg = f"Cannot create or write to save_path '{save_path}': {e}"
        raise OSError(msg) from e

    # Validate image dimensions match
    fixed_dims = fixed_image.GetImageDimension()
    moving_dims = moving_image.GetImageDimension()
    if fixed_dims != moving_dims:
        msg = (
            f"Image dimension mismatch: fixed_image has {fixed_dims}D, "
            f"moving_image has {moving_dims}D"
        )
        raise ValueError(msg)

    # Perform three-stage registration: rigid, affine, B-spline
    _, _, rigid_file = _perform_rigid_registration(fixed_image, moving_image, save_path)
    _, _, affine_file = _perform_affine_registration(
        fixed_image, moving_image, rigid_file, save_path
    )
    bspline_image, _, bspline_file = _perform_bspline_registration(
        fixed_image, moving_image, affine_file, save_path
    )

    # Load the composed transform from file (includes chain information via
    # InitialTransformParameterFileName). The files must remain accessible
    # for transformix to resolve the transform chain.
    composed_transform = itk.ParameterObject.New()
    composed_transform.ReadParameterFile(str(bspline_file))
    composed_transform_image = bspline_image

    return (
        composed_transform_image,
        composed_transform,
        [rigid_file, affine_file, bspline_file],
    )


def inverse_register(
    fixed_image: itk.Image,
    forward_transform_file: Path,
    save_path: Path,
) -> tuple[itk.ParameterObject, Path]:
    """Compute inverse transform using DisplacementMagnitudePenalty method.

    From elastix manual section 6.1.6:
    The DisplacementMagnitudePenalty is a cost function that penalizes ||T(x) - x||^2.

    This function saves the inverse transform file to disk as a side effect.

    Args:
        fixed_image: The fixed (target) image.
        forward_transform_file: Path to the forward transform parameter file.
        save_path: Path to directory for saving transform files. Must be writable.

    Returns:
        A tuple of (inverse_transform, inverse_transform_file_path).

    Raises:
        FileNotFoundError: If forward_transform_file does not exist.
        OSError: If save_path cannot be created or is not writable.
    """
    if not forward_transform_file.exists():
        msg = f"Forward transform file not found: {forward_transform_file}"
        raise FileNotFoundError(msg)

    # Validate save_path can be created
    try:
        save_path.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = save_path / ".write_test"
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        msg = f"Cannot create or write to save_path '{save_path}': {e}"
        raise OSError(msg) from e

    # Use B-spline for inverse to handle non-linear deformations

    _, inverse_params = itk.elastix_registration_method(
        fixed_image,
        fixed_image,
        parameter_object=INVERSE_PARAM_OBJECT,
        initial_transform_parameter_file_name=str(forward_transform_file),
        log_to_console=False,
    )

    # Extract standalone inverse
    inverse_map = dict(inverse_params.GetParameterMap(0))
    inverse_map["InitialTransformParameterFileName"] = ("NoInitialTransform",)

    standalone_inverse = itk.ParameterObject.New()
    standalone_inverse.AddParameterMap(inverse_map)

    # Save inverse transform
    inverse_file = save_path / INVERSE_TRANSFORM_FILENAME
    _save_transform_to_file(standalone_inverse, inverse_file)

    return standalone_inverse, inverse_file


def transform_points_moving_to_fixed(
    fixed_image: itk.Image,
    forward_transform_file: Path,
    points_csv_path: Path,
    save_path: Path | None = None,
) -> np.ndarray:
    """Transform points from moving image space to fixed image space.

    This function reads points from a CSV file, computes the inverse transform
    internally from the forward transform file, and uses the inverse transform
    to transform points from moving image space to fixed image space.

    Note: transformix point transformation direction is opposite to image
    transformation. To map points from moving->fixed space, the inverse transform
    is used.

    This function may save transformed points to CSV as a side effect if save_path
    is provided.

    Args:
        fixed_image: The fixed (target) image used as reference for transformix.
        forward_transform_file: Path to the forward transform parameter file.
            This is used to compute the inverse transform internally.
        points_csv_path: Path to CSV file containing points to transform.
            CSV must have a header row (e.g., "X,Y" for 2D or "R,A,S" for 3D)
            followed by rows of comma-separated coordinates.
        save_path: Optional path to directory for saving transformed points CSV.
            If None, no CSV file is created. If provided, saves to
            save_path / "transformed_points.csv".

    Returns:
        Numpy array of transformed points with shape [n_points, n_dims].
        Dimensions are automatically detected from CSV (2D or 3D).

    Raises:
        FileNotFoundError: If points_csv_path or forward_transform_file doesn't exist.
        ValueError: If CSV file is malformed, has invalid coordinates, or transformix
            output parsing fails.
    """
    if not forward_transform_file.exists():
        msg = f"Forward transform file not found: {forward_transform_file}"
        raise FileNotFoundError(msg)

    # Load and validate points from CSV
    points, header, n_dims = load_points_from_csv(points_csv_path)

    # Compute inverse transform
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_save_path = Path(tmpdir)
        inverse_transform, _ = inverse_register(
            fixed_image, forward_transform_file, save_path=temp_save_path
        )

        # Create temporary directory for transformix I/O
        transformix_dir = Path(tmpdir) / "transformix_points"
        transformix_dir.mkdir()

        # Write points to transformix input format
        points_file = transformix_dir / "inputpoints.txt"
        write_points_to_transformix_format(points, n_dims, points_file)

        # Use TransformixFilter with inverse transform
        transformix_obj = itk.TransformixFilter.New(fixed_image)
        transformix_obj.SetTransformParameterObject(inverse_transform)
        transformix_obj.SetFixedPointSetFileName(str(points_file))
        transformix_obj.SetOutputDirectory(str(transformix_dir))
        transformix_obj.Update()

        # Read transformed points from output file
        output_file = transformix_dir / "outputpoints.txt"
        transformed_points = _parse_transformix_output_points(
            output_file, expected_count=len(points)
        )

    # Convert to numpy array
    transformed_array = np.array(transformed_points, dtype=np.float64)

    # Save to CSV if save_path is provided
    if save_path is not None:
        output_csv_path = save_path / TRANSFORMED_POINTS_FILENAME
        save_transformed_points_to_csv(transformed_points, header, output_csv_path)

    return transformed_array


def register_atlas(
    moving_image: Path,
    fixed_image: Path,
    result_image_save_path: Path,
    transform_params_save_path: Path,
) -> list[Path]:
    """Register the moving image to the fixed image using file paths.

    This is a convenience wrapper around the `register` function that handles
    loading images from file paths and saving the results to specified location.

    This function performs file I/O as side effects: it loads images from disk,
    performs registration (which saves intermediate files), and saves the final
    registered image.

    Args:
        moving_image: Path to the moving (source) image to register.
        fixed_image: Path to the fixed (target) reference image.
        result_image_save_path: Path to save the registered result image.
        transform_params_save_path: Path to save the transformation
            parameters files (Rigid_Transform.txt, Affine_Transform.txt, BSpline_Transform.txt).

    Returns:
        A list of paths to the transform parameter files.

    Raises:
        FileNotFoundError: If moving_image or fixed_image does not exist.
        OSError: If result_image_save_path or transform_params_save_path cannot be written to.
    """
    # Validate input files exist
    if not moving_image.exists():
        msg = f"Moving image file not found: {moving_image}"
        raise FileNotFoundError(msg)
    if not fixed_image.exists():
        msg = f"Fixed image file not found: {fixed_image}"
        raise FileNotFoundError(msg)
    # Load images from file paths
    moving_image_itk = itk.imread(str(moving_image), itk.F)
    fixed_image_itk = itk.imread(str(fixed_image), itk.F)

    # Use the transform_params_save_path's parent directory for registration
    # This ensures all transform files are in the same directory
    transform_dir = transform_params_save_path.parent
    try:
        transform_dir.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = transform_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        msg = f"Cannot create or write to transform directory '{transform_dir}': {e}"
        raise OSError(msg) from e

    # Perform registration
    registered_moving_image, _, transform_files = register(
        fixed_image=fixed_image_itk,
        moving_image=moving_image_itk,
        save_path=transform_dir,
    )

    # Save the registered result image
    try:
        result_image_save_path.parent.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = result_image_save_path.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        msg = (
            f"Cannot create or write to result image directory "
            f"'{result_image_save_path.parent}': {e}"
        )
        raise OSError(msg) from e

    itk.imwrite(registered_moving_image, str(result_image_save_path))

    return transform_files

"""Module for performing elastix registration of phantom data.

Registration currently consists of an Euler (Rigid) Transform, followed by an Affine
Transform which is followed by a B-Splines transform. The default elastix transforms
loaded and are customized to the use case.

"""

import logging
from importlib import resources
from pathlib import Path
from typing import NamedTuple

import itk

logger = logging.getLogger(__name__)
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
TRANSFORMED_POINTS_FILENAME = "transformed_points.txt"


class RegistrationResult(NamedTuple):
    """Result of atlas registration.

    Attributes:
        rigid_image_path: Path to the rigid-registered intermediate image.
        affine_image_path: Path to the affine-registered intermediate image.
        bspline_image_path: Path to the B-spline registered (final) image.
        rigid_parameters_path: Path to the input parameters used for rigid registration.
        affine_parameters_path: Path to the input parameters used for affine registration.
        bspline_parameters_path: Path to the input parameters used for B-spline registration.
        rigid_transform_path: Path to the rigid transform output file.
        affine_transform_path: Path to the affine transform output file.
        bspline_transform_path: Path to the B-spline transform output file.
        registered_image_path: Alias for bspline_image_path (the final registered image).
        registration_transform_path: Alias for bspline_transform_path (the final transform).
    """

    rigid_image_path: Path
    affine_image_path: Path
    bspline_image_path: Path
    rigid_parameters_path: Path
    affine_parameters_path: Path
    bspline_parameters_path: Path
    rigid_transform_path: Path
    affine_transform_path: Path
    bspline_transform_path: Path
    registered_image_path: Path
    registration_transform_path: Path


class _StageResult(NamedTuple):
    """Result of a single registration stage (internal use)."""

    image: itk.Image
    transform: itk.ParameterObject
    image_path: Path
    parameters_path: Path
    transform_path: Path


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


def _save_transform_to_file(
    transform_params: itk.ParameterObject,
    filename: Path,
    initial_transform_file: Path | None = None,
) -> Path:
    """Save transform parameters to a file, optionally with reference to initial transform.

    The `initial_transform_file` parameter allows the transform to reference a
    previously run transform, enabling chaining of transforms. When provided,
    the transform will be saved with an `InitialTransformParameterFileName`
    parameter that points to the previous transform file.

    This saved reference is used by transformix (not during registration) when
    applying the transform. Transformix will recursively resolve the transform
    chain by following the `InitialTransformParameterFileName` references.
    For this to work correctly, all transform files referenced in the chain must
    be available to transformix at the specified file paths when the transform
    is applied.

    Args:
        transform_params: The transform parameters to save.
        filename: The path to save the transform parameters to.
        initial_transform_file: Optional path to a previously run transform file.
            When provided, this transform will reference it to enable transform
            chaining. If None, the transform will be saved as standalone.

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
) -> _StageResult:
    """Perform rigid registration stage.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        save_path: Path to directory for saving transform files.

    Returns:
        _StageResult containing the registered image, transform, and all file paths.
    """
    logger.info("Perform rigid registration: start")

    # Save the parameters used by elastix to perform the rigid transform
    parameters_path = save_path / RIGID_PARAMETERS_IN_FILENAME
    RIGID_PARAM_OBJECT.WriteParameterFile(
        RIGID_PARAM_OBJECT.GetParameterMap(0),
        str(parameters_path),
    )
    rigid_image, rigid_transform = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=RIGID_PARAM_OBJECT,
        log_to_console=False,
    )

    # Save rigid registered atlas image to file
    image_path = save_path / RIGID_IMAGE_FILENAME
    itk.imwrite(rigid_image, str(image_path))

    # Save the rigid transform produced by elastix to file
    transform_path = save_path / RIGID_TRANSFORM_FILENAME
    _save_transform_to_file(rigid_transform, transform_path)

    logger.info("Perform rigid registration: end")

    return _StageResult(
        image=rigid_image,
        transform=rigid_transform,
        image_path=image_path,
        parameters_path=parameters_path,
        transform_path=transform_path,
    )


def _perform_affine_registration(
    fixed_image: itk.Image,
    moving_image: itk.Image,
    initial_transform_file: Path,
    save_path: Path,
) -> _StageResult:
    """Perform affine registration stage.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        initial_transform_file: Path to the initial transform file (rigid).
        save_path: Path to directory for saving transform files.

    Returns:
        _StageResult containing the registered image, transform, and all file paths.
    """
    logger.info("Perform affine registration: start")
    # Save the parameters used by elastix to perform the affine transform
    parameters_path = save_path / AFFINE_PARAMETERS_IN_FILENAME
    AFFINE_PARAM_OBJECT.WriteParameterFile(
        AFFINE_PARAM_OBJECT.GetParameterMap(0),
        str(parameters_path),
    )

    affine_image, affine_transform = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=AFFINE_PARAM_OBJECT,
        initial_transform_parameter_file_name=str(initial_transform_file),
        log_to_console=False,
    )

    # Save affine-registered atlas image to file
    image_path = save_path / AFFINE_IMAGE_FILENAME
    itk.imwrite(affine_image, str(image_path))

    # Save the affine transform produced by elastix to file
    transform_path = save_path / AFFINE_TRANSFORM_FILENAME
    _save_transform_to_file(
        affine_transform, transform_path, initial_transform_file=initial_transform_file
    )
    logger.info("Perform affine registration: end")
    return _StageResult(
        image=affine_image,
        transform=affine_transform,
        image_path=image_path,
        parameters_path=parameters_path,
        transform_path=transform_path,
    )


def _perform_bspline_registration(
    fixed_image: itk.Image,
    moving_image: itk.Image,
    initial_transform_file: Path,
    save_path: Path,
) -> _StageResult:
    """Perform B-spline registration stage.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        initial_transform_file: Path to the initial transform file (affine).
        save_path: Path to directory for saving transform files.

    Returns:
        _StageResult containing the registered image, transform, and all file paths.
    """
    logger.info("Perform b-spline registration: start")
    # Save parameters used by elastix to perform the B-Spline transform
    parameters_path = save_path / BSPLINE_PARAMETERS_IN_FILENAME
    BSPLINE_PARAM_OBJECT.WriteParameterFile(
        BSPLINE_PARAM_OBJECT.GetParameterMap(0),
        str(parameters_path),
    )

    bspline_image, bspline_transform = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=BSPLINE_PARAM_OBJECT,
        initial_transform_parameter_file_name=str(initial_transform_file),
        log_to_console=False,
    )

    # Save bspline-registered image to file
    image_path = save_path / BSPLINE_IMAGE_FILENAME
    itk.imwrite(bspline_image, str(image_path))

    # Save the B-spline transform produced by elastix to file
    transform_path = save_path / BSPLINE_TRANSFORM_FILENAME
    _save_transform_to_file(
        bspline_transform, transform_path, initial_transform_file=initial_transform_file
    )

    logger.info("Perform b-spline registration: end")

    return _StageResult(
        image=bspline_image,
        transform=bspline_transform,
        image_path=image_path,
        parameters_path=parameters_path,
        transform_path=transform_path,
    )


def _register(
    fixed_image: itk.Image,
    moving_image: itk.Image,
    save_path: Path,
) -> tuple[itk.Image, itk.ParameterObject, list[Path]]:
    """Private register method that makes use of the three stage methods.

    This function performs a three-stage registration (rigid, affine, B-spline)
    of image objects in memory and and saves intermediate results and transform
    files to disk as side effects.

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
    logger.info("Registration flow: start")

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
    rigid_result = _perform_rigid_registration(fixed_image, moving_image, save_path)
    affine_result = _perform_affine_registration(
        fixed_image, moving_image, rigid_result.transform_path, save_path
    )
    bspline_result = _perform_bspline_registration(
        fixed_image, moving_image, affine_result.transform_path, save_path
    )

    # Load the composed transform from file (includes chain information via
    # InitialTransformParameterFileName). The files must remain accessible
    # for transformix to resolve the transform chain.
    composed_transform = itk.ParameterObject.New()
    composed_transform.ReadParameterFile(str(bspline_result.transform_path))
    logger.info("Registration flow: end")
    return (
        bspline_result.image,
        composed_transform,
        [
            rigid_result.transform_path,
            affine_result.transform_path,
            bspline_result.transform_path,
        ],
    )


def register_atlas(
    *,
    moving_image: Path,
    fixed_image: Path,
    output_directory: Path,
) -> RegistrationResult:
    """Register the moving image to the fixed image using file paths.

    This is a convenience wrapper that handles loading images from file paths
    and saving the results to the specified output directory.

    This function performs file I/O as side effects: it loads images from disk,
    performs three-stage registration (rigid, affine, B-spline), and saves all
    intermediate and final results.

    Args:
        moving_image: Path to the moving (source) image file to register.
        fixed_image: Path to the fixed (target) reference image file.
        output_directory: Directory for all output files. Intermediate images,
            input parameters, and transform files will all be saved here.

    Returns:
        RegistrationResult containing paths to all output files including
        intermediate images, input parameters, and transform files for each
        registration stage (rigid, affine, B-spline).

    Raises:
        FileNotFoundError: If moving_image or fixed_image does not exist.
        OSError: If output_directory cannot be created or written to.
    """
    # Validate input files exist
    if not moving_image.exists():
        msg = f"Moving image file not found: {moving_image}"
        raise FileNotFoundError(msg)
    if not fixed_image.exists():
        msg = f"Fixed image file not found: {fixed_image}"
        raise FileNotFoundError(msg)

    # Validate output directory can be created
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = output_directory / ".write_test"
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        msg = f"Cannot create or write to output directory '{output_directory}': {e}"
        raise OSError(msg) from e

    # Load images from file paths
    moving_image_itk = itk.imread(str(moving_image), itk.F)
    fixed_image_itk = itk.imread(str(fixed_image), itk.F)

    # Perform three-stage registration: rigid, affine, B-spline
    rigid_result = _perform_rigid_registration(
        fixed_image_itk, moving_image_itk, output_directory
    )
    affine_result = _perform_affine_registration(
        fixed_image_itk, moving_image_itk, rigid_result.transform_path, output_directory
    )
    bspline_result = _perform_bspline_registration(
        fixed_image_itk,
        moving_image_itk,
        affine_result.transform_path,
        output_directory,
    )

    return RegistrationResult(
        rigid_image_path=rigid_result.image_path,
        affine_image_path=affine_result.image_path,
        bspline_image_path=bspline_result.image_path,
        rigid_parameters_path=rigid_result.parameters_path,
        affine_parameters_path=affine_result.parameters_path,
        bspline_parameters_path=bspline_result.parameters_path,
        rigid_transform_path=rigid_result.transform_path,
        affine_transform_path=affine_result.transform_path,
        bspline_transform_path=bspline_result.transform_path,
        registered_image_path=bspline_result.image_path,
        registration_transform_path=bspline_result.transform_path,
    )

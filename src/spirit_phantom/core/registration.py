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

from spirit_phantom.core.registration_optimizations import (
    AffineOverrides,
    BSplineOverrides,
    RegistrationOverrides,
    RigidOverrides,
)

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


def _load_stage_parameter_object(*, parameter_filename: str) -> itk.ParameterObject:
    """Load a stage parameter object from a packaged/default parameter file.

    Args:
        parameter_filename: The registration parameter filename to load.

    Returns:
        A new parameter object loaded from the provided filename.
    """
    parameter_file = _get_parameter_file_path(parameter_filename)
    parameter_object = itk.ParameterObject.New()
    parameter_object.ReadParameterFile(str(parameter_file))
    return parameter_object


def _apply_parameter_overrides(
    *,
    parameter_object: itk.ParameterObject,
    parameter_updates: dict[str, tuple[str, ...]],
) -> itk.ParameterObject:
    """Apply curated parameter overrides and return an updated object.

    Args:
        parameter_object: The loaded elastix parameter object.
        parameter_updates: Mapping of elastix keys to tuple-formatted values.

    Returns:
        A new parameter object with the provided updates applied.

    Raises:
        ValueError: If an override key is not present in the default map.
    """
    if not parameter_updates:
        return parameter_object

    parameter_map = dict(parameter_object.GetParameterMap(0))
    for parameter_name in parameter_updates:
        if parameter_name not in parameter_map:
            msg = (
                f"Unknown elastix parameter override '{parameter_name}'. "
                "Only curated known keys are allowed."
            )
            raise ValueError(msg)
    parameter_map.update(parameter_updates)

    updated_parameter_object = itk.ParameterObject.New()
    updated_parameter_object.AddParameterMap(parameter_map)
    return updated_parameter_object


def _perform_rigid_registration(
    fixed_image: itk.Image,
    moving_image: itk.Image,
    save_path: Path,
    overrides: RigidOverrides | None = None,
) -> _StageResult:
    """Perform rigid registration stage.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        save_path: Path to directory for saving transform files.
        overrides: Optional curated rigid-stage parameter overrides.

    Returns:
        _StageResult containing the registered image, transform, and all file paths.
    """
    logger.info("Perform rigid registration: start")
    rigid_param_object = _load_stage_parameter_object(
        parameter_filename="parameters_Rigid.txt"
    )
    rigid_param_object = _apply_parameter_overrides(
        parameter_object=rigid_param_object,
        parameter_updates=(
            overrides.to_elastix_updates() if overrides is not None else {}
        ),
    )

    # Save the parameters used by elastix to perform the rigid transform
    parameters_path = save_path / RIGID_PARAMETERS_IN_FILENAME
    rigid_param_object.WriteParameterFile(
        rigid_param_object.GetParameterMap(0),
        str(parameters_path),
    )
    rigid_image, rigid_transform = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=rigid_param_object,
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
    overrides: AffineOverrides | None = None,
) -> _StageResult:
    """Perform affine registration stage.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        initial_transform_file: Path to the initial transform file (rigid).
        save_path: Path to directory for saving transform files.
        overrides: Optional curated affine-stage parameter overrides.

    Returns:
        _StageResult containing the registered image, transform, and all file paths.
    """
    logger.info("Perform affine registration: start")
    affine_param_object = _load_stage_parameter_object(
        parameter_filename="parameters_Affine.txt"
    )
    affine_param_object = _apply_parameter_overrides(
        parameter_object=affine_param_object,
        parameter_updates=(
            overrides.to_elastix_updates() if overrides is not None else {}
        ),
    )
    # Save the parameters used by elastix to perform the affine transform
    parameters_path = save_path / AFFINE_PARAMETERS_IN_FILENAME
    affine_param_object.WriteParameterFile(
        affine_param_object.GetParameterMap(0),
        str(parameters_path),
    )

    affine_image, affine_transform = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=affine_param_object,
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
    overrides: BSplineOverrides | None = None,
) -> _StageResult:
    """Perform B-spline registration stage.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        initial_transform_file: Path to the initial transform file (affine).
        save_path: Path to directory for saving transform files.
        overrides: Optional curated B-spline-stage parameter overrides.

    Returns:
        _StageResult containing the registered image, transform, and all file paths.
    """
    logger.info("Perform b-spline registration: start")
    bspline_param_object = _load_stage_parameter_object(
        parameter_filename="parameters_B_Spline.txt"
    )
    bspline_param_object = _apply_parameter_overrides(
        parameter_object=bspline_param_object,
        parameter_updates=(
            overrides.to_elastix_updates() if overrides is not None else {}
        ),
    )
    # Save parameters used by elastix to perform the B-Spline transform
    parameters_path = save_path / BSPLINE_PARAMETERS_IN_FILENAME
    bspline_param_object.WriteParameterFile(
        bspline_param_object.GetParameterMap(0),
        str(parameters_path),
    )

    bspline_image, bspline_transform = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=bspline_param_object,
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
    overrides: RegistrationOverrides | None = None,
) -> tuple[itk.Image, itk.ParameterObject, list[Path]]:
    """Private register method that makes use of the three stage methods.

    This function performs a three-stage registration (rigid, affine, B-spline)
    of image objects in memory and and saves intermediate results and transform
    files to disk as side effects.

    Args:
        fixed_image: The fixed (target) image.
        moving_image: The moving (source) image to register.
        save_path: Path to directory for saving transform files. Must be writable.
        overrides: Optional curated stage-level parameter overrides.

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
    rigid_result = _perform_rigid_registration(
        fixed_image=fixed_image,
        moving_image=moving_image,
        save_path=save_path,
        overrides=overrides.RigidOverrides if overrides is not None else None,
    )
    affine_result = _perform_affine_registration(
        fixed_image=fixed_image,
        moving_image=moving_image,
        initial_transform_file=rigid_result.transform_path,
        save_path=save_path,
        overrides=overrides.AffineOverrides if overrides is not None else None,
    )
    bspline_result = _perform_bspline_registration(
        fixed_image=fixed_image,
        moving_image=moving_image,
        initial_transform_file=affine_result.transform_path,
        save_path=save_path,
        overrides=overrides.BSplineOverrides if overrides is not None else None,
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
    cli_user: bool = False,
    overrides: RegistrationOverrides | None = None,
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
        cli_user: Whether the function is being called from the CLI. If True, will print logging messages to the console.
        overrides: Optional curated stage-level registration parameter overrides.
            For each stage, only non-``None`` override values are applied.

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

    if cli_user:
        print("Performing rigid part of registration...")
    # Perform three-stage registration: rigid, affine, B-spline
    rigid_result = _perform_rigid_registration(
        fixed_image=fixed_image_itk,
        moving_image=moving_image_itk,
        save_path=output_directory,
        overrides=overrides.RigidOverrides if overrides is not None else None,
    )
    if cli_user:
        print("Performing affine part of registration...")
    affine_result = _perform_affine_registration(
        fixed_image=fixed_image_itk,
        moving_image=moving_image_itk,
        initial_transform_file=rigid_result.transform_path,
        save_path=output_directory,
        overrides=overrides.AffineOverrides if overrides is not None else None,
    )
    if cli_user:
        print("Performing B-spline part of registration...")
    bspline_result = _perform_bspline_registration(
        fixed_image=fixed_image_itk,
        moving_image=moving_image_itk,
        initial_transform_file=affine_result.transform_path,
        save_path=output_directory,
        overrides=overrides.BSplineOverrides if overrides is not None else None,
    )
    if cli_user:
        print("Registration complete!")

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

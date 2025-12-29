"""Module for performing elastix registration of SPIRIT phantom data."""

import logging
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any

import itk

logging.basicConfig(level=logging.INFO)


class TransformType(str, Enum):
    """Enumeration of supported elastix transform types.

    Values correspond to the transform type names used in parameter files.
    """

    RIGID = "Rigid"
    AFFINE = "Affine"
    # Parameters tuned to simple unit test objects with low complexity and with fast convergence.
    UNIT_TEST = "UnitTest"


def _get_parameter_file_path(transform_type: TransformType) -> Path:
    """Get the path to a registration parameters file for a given transform type.

    Args:
        transform_type: The type of transform (Rigid, Affine, BSpline, etc.).

    Returns:
        Path to the registration parameters file, either from package data
        or as a fallback from the source directory.

    Raises:
        FileNotFoundError: If the parameter file cannot be found in either
            the installed package or the source directory.
    """
    filename = f"parameters_{transform_type.value}.txt"

    try:
        # Try to access from installed package
        with resources.path("spirit_phantom.core", filename) as param_path:
            return Path(param_path)
    except (ModuleNotFoundError, FileNotFoundError):
        # Fallback to source location for development
        source_path = Path(__file__).parent / filename
        if not source_path.exists():
            msg = (
                f"Parameter file '{filename}' not found. "
                f"Checked package data and source location: {source_path}"
            )
            raise FileNotFoundError(msg) from None
        return source_path


def register_atlas(
    moving_image: Path,
    fixed_image: Path,
    result_image_save_path: Path | None,
    transform_params_save_path: Path | None,
    transform: TransformType | Path = TransformType.RIGID,
) -> Any:
    """Register the moving image to the fixed image.

    Args:
        moving_image: The moving image to register.
        fixed_image: The fixed image to register to.
        result_image_save_path: The path to save the registered image to. If None, the image is not saved.
        transform_params_save_path: The path to save the transform parameters to. If None, the transform parameters are not saved.
        transform: The type of transform to use. If a Path is provided, it is used as the path to the parameter file.
        If a TransformType is provided, the corresponding parameter file is used. Default is Rigid.

    Returns:
        The transform parameters.

    """
    moving_image = itk.imread(moving_image, itk.F)
    fixed_image = itk.imread(fixed_image, itk.F)

    # load the parameters for the registration
    parameter_object = itk.ParameterObject.New()
    if isinstance(transform, TransformType):
        parameters_path = _get_parameter_file_path(transform)
    else:
        parameters_path = transform
    parameter_object.AddParameterFile(str(parameters_path))

    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=False,
    )

    parameter_map = result_transform_parameters.GetParameterMap(0)

    if result_image_save_path is not None:
        itk.imwrite(result_image, result_image_save_path)

    if transform_params_save_path is not None:
        result_transform_parameters.WriteParameterFile(
            parameter_map, str(transform_params_save_path)
        )
    return result_transform_parameters

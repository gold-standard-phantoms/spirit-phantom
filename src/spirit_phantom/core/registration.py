"""Module for performing elastix registration of SPIRIT phantom data."""

import logging
from pathlib import Path
from typing import Any

import itk

logging.basicConfig(level=logging.INFO)

REGISTRATION_PARAMETERS_PATH = (
    Path(__file__).parent / "registration_parameters_Rigid.txt"
)


def register_atlas(
    moving_image: Path,
    fixed_image: Path,
    result_image_save_path: Path | None,
    transform_params_save_path: Path | None,
) -> Any:
    """Register the moving image to the fixed image.

    Args:
        moving_image: The moving image to register.
        fixed_image: The fixed image to register to.
        result_image_save_path: The path to save the registered image to. If None, the image is not saved.
        transform_params_save_path: The path to save the transform parameters to. If None, the transform parameters are not saved.

    Returns:
        The transform parameters.

    """
    moving_image = itk.imread(moving_image, itk.F)
    fixed_image = itk.imread(fixed_image, itk.F)

    # load the parameters for the registration
    parameter_object = itk.ParameterObject.New()

    parameter_object.AddParameterFile(str(REGISTRATION_PARAMETERS_PATH))

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

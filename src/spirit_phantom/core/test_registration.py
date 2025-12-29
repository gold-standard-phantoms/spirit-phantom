"""Tests for the registration module.

These are basic tests for the use of the registration module.
In depth tests for the phantom type and parameter files should be dealt with in separate tests.
"""

from pathlib import Path

import itk
import nibabel as nib
from scipy.spatial.distance import dice

from spirit_phantom.core.registration import TransformType, register_atlas
from spirit_phantom.utils.digital_phantom import (
    apply_rotation,
    apply_translation,
    create_synthetic_volume,
    save_nifti_from_array,
)


def test_registration_with_translation(tmp_path: Path) -> None:
    """Test registration with translation-only transformation.

    Creates a synthetic volume, applies a random translation, performs
    registration, and verifies the recovered translation matches the applied one.

    Note- in the scanner scenario, the "fixed image" is one of the phantom in the
    scanner with a non-ideal positioning. The "moving image" is the phantom atlas.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    # Create base synthetic volume
    volume = create_synthetic_volume((100, 100, 100), (30, 35, 40))

    # Apply translation to simulate scanning to produce the fixed volume.
    translation = (10, 10, 10)
    # Apply translation to the volume data only (not the affine)
    # This creates a misalignment that elastix should detect
    fixed_volume = apply_translation(volume, translation)

    # Save both as NIfTI files
    # Use identity affine for both so elastix sees the data misalignment
    fixed_image_path = tmp_path / "fixed_image.nii.gz"
    moving_image_path = tmp_path / "moving_image.nii.gz"
    save_nifti_from_array(fixed_volume, fixed_image_path)
    save_nifti_from_array(volume, moving_image_path)

    # Perform registration
    result_image_path = tmp_path / "result_image.nii.gz"
    transform_params_path = tmp_path / "TransformParameters.0.txt"
    _ = register_atlas(
        moving_image=moving_image_path,
        fixed_image=fixed_image_path,
        result_image_save_path=result_image_path,
        transform_params_save_path=transform_params_path,
        transform=TransformType.UNIT_TEST,
    )

    result_nifti = nib.Nifti1Image.load(result_image_path)
    result_volume = result_nifti.get_fdata()

    dice_original = 1 - dice(fixed_volume[:].ravel(), volume[:].ravel())
    # There should originally be a large difference between the original and fixed volume seen with a lower dice score.
    assert dice_original < 0.6, f"Dice score {dice_original} is not less than 0.6"
    dice_result = 1 - dice(fixed_volume[:].ravel(), result_volume[:].ravel())
    # After registration, the dice score should be close to 1 indicating that the registration has been successful.
    assert dice_result > 0.95, f"Dice score {dice_result} is not greater than 0.95"


def test_registration_with_rotation_and_translation(tmp_path: Path) -> None:
    """Test registration with rotation-only transformation.

    Creates a synthetic volume, applies a random rotation and translation, performs
    registration, and verifies the recovered rotation matches the applied one.
    """
    volume = create_synthetic_volume((200, 200, 200), (60, 70, 80))
    translation = (15, 15, 15)
    rotation = (10, 10, 10)

    fixed_volume = apply_rotation(volume, rotation)
    fixed_volume = apply_translation(fixed_volume, translation)
    fixed_image_path = tmp_path / "fixed_image.nii.gz"
    moving_image_path = tmp_path / "moving_image.nii.gz"
    save_nifti_from_array(fixed_volume, fixed_image_path)
    save_nifti_from_array(volume, moving_image_path)

    result_image_path = tmp_path / "result_image.nii.gz"
    transform_params_path = tmp_path / "TransformParameters.0.txt"
    _ = register_atlas(
        moving_image=moving_image_path,
        fixed_image=fixed_image_path,
        result_image_save_path=result_image_path,
        transform_params_save_path=transform_params_path,
        transform=TransformType.UNIT_TEST,
    )

    result_nifti = nib.Nifti1Image.load(result_image_path)
    result_volume = result_nifti.get_fdata()

    dice_original = 1 - dice(fixed_volume[:].ravel(), volume[:].ravel())
    # There should originally be a large difference between the original and fixed volume seen with a lower dice score.
    assert dice_original < 0.6, f"Dice score {dice_original} is not less than 0.6"
    dice_result = 1 - dice(fixed_volume[:].ravel(), result_volume[:].ravel())
    # After registration, the dice score should be close to 1 indicating that the registration has been successful.
    assert dice_result > 0.95, f"Dice score {dice_result} is not greater than 0.95"


def test_registration_with_custom_parameter_file(tmp_path: Path) -> None:
    """Test that the registration runs with a custom parameter file."""
    volume = create_synthetic_volume((100, 100, 100), (30, 35, 40))
    translation = (15, 15, 15)

    fixed_volume = apply_translation(volume, translation)
    fixed_image_path = tmp_path / "fixed_image.nii.gz"
    moving_image_path = tmp_path / "moving_image.nii.gz"
    save_nifti_from_array(fixed_volume, fixed_image_path)
    save_nifti_from_array(volume, moving_image_path)

    result_image_path = tmp_path / "result_image.nii.gz"
    transform_params_path = tmp_path / "TransformParameters.0.txt"

    par_obj = itk.ParameterObject.New()
    par_map = par_obj.GetDefaultParameterMap("rigid")
    par_obj.AddParameterMap(par_map)

    tmp_file_path = tmp_path / "rigid_parameters.txt"
    par_obj.WriteParameterFile(par_map, str(tmp_file_path))
    _ = register_atlas(
        moving_image=moving_image_path,
        fixed_image=fixed_image_path,
        result_image_save_path=result_image_path,
        transform_params_save_path=transform_params_path,
        transform=tmp_file_path,
    )

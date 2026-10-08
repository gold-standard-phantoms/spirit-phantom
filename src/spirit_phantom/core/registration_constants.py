"""Lightweight registration constants that must not import ITK.

The CLI parent process resolves parameter-set names and output filenames
before spawning the isolated elastix worker. Keeping these symbols free of
``import itk`` avoids loading ITK into the parent and doubling peak RAM.
"""

from enum import StrEnum

# Filename constants for registration outputs
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
TRANSFORMED_COMPONENT_ATLAS_FILENAME = "transformed_component_atlas.nii.gz"
REGISTRATION_LOG_FILENAME = "registration.log"


class ParameterSet(StrEnum):
    """Elastix parameter-file set used for registration stages."""

    REGULAR = "regular"
    SPEEDY = "speedy"

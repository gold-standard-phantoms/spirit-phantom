"""Helpers for building initial registration transforms.

This module supports pre-registration orientation correction for SPIRIT phantom
workflows where the phantom may be scanned inverted.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import itk
import numpy as np
import numpy.typing as npt

INITIAL_FLIP_TRANSFORM_FILENAME = "Initial_Flip_Transform.txt"


def image_geometric_centre(*, image: itk.Image) -> npt.NDArray[np.float64]:
    """Compute an ITK image geometric centre in world coordinates.

    Args:
        image: ITK image used to derive origin, spacing, size, and direction.

    Returns:
        Numpy vector containing the geometric centre in physical coordinates.
    """
    origin = np.asarray(image.GetOrigin(), dtype=float)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    size = np.asarray(image.GetLargestPossibleRegion().GetSize(), dtype=float)
    direction = cast(
        "npt.NDArray[np.float64]",
        np.asarray(itk.array_from_matrix(image.GetDirection()), dtype=float),
    )
    centre = origin + direction @ (spacing * (size - 1.0) / 2.0)
    return cast("npt.NDArray[np.float64]", centre)


def _format_sequence(*, values: Sequence[float], precision: int = 15) -> str:
    """Format numeric values for elastix parameter file output.

    Args:
        values: Sequence of float-compatible values.
        precision: Significant figure precision.

    Returns:
        Space-delimited values formatted for elastix text parameters.
    """
    return " ".join(f"{float(value):.{precision}g}" for value in values)


def write_initial_flip_transform(
    filepath: Path,
    *,
    image: itk.Image,
    rotation: tuple[float, float, float] = (0.0, np.pi, 0.0),
) -> Path:
    """Write an Euler initial transform parameter file for elastix.

    The default transform applies a 180-degree rotation around the Y axis and
    uses the fixed-image geometric centre as the rotation centre.

    Args:
        filepath: Destination transform file path.
        image: Fixed ITK image used to populate transform geometry.
        rotation: Euler rotation in radians ordered as (x, y, z).

    Returns:
        Path to the written transform parameter file.
    """
    centre = image_geometric_centre(image=image)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    size = image.GetLargestPossibleRegion().GetSize()
    direction = itk.array_from_matrix(image.GetDirection())
    transform_parameters = (*rotation, 0.0, 0.0, 0.0)

    lines = [
        '(Transform "EulerTransform")',
        "(NumberOfParameters 6)",
        f"(TransformParameters {_format_sequence(values=transform_parameters)})",
        f"(CenterOfRotationPoint {_format_sequence(values=centre.tolist())})",
        '(ComputeZYX "false")',
        '(InitialTransformParameterFileName "NoInitialTransform")',
        '(HowToCombineTransforms "Compose")',
        "(FixedImageDimension 3)",
        "(MovingImageDimension 3)",
        '(FixedInternalImagePixelType "float")',
        '(MovingInternalImagePixelType "float")',
        f"(Size {' '.join(str(component) for component in size)})",
        "(Index 0 0 0)",
        f"(Spacing {_format_sequence(values=spacing)})",
        f"(Origin {_format_sequence(values=origin)})",
        f"(Direction {_format_sequence(values=direction.flatten())})",
        '(UseDirectionCosines "true")',
        '(ResampleInterpolator "FinalBSplineInterpolator")',
        "(FinalBSplineInterpolationOrder 0)",
        '(Resampler "DefaultResampler")',
        "(DefaultPixelValue 0)",
        '(CompressResultImage "false")',
        '(ResultImageFormat "nii.gz")',
        '(ResultImagePixelType "short")',
    ]

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath

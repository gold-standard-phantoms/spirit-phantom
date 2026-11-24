"""Tests for transformix point and index I/O."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal, cast

import itk
import nibabel
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from spirit_phantom.io.points import (
    InversionMethod,
    invert_points_through_transformix,
    itk_physical_to_nifti_world,
    load_points,
    nifti_world_to_itk_physical,
    save_points,
    transform_points_with_transformix,
)

PointType = Literal["point", "index"]


def _generate_random_points(
    rng: random.Random,
    n_points: int,
    n_dims: int,
) -> list[list[float]]:
    """Generate deterministic random point coordinates.

    Args:
        rng: Random number generator used for reproducibility.
        n_points: Number of points to generate.
        n_dims: Dimensionality of each point.

    Returns:
        List of points, where each point is a list of floats.
    """
    return [[rng.random() for _ in range(n_dims)] for _ in range(n_points)]


def _round_trip_points(
    tmp_dir: Path,
    point_type: PointType,
    dims: int,
    filename: str,
    rng: random.Random,
) -> None:
    """Helper for saving and loading random points.

    Args:
        tmp_dir: Temporary directory to save the points file.
        point_type: Type of points to save.
        dims: Dimensionality of the points.
        filename: Name of the points file.
        rng: Random number generator used for reproducibility.
    """
    points = _generate_random_points(rng=rng, n_points=10, n_dims=dims)
    output_path = tmp_dir / filename

    save_points(points=points, output_path=output_path, point_type=point_type)
    loaded = load_points(points_path=output_path)

    assert loaded == points


def test_save_and_load_points_and_index(tmp_path: Path) -> None:
    """Save and load 2D/3D point and index files via transformix format.

    This test creates four files, each containing ten random values:

    - 2D points
    - 3D points
    - 2D index coordinates
    - 3D index coordinates
    """
    rng = random.Random(1234)  # noqa: S311

    cases = (
        ("point", 2, "points_2d.txt"),
        ("point", 3, "points_3d.txt"),
        ("index", 2, "index_2d.txt"),
        ("index", 3, "index_3d.txt"),
    )

    for point_type_str, dims, filename in cases:
        assert point_type_str in ("point", "index")
        point_type = cast("PointType", point_type_str)
        _round_trip_points(
            tmp_dir=tmp_path,
            point_type=point_type,
            dims=dims,
            filename=filename,
            rng=rng,
        )


def test_nifti_itk_physical_conversion_flips_xy() -> None:
    """NIfTI world and ITK physical differ by an X/Y sign flip (RAS↔LPS)."""
    assert nifti_world_to_itk_physical([1.0, -2.0, 3.0]) == [-1.0, 2.0, 3.0]
    assert itk_physical_to_nifti_world([-1.0, 2.0, 3.0]) == [1.0, -2.0, 3.0]
    assert itk_physical_to_nifti_world(
        nifti_world_to_itk_physical([4.5, -6.25, 0.0])
    ) == [4.5, -6.25, 0.0]


def _write_identity_euler_transform(
    *,
    image_path: Path,
    transform_path: Path,
) -> None:
    """Write an identity Euler TransformParameters file for ``image_path``."""
    _write_euler_transform(
        image_path=image_path,
        transform_path=transform_path,
        parameters=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _write_euler_transform(
    *,
    image_path: Path,
    transform_path: Path,
    parameters: tuple[float, float, float, float, float, float],
) -> None:
    """Write an Euler TransformParameters file (rotations rad, translations mm).

    The centre of rotation is the ITK physical origin ``(0, 0, 0)``.
    """
    itk_image = itk.imread(str(image_path), itk.F)
    origin = [float(value) for value in itk.origin(itk_image)]
    spacing = [float(value) for value in itk.spacing(itk_image)]
    direction = np.asarray(
        itk.array_from_matrix(itk_image.GetDirection()), dtype=np.float64
    ).ravel()
    size = [int(value) for value in itk.size(itk_image)]
    lines = [
        '(Transform "EulerTransform")',
        "(NumberOfParameters 6)",
        f"(TransformParameters {' '.join(str(value) for value in parameters)})",
        '(InitialTransformParametersFileName "NoInitialTransform")',
        '(HowToCombineTransforms "Compose")',
        "(FixedImageDimension 3)",
        "(MovingImageDimension 3)",
        '(FixedInternalImagePixelType "float")',
        '(MovingInternalImagePixelType "float")',
        f"(Size {' '.join(str(value) for value in size)})",
        "(Index 0 0 0)",
        f"(Spacing {' '.join(str(value) for value in spacing)})",
        f"(Origin {' '.join(str(value) for value in origin)})",
        f"(Direction {' '.join(str(value) for value in direction)})",
        "(CenterOfRotationPoint 0 0 0)",
        '(ComputeZYX "false")',
        '(ResampleInterpolator "FinalNearestNeighborInterpolator")',
        '(Resampler "DefaultResampler")',
        "(DefaultPixelValue 0)",
        '(ResultImageFormat "nii.gz")',
        '(ResultImagePixelType "float")',
        '(CompressResultImage "true")',
    ]
    transform_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_transformix_identity_preserves_nifti_world_points(tmp_path: Path) -> None:
    """Identity transformix must not flip X/Y relative to nibabel world coords."""
    data = np.zeros((16, 16, 12), dtype=np.float32)
    affine = np.eye(4)
    affine[0, 0] = 0.5
    affine[1, 1] = 0.5
    affine[2, 2] = 0.5
    affine[:3, 3] = [-4.0, -4.0, -3.0]
    image_path = tmp_path / "moving.nii.gz"
    nibabel.save(nibabel.Nifti1Image(dataobj=data, affine=affine), str(image_path))

    transform_path = tmp_path / "TransformParameters.txt"
    _write_identity_euler_transform(
        image_path=image_path, transform_path=transform_path
    )

    points = [[1.0, -2.0, 0.5], [0.0, 0.0, 0.0], [-1.5, 2.25, -0.25]]
    mapped = transform_points_with_transformix(
        points=points,
        transform_parameter_path=transform_path,
        moving_image_path=image_path,
        output_directory=tmp_path / "transformix_out",
    )
    for source, result in zip(points, mapped, strict=True):
        assert result == pytest.approx(source, abs=1e-6)

    # Without the RAS↔LPS bridge, transformix would return the flipped coords.
    flipped = [nifti_world_to_itk_physical(point) for point in points]
    assert any(
        result != pytest.approx(wrong, abs=1e-6)
        for result, wrong in zip(mapped, flipped, strict=True)
    )


@pytest.mark.parametrize("method", list(InversionMethod))
def test_invert_identity_preserves_nifti_world_points(
    tmp_path: Path,
    method: InversionMethod,
) -> None:
    """Moving→fixed inversion through identity must keep nibabel world points."""
    data = np.zeros((16, 16, 12), dtype=np.float32)
    affine = np.eye(4)
    affine[0, 0] = 0.5
    affine[1, 1] = 0.5
    affine[2, 2] = 0.5
    affine[:3, 3] = [-4.0, -4.0, -3.0]
    image_path = tmp_path / "moving.nii.gz"
    nibabel.save(nibabel.Nifti1Image(dataobj=data, affine=affine), str(image_path))

    transform_path = tmp_path / "TransformParameters.txt"
    _write_identity_euler_transform(
        image_path=image_path, transform_path=transform_path
    )

    points = [[25.0, 0.0, 0.0], [25.0, 5.0, 0.0], [-25.0, -1.0, 0.0]]
    fixed = invert_points_through_transformix(
        moving_points=points,
        transform_parameter_path=transform_path,
        fixed_image_path=image_path,
        moving_image_path=image_path,
        output_directory=tmp_path / "invert_out",
        method=method,
    )
    for source, result in zip(points, fixed, strict=True):
        assert result == pytest.approx(source, abs=1e-5)
    assert (tmp_path / "invert_out" / "fixed_space_points.txt").exists()


def _write_centred_image(*, image_path: Path) -> None:
    """Write a 32 mm cube of 1 mm voxels centred on the world origin."""
    data = np.zeros((32, 32, 32), dtype=np.float32)
    affine = np.eye(4)
    affine[:3, 3] = [-15.5, -15.5, -15.5]
    nibabel.save(nibabel.Nifti1Image(dataobj=data, affine=affine), str(image_path))


@pytest.mark.parametrize(
    "parameters",
    [
        (0.0, 0.0, 0.2, 1.5, -2.0, 0.5),
        (0.0, np.pi, 0.0, 0.0, 0.0, 0.0),
    ],
    ids=["small-rotation", "inverted-phantom-flip"],
)
def test_elastix_inversion_round_trips_euler_transform(
    tmp_path: Path,
    parameters: tuple[float, float, float, float, float, float],
) -> None:
    """The elastix inverse must map points back so that ``T(x) ≈ p``."""
    image_path = tmp_path / "fixed.nii.gz"
    _write_centred_image(image_path=image_path)
    transform_path = tmp_path / "TransformParameters.txt"
    _write_euler_transform(
        image_path=image_path,
        transform_path=transform_path,
        parameters=parameters,
    )

    points = [[5.0, -3.0, 2.0], [-6.0, 4.0, -1.0], [0.5, 7.0, 3.5]]
    fixed = invert_points_through_transformix(
        moving_points=points,
        transform_parameter_path=transform_path,
        fixed_image_path=image_path,
        moving_image_path=image_path,
        output_directory=tmp_path / "invert_out",
    )
    round_trip = transform_points_with_transformix(
        points=fixed,
        transform_parameter_path=transform_path,
        moving_image_path=image_path,
        output_directory=tmp_path / "round_trip",
    )
    for source, result in zip(points, round_trip, strict=True):
        assert result == pytest.approx(source, abs=0.05)


def test_picard_inversion_fails_for_inverted_phantom_flip(tmp_path: Path) -> None:
    """Picard iteration cannot invert a 180° rotation (not a contraction)."""
    image_path = tmp_path / "fixed.nii.gz"
    _write_centred_image(image_path=image_path)
    transform_path = tmp_path / "TransformParameters.txt"
    _write_euler_transform(
        image_path=image_path,
        transform_path=transform_path,
        parameters=(0.0, np.pi, 0.0, 0.0, 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        invert_points_through_transformix(
            moving_points=[[5.0, -3.0, 2.0]],
            transform_parameter_path=transform_path,
            fixed_image_path=image_path,
            moving_image_path=image_path,
            output_directory=tmp_path / "invert_out",
            method=InversionMethod.PICARD,
        )

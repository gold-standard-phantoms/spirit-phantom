"""Module for point I/O operations using transformix format.

This module provides utilities for loading and saving points in the transformix
native format, which uses space-separated coordinates with a simple header.

Transformix format:
    point|index
    <number_of_points>
    x1 y1 [z1]
    x2 y2 [z2]
    ...

Coordinate convention
---------------------
Public helpers that talk to transformix
(:func:`transform_points_with_transformix`,
:func:`invert_points_through_transformix`) accept and return **NIfTI / nibabel
world coordinates** — the same millimetre frame as ``image.affine @ (i, j, k,
1)``.

ITK / elastix / transformix use **LPS** physical space when reading NIfTI
files (``itk.imread``), which differs from nibabel's RAS-style affine frame by
a sign flip on X and Y. Conversion is applied at the transformix boundary so
callers (atlas wedge ROIs, nibabel sampling/masks) stay consistent.
"""

import logging
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

import itk

from spirit_phantom.core.registration import compute_inverse_transform

logger = logging.getLogger(__name__)

# Supported dimensions for point transformations
DIM_2D = 2
DIM_3D = 3

# Minimum lines in a valid transformix points file (type line + count line)
_MIN_HEADER_LINES = 2

# Round-trip residual above which an elastix-estimated inverse is reported
_INVERSE_RESIDUAL_WARNING_MM = 0.1

_FIXED_SPACE_POINTS_FILENAME = "fixed_space_points.txt"

_PointType = Literal["point", "index"]
_Point = Sequence[float]
_Points = Sequence[_Point]


class InversionMethod(StrEnum):
    """Method used to map moving-domain points into the fixed domain.

    Attributes:
        ELASTIX: Estimate an inverse transform with elastix
            (``DisplacementMagnitudePenalty``) and apply it with transformix.
        PICARD: Solve ``T(x) = p`` per point by fixed-point (Picard)
            iteration through the forward transform. Only converges when the
            displacement field is a contraction (e.g. small rotations).
    """

    ELASTIX = "elastix"
    PICARD = "picard"


def nifti_world_to_itk_physical(point: _Point) -> list[float]:
    """Convert a NIfTI/nibabel world point to ITK/transformix physical space.

    For NIfTI images, ITK applies a RAS→LPS flip: ``(x, y, z) → (-x, -y, z)``.

    Args:
        point: World coordinates from a nibabel affine (2-D or 3-D).

    Returns:
        Coordinates in ITK physical space.
    """
    coords = [float(value) for value in point]
    if len(coords) >= DIM_2D:
        coords[0] = -coords[0]
        coords[1] = -coords[1]
    return coords


def itk_physical_to_nifti_world(point: _Point) -> list[float]:
    """Convert an ITK/transformix physical point to NIfTI/nibabel world space.

    Inverse of :func:`nifti_world_to_itk_physical` (same X/Y sign flip).

    Args:
        point: Physical coordinates from ITK / transformix (2-D or 3-D).

    Returns:
        Coordinates in the nibabel affine world frame.
    """
    return nifti_world_to_itk_physical(point)


def _parse_header(lines: list[str]) -> int:
    """Parse and validate the transformix file header.

    Args:
        lines: Non-empty lines from the file.

    Returns:
        Number of points specified in the header.

    Raises:
        ValueError: If header is invalid.
    """
    if len(lines) < _MIN_HEADER_LINES:
        msg = "Points file must have at least 2 lines (header and count)"
        raise ValueError(msg)

    # First line should be "point" or "index"
    point_type = lines[0].lower()
    if point_type not in ("point", "index"):
        msg = f"Invalid point type '{lines[0]}'. Expected 'point' or 'index'."
        raise ValueError(msg)

    # Second line is the number of points
    try:
        return int(lines[1])
    except ValueError as e:
        msg = f"Invalid point count '{lines[1]}': {e}"
        raise ValueError(msg) from e


def _parse_point_line(
    line: str,
    line_num: int,
    expected_dims: int | None,
) -> list[float]:
    """Parse a single point line.

    Args:
        line: The line containing space-separated coordinates.
        line_num: Line number for error reporting.
        expected_dims: Expected number of dimensions, or None if not yet known.

    Returns:
        List of float coordinates.

    Raises:
        ValueError: If line is malformed or has wrong dimensions.
    """
    try:
        coords = [float(x) for x in line.split()]
    except ValueError as e:
        msg = f"Invalid coordinates at line {line_num}: '{line}'. Error: {e}"
        raise ValueError(msg) from e

    n_dims = len(coords)
    if expected_dims is None:
        if n_dims not in (DIM_2D, DIM_3D):
            msg = f"Unsupported dimensionality: {n_dims}. Expected 2D or 3D."
            raise ValueError(msg)
    elif n_dims != expected_dims:
        msg = f"Point at line {line_num} has {n_dims} dimensions, expected {expected_dims}"
        raise ValueError(msg)

    return coords


def load_points(points_path: Path) -> list[list[float]]:
    """Load points from a transformix format file.

    Args:
        points_path: Path to file containing points in transformix format.

    Returns:
        List of point coordinates, each point is a list of floats.

    Raises:
        FileNotFoundError: If points_path does not exist.
        ValueError: If file is malformed or has invalid coordinates.
    """
    if not points_path.exists():
        msg = f"Points file not found: {points_path}"
        raise FileNotFoundError(msg)

    with points_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    num_points = _parse_header(lines)

    if len(lines) < _MIN_HEADER_LINES + num_points:
        msg = f"Expected {num_points} points but found {len(lines) - _MIN_HEADER_LINES}"
        raise ValueError(msg)

    # Parse points
    points: list[list[float]] = []
    n_dims: int | None = None
    for i, line in enumerate(lines[_MIN_HEADER_LINES : _MIN_HEADER_LINES + num_points]):
        coords = _parse_point_line(line, line_num=i + 3, expected_dims=n_dims)
        if n_dims is None:
            n_dims = len(coords)
        points.append(coords)

    if not points:
        msg = "No valid points found in file"
        raise ValueError(msg)

    return points


def save_points(
    points: _Points,
    output_path: Path,
    point_type: _PointType = "point",
) -> None:
    r"""Save points or index coordinates to a transformix format file.

    The first header line in the resulting file is either ``point`` or
    ``index`` depending on ``point_type``.

    Args:
        points: Iterable of point coordinates, each point is a sequence of
            floats.
        output_path: Path to the output file.
        point_type: Indicates whether the coordinates represent physical
            points (``\"point\"``) or voxel indices (``\"index\"``).

    Raises:
        ValueError: If ``points`` is empty, has inconsistent dimensions, or
            ``point_type`` is not supported.
    """
    points_list = [list(pt) for pt in points]
    if not points_list:
        msg = "Cannot save empty points list"
        raise ValueError(msg)

    n_dims = len(points_list[0])
    if n_dims not in (DIM_2D, DIM_3D):
        msg = f"Unsupported dimensionality: {n_dims}. Expected 2D or 3D."
        raise ValueError(msg)

    if point_type not in ("point", "index"):
        msg = f"Unsupported point_type: {point_type}. Expected 'point' or 'index'."
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"{point_type}\n")
        f.write(f"{len(points_list)}\n")
        for i, pt in enumerate(points_list):
            if len(pt) != n_dims:
                msg = f"Point {i} has {len(pt)} dimensions, expected {n_dims}"
                raise ValueError(msg)
            coords_str = " ".join(str(coord) for coord in pt)
            f.write(f"{coords_str}\n")


def transform_points_with_transformix(
    *,
    points: _Points,
    transform_parameter_path: Path,
    moving_image_path: Path,
    output_directory: Path,
    input_points_filename: str = "inputpoints.txt",
) -> list[list[float]]:
    """Transform points with transformix (fixed domain → moving domain).

    Elastix stores ``T`` mapping fixed → moving. Transformix therefore takes
    input points in the fixed image domain and returns moving-domain points.
    The moving image is only used to establish dimensionality/geometry.

    Args:
        points: NIfTI/nibabel world-coordinate points in the fixed image
            domain (same frame as ``fixed_image.affine``).
        transform_parameter_path: Elastix transform parameter file. May chain
            further transforms via ``InitialTransformParameterFileName``.
        moving_image_path: Moving image path used by transformix for geometry.
        output_directory: Directory for transformix ``inputpoints.txt`` and
            ``outputpoints.txt``.
        input_points_filename: Filename for the written input point set.

    Returns:
        Transformed points in the moving image domain, still in NIfTI/nibabel
        world coordinates.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    input_points_path = output_directory / input_points_filename
    # Transformix consumes ITK LPS physical coordinates.
    itk_points = [nifti_world_to_itk_physical(point) for point in points]
    save_points(points=itk_points, output_path=input_points_path, point_type="point")

    moving_image = itk.imread(str(moving_image_path), itk.F)
    transform_parameters = itk.ParameterObject.New()
    transform_parameters.ReadParameterFile(str(transform_parameter_path))
    itk.transformix_pointset(
        moving_image,
        transform_parameters,
        fixed_point_set_file_name=str(input_points_path),
        output_directory=str(output_directory),
    )
    itk_output = parse_transformix_output(
        output_path=output_directory / "outputpoints.txt"
    )
    return [itk_physical_to_nifti_world(point) for point in itk_output]


def _max_residual_mm(*, mapped: _Points, targets: _Points) -> float:
    """Return the largest per-axis absolute difference between point sets.

    Args:
        mapped: Points produced by a transform.
        targets: Expected points, parallel to ``mapped``.

    Returns:
        Maximum absolute coordinate difference in millimetres.
    """
    return max(
        abs(mapped_point[dim] - target[dim])
        for mapped_point, target in zip(mapped, targets, strict=True)
        for dim in range(len(target))
    )


def _invert_points_with_elastix(
    *,
    targets: list[list[float]],
    transform_parameter_path: Path,
    fixed_image_path: Path,
    moving_image_path: Path,
    output_directory: Path,
    inverse_transform_path: Path | None = None,
) -> list[list[float]]:
    """Map moving-domain points to fixed space via an elastix-estimated inverse.

    When ``inverse_transform_path`` is omitted, estimates ``T⁻¹`` with
    :func:`compute_inverse_transform` under ``output_directory``. Prefer
    :func:`~spirit_phantom.core.registration.ensure_inverse_transform` so
    analyses share one inverse in the registration directory.

    Args:
        targets: NIfTI/nibabel world-coordinate points in the moving domain.
        transform_parameter_path: Forward elastix transform (fixed → moving).
        fixed_image_path: Fixed image of the forward registration.
        moving_image_path: Moving image path used by transformix for geometry
            of the round-trip check.
        output_directory: Directory for point I/O (and a local inverse when
            ``inverse_transform_path`` is omitted).
        inverse_transform_path: Optional precomputed inverse transform head.

    Returns:
        Fixed-domain NIfTI/nibabel world-coordinate points.
    """
    resolved_inverse = (
        inverse_transform_path
        if inverse_transform_path is not None
        else compute_inverse_transform(
            transform_parameter_path=transform_parameter_path,
            fixed_image_path=fixed_image_path,
            output_directory=output_directory / "inverse_transform",
        )
    )
    fixed_points = transform_points_with_transformix(
        points=targets,
        transform_parameter_path=resolved_inverse,
        moving_image_path=fixed_image_path,
        output_directory=output_directory / "inverse_points",
    )
    round_trip = transform_points_with_transformix(
        points=fixed_points,
        transform_parameter_path=transform_parameter_path,
        moving_image_path=moving_image_path,
        output_directory=output_directory / "round_trip_check",
    )
    residual = _max_residual_mm(mapped=round_trip, targets=targets)
    if residual > _INVERSE_RESIDUAL_WARNING_MM:
        logger.warning(
            "Elastix inverse round-trip residual %.4g mm exceeds %.4g mm.",
            residual,
            _INVERSE_RESIDUAL_WARNING_MM,
        )
    else:
        logger.info("Elastix inverse round-trip residual: %.4g mm.", residual)
    return fixed_points


def _invert_points_with_picard_iteration(
    *,
    targets: list[list[float]],
    transform_parameter_path: Path,
    moving_image_path: Path,
    output_directory: Path,
    max_iterations: int,
    tolerance_mm: float,
) -> list[list[float]]:
    """Map moving-domain points to fixed space by fixed-point iteration.

    Finds fixed points ``x`` such that ``T(x) ≈ p_moving`` using fixed-point
    iteration ``x ← p - (T(x) - x)``.

    Args:
        targets: NIfTI/nibabel world-coordinate points in the moving domain.
        transform_parameter_path: Forward elastix transform (fixed → moving).
        moving_image_path: Moving image path used by transformix for geometry.
        output_directory: Directory for per-iteration transformix outputs.
        max_iterations: Maximum fixed-point iterations.
        tolerance_mm: Stop when max absolute residual in millimetres is below
            this threshold.

    Returns:
        Fixed-domain NIfTI/nibabel world-coordinate points.

    Raises:
        RuntimeError: If the iteration fails to converge.
    """
    estimates = [point.copy() for point in targets]
    residual = float("inf")
    for iteration in range(max_iterations):
        iteration_directory = output_directory / f"iteration_{iteration:02d}"
        mapped = transform_points_with_transformix(
            points=estimates,
            transform_parameter_path=transform_parameter_path,
            moving_image_path=moving_image_path,
            output_directory=iteration_directory,
        )
        residual = _max_residual_mm(mapped=mapped, targets=targets)
        estimates = [
            [
                target[dim] - (mapped_point[dim] - estimate[dim])
                for dim in range(len(target))
            ]
            for target, estimate, mapped_point in zip(
                targets, estimates, mapped, strict=True
            )
        ]
        if residual <= tolerance_mm:
            return estimates

    msg = (
        f"Moving-to-fixed inversion did not converge after {max_iterations} "
        f"iterations (residual={residual:.6g} mm, tolerance={tolerance_mm} mm)."
    )
    raise RuntimeError(msg)


def invert_points_through_transformix(  # noqa: PLR0913
    *,
    moving_points: _Points,
    transform_parameter_path: Path,
    fixed_image_path: Path,
    moving_image_path: Path,
    output_directory: Path,
    method: InversionMethod = InversionMethod.ELASTIX,
    inverse_transform_path: Path | None = None,
    max_iterations: int = 10,
    tolerance_mm: float = 1e-4,
) -> list[list[float]]:
    """Map moving-domain points into fixed space by inverting transformix ``T``.

    Useful for atlas landmarks that should land in scan (fixed) space after
    registration. By default the inverse is estimated with elastix
    (``DisplacementMagnitudePenalty``); see :class:`InversionMethod` for the
    alternatives. Pass ``inverse_transform_path`` from
    :func:`~spirit_phantom.core.registration.ensure_inverse_transform` to reuse
    a shared inverse. The result is also written to ``fixed_space_points.txt``.

    Args:
        moving_points: NIfTI/nibabel world-coordinate points in the
            moving/atlas domain (same frame as the atlas affine).
        transform_parameter_path: Forward elastix transform (fixed → moving).
        fixed_image_path: Fixed image of the forward registration. Defines the
            domain of the elastix inverse; unused by Picard iteration.
        moving_image_path: Moving image path used by transformix for geometry.
        output_directory: Directory for intermediate transformix outputs.
        method: Inversion method.
        inverse_transform_path: Optional precomputed inverse transform head
            (elastix method only). When omitted, a local inverse is estimated
            under ``output_directory``.
        max_iterations: Maximum fixed-point iterations (Picard only).
        tolerance_mm: Stop when max absolute residual in millimetres is below
            this threshold (Picard only).

    Returns:
        Corresponding NIfTI/nibabel world-coordinate points in the fixed image
        domain (compatible with ``fixed_image.affine`` and nibabel masks).

    Raises:
        ValueError: If ``moving_points`` is empty.
        RuntimeError: If Picard iteration fails to converge.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    targets = [list(map(float, point)) for point in moving_points]
    if not targets:
        msg = "Cannot invert an empty point set"
        raise ValueError(msg)

    if InversionMethod(method) is InversionMethod.PICARD:
        fixed_points = _invert_points_with_picard_iteration(
            targets=targets,
            transform_parameter_path=transform_parameter_path,
            moving_image_path=moving_image_path,
            output_directory=output_directory,
            max_iterations=max_iterations,
            tolerance_mm=tolerance_mm,
        )
    else:
        fixed_points = _invert_points_with_elastix(
            targets=targets,
            transform_parameter_path=transform_parameter_path,
            fixed_image_path=fixed_image_path,
            moving_image_path=moving_image_path,
            output_directory=output_directory,
            inverse_transform_path=inverse_transform_path,
        )

    save_points(
        points=fixed_points,
        output_path=output_directory / _FIXED_SPACE_POINTS_FILENAME,
        point_type="point",
    )
    return fixed_points


def parse_transformix_output(output_path: Path) -> list[list[float]]:
    """Parse transformed points from transformix output file.

    Args:
        output_path: Path to the transformix outputpoints.txt file.

    Returns:
        List of transformed point coordinates, each point is a list of floats.

    Raises:
        FileNotFoundError: If output_path does not exist.
        ValueError: If output file is malformed or parsing fails.
    """
    if not output_path.exists():
        msg = (
            f"Transformix output file not found: {output_path}. "
            "Transformix may have failed to process points."
        )
        raise FileNotFoundError(msg)

    transformed_points: list[list[float]] = []
    output_point_marker = "OutputPoint = ["

    with output_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if "OutputPoint" not in line:
                continue

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

    return transformed_points

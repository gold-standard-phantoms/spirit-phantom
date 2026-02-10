"""Module for point I/O operations using transformix format.

This module provides utilities for loading and saving points in the transformix
native format, which uses space-separated coordinates with a simple header.

Transformix format:
    point|index
    <number_of_points>
    x1 y1 [z1]
    x2 y2 [z2]
    ...
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

# Supported dimensions for point transformations
DIM_2D = 2
DIM_3D = 3

# Minimum lines in a valid transformix points file (type line + count line)
_MIN_HEADER_LINES = 2

_PointType = Literal["point", "index"]
_Point = Sequence[float]
_Points = Sequence[_Point]


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

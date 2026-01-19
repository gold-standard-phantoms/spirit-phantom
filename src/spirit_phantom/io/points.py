"""Module for point I/O operations.

This module provides utilities for loading points from CSV files, validating them,
and writing them in various formats (CSV, transformix input format).
"""

import csv
from pathlib import Path

# Supported dimensions for point transformations
DIM_2D = 2
DIM_3D = 3


def load_points_from_csv(
    points_csv_path: Path,
) -> tuple[list[list[float]], list[str], int]:
    """Load and validate points from a CSV file.

    Args:
        points_csv_path: Path to CSV file containing points to load.
            CSV must have a header row (e.g., "X,Y" for 2D or "R,A,S" for 3D)
            followed by rows of comma-separated coordinates.

    Returns:
        A tuple of (points, header, n_dims) where:
        - points: List of point coordinates, each point is a list of floats.
        - header: List of header column names from the CSV file.
        - n_dims: Number of dimensions (2 or 3).

    Raises:
        FileNotFoundError: If points_csv_path doesn't exist.
        ValueError: If CSV file is malformed or has invalid coordinates.
    """
    if not points_csv_path.exists():
        msg = f"Points CSV file not found: {points_csv_path}"
        raise FileNotFoundError(msg)

    points = []
    header = None
    with points_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header row
        for row in reader:
            if row:  # Skip empty rows
                try:
                    coords = [float(x) for x in row]
                    points.append(coords)
                except ValueError as e:
                    msg = f"Invalid coordinate in CSV row {row}: {e}"
                    raise ValueError(msg) from e

    if not points:
        msg = "No valid points found in CSV file"
        raise ValueError(msg)

    # Detect dimensionality from first point
    n_dims = len(points[0])
    if n_dims not in (DIM_2D, DIM_3D):
        msg = f"Unsupported dimensionality: {n_dims}. Expected 2D or 3D points."
        raise ValueError(msg)

    # Verify all points have same dimensionality
    for i, pt in enumerate(points):
        if len(pt) != n_dims:
            msg = f"Point {i} has {len(pt)} dimensions, expected {n_dims}"
            raise ValueError(msg)

    return points, header, n_dims


def write_points_to_transformix_format(
    points: list[list[float]],
    n_dims: int,
    output_file: Path,
) -> None:
    """Write points to transformix input format file.

    Args:
        points: List of point coordinates, each point is a list of floats.
        n_dims: Number of dimensions (2 or 3).
        output_file: Path to the output file to write transformix format points.

    Raises:
        ValueError: If n_dims is not 2 or 3, or if points have inconsistent dimensions.
    """
    if n_dims not in (DIM_2D, DIM_3D):
        msg = f"Unsupported dimensionality: {n_dims}. Expected 2D or 3D points."
        raise ValueError(msg)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("point\n")  # Points in physical coordinates
        f.write(f"{len(points)}\n")
        for pt in points:
            if len(pt) != n_dims:
                msg = f"Point has {len(pt)} dimensions, expected {n_dims}"
                raise ValueError(msg)
            if n_dims == DIM_2D:
                f.write(f"{pt[0]} {pt[1]}\n")
            else:  # n_dims == DIM_3D
                f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")


def save_transformed_points_to_csv(
    points: list[list[float]],
    header: list[str],
    output_path: Path,
) -> None:
    """Save transformed points to a CSV file.

    Args:
        points: List of transformed point coordinates, each point is a list of floats.
        header: Header row to write to CSV (preserves original format).
        output_path: Path to the output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Write header (preserve original format)
        writer.writerow(header)
        # Write transformed points
        for pt in points:
            writer.writerow(pt)

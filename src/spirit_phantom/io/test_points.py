"""Tests for transformix point and index I/O."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from spirit_phantom.io.points import load_points, parse_transformix_output, save_points

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


def test_parse_transformix_output_reads_output_point_for_point_type(
    tmp_path: Path,
) -> None:
    """Parse transformed physical points from transformix output."""
    output_path = tmp_path / "outputpoints.txt"
    output_path.write_text(
        "\n".join(
            (
                "Point 0 ; InputPoint = [ 1.0 2.0 ] ; OutputPoint = [ 3.5 4.5 ] ;",
                "Point 1 ; InputPoint = [ 5.0 6.0 ] ; OutputPoint = [ 7.5 8.5 ] ;",
            ),
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_transformix_output(output_path=output_path, point_type="point")

    assert parsed == [[3.5, 4.5], [7.5, 8.5]]


def test_parse_transformix_output_reads_output_index_moving_for_index_type(
    tmp_path: Path,
) -> None:
    """Parse transformed voxel indices from transformix output."""
    output_path = tmp_path / "outputpoints.txt"
    output_path.write_text(
        "\n".join(
            (
                "Point 0 ; OutputPoint = [ 9.9 9.9 ] ; OutputIndexMoving = [ 10 20 ] ;",
                "Point 1 ; OutputPoint = [ 8.8 8.8 ] ; OutputIndexMoving = [ 30 40 ] ;",
            ),
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_transformix_output(output_path=output_path, point_type="index")

    assert parsed == [[10.0, 20.0], [30.0, 40.0]]


def test_parse_transformix_output_rejects_unsupported_point_type(
    tmp_path: Path,
) -> None:
    """Raise an error for unsupported point type values."""
    output_path = tmp_path / "outputpoints.txt"
    output_path.write_text(
        "Point 0 ; OutputPoint = [ 1.0 2.0 ] ;\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported point_type"):
        parse_transformix_output(
            output_path=output_path,
            point_type=cast("PointType", "unsupported"),
        )

"""Interpolation helpers implemented with NumPy only."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt


def trilinear_sample(
    volume: npt.ArrayLike,
    coordinates: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Sample a 3-D volume at fractional voxel coordinates.

    Uses trilinear interpolation. Coordinates outside the volume are clamped to
    the nearest edge voxel, matching SciPy's
    ``map_coordinates(..., order=1, mode="nearest")``.

    Args:
        volume: Array of shape ``(nx, ny, nz)``.
        coordinates: Voxel coordinates of shape ``(3, ...)`` in array axis
            order. Any trailing shape is allowed.

    Returns:
        Interpolated values with shape ``coordinates.shape[1:]``.

    Raises:
        ValueError: If ``volume`` is not 3-D or ``coordinates`` does not have
            a leading axis of length 3.
    """
    data = np.asarray(volume, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    if data.ndim != 3:  # noqa: PLR2004
        msg = f"volume must be 3-D, got {data.ndim}-D."
        raise ValueError(msg)
    if coords.ndim < 1 or coords.shape[0] != 3:  # noqa: PLR2004
        msg = f"coordinates must have shape (3, ...), got {coords.shape}."
        raise ValueError(msg)

    upper = (np.asarray(data.shape) - 1).reshape(3, *([1] * (coords.ndim - 1)))
    clamped = np.clip(coords, 0, upper)
    lower_index = np.floor(clamped).astype(np.intp)
    upper_index = np.minimum(lower_index + 1, upper)
    fraction = clamped - lower_index

    result = np.zeros(coords.shape[1:], dtype=np.float64)
    for corner in itertools.product((0, 1), repeat=3):
        index = tuple(
            upper_index[axis] if use_upper else lower_index[axis]
            for axis, use_upper in enumerate(corner)
        )
        weight = np.ones(coords.shape[1:], dtype=np.float64)
        for axis, use_upper in enumerate(corner):
            weight *= fraction[axis] if use_upper else 1.0 - fraction[axis]
        result += weight * data[index]
    return result

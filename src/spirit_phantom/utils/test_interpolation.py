"""Tests for NumPy-based interpolation helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from spirit_phantom.utils.interpolation import trilinear_sample


def test_trilinear_sample_exact_at_voxel_centres() -> None:
    """Integer coordinates should return the stored voxel values."""
    volume = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    coords = np.asarray([[1, 0], [2, 1], [3, 0]], dtype=np.float64)

    np.testing.assert_array_equal(
        trilinear_sample(volume, coords), [volume[1, 2, 3], volume[0, 1, 0]]
    )


def test_trilinear_sample_reproduces_linear_field() -> None:
    """Trilinear interpolation is exact for a field linear in each axis."""
    i, j, k = np.indices((4, 5, 6), dtype=np.float64)
    volume = 2.0 * i - 3.0 * j + 0.5 * k + 1.0
    coords = np.asarray([[0.25, 2.5], [1.75, 3.1], [4.5, 0.2]])

    expected = 2.0 * coords[0] - 3.0 * coords[1] + 0.5 * coords[2] + 1.0
    np.testing.assert_allclose(trilinear_sample(volume, coords), expected)


@pytest.mark.parametrize("shape", [(20, 30, 1), (15, 17, 9)])
def test_trilinear_sample_matches_scipy_nearest_mode(
    shape: tuple[int, int, int],
) -> None:
    """Results should match ``map_coordinates(order=1, mode="nearest")``."""
    rng = np.random.default_rng(seed=0)
    volume = rng.random(shape)
    coords = rng.uniform(-3.0, max(shape) + 3.0, size=(3, 64, 5))

    expected = map_coordinates(volume, coords, order=1, mode="nearest")
    np.testing.assert_allclose(trilinear_sample(volume, coords), expected, atol=1e-12)


def test_trilinear_sample_rejects_bad_shapes() -> None:
    """Non-3-D volumes and malformed coordinates should raise."""
    with pytest.raises(ValueError, match="3-D"):
        trilinear_sample(np.zeros((2, 2)), np.zeros((3, 1)))
    with pytest.raises(ValueError, match=r"\(3, \.\.\.\)"):
        trilinear_sample(np.zeros((2, 2, 2)), np.zeros((2, 1)))

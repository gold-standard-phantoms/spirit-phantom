"""Tests for NumPy-based binary morphology helpers."""

from __future__ import annotations

import numpy as np
import pytest

from spirit_phantom.utils.binary_morphology import binary_dilation_connectivity_one


def test_binary_dilation_connectivity_one_2d_single_iteration() -> None:
    """Single-iteration 2D dilation should use axis-connected neighbours."""
    mask = np.zeros(shape=(5, 5), dtype=np.uint8)
    mask[2, 2] = np.uint8(1)

    dilated = binary_dilation_connectivity_one(mask=mask, iterations=1)

    expected = np.zeros(shape=(5, 5), dtype=np.uint8)
    expected[2, 2] = np.uint8(1)
    expected[1, 2] = np.uint8(1)
    expected[3, 2] = np.uint8(1)
    expected[2, 1] = np.uint8(1)
    expected[2, 3] = np.uint8(1)
    np.testing.assert_array_equal(dilated, expected)


def test_binary_dilation_connectivity_one_3d_single_iteration() -> None:
    """Single-iteration 3D dilation should add six axis neighbours."""
    mask = np.zeros(shape=(3, 3, 3), dtype=np.uint8)
    mask[1, 1, 1] = np.uint8(2)

    dilated = binary_dilation_connectivity_one(mask=mask, iterations=1)

    assert int(np.count_nonzero(dilated)) == 7
    assert int(dilated[1, 1, 1]) == 1
    assert int(dilated[0, 1, 1]) == 1
    assert int(dilated[2, 1, 1]) == 1
    assert int(dilated[1, 0, 1]) == 1
    assert int(dilated[1, 2, 1]) == 1
    assert int(dilated[1, 1, 0]) == 1
    assert int(dilated[1, 1, 2]) == 1


def test_binary_dilation_connectivity_one_zero_iterations_binarises() -> None:
    """Zero iterations should only binarise the input mask."""
    mask = np.array([[0, 2], [3, 0]], dtype=np.uint8)

    dilated = binary_dilation_connectivity_one(mask=mask, iterations=0)

    np.testing.assert_array_equal(
        dilated,
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
    )


def test_binary_dilation_connectivity_one_rejects_negative_iterations() -> None:
    """Negative iteration count should be rejected."""
    mask = np.zeros(shape=(2, 2), dtype=np.uint8)

    with pytest.raises(
        ValueError, match=r"iterations must be greater than or equal to 0\."
    ):
        binary_dilation_connectivity_one(mask=mask, iterations=-1)

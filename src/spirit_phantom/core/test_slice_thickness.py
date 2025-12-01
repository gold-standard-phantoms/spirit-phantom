"""Tests for the slice thickness module."""

import math

import numpy as np
import pytest
from scipy import special

from spirit_phantom.core.slice_thickness import (
    calculate_slice_profile,
    full_width_half_maximum,
    nema_slice_thickness,
)

NEMA_MS_5_2018_FIGURE_2_3_SLICE_PROFILE = np.array(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
    ]
)

EDGE_WITH_GAUSSIAN_PROFILE = 10 * (
    0.5 * (1 + special.erf((np.arange(50) - 10) / (2.5 * np.sqrt(2))))
)


TEST_ANGLE_DEGREES = 15.0


def test_calculate_slice_profile() -> None:
    """Test the calculate_slice_profile function."""
    # Test case 1: Simple linear edge response function
    edge_response_function = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    slice_profile = calculate_slice_profile(edge_response_function, 1.0)
    assert len(slice_profile) == len(edge_response_function) - 1
    assert np.allclose(slice_profile, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    # Test case 2: Edge response function with a plateau
    edge_response_function = np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10])
    slice_profile = calculate_slice_profile(edge_response_function, 1.0)
    assert len(slice_profile) == len(edge_response_function) - 1
    assert np.allclose(
        slice_profile, np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    )

    # Test case 3: Edge response function with ramp as in NEME MS 5-2018 figure 2-3
    slice_profile = calculate_slice_profile(
        NEMA_MS_5_2018_FIGURE_2_3_SLICE_PROFILE, 1.0
    )
    assert len(slice_profile) == len(NEMA_MS_5_2018_FIGURE_2_3_SLICE_PROFILE) - 1
    assert np.allclose(
        slice_profile,
        np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        ),
    )

    # Test case 4: Edge response function with ramp as in NEME MS 5-2018 figure 2-3 with pixel size 2 mm
    slice_profile = calculate_slice_profile(NEMA_MS_5_2018_FIGURE_2_3_SLICE_PROFILE, 2)
    assert len(slice_profile) == len(NEMA_MS_5_2018_FIGURE_2_3_SLICE_PROFILE) - 1
    # With 2x the pixel shize, the slice to produce the same profile should have half the slope
    assert np.allclose(
        slice_profile,
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    )


def test_full_width_half_maximum() -> None:
    """Test the full_width_half_maximum function."""
    # Test case 1: Edge response function with ramp as in NEME MS 5-2018 figure 2-3
    slice_profile = calculate_slice_profile(NEMA_MS_5_2018_FIGURE_2_3_SLICE_PROFILE, 1)
    fwhm = full_width_half_maximum(slice_profile)
    assert fwhm == pytest.approx(9, abs=0.1)

    # Test case 2: Edge response function with Gaussian profile
    slice_profile = calculate_slice_profile(EDGE_WITH_GAUSSIAN_PROFILE, 1)
    fwhm = full_width_half_maximum(slice_profile)
    assert fwhm == pytest.approx(6, abs=0.1)


def test_nema_slice_thickness() -> None:
    """Test the nema_slice_thickness function."""
    # Test case 1: Edge response function with ramp as in NEME MS 5-2018 figure 2-3
    assert nema_slice_thickness(
        NEMA_MS_5_2018_FIGURE_2_3_SLICE_PROFILE, 1, TEST_ANGLE_DEGREES
    ) == pytest.approx(9 * math.tan(math.radians(TEST_ANGLE_DEGREES)), abs=0.1)

    # Test case 2: Edge response function with Gaussian profile
    assert nema_slice_thickness(
        EDGE_WITH_GAUSSIAN_PROFILE, 1, TEST_ANGLE_DEGREES
    ) == pytest.approx(6 * math.tan(math.radians(TEST_ANGLE_DEGREES)), abs=0.1)

    # Test case 3: input is 2d array with three edge response functions as column vectors
    np.random.seed(42)  # Set seed for predictable testing
    input_2d_array = np.column_stack(
        [
            EDGE_WITH_GAUSSIAN_PROFILE
            + np.random.normal(0, 0.1, len(EDGE_WITH_GAUSSIAN_PROFILE)),
            EDGE_WITH_GAUSSIAN_PROFILE
            + np.random.normal(0, 0.1, len(EDGE_WITH_GAUSSIAN_PROFILE)),
            EDGE_WITH_GAUSSIAN_PROFILE
            + np.random.normal(0, 0.1, len(EDGE_WITH_GAUSSIAN_PROFILE)),
        ]
    )
    assert input_2d_array.shape == (len(EDGE_WITH_GAUSSIAN_PROFILE), 3)
    assert nema_slice_thickness(input_2d_array, 1, TEST_ANGLE_DEGREES) == pytest.approx(
        6 * math.tan(math.radians(TEST_ANGLE_DEGREES)), abs=0.1
    )

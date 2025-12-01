r"""Module for calculating slice thickness.

This module implements the Wedge Method (Section 2.2) of the NEMA MS 5-2018 standard
for determining slice thickness in diagnostic magnetic resonance imaging. The standard
is available at:

<https://www.nema.org/Standards/view/Determination-of-Slice-Thickness-in-Diagnostic-Magnetic-Resonance-Imaging>


The implementation follows the measurement procedure described in Section 2.2.2:

1. Numerical differentiation of the edge response function (ERF) to calculate the
   stretched (projected) slice profile (Section 2.2.2, step c, page 4; Equation 4).
2. Averaging multiple edge response functions to improve signal-to-noise ratio
   (Section 3).
3. Calculation of the full width at half maximum (FWHM) of the slice profile using
   linear interpolation (Section 2.2.2, step b, page 5).
4. Scaling the projected FWHM by $\tan(\alpha)$ to obtain the slice thickness in millimetres
   (Section 2.2.2, step c, page 5).


Definitions:

- The slice thickness is defined as the FWHM of the slice profile (Section 1.2.7).
- The edge response function is the integral of the slice profile (Section 1.2.3).

Requirements:

- For reliable results, the SNR of the slice profile must be greater than 10 (Section 3).
  Averaging the slice profile over multiple edge response functions will in general be
  required to improve the signal-to-noise ratio.
- The spatial resolution must include at least six intrinsic pixels across the FWHM of the projected slice profile (Section 2.2.2, Equation 1).

Not Yet Implemented:

- Baseline pixel offset correction (Section 1.2.1): The standard defines the
  baseline pixel offset value as the pixel value representing a noise-free
  signal level of zero. This correction is not currently applied automatically
  and should be handled by the caller if the MR system uses a non-zero baseline.
- Rotational error correction (Section 4): Correction for rotational errors
  about the y-axis is not implemented. The SPIRIT phantom is designed with matching
  alignment hardware to remove the need for such corrections. The caller calculates
  slice thickness for both wedges and if discrepancies are found that warrant inclusion
  if the rotational correction please raise an issue with Gold Standard Phantoms.
"""

import logging
import math

import numpy as np

logging.basicConfig(level=logging.INFO)


def nema_slice_thickness(
    edge_response_function: np.ndarray,
    pixel_size: float,
    ramp_slope_degrees: float = 15.0,
) -> float:
    r"""Calculate the slice thickness from a given edge response function.

    This function implements the Wedge Method measurement procedure (NEMA MS 5-2018,
    Section 2.2.2). The slice thickness is defined as the full width at half maximum
    (FWHM) of the slice profile (Section 1.2.7). The slice profile is calculated by
    numerically differentiating the edge response function (Equation 4), and the
    projected FWHM is then scaled by $\tan(\alpha)$ to obtain the slice thickness in
    millimetres Section 2.2.2, step c, page 5).

    The edge response function is the integral of the slice profile versus the slice
    coordinate z (Section 1.2.3). When multiple edge response functions are provided,
    they are averaged to improve the signal-to-noise ratio (Section 3).

    Args:
        edge_response_function: The edge response function, i.e. the intensity as a
            function of the edge position. Can be a 1D array of shape (N,) or a 2D
            array of shape (N, M) where M is the number of edge response functions
            stored as column vectors. If a 2D array is provided, the average of the
            slice profile functions will be calculated. This is useful to improve the
            signal-to-noise ratio of the slice profile, making it easier to clearly
            identify the FWHM (Section 3).
        pixel_size: The size of a pixel in the edge response function in millimetres.
            The spatial resolution must include at least six intrinsic pixels across
            the FWHM of the projected slice profile (Section 2.2.2, Equation 1).
        ramp_slope_degrees: The angle $\alpha$ between the inclined surface of the wedge and
            the slice plane, in degrees [$\degree$]. Default is 15$\degree$ as this matches the
            design of the GSP SPIRIT phantom.

    Returns:
        The slice thickness in millimetres, calculated as the FWHM of the slice profile.

    Raises:
        ValueError: If the edge response function is invalid or if the FWHM cannot be
            determined.

    """
    # if a single edge profile is provided, only one profile is calculated
    if len(np.shape(edge_response_function)) == 1:  # single edge profile
        slice_profile = calculate_slice_profile(edge_response_function, pixel_size)
    else:
        # Handle column vectors: shape (N, M) where M is the number of edge response functions
        slice_profile_set = [
            calculate_slice_profile(edge_response_function[:, i], pixel_size)
            for i in range(edge_response_function.shape[1])
        ]
        slice_profile = np.mean(slice_profile_set, axis=0)

    fwhm_slope = full_width_half_maximum(slice_profile)

    ramp_slope_radians = math.radians(ramp_slope_degrees)
    return fwhm_slope * math.tan(ramp_slope_radians)


def full_width_half_maximum(slice_profile: np.ndarray) -> float:
    r"""Calculate the full width at half maximum (FWHM) of a given slice profile.

    The FWHM is defined as the width of the slice profile at one half of its maximum
    value (NEMA MS 5-2018, Section 1.2.7). This function implements the measurement
    procedure described in Section 2.2.2, step b, page 5, using linear interpolation to find
    the half-maximum crossings when necessary.

    The function finds the first crossing from below to above the half-maximum value
    and the last crossing from above to below, then calculates the width between these
    points. Linear interpolation is used to determine the exact crossing points when
    the half-maximum value falls between two sample points.

    Args:
        slice_profile: The slice profile, i.e. the intensity as a function of the
            slice position. This should be the stretched (projected) slice profile
            obtained by differentiating the edge response function.

    Returns:
        The full width at half maximum of the slice profile in pixel indices. Note
            that this is the projected FWHM in the image coordinate system, not the
            actual slice thickness. To obtain the slice thickness in millimetres, this
            value must be multiplied by $\tan(\alpha)$, where $\alpha$ is the wedge angle.

    Raises:
        ValueError: If the slice profile is invalid or if both half-maximum crossings
            cannot be found.


    """
    max_value = np.max(slice_profile)
    half_max_value = max_value / 2

    # Find the first crossing (from below to above)
    left_cross = None
    for i in range(len(slice_profile) - 1):
        if slice_profile[i] < half_max_value and slice_profile[i + 1] >= half_max_value:
            # Linear interpolation:
            x1, x2 = i, i + 1
            y1, y2 = slice_profile[x1], slice_profile[x2]
            logging.info(
                "Left cross: %s, x1: %s, x2: %s, y1: %s, y2: %s, ",
                left_cross,
                x1,
                x2,
                y1,
                y2,
            )
            if y2 == y1:
                left_cross = x1 + 1  # Flat step, pick upper
                logging.info("left is flat")
            else:
                frac = (half_max_value - y1) / (y2 - y1)
                left_cross = x1 + frac
                # Conservative round up (ceil)
                left_cross = int(np.ceil(left_cross))
                logging.info("Left not flat, frac: %s", frac)
            break

    # Find the last crossing (from above to below)
    right_cross = None
    for i in range(
        len(slice_profile) - 1, 0, -1
    ):  # iterate from the end of the profile towards the beginning
        if slice_profile[i] < half_max_value and slice_profile[i - 1] >= half_max_value:
            x1, x2 = i - 1, i
            y1, y2 = slice_profile[x1], slice_profile[x2]
            logging.info(
                "Right cross: %s, x1: %s, x2: %s, y1: %s, y2: %s",
                right_cross,
                x1,
                x2,
                y1,
                y2,
            )
            logging.info("Slice profile: %s", slice_profile)
            if y2 == y1:
                right_cross = x1  # Flat step, pick upper
                logging.info("right is flat")
            else:
                frac = (half_max_value - y1) / (y2 - y1)
                right_cross = x1 + frac
                # Conservative round up (ceil)
                right_cross = int(np.ceil(right_cross))
                logging.info("Right not flat, frac: %s", frac)
            break

    if left_cross is None or right_cross is None:
        msg = "Unable to determine FWHM: cannot find both half-max crossings."
        raise ValueError(msg)

    return right_cross - left_cross


def calculate_slice_profile(
    edge_response_function: np.ndarray, pixel_size: float
) -> np.ndarray:
    r"""Calculate the slice profile from a given edge response function.

    This function implements the numerical differentiation step of the Wedge Method
    (NEMA MS 5-2018, Section 2.2.2, step c, page 4; Equation 4). The edge response
    function (ERF) is numerically differentiated with respect to the x-direction to
    obtain a stretched (projected) slice profile $D(x)$ as specified in Equation 4:

    \[
    D(x_i)=\frac{I(x_i)-I(x_{i-1})}{x_i-x_{i-1}}
    \]

    The edge response function is the integral of the slice profile versus the slice
    coordinate z (Section 1.2.3). The resulting slice profile represents the intensity
    as a function of position perpendicular to the slice plane, but projected onto the
    image plane along the x-direction. This "stretched" profile must be scaled by
    $\tan(\alpha)$ to obtain the actual slice thickness, where $\alpha$ is the angle between the
    wedge surface and the slice plane.

    Args:
        edge_response_function: The edge response function, i.e. the intensity as a
            function of the edge position. This is the integral of the slice profile
            (Section 1.2.3).
        pixel_size: The size of a pixel in the edge response function in millimetres.
            This is used to convert pixel indices to physical distances for the
            differentiation.

    Returns:
        The stretched (projected) slice profile, i.e. the intensity as a function of
            the slice position in the image coordinate system. The length of the returned
            array is one less than the input edge response function due to the numerical
            differentiation.
    """
    x = np.arange(len(edge_response_function)) * pixel_size
    return np.diff(edge_response_function) / np.diff(x)

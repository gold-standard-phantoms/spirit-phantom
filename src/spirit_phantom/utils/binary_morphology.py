"""Binary morphology helpers implemented with NumPy only."""

from __future__ import annotations

import numpy as np


def binary_dilation_connectivity_one(
    mask: np.ndarray,
    *,
    iterations: int,
) -> np.ndarray:
    """Dilate a binary mask using axis-connected neighbourhoods.

    This uses connectivity-one neighbourhoods (centre voxel plus immediate
    neighbours along each axis), equivalent to SciPy's default binary dilation
    structuring element.

    Args:
        mask: Input mask where non-zero values are treated as foreground.
        iterations: Number of dilation iterations. ``0`` returns the input mask
            converted to binary form.

    Returns:
        Binary mask with ``uint8`` dtype and the same shape as the input.
    """
    if iterations < 0:
        msg = "iterations must be greater than or equal to 0."
        raise ValueError(msg)

    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim == 0:
        msg = "mask must have at least one dimension."
        raise ValueError(msg)

    dilated_foreground = foreground
    for _ in range(iterations):
        expanded = np.array(dilated_foreground, copy=True)
        for axis in range(dilated_foreground.ndim):
            shifted_plus = np.roll(dilated_foreground, shift=1, axis=axis)
            leading_index: list[slice | int] = [slice(None)] * dilated_foreground.ndim
            leading_index[axis] = 0
            shifted_plus[tuple(leading_index)] = False

            shifted_minus = np.roll(dilated_foreground, shift=-1, axis=axis)
            trailing_index: list[slice | int] = [slice(None)] * dilated_foreground.ndim
            trailing_index[axis] = -1
            shifted_minus[tuple(trailing_index)] = False

            expanded |= shifted_plus | shifted_minus
        dilated_foreground = expanded

    return np.asarray(dilated_foreground, dtype=np.uint8)

"""Tests for ``segmentations.py`` label statistics utilities."""

from __future__ import annotations

from typing import Final

import numpy as np

from spirit_phantom.io.segmentations import (
    _compute_label_stats_reference,
    compute_label_stats,
)

_VOLUME_SHAPE: Final = (100, 101, 102)
_SPHERE_RADIUS: Final = 5.0
_GRID_COORDINATES: Final = (20.0, 40.0, 60.0, 80.0)


def _create_spherical_label_volume() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic 3D label volume containing a grid of spheres.

    The returned volume is 100 x 100 x 100 with background value 0.0 (float32).
    It contains a 4 x 4 x 4 grid of non-overlapping spheres with radius 5 voxels
    and centre-to-centre spacing of 20 voxels.

    Returns:
        seg: 3D segmentation volume with float32 labels.
        centres: Array of shape (n_labels, 3) containing (z, y, x) centres.
        voxel_counts: Array of shape (n_labels,) with voxel counts per label.
    """
    seg = np.zeros(_VOLUME_SHAPE, dtype=np.float32)

    x_indices, y_indices, z_indices = np.indices(_VOLUME_SHAPE)

    centres_list: list[tuple[float, float, float]] = []
    counts_list: list[int] = []

    label_value = 1.0
    radius_sq = _SPHERE_RADIUS**2

    for z_centre in _GRID_COORDINATES:
        for y_centre in _GRID_COORDINATES:
            for x_centre in _GRID_COORDINATES:
                distance_sq = (
                    (x_indices - x_centre) ** 2
                    + (y_indices - y_centre) ** 2
                    + (z_indices - z_centre) ** 2
                )
                mask = distance_sq <= radius_sq

                # Assign unique label to this spherical region.
                seg[mask] = label_value

                counts_list.append(int(mask.sum()))
                centres_list.append((x_centre, y_centre, z_centre))

                label_value += 1.0

    centres = np.asarray(centres_list, dtype=np.float64)
    voxel_counts = np.asarray(counts_list, dtype=np.float64)

    return seg, centres, voxel_counts


def test_compute_label_stats_reference_on_spherical_grid() -> None:
    """Validate reference label statistics for a synthetic spherical grid phantom.

    This test constructs a reference segmentation volume consisting of a 4 x 4 x 4
    grid of spheres. It then verifies that the reference implementation of
    ``compute_label_stats``:

    - recover the correct label values
    - compute centres that match the known sphere centres
    - count the correct number of voxels per label
    """
    seg, expected_centres, expected_counts = _create_spherical_label_volume()

    n_labels = expected_centres.shape[0]
    expected_labels = np.arange(1.0, float(n_labels) + 1.0, dtype=np.float64)

    stats_reference = _compute_label_stats_reference(seg_data=seg)

    # Sort rows by label value to ensure deterministic comparison.
    order_reference = np.argsort(stats_reference[:, 0])

    stats_reference = stats_reference[order_reference]

    labels = stats_reference[:, 0]
    centres = stats_reference[:, 1:4]
    voxel_counts = stats_reference[:, 4]

    assert labels.shape == expected_labels.shape
    assert centres.shape == expected_centres.shape
    assert voxel_counts.shape == expected_counts.shape

    # Labels and voxel counts must match the synthetic phantom definition.
    np.testing.assert_allclose(labels, expected_labels)
    np.testing.assert_allclose(voxel_counts, expected_counts)

    # Centres of mass should coincide with the exact sphere centres.
    np.testing.assert_allclose(centres, expected_centres, rtol=0.0, atol=1e-6)


def test_compute_label_stats_performant_on_spherical_grid() -> None:
    """Validate performant label statistics for a synthetic spherical grid phantom.

    This test constructs a reference segmentation volume consisting of a 4 x 4 x 4
    grid of spheres. It then verifies that the performant implementation of
    ``compute_label_stats``:

    - recover the correct label values
    - compute centres that match the known sphere centres
    - count the correct number of voxels per label
    """
    seg, expected_centres, expected_counts = _create_spherical_label_volume()

    n_labels = expected_centres.shape[0]
    expected_labels = np.arange(1.0, float(n_labels) + 1.0, dtype=np.float64)

    stats_fast = compute_label_stats(seg_data=seg)

    # Sort rows by label value to ensure deterministic comparison.
    order_fast = np.argsort(stats_fast[:, 0])
    stats_fast = stats_fast[order_fast]

    labels = stats_fast[:, 0]
    centres = stats_fast[:, 1:4]
    voxel_counts = stats_fast[:, 4]

    assert labels.shape == expected_labels.shape
    assert centres.shape == expected_centres.shape
    assert voxel_counts.shape == expected_counts.shape

    # Labels and voxel counts must match the synthetic phantom definition.
    np.testing.assert_allclose(labels, expected_labels)
    np.testing.assert_allclose(voxel_counts, expected_counts)

    # Centres of mass should coincide with the exact sphere centres.
    np.testing.assert_allclose(centres, expected_centres, rtol=0.0, atol=1e-6)

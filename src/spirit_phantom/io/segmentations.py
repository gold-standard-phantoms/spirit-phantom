"""Module for computing label statistics from segmentations.

This module provides functions to compute the centre of mass and voxel count for each non-zero label in a segmentation.
"""

import numpy as np


def compute_label_stats(seg_data: np.ndarray) -> np.ndarray:
    """A performant implementation to compute centre of mass and voxel count for each non-zero label.

    Args:
        seg_data: 3D array of integer labels (e.g. from a labelled segmentation NIfTI).

    Returns:
        2D array of shape (n_labels, 5) with columns
        [label_value, x_center, y_center, z_center, num_voxels] in voxel indices.
    """
    # 1. Cast to unsigned int (or int32/64)
    # Ensure background is 0
    data_int = seg_data.astype(np.int32)
    flat_labels = data_int.ravel()

    # 2. Get counts for every integer from 0 to max_label
    counts = np.bincount(flat_labels)

    # Find which labels actually exist in the data (excluding 0)
    labels = np.where(counts > 0)[0]
    labels = labels[labels > 0]
    final_counts = counts[labels]

    if len(labels) == 0:
        return np.empty((0, 5))

    # 3. Use the coordinate-summing trick
    shape = seg_data.shape
    z_coords, y_coords, x_coords = np.indices(shape)  # Simplest, but uses more RAM

    # Sum coordinates for every label bucket
    sum_z = np.bincount(flat_labels, weights=z_coords.ravel())[labels]
    sum_y = np.bincount(flat_labels, weights=y_coords.ravel())[labels]
    sum_x = np.bincount(flat_labels, weights=x_coords.ravel())[labels]

    # 4. Result: [label, z, y, x, count]
    return np.column_stack(
        (
            labels.astype(float),
            sum_z / final_counts,
            sum_y / final_counts,
            sum_x / final_counts,
            final_counts.astype(float),
        )
    )


def _compute_label_stats_reference(seg_data: np.ndarray) -> np.ndarray:
    """A slower and easier to read implementation to compute centre of mass and voxel count for each non-zero label.

    Args:
        seg_data: 3D array of integer labels (e.g. from a labelled segmentation NIfTI).

    Returns:
        2D array of shape (n_labels, 5) with columns
        [label_value, x_center, y_center, z_center, num_voxels] in voxel indices.
    """
    labels = np.unique(seg_data)
    labels = labels[labels > 0]
    rows: list[list[float]] = []
    for value in labels:
        mask = seg_data == value
        num_voxels = int(np.sum(mask))
        indices = np.argwhere(mask)
        center = indices.mean(axis=0)
        x_center, y_center, z_center = center[0], center[1], center[2]
        rows.append([float(value), x_center, y_center, z_center, float(num_voxels)])
    return np.array(rows)

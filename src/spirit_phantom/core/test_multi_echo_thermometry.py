"""Tests for multi-echo thermometry EG mask generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import nibabel
import numpy as np

from spirit_phantom.core.multi_echo_thermometry import (
    coordinate_ethylene_glycol_vial_segmentation,
    dilate_segmentation_mask,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_dilate_segmentation_mask_expands_foreground_voxels() -> None:
    """Binary dilation should grow the initial foreground support."""
    mask = np.zeros(shape=(5, 5, 1), dtype=np.uint8)
    mask[2, 2, 0] = np.uint8(1)

    dilated_mask = dilate_segmentation_mask(mask=mask, iterations=1)

    assert dilated_mask.shape == mask.shape
    assert dilated_mask.dtype == np.uint8
    assert int(np.count_nonzero(dilated_mask)) > int(np.count_nonzero(mask))


def test_dilate_segmentation_mask_zero_iterations_keeps_mask_unchanged() -> None:
    """Zero iterations should return the original binary foreground mask."""
    mask = np.zeros(shape=(5, 5, 1), dtype=np.uint8)
    mask[2, 2, 0] = np.uint8(3)
    mask[1, 3, 0] = np.uint8(1)

    undilated_mask = dilate_segmentation_mask(mask=mask, iterations=0)

    expected_binary_mask = (mask != 0).astype(np.uint8)
    assert undilated_mask.shape == mask.shape
    assert undilated_mask.dtype == np.uint8
    np.testing.assert_array_equal(undilated_mask, expected_binary_mask)


def test_coordinate_eg_mask_applies_sad_filter_and_writes_plot(
    tmp_path: Path,
) -> None:
    """Generated EG mask should retain only voxels above the SAD threshold.

    For a voxel trace ``S`` with 5 time points, the Sum of Absolute
    Differences (SAD) is:

    ``SAD = |S1 - S0| + |S2 - S1| + |S3 - S2| + |S4 - S3|``.

    This test uses:

    - ``[0,  20, 0,  20, 0]``   -> `` 20 +  20 +  20 +  20 =  80``  (excluded)
    - ``[0, 100, 80, 100, 80]`` -> ``100 + 20 + 20 + 20 = 160``     (excluded)
    - ``[0, 260, 0, 260, 0]``   -> ``260 + 260 + 260 + 260 = 1040`` (included)
    - ``[0, 300, 0, 300, 0]``   -> ``300 + 300 + 300 + 300 = 1200`` (included)
    """
    atlas_data = np.zeros(shape=(5, 5, 1), dtype=np.uint8)
    atlas_data[1, 1, 0] = np.uint8(21)
    atlas_data[2, 2, 0] = np.uint8(21)
    atlas_data[3, 3, 0] = np.uint8(22)
    atlas_data[4, 4, 0] = np.uint8(22)
    atlas_image = nibabel.Nifti1Image(dataobj=atlas_data, affine=np.eye(4))
    atlas_path = tmp_path / "atlas.nii.gz"
    nibabel.save(atlas_image, str(atlas_path))

    thermometry_data = np.zeros(shape=(5, 5, 1, 5), dtype=np.float32)
    thermometry_data[1, 1, 0, :] = np.array(
        [0.0, 20.0, 0.0, 20.0, 0.0], dtype=np.float32
    )
    thermometry_data[2, 2, 0, :] = np.array(
        [0.0, 100.0, 80.0, 100.0, 80.0], dtype=np.float32
    )
    thermometry_data[3, 3, 0, :] = np.array(
        [0.0, 260.0, 0.0, 260.0, 0.0], dtype=np.float32
    )
    thermometry_data[4, 4, 0, :] = np.array(
        [0.0, 300.0, 0.0, 300.0, 0.0], dtype=np.float32
    )
    thermometry_image = nibabel.Nifti1Image(dataobj=thermometry_data, affine=np.eye(4))
    thermometry_path = tmp_path / "thermometry.nii.gz"
    nibabel.save(thermometry_image, str(thermometry_path))

    output_mask_path = tmp_path / "ethylene_glycol_mask.nii.gz"
    saved_mask_path = coordinate_ethylene_glycol_vial_segmentation(
        registered_component_atlas_image_path=atlas_path,
        multi_echo_gradient_echo_scan_image_path=thermometry_path,
        minimum_sad_counts=1000.0,
        dilation_iterations=0,
        output_mask_image_path=output_mask_path,
        generate_sad_visualisation=True,
    )

    saved_mask_image = cast(
        "nibabel.nifti1.Nifti1Image", nibabel.load(str(saved_mask_path))
    )
    saved_mask = np.asarray(saved_mask_image.get_fdata(), dtype=np.uint8)
    assert int(np.count_nonzero(saved_mask)) == 2
    assert float(saved_mask[1, 1, 0]) == 0.0  # excluded with SAD < 1000
    assert float(saved_mask[2, 2, 0]) == 0.0  # excluded with SAD < 1000
    assert float(saved_mask[3, 3, 0]) == 22.0  # included with SAD >= 1000
    assert float(saved_mask[4, 4, 0]) == 22.0  # included with SAD >= 1000
    assert set(np.unique(saved_mask).tolist()) == {0, 22}

    expected_plot_path = tmp_path / "ethylene_glycol_mask_sad_filter_plot.png"
    assert expected_plot_path.exists()

    expected_mapped_label_mask_path = (
        tmp_path / "ethylene_glycol_mask_mapped_label_mask.nii.gz"
    )
    assert expected_mapped_label_mask_path.exists()
    mapped_label_mask_image = cast(
        "nibabel.nifti1.Nifti1Image", nibabel.load(str(expected_mapped_label_mask_path))
    )
    mapped_label_mask = np.asarray(mapped_label_mask_image.get_fdata(), dtype=np.uint8)
    assert float(mapped_label_mask[1, 1, 0]) == 21.0
    assert float(mapped_label_mask[2, 2, 0]) == 21.0
    assert float(mapped_label_mask[3, 3, 0]) == 22.0
    assert float(mapped_label_mask[4, 4, 0]) == 22.0

"""Tests for atlas transfer between NIfTI image spaces.

Synthetic geometry assumptions used throughout this module:

- Voxel indices are treated in NumPy order ``(x, y, z)``.
- Transfer uses pull resampling via ``inv(atlas_affine) @ target_affine``.
- Identity affines represent index-aligned atlas and target spaces.
- Integer affine translations are interpreted as whole-voxel shifts.
- In-memory NIfTI images are sufficient because tests exercise mapping logic,
  not file I/O.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from spirit_phantom.io.atlas_transfer import (
    build_atlas_in_target_space,
    transfer_atlas_labels_to_image_space,
)


def _make_nifti(*, data: np.ndarray, affine: np.ndarray) -> nib.Nifti1Image:
    """Create an in-memory NIfTI image.

    Args:
        data: Image data array.
        affine: Homogeneous 4x4 affine matrix.

    Returns:
        NIfTI image with the given data and affine.
    """
    return nib.Nifti1Image(dataobj=data, affine=affine)


def _make_base_atlas_data() -> np.ndarray:
    """Create a deterministic labelled atlas volume.

    Returns:
        Small 3-D atlas with two non-zero labels in known regions.
    """
    atlas_data = np.zeros((5, 5, 5), dtype=np.uint8)
    atlas_data[1:4, 1:4, 1:4] = 1
    atlas_data[4, 0, 0] = 2
    return atlas_data


def test_transfer_identity_affine_preserves_labels_and_shape() -> None:
    """Map labels unchanged when atlas and target spaces are identical.

    This is the baseline sanity check for transfer behaviour. Both atlas and
    target use the identity affine and have the same shape, so
    ``inv(atlas_affine) @ target_affine`` is also identity. Every target voxel
    therefore maps to the same atlas voxel index, and the output must exactly
    match the atlas data with no shifts, clipping, or filtering artefacts.
    """
    atlas_data = _make_base_atlas_data()
    identity_affine = np.eye(4)
    atlas_image = _make_nifti(data=atlas_data, affine=identity_affine)
    target_image = _make_nifti(
        data=np.zeros_like(atlas_data, dtype=np.float32),
        affine=identity_affine,
    )

    output = transfer_atlas_labels_to_image_space(
        atlas_image=atlas_image,
        target_image=target_image,
    )

    assert output.dtype == np.uint8
    assert output.shape == atlas_data.shape
    np.testing.assert_array_equal(output, atlas_data)


def test_transfer_translation_respects_inverse_warp_direction() -> None:
    """Shift a single label according to inverse-warp mapping direction.

    The atlas affine is identity and the target affine has +1 translation in x.
    The composite affine is therefore +1 in x, meaning target index x samples
    atlas index x + 1. A label at atlas (2, 2, 2) must appear at target
    (1, 2, 2). This confirms the pull resampling direction rather than a push
    interpretation.
    """
    atlas_data = np.zeros((5, 5, 5), dtype=np.uint8)
    atlas_data[2, 2, 2] = 7
    atlas_affine = np.eye(4)
    target_affine = np.eye(4)
    target_affine[0, 3] = 1.0
    atlas_image = _make_nifti(data=atlas_data, affine=atlas_affine)
    target_image = _make_nifti(
        data=np.zeros_like(atlas_data, dtype=np.float32),
        affine=target_affine,
    )

    output = transfer_atlas_labels_to_image_space(
        atlas_image=atlas_image,
        target_image=target_image,
    )

    expected = np.zeros_like(atlas_data)
    expected[1, 2, 2] = 7
    np.testing.assert_array_equal(output, expected)


def test_transfer_out_of_bounds_voxels_are_zero() -> None:
    """Write zeros where mapped target voxels fall outside atlas extent.

    The atlas is a 2x2x2 block of ones (valid atlas x indices 0..1). The
    target is larger in x with shape 4x2x2 and has +1 translation in x. Mapped
    atlas x indices for target x=0..3 are 1, 2, 3, 4 respectively, so only the
    first target x-plane is in range and should remain one; all other target
    planes are out of bounds and must be zero.
    """
    atlas_data = np.ones((2, 2, 2), dtype=np.uint8)
    atlas_image = _make_nifti(data=atlas_data, affine=np.eye(4))
    translated_target_affine = np.eye(4)
    translated_target_affine[0, 3] = 1.0
    translated_target = _make_nifti(
        data=np.zeros((4, 2, 2), dtype=np.float32),
        affine=translated_target_affine,
    )

    output = transfer_atlas_labels_to_image_space(
        atlas_image=atlas_image,
        target_image=translated_target,
    )

    expected = np.zeros((4, 2, 2), dtype=np.uint8)
    expected[0, :, :] = 1
    np.testing.assert_array_equal(output, expected)


def test_transfer_label_filtering_keeps_only_requested_labels() -> None:
    """Keep only requested labels and zero all other non-zero labels.

    The base atlas contains label 1 in a central cube and label 2 in a single
    corner voxel. Passing ``labels=[2]`` should keep only that single label-2
    voxel in the output while removing label-1 voxels entirely.
    """
    atlas_data = _make_base_atlas_data()
    atlas_image = _make_nifti(data=atlas_data, affine=np.eye(4))
    target_image = _make_nifti(
        data=np.zeros_like(atlas_data, dtype=np.float32),
        affine=np.eye(4),
    )

    output = transfer_atlas_labels_to_image_space(
        atlas_image=atlas_image,
        target_image=target_image,
        labels=[2],
    )

    expected = np.zeros_like(atlas_data)
    expected[4, 0, 0] = 2
    np.testing.assert_array_equal(output, expected)


def test_transfer_uses_first_volume_for_4d_atlas() -> None:
    """Use the first volume when a 4-D atlas image is supplied.

    The implementation projects a 4-D atlas to spatial data via ``[..., 0]``.
    This test sets volume 0 to meaningful labels and volume 1 to a constant
    value so the assertion can verify that output is derived strictly from the
    first volume.
    """
    first_volume = _make_base_atlas_data()
    second_volume = np.full_like(first_volume, fill_value=9)
    fourth_dim_data = np.stack([first_volume, second_volume], axis=3)
    atlas_image = _make_nifti(data=fourth_dim_data, affine=np.eye(4))
    target_image = _make_nifti(
        data=np.zeros_like(first_volume, dtype=np.float32),
        affine=np.eye(4),
    )

    output = transfer_atlas_labels_to_image_space(
        atlas_image=atlas_image,
        target_image=target_image,
    )

    np.testing.assert_array_equal(output, first_volume)


def test_build_atlas_in_target_space_returns_3d_for_3d_target() -> None:
    """Return a 3-D mask unchanged when the target has no volume axis.

    For a 3-D target, ``build_atlas_in_target_space`` should directly return
    the transferred spatial mask rather than stacking along a fourth axis.
    """
    atlas_data = _make_base_atlas_data()
    atlas_image = _make_nifti(data=atlas_data, affine=np.eye(4))
    target_3d = _make_nifti(
        data=np.zeros_like(atlas_data, dtype=np.float32),
        affine=np.eye(4),
    )

    output = build_atlas_in_target_space(
        atlas_image=atlas_image,
        target_image=target_3d,
    )

    assert output.shape == atlas_data.shape
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output, atlas_data)


def test_build_atlas_in_target_space_returns_3d_for_4d_target() -> None:
    """Return a 3-D spatial mask even when the target image is 4-D.

    With a 4-D target shape (5, 5, 5, 3), the function should ignore the
    non-spatial axis and return shape (5, 5, 5). Using ``labels=[1]`` also
    verifies that label filtering is applied to the returned spatial mask.
    """
    atlas_data = _make_base_atlas_data()
    atlas_image = _make_nifti(data=atlas_data, affine=np.eye(4))
    target_4d = _make_nifti(
        data=np.zeros((5, 5, 5, 3), dtype=np.float32),
        affine=np.eye(4),
    )

    output = build_atlas_in_target_space(
        atlas_image=atlas_image,
        target_image=target_4d,
        labels=[1],
    )

    expected_spatial = np.where(atlas_data == 1, 1, 0).astype(np.uint8)
    assert output.shape == (5, 5, 5)
    assert output.dtype == np.uint8
    np.testing.assert_array_equal(output, expected_spatial)

"""Atlas label transfer between NIfTI image spaces via affine mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import nibabel

_ATLAS_SPATIAL_NDIM: int = 3


def transfer_atlas_labels_to_image_space(
    *,
    atlas_image: nibabel.nifti1.Nifti1Image,
    target_image: nibabel.nifti1.Nifti1Image,
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    """Map atlas label voxels into the spatial voxel space of a target image.

    Uses inverse-warp (pull) resampling with nearest-neighbour interpolation:
    every target voxel is mapped back into the atlas via the composite affine
    ``inv(atlas_affine) @ target_affine`` and the nearest atlas label is read.
    Pulling rather than pushing guarantees every target voxel is assigned a
    value, so no striped/grid holes appear when the target grid is finer than
    the atlas. Nearest-neighbour preserves discrete label values.

    Works for any combination of 3-D or 4-D atlas and target; only the first
    three spatial axes of each image are used.

    Args:
        atlas_image: Registered atlas NIfTI image containing integer labels.
        target_image: Target NIfTI image whose spatial grid defines the output.
        labels: Integer label values to keep. When ``None``, all non-zero
            atlas labels are kept.

    Returns:
        ``uint8`` array of shape ``target_image.shape[:3]`` with atlas label
        values at the corresponding target voxel positions; zero elsewhere.
    """
    atlas_data = np.asarray(atlas_image.get_fdata(), dtype=np.float64)
    spatial_atlas_data = (
        atlas_data[..., 0] if atlas_data.ndim > _ATLAS_SPATIAL_NDIM else atlas_data
    )

    output_shape = tuple(int(s) for s in target_image.shape[:3])
    output = np.zeros(output_shape, dtype=np.uint8)

    composite_affine = np.linalg.inv(atlas_image.affine) @ target_image.affine
    linear = composite_affine[:3, :3]
    translation = composite_affine[:3, 3]

    target_indices = np.indices(output_shape, dtype=np.float64)  # (3, Nx, Ny, Nz)
    atlas_continuous = (
        np.tensordot(linear, target_indices, axes=([1], [0]))
        + translation[:, None, None, None]
    )  # (3, Nx, Ny, Nz)
    atlas_voxels = np.rint(atlas_continuous).astype(np.int64)

    atlas_shape = spatial_atlas_data.shape
    in_bounds = (
        (atlas_voxels[0] >= 0)
        & (atlas_voxels[0] < atlas_shape[0])
        & (atlas_voxels[1] >= 0)
        & (atlas_voxels[1] < atlas_shape[1])
        & (atlas_voxels[2] >= 0)
        & (atlas_voxels[2] < atlas_shape[2])
    )

    clipped_x = np.clip(atlas_voxels[0], 0, atlas_shape[0] - 1)
    clipped_y = np.clip(atlas_voxels[1], 0, atlas_shape[1] - 1)
    clipped_z = np.clip(atlas_voxels[2], 0, atlas_shape[2] - 1)
    sampled = spatial_atlas_data[clipped_x, clipped_y, clipped_z]

    keep = (
        in_bounds & (sampled != 0)
        if labels is None
        else in_bounds & np.isin(sampled, labels)
    )
    output[keep] = sampled[keep].astype(np.uint8)
    return output


def build_atlas_in_target_space(
    *,
    atlas_image: nibabel.nifti1.Nifti1Image,
    target_image: nibabel.nifti1.Nifti1Image,
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    """Build a labelled atlas array in the spatial grid of a target image.

    Transfers atlas labels into the spatial grid of ``target_image`` and
    always returns a 3-D spatial mask with shape ``target_image.shape[:3]``.
    For 4-D targets (e.g. echo or time series), the non-spatial axis is
    intentionally ignored by this function.

    Args:
        atlas_image: Registered atlas NIfTI image containing integer labels.
        target_image: Target NIfTI image whose shape defines the output.
        labels: Integer label values to transfer. When ``None``, all non-zero
            atlas labels are transferred.

    Returns:
        ``uint8`` array matching ``target_image.shape[:3]``.
    """
    return transfer_atlas_labels_to_image_space(
        atlas_image=atlas_image,
        target_image=target_image,
        labels=labels,
    )

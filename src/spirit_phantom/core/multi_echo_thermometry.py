"""Multi-echo thermometry helpers for ethylene glycol vial segmentation.

The EG mask refinement uses Sum of Absolute Differences (SAD) filtering on each
voxel time series:

    SAD = sum(abs(S[n] - S[n - 1])) for n in 1..N
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import nibabel
import numpy as np

from spirit_phantom.utils.binary_morphology import binary_dilation_connectivity_one

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

ETHYLENE_GLYCOL_LABELS: tuple[int, int] = (21, 22)
_MIN_THERMOMETRY_NDIM_FOR_TIME_AXIS: int = 4
_EG_MASK_DILATION_ITERATIONS: int = 1
_EG_MASK_MIN_SAD_COUNTS: float = 1000.0
_DIAGNOSTIC_PLOT_DPI: int = 160


def _get_mask_output_stem(*, output_mask_image_path: Path) -> str:
    """Return the output mask filename stem with ``.nii.gz`` handled."""
    if output_mask_image_path.name.endswith(".nii.gz"):
        return output_mask_image_path.name[: -len(".nii.gz")]
    return output_mask_image_path.stem


def dilate_segmentation_mask(
    mask: np.ndarray,
    *,
    iterations: int,
) -> np.ndarray:
    """Apply binary dilation to a label or mask array.

    Args:
        mask: Non-zero entries treated as foreground (2D or 3D).
        iterations: Number of dilation steps (voxel layers added per step).
            A value of ``0`` leaves the original binary mask unchanged.

    Returns:
        ``uint8`` array of the same shape as ``mask`` with values 0 or 1.
    """
    if iterations < 0:
        msg = "iterations must be greater than or equal to 0."
        raise ValueError(msg)

    foreground = np.asarray(mask) != 0
    if iterations == 0:
        return np.asarray(foreground, dtype=np.uint8)

    dilated = binary_dilation_connectivity_one(
        mask=foreground,
        iterations=iterations,
    )
    return np.asarray(dilated, dtype=np.uint8)


def _get_mask_output_plot_path(*, output_mask_image_path: Path) -> Path:
    """Build the diagnostic plot path alongside the saved mask image."""
    mask_stem = _get_mask_output_stem(output_mask_image_path=output_mask_image_path)
    return output_mask_image_path.parent / f"{mask_stem}_sad_filter_plot.png"


def _get_output_mapped_label_mask_path(*, output_mask_image_path: Path) -> Path:
    """Build the mapped-label mask path alongside the saved final mask image."""
    mask_stem = _get_mask_output_stem(output_mask_image_path=output_mask_image_path)
    return output_mask_image_path.parent / f"{mask_stem}_mapped_label_mask.nii.gz"


def _save_labelled_mask_image(
    *,
    mask: np.ndarray,
    reference_image: nibabel.nifti1.Nifti1Image,
    output_image_path: Path,
    description: str,
) -> None:
    """Save a labelled mask NIfTI with copied affine metadata.

    Args:
        mask: Labelled mask to save.
        reference_image: Image whose affine and header metadata are reused.
        output_image_path: Output NIfTI path.
        description: Short NIfTI header description.
    """
    mask_header = reference_image.header.copy()
    mask_header.set_data_dtype(np.uint8)
    mask_header["descrip"] = np.array(description, dtype="|S80")
    mask_image = nibabel.Nifti1Image(
        dataobj=np.asarray(mask, dtype=np.uint8),
        affine=reference_image.affine,
        header=mask_header,
    )
    mask_image.set_qform(reference_image.affine, code=1)
    mask_image.set_sform(reference_image.affine, code=1)
    nibabel.save(mask_image, str(output_image_path))


def _dilate_labelled_mask(*, labelled_mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate each EG label independently while preserving label identities."""
    dilated_labelled_mask = np.asarray(labelled_mask, dtype=np.uint8).copy()
    if iterations == 0:
        return dilated_labelled_mask

    for label in ETHYLENE_GLYCOL_LABELS:
        label_mask = np.asarray(labelled_mask == label, dtype=np.uint8)
        dilated_label_mask = dilate_segmentation_mask(
            mask=label_mask,
            iterations=iterations,
        )
        addable_voxels = np.logical_and(
            dilated_label_mask != 0,
            dilated_labelled_mask == 0,
        )
        dilated_labelled_mask[addable_voxels] = np.uint8(label)
    return dilated_labelled_mask


def _save_sad_filter_plot(
    *,
    included_traces: Sequence[np.ndarray],
    excluded_traces: Sequence[np.ndarray],
    output_plot_path: Path,
    minimum_sad_counts: float,
) -> None:
    """Save a Sum of Absolute Differences (SAD) diagnostic trace plot."""
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(10, 6))

    if included_traces:
        for trace in included_traces:
            axis.plot(
                trace,
                color="tab:green",
                alpha=0.20,
                linewidth=1.0,
            )
    if excluded_traces:
        for trace in excluded_traces:
            axis.plot(
                trace,
                color="tab:red",
                alpha=0.08,
                linewidth=0.8,
            )

    axis.set_xlabel("Time / echo index")
    axis.set_ylabel("Signal intensity")
    axis.set_title(
        "EG mask SAD filter diagnostic "
        f"(included={len(included_traces)}, excluded={len(excluded_traces)})"
    )
    axis.grid(visible=True, alpha=0.3)
    axis.text(
        x=0.01,
        y=0.98,
        s=f"Threshold: SAD >= {minimum_sad_counts:.1f} counts",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=_DIAGNOSTIC_PLOT_DPI)
    plt.close()


def _filter_mask_by_sad_threshold(
    *,
    thermometry_image: nibabel.nifti1.Nifti1Image,
    labelled_mask: np.ndarray,
    minimum_sad_counts: float,
    output_plot_path: Path | None,
) -> tuple[np.ndarray, int, int]:
    """Keep only mask voxels with sufficient Sum of Absolute Differences (SAD).

    Args:
        thermometry_image: Multi-echo thermometry image.
        labelled_mask: Labelled mask in thermometry voxel space.
        minimum_sad_counts: Minimum Sum of Absolute Differences value required
            to retain a voxel.
        output_plot_path: Path where the diagnostic plot should be written. If
            ``None``, no diagnostic plot is generated.

    Returns:
        Tuple of ``(filtered_mask, included_voxel_count, excluded_voxel_count)``.
    """
    thermometry_data = np.asarray(thermometry_image.get_fdata(), dtype=np.float64)
    if thermometry_data.ndim < _MIN_THERMOMETRY_NDIM_FOR_TIME_AXIS:
        msg = (
            f"Expected thermometry image to have at least "
            f"{_MIN_THERMOMETRY_NDIM_FOR_TIME_AXIS} dimensions; "
            f"got shape {thermometry_data.shape}."
        )
        raise ValueError(msg)

    candidate_voxels = np.argwhere(np.asarray(labelled_mask) != 0)
    filtered_mask = np.zeros(shape=labelled_mask.shape, dtype=np.uint8)
    included_traces: list[np.ndarray] = []
    excluded_traces: list[np.ndarray] = []

    for candidate_voxel in candidate_voxels:
        x_index, y_index, z_index = (
            int(candidate_voxel[0]),
            int(candidate_voxel[1]),
            int(candidate_voxel[2]),
        )
        voxel_trace = np.asarray(
            thermometry_data[x_index, y_index, z_index, :],
            dtype=np.float64,
        ).ravel()
        sad_counts = float(np.sum(np.abs(np.diff(voxel_trace))))
        if sad_counts >= minimum_sad_counts:
            filtered_mask[x_index, y_index, z_index] = np.uint8(
                labelled_mask[x_index, y_index, z_index]
            )
            included_traces.append(voxel_trace)
        else:
            excluded_traces.append(voxel_trace)

    if output_plot_path is not None:
        _save_sad_filter_plot(
            included_traces=included_traces,
            excluded_traces=excluded_traces,
            output_plot_path=output_plot_path,
            minimum_sad_counts=minimum_sad_counts,
        )
    return filtered_mask, len(included_traces), len(excluded_traces)


def world_coordinate_to_eg_thermometry_indices(
    *,
    thermometry_image: nibabel.nifti1.Nifti1Image,
    world_coordinates: Sequence[tuple[float, float, float]],
) -> Sequence[tuple[int, int, int]]:
    """Convert world-space coordinates into thermometry voxel indices.

    Args:
        thermometry_image: Thermometry NIfTI image with affine metadata.
        world_coordinates: World-space ``(x, y, z)`` coordinates.

    Returns:
        Sequence of voxel index tuples ``(x_index, y_index, z_index)``.
    """
    inverse_affine = np.linalg.inv(thermometry_image.affine)
    voxel_indices: list[tuple[int, int, int]] = []
    for world_coordinate in world_coordinates:
        voxel_coordinate = nibabel.affines.apply_affine(
            inverse_affine,
            np.array(world_coordinate, dtype=float),
        )
        x_index = int(np.rint(voxel_coordinate[0]))
        y_index = int(np.rint(voxel_coordinate[1]))
        z_index = int(np.rint(voxel_coordinate[2]))

        x_index = int(np.clip(x_index, a_min=0, a_max=thermometry_image.shape[0] - 1))
        y_index = int(np.clip(y_index, a_min=0, a_max=thermometry_image.shape[1] - 1))
        z_index = int(np.clip(z_index, a_min=0, a_max=thermometry_image.shape[2] - 1))
        voxel_indices.append((x_index, y_index, z_index))
    return voxel_indices


def voxel_xy_to_world_coordinate_at_z(
    *,
    image: nibabel.nifti1.Nifti1Image,
    x_index: int,
    y_index: int,
    world_z: float,
) -> tuple[float, float, float]:
    """Convert voxel XY indices to a world-space coordinate at ``world_z``.

    Args:
        image: NIfTI image containing affine orientation metadata.
        x_index: X voxel index within the image grid.
        y_index: Y voxel index within the image grid.
        world_z: Z value in world space.

    Returns:
        The ``(x, y, z)`` coordinate in world space.
    """
    if x_index < 0 or x_index >= image.shape[0]:
        raise IndexError
    if y_index < 0 or y_index >= image.shape[1]:
        raise IndexError

    inverse_affine = np.linalg.inv(image.affine)
    reference_voxel = nibabel.affines.apply_affine(
        inverse_affine,
        np.array([0.0, 0.0, world_z], dtype=float),
    )
    z_index = int(np.rint(reference_voxel[2]))
    z_index = int(np.clip(z_index, a_min=0, a_max=image.shape[2] - 1))

    world_coordinate = nibabel.affines.apply_affine(
        image.affine,
        np.array([x_index, y_index, z_index], dtype=float),
    )
    return (
        float(world_coordinate[0]),
        float(world_coordinate[1]),
        float(world_coordinate[2]),
    )


def get_atlas_world_coords(
    *,
    image: nibabel.nifti1.Nifti1Image,
    world_z: float,
    label: int,
) -> Sequence[tuple[float, float, float]]:
    """Map labelled voxels on a world-space Z plane to world coordinates.

    Args:
        image: Registered atlas image providing affine and label information.
        world_z: Z value in world space.
        label: Integer label of voxels to extract.

    Returns:
        Sequence of ``(x, y, z)`` world coordinates for matching label voxels.
    """
    inverse_affine = np.linalg.inv(image.affine)
    reference_voxel = nibabel.affines.apply_affine(
        inverse_affine,
        np.array([0.0, 0.0, world_z], dtype=float),
    )
    z_index = int(np.rint(reference_voxel[2]))
    z_index = int(np.clip(z_index, a_min=0, a_max=image.shape[2] - 1))

    atlas_data = image.get_fdata()
    atlas_slice = atlas_data[:, :, z_index]
    matching_indices = np.argwhere(atlas_slice == float(label))

    world_coordinates: list[tuple[float, float, float]] = []
    for x_idx, y_idx in matching_indices:
        world_coordinates.append(
            voxel_xy_to_world_coordinate_at_z(
                image=image,
                x_index=int(x_idx),
                y_index=int(y_idx),
                world_z=world_z,
            )
        )
    return world_coordinates


def _get_world_z_for_voxel_slice(
    *, image: nibabel.nifti1.Nifti1Image, z_index: int
) -> float:
    """Return the world-space Z value for a voxel-space slice index."""
    world_coordinate = nibabel.affines.apply_affine(
        image.affine, np.array([0.0, 0.0, float(z_index)], dtype=float)
    )
    return float(world_coordinate[2])


def _get_nearest_atlas_world_z(
    *,
    registered_atlas_image: nibabel.nifti1.Nifti1Image,
    target_world_z: float,
) -> float:
    """Find the atlas slice world-Z nearest to the thermometry target Z."""
    atlas_slice_count = int(registered_atlas_image.shape[2])
    atlas_world_zs = np.array(
        [
            _get_world_z_for_voxel_slice(
                image=registered_atlas_image,
                z_index=slice_index,
            )
            for slice_index in range(atlas_slice_count)
        ],
        dtype=float,
    )
    nearest_slice_index = int(np.argmin(np.abs(atlas_world_zs - target_world_z)))
    return float(atlas_world_zs[nearest_slice_index])


def coordinate_ethylene_glycol_vial_segmentation(
    *,
    registered_component_atlas_image_path: Path,
    multi_echo_gradient_echo_scan_image_path: Path,
    output_mask_image_path: Path | None = None,
    minimum_sad_counts: float = _EG_MASK_MIN_SAD_COUNTS,
    dilation_iterations: int = _EG_MASK_DILATION_ITERATIONS,
    generate_sad_visualisation: bool = False,
) -> Path:
    """Create a mapped ethylene glycol segmentation mask from atlas and multi-echo data.

    Args:
        registered_component_atlas_image_path: Path to the registered component atlas NIfTI.
        multi_echo_gradient_echo_scan_image_path: Path to the multi-echo GRE NIfTI.
        output_mask_image_path: Optional explicit output mask path.
        minimum_sad_counts: SAD inclusion threshold in counts.
        dilation_iterations: Number of binary-dilation iterations.
        generate_sad_visualisation: Whether to save the SAD diagnostic plot.

    Returns:
        Path to the saved segmentation mask NIfTI file.
    """
    if minimum_sad_counts < 0.0:
        msg = "minimum_sad_counts must be greater than or equal to 0."
        raise ValueError(msg)
    if dilation_iterations < 0:
        msg = "dilation_iterations must be greater than or equal to 0."
        raise ValueError(msg)

    registered_atlas_image = nibabel.nifti1.load(
        filename=str(registered_component_atlas_image_path)
    )
    eg_thermometry_image = nibabel.nifti1.load(
        filename=str(multi_echo_gradient_echo_scan_image_path)
    )

    target_world_z = _get_world_z_for_voxel_slice(image=eg_thermometry_image, z_index=0)
    nearest_atlas_world_z = _get_nearest_atlas_world_z(
        registered_atlas_image=registered_atlas_image,
        target_world_z=target_world_z,
    )

    if len(eg_thermometry_image.shape) < _MIN_THERMOMETRY_NDIM_FOR_TIME_AXIS:
        msg = (
            "Expected the multi-echo thermometry image to include a fourth "
            f"time dimension; got shape {eg_thermometry_image.shape}."
        )
        raise ValueError(msg)

    thermometry_spatial_shape = tuple(
        int(size) for size in eg_thermometry_image.shape[:3]
    )
    mapped_labelled_mask = np.zeros(shape=thermometry_spatial_shape, dtype=np.uint8)

    for label in ETHYLENE_GLYCOL_LABELS:
        mapped_atlas_world_coords = get_atlas_world_coords(
            image=registered_atlas_image,
            world_z=nearest_atlas_world_z,
            label=label,
        )
        mapped_thermometry_indices = world_coordinate_to_eg_thermometry_indices(
            thermometry_image=eg_thermometry_image,
            world_coordinates=mapped_atlas_world_coords,
        )
        for therm_x, therm_y, therm_z in mapped_thermometry_indices:
            mapped_labelled_mask[therm_x, therm_y, therm_z] = np.uint8(label)

    if output_mask_image_path is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_mask_image_path = (
            multi_echo_gradient_echo_scan_image_path.parent
            / f"ethylene_glycol_mask_{timestamp}.nii.gz"
        )
    output_mask_image_path.parent.mkdir(parents=True, exist_ok=True)
    mapped_label_mask_output_path = _get_output_mapped_label_mask_path(
        output_mask_image_path=output_mask_image_path
    )
    _save_labelled_mask_image(
        mask=mapped_labelled_mask,
        reference_image=eg_thermometry_image,
        output_image_path=mapped_label_mask_output_path,
        description="Mapped EG labels from component atlas before dilation/SAD",
    )

    dilated_labelled_mask = _dilate_labelled_mask(
        labelled_mask=mapped_labelled_mask,
        iterations=dilation_iterations,
    )

    output_plot_path = (
        _get_mask_output_plot_path(output_mask_image_path=output_mask_image_path)
        if generate_sad_visualisation
        else None
    )

    filtered_mask, _, _ = _filter_mask_by_sad_threshold(
        thermometry_image=eg_thermometry_image,
        labelled_mask=dilated_labelled_mask,
        minimum_sad_counts=minimum_sad_counts,
        output_plot_path=output_plot_path,
    )
    _save_labelled_mask_image(
        mask=filtered_mask,
        reference_image=eg_thermometry_image,
        output_image_path=output_mask_image_path,
        description="Mapped EG labels, dilated and Sum of Absolute Differences-filtered",
    )
    return output_mask_image_path

"""Generate wedge ROI masks and corner points for slice-thickness analysis.

World-space (mm) rectangular ROIs on the Z = 0 mm atlas plane define the two
SPIRIT slice-thickness wedges. Labels are 1 and 2.

Mask generation walks one voxel-Z plane at a time so peak memory stays on the
order of a single slice rather than a full-volume world-coordinate grid
(which for the default 0.25 mm atlas is tens of gigabytes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import nibabel
import numpy as np
import numpy.typing as npt

from spirit_phantom.io.points import save_points

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# World-space (mm) axis-aligned bounds for each wedge on the Z = 0 plane.
# Y extents (high→low listing): 5.766, 0.01592, -0.2341, -5.984.
WEDGE_ROIS: dict[int, tuple[tuple[float, float], tuple[float, float], float]] = {
    1: ((-25.0, 25.0), (0.01592, 5.766), 0.0),
    2: ((-25.0, 25.0), (-5.984, -0.2341), 0.0),
}


class WedgeCorner(NamedTuple):
    """One rectangle corner of a wedge ROI."""

    label: int
    name: str
    point_mm: tuple[float, float, float]


def _ordered_range(*, bounds: tuple[float, float]) -> tuple[float, float]:
    """Return ``bounds`` ordered as ``(low, high)``.

    Args:
        bounds: Inclusive endpoint pair in either order.

    Returns:
        Ordered ``(low, high)`` pair.
    """
    low, high = bounds
    return (low, high) if low <= high else (high, low)


def _half_voxel_extent_along_world_z(*, affine: npt.NDArray[np.float64]) -> float:
    """Estimate half a voxel's extent projected onto world Z.

    Args:
        affine: 4x4 voxel-to-world affine.

    Returns:
        Half-voxel tolerance in millimetres along world Z.
    """
    axis_vectors = affine[:3, :3]
    half_spacings = 0.5 * np.linalg.norm(axis_vectors, axis=0)
    world_z_components = np.abs(axis_vectors[2, :])
    contributing = world_z_components > 0.0
    if not np.any(contributing):
        return 1e-6
    return float(np.max(half_spacings[contributing]))


def _plane_world_z_range(
    *,
    nx: int,
    ny: int,
    z_index: int,
    affine: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Return min/max world Z on the four corners of a fixed voxel-Z plane."""
    corners = (
        (0.0, 0.0, float(z_index)),
        (float(nx - 1), 0.0, float(z_index)),
        (0.0, float(ny - 1), float(z_index)),
        (float(nx - 1), float(ny - 1), float(z_index)),
    )
    world_z_values = [
        float(affine[2, 0] * i + affine[2, 1] * j + affine[2, 2] * k + affine[2, 3])
        for i, j, k in corners
    ]
    return min(world_z_values), max(world_z_values)


def _world_coordinates_plane(
    *,
    nx: int,
    ny: int,
    z_index: int,
    affine: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Map one voxel-Z plane to world coordinates.

    Args:
        nx: Number of voxels along axis 0.
        ny: Number of voxels along axis 1.
        z_index: Voxel index along axis 2.
        affine: 4x4 voxel-to-world affine.

    Returns:
        Array of shape ``(3, nx, ny)`` with world X, Y, Z in millimetres.
    """
    indices = np.indices(dimensions=(nx, ny), dtype=np.float64)
    linear = affine[:3, :3]
    translation = affine[:3, 3]
    # tensordot over in-plane axes; add the fixed-Z column contribution.
    world = (
        np.tensordot(linear[:, :2], indices, axes=([1], [0]))
        + linear[:, 2, None, None] * float(z_index)
        + translation[:, None, None]
    )
    return cast("npt.NDArray[np.float64]", world)


def _mask_plane_for_wedge_roi(
    *,
    world_xyz: npt.NDArray[np.float64],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_plane_mm: float,
    z_tolerance: float,
) -> npt.NDArray[np.bool_]:
    """Build a boolean mask for one wedge ROI on a single Z plane."""
    x_low, x_high = _ordered_range(bounds=x_bounds)
    y_low, y_high = _ordered_range(bounds=y_bounds)
    return cast(
        "npt.NDArray[np.bool_]",
        (world_xyz[0] >= x_low)
        & (world_xyz[0] <= x_high)
        & (world_xyz[1] >= y_low)
        & (world_xyz[1] <= y_high)
        & (np.abs(world_xyz[2] - z_plane_mm) <= z_tolerance),
    )


def _nifti_from_mask(
    *,
    mask: npt.NDArray[np.uint8],
    reference_image: nibabel.nifti1.Nifti1Image,
    description: str,
) -> nibabel.nifti1.Nifti1Image:
    """Wrap a mask array as a NIfTI with the reference header and affine.

    Args:
        mask: Labelled mask array.
        reference_image: Image whose affine and header metadata are reused.
        description: Short NIfTI header description.

    Returns:
        NIfTI image sharing the reference geometry.
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
    return mask_image


def wedge_roi_corners() -> list[WedgeCorner]:
    """Return the four rectangle corners of each wedge ROI in atlas/world mm.

    Corner order per wedge is:
    ``(x_min, y_min)``, ``(x_max, y_min)``, ``(x_max, y_max)``, ``(x_min, y_max)``
    at the wedge Z plane. The long edge runs from corner 0 to corner 1.

    Returns:
        Corner descriptors for both wedges (eight points total).
    """
    corners: list[WedgeCorner] = []
    for label, (x_bounds, y_bounds, z_plane_mm) in WEDGE_ROIS.items():
        x_low, x_high = _ordered_range(bounds=x_bounds)
        y_low, y_high = _ordered_range(bounds=y_bounds)
        xy_corners = (
            (x_low, y_low, "x_min_y_min"),
            (x_high, y_low, "x_max_y_min"),
            (x_high, y_high, "x_max_y_max"),
            (x_low, y_high, "x_min_y_max"),
        )
        for x_mm, y_mm, corner_name in xy_corners:
            corners.append(
                WedgeCorner(
                    label=label,
                    name=f"wedge{label}_{corner_name}",
                    point_mm=(float(x_mm), float(y_mm), float(z_plane_mm)),
                )
            )
    return corners


def generate_slice_mask(
    *,
    image: nibabel.nifti1.Nifti1Image,
) -> nibabel.nifti1.Nifti1Image:
    """Generate a labelled wedge ROI mask in the voxel grid of ``image``.

    Processes one voxel-Z plane at a time so peak RAM stays proportional to a
    single slice (safe for the default ~0.4e9-voxel SPIRIT atlas on 12 GB hosts).

    Args:
        image: Reference NIfTI whose spatial grid and header define the mask.

    Returns:
        Single NIfTI mask with labels 1 and 2, matching ``image`` geometry.
    """
    nx = int(image.shape[0])
    ny = int(image.shape[1])
    nz = int(image.shape[2])
    spatial_shape = (nx, ny, nz)
    affine = np.asarray(image.affine, dtype=np.float64)
    z_tolerance = _half_voxel_extent_along_world_z(affine=affine)

    labelled_mask = np.zeros(spatial_shape, dtype=np.uint8)
    for z_index in range(nz):
        z_min, z_max = _plane_world_z_range(
            nx=nx, ny=ny, z_index=z_index, affine=affine
        )
        # Skip planes that cannot intersect any wedge Z slab.
        plane_needed = False
        for _label, (_x_bounds, _y_bounds, z_plane_mm) in WEDGE_ROIS.items():
            if z_min - z_tolerance <= z_plane_mm <= z_max + z_tolerance:
                plane_needed = True
                break
        if not plane_needed:
            continue

        world_xyz = _world_coordinates_plane(
            nx=nx, ny=ny, z_index=z_index, affine=affine
        )
        for label, (x_bounds, y_bounds, z_plane_mm) in WEDGE_ROIS.items():
            inside = _mask_plane_for_wedge_roi(
                world_xyz=world_xyz,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                z_plane_mm=z_plane_mm,
                z_tolerance=z_tolerance,
            )
            labelled_mask[:, :, z_index][inside] = np.uint8(label)

    return _nifti_from_mask(
        mask=labelled_mask,
        reference_image=image,
        description="Slice-thickness wedge masks (labels 1, 2)",
    )


def _resolve_nifti_output_path(*, output_path: Path) -> Path:
    """Ensure ``output_path`` has a ``.nii`` / ``.nii.gz`` suffix."""
    resolved = output_path.resolve()
    if not (resolved.name.endswith(".nii.gz") or resolved.suffix == ".nii"):
        resolved = resolved.with_name(f"{resolved.name}.nii.gz")
    return resolved


def save_slice_mask(
    *,
    image: nibabel.nifti1.Nifti1Image,
    output_path: Path,
) -> Path:
    """Generate the labelled wedge mask and write it to ``output_path``.

    Args:
        image: Reference NIfTI whose spatial grid and header define the mask.
        output_path: Destination NIfTI path (``.nii.gz`` is appended when missing).

    Returns:
        Absolute path to the saved mask file.

    Raises:
        RuntimeError: If the file is not present after writing.
    """
    mask_image = generate_slice_mask(image=image)
    resolved = _resolve_nifti_output_path(output_path=output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    nibabel.save(mask_image, str(resolved))
    if not resolved.is_file():
        message = f"Failed to write wedge mask to {resolved}"
        raise RuntimeError(message)
    return resolved


def generate_world_points_mask(
    *,
    image: nibabel.nifti1.Nifti1Image,
    points_mm: Sequence[Sequence[float]],
    mark_radius_voxels: int = 2,
) -> nibabel.nifti1.Nifti1Image:
    """Paint world-space points onto ``image``'s grid with 1-based order labels.

    Voxel value ``k`` marks the ``k``-th point in ``points_mm`` (processing order).
    Each point is drawn as an axis-aligned cross of radius ``mark_radius_voxels``
    so the markers remain visible when overlaid on high-resolution atlases.

    Args:
        image: Reference NIfTI whose spatial grid and header define the mask.
        points_mm: World-coordinate points in millimetres, in the desired order.
        mark_radius_voxels: Half-extent of the cross painted around each point.

    Returns:
        NIfTI mask matching ``image`` geometry; background is 0.

    Raises:
        ValueError: If ``mark_radius_voxels`` is negative or a point is not 3-D.
    """
    if mark_radius_voxels < 0:
        msg = "mark_radius_voxels must be non-negative"
        raise ValueError(msg)

    nx = int(image.shape[0])
    ny = int(image.shape[1])
    nz = int(image.shape[2])
    labelled = np.zeros((nx, ny, nz), dtype=np.uint8)
    affine = np.asarray(image.affine, dtype=np.float64)
    inverse_affine = np.linalg.inv(affine)

    for order_index, point in enumerate(points_mm, start=1):
        point_array = np.asarray(point, dtype=np.float64)
        if point_array.shape != (3,):
            msg = f"Expected a 3-D point, got shape {point_array.shape}"
            raise ValueError(msg)
        voxel = inverse_affine @ np.array(
            [point_array[0], point_array[1], point_array[2], 1.0],
            dtype=np.float64,
        )
        centre = np.rint(voxel[:3]).astype(np.int64)
        i0, j0, k0 = (int(centre[0]), int(centre[1]), int(centre[2]))
        label = np.uint8(order_index)
        for delta in range(-mark_radius_voxels, mark_radius_voxels + 1):
            i = i0 + delta
            if 0 <= i < nx and 0 <= j0 < ny and 0 <= k0 < nz:
                labelled[i, j0, k0] = label
            j = j0 + delta
            if 0 <= i0 < nx and 0 <= j < ny and 0 <= k0 < nz:
                labelled[i0, j, k0] = label
            k = k0 + delta
            if 0 <= i0 < nx and 0 <= j0 < ny and 0 <= k < nz:
                labelled[i0, j0, k] = label

    return _nifti_from_mask(
        mask=labelled,
        reference_image=image,
        description="Wedge corner points (1-based processing order)",
    )


def save_world_points_mask(
    *,
    image: nibabel.nifti1.Nifti1Image,
    points_mm: Sequence[Sequence[float]],
    output_path: Path,
    mark_radius_voxels: int = 2,
) -> Path:
    """Write a NIfTI mask of world points labelled by processing order.

    Args:
        image: Reference NIfTI whose spatial grid and header define the mask.
        points_mm: World-coordinate points in millimetres, in the desired order.
        output_path: Destination NIfTI path (``.nii.gz`` is appended when missing).
        mark_radius_voxels: Half-extent of the cross painted around each point.

    Returns:
        Absolute path to the saved mask file.

    Raises:
        RuntimeError: If the file is not present after writing.
    """
    mask_image = generate_world_points_mask(
        image=image,
        points_mm=points_mm,
        mark_radius_voxels=mark_radius_voxels,
    )
    resolved = _resolve_nifti_output_path(output_path=output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    nibabel.save(mask_image, str(resolved))
    if not resolved.is_file():
        message = f"Failed to write corner-points mask to {resolved}"
        raise RuntimeError(message)
    return resolved


def save_wedge_corner_points_mask(
    *,
    image: nibabel.nifti1.Nifti1Image,
    output_path: Path,
    mark_radius_voxels: int = 2,
) -> Path:
    """Save atlas wedge corners as a labelled NIfTI in ``wedge_roi_corners`` order.

    Labels are ``1..N`` matching the flat processing order used for transformix
    inversion and profile sampling (wedge 1 corners, then wedge 2 corners;
    within each wedge: ``x_min_y_min``, ``x_max_y_min``, ``x_max_y_max``,
    ``x_min_y_max``).

    Args:
        image: Reference NIfTI (typically the atlas) defining the voxel grid.
        output_path: Destination NIfTI path.
        mark_radius_voxels: Half-extent of the cross painted around each corner.

    Returns:
        Absolute path to the saved mask file.
    """
    corners = wedge_roi_corners()
    return save_world_points_mask(
        image=image,
        points_mm=[corner.point_mm for corner in corners],
        output_path=output_path,
        mark_radius_voxels=mark_radius_voxels,
    )


def save_wedge_roi_corner_points(*, output_path: Path) -> Path:
    """Write wedge rectangle corners as a transformix ``point`` file.

    Args:
        output_path: Destination transformix points path.

    Returns:
        Absolute path to the written points file.
    """
    corners = wedge_roi_corners()
    points = [list(corner.point_mm) for corner in corners]
    resolved = output_path.resolve()
    save_points(points=points, output_path=resolved, point_type="point")
    return resolved

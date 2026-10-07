"""Generate wedge ROI masks and corner points for slice-thickness analysis.

World-space (mm) rectangular ROIs on the Z = 0 mm atlas plane define the two
SPIRIT slice-thickness wedges. Labels are 1 and 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import nibabel
import numpy as np
import numpy.typing as npt

from spirit_phantom.io.points import save_points

if TYPE_CHECKING:
    from pathlib import Path

# World-space (mm) axis-aligned bounds for each wedge on the Z = 0 plane.
WEDGE_ROIS: dict[int, tuple[tuple[float, float], tuple[float, float], float]] = {
    1: ((-25.0, 25.0), (0.0, 5.0), 0.0),
    2: ((-25.0, 25.0), (-6.0, -1.0), 0.0),
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


def _world_coordinates(
    *,
    shape: tuple[int, int, int],
    affine: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Map voxel centres to world coordinates via the image affine.

    Args:
        shape: Spatial shape ``(nx, ny, nz)``.
        affine: 4x4 voxel-to-world affine.

    Returns:
        Array of shape ``(3, nx, ny, nz)`` with world X, Y, Z in millimetres.
    """
    indices = np.indices(dimensions=shape, dtype=np.float64)
    linear = affine[:3, :3]
    translation = affine[:3, 3]
    world = (
        np.tensordot(linear, indices, axes=([1], [0]))
        + translation[:, None, None, None]
    )
    return cast("npt.NDArray[np.float64]", world)


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


def _mask_for_wedge_roi(
    *,
    world_xyz: npt.NDArray[np.float64],
    affine: npt.NDArray[np.float64],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_plane_mm: float,
) -> npt.NDArray[np.bool_]:
    """Build a boolean mask for one world-space wedge ROI.

    Args:
        world_xyz: World coordinates of shape ``(3, nx, ny, nz)``.
        affine: Image affine used to derive the Z-plane tolerance.
        x_bounds: Inclusive X range in millimetres.
        y_bounds: Inclusive Y range in millimetres.
        z_plane_mm: World Z plane in millimetres.

    Returns:
        Boolean array that is True inside the ROI.
    """
    x_low, x_high = _ordered_range(bounds=x_bounds)
    y_low, y_high = _ordered_range(bounds=y_bounds)
    z_tolerance = _half_voxel_extent_along_world_z(affine=affine)
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
    world_xyz = _world_coordinates(shape=spatial_shape, affine=affine)

    labelled_mask = np.zeros(spatial_shape, dtype=np.uint8)
    for label, (x_bounds, y_bounds, z_plane_mm) in WEDGE_ROIS.items():
        inside = _mask_for_wedge_roi(
            world_xyz=world_xyz,
            affine=affine,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_plane_mm=z_plane_mm,
        )
        labelled_mask[inside] = np.uint8(label)

    return _nifti_from_mask(
        mask=labelled_mask,
        reference_image=image,
        description="Slice-thickness wedge masks (labels 1, 2)",
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

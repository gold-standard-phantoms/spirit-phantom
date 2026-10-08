"""Slice-thickness analysis via registered wedge ROIs.

Maps atlas wedge corners into fixed (scan) space by inverting the elastix
transform with transformix, samples edge-response profiles from the fixed
image rectangles, and computes NEMA MS 5-2018 slice thickness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import nibabel
import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates

from spirit_phantom.core.generate_slice_mask import (
    WedgeCorner,
    save_slice_mask,
    save_wedge_corner_points_mask,
    save_world_points_mask,
    wedge_roi_corners,
)
from spirit_phantom.core.slice_thickness import nema_slice_thickness
from spirit_phantom.io.points import (
    invert_points_through_transformix,
    save_points,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_CORNERS_PER_WEDGE = 4
_DEFAULT_PROFILE_LINES = 5
_MIN_SAMPLES_PER_LINE = 32
_SPATIAL_NDIM = 3


@dataclass(frozen=True, slots=True)
class WedgeThicknessResult:
    """Slice-thickness result for one wedge.

    Attributes:
        label: Wedge label (1 or 2).
        thickness_mm: Estimated slice thickness in millimetres.
        pixel_size_mm: Sample spacing along the edge-response profile.
        n_lines: Number of averaged profile lines.
        n_samples: Samples per profile line.
        corners_fixed_mm: Four rectangle corners in fixed-space millimetres.
    """

    label: int
    thickness_mm: float
    pixel_size_mm: float
    n_lines: int
    n_samples: int
    corners_fixed_mm: tuple[tuple[float, float, float], ...]


def _as_point3(*, point: Sequence[float]) -> npt.NDArray[np.float64]:
    """Convert a length-3 sequence to a float64 vector.

    Args:
        point: Three-element coordinate sequence.

    Returns:
        Shape ``(3,)`` array.
    """
    array = np.asarray(point, dtype=np.float64)
    if array.shape != (3,):
        msg = f"Expected a 3-D point, got shape {array.shape}"
        raise ValueError(msg)
    return array


def sample_rectangle_edge_response(
    *,
    image: nibabel.nifti1.Nifti1Image,
    corners_mm: Sequence[Sequence[float]],
    n_lines: int = _DEFAULT_PROFILE_LINES,
    samples_per_line: int | None = None,
) -> tuple[npt.NDArray[np.float64], float]:
    """Sample edge-response profiles along the long axis of a rectangle.

    Corner order must be ``(p00, p10, p11, p01)`` where the long edge runs
    ``p00 → p10`` and the short edge runs ``p00 → p01``.

    Args:
        image: Fixed-space NIfTI to sample.
        corners_mm: Four rectangle corners in world millimetres.
        n_lines: Number of parallel lines across the short edge to average.
        samples_per_line: Optional explicit sample count along the long edge.

    Returns:
        Tuple of ``(profiles, pixel_size_mm)`` where ``profiles`` has shape
        ``(n_samples, n_lines)`` suitable for :func:`nema_slice_thickness`.
    """
    if len(corners_mm) != _CORNERS_PER_WEDGE:
        msg = f"Expected {_CORNERS_PER_WEDGE} corners, got {len(corners_mm)}"
        raise ValueError(msg)
    if n_lines < 1:
        msg = "n_lines must be at least 1"
        raise ValueError(msg)

    p00 = _as_point3(point=corners_mm[0])
    p10 = _as_point3(point=corners_mm[1])
    p01 = _as_point3(point=corners_mm[3])

    long_vec = p10 - p00
    short_vec = p01 - p00
    long_length = float(np.linalg.norm(long_vec))
    if long_length <= 0.0:
        msg = "Rectangle long edge has zero length"
        raise ValueError(msg)

    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    mean_spacing = float(np.mean(spacing))
    n_samples = (
        samples_per_line
        if samples_per_line is not None
        else max(_MIN_SAMPLES_PER_LINE, int(np.ceil(long_length / mean_spacing)) + 1)
    )
    pixel_size_mm = long_length / float(n_samples - 1)

    long_fractions = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    if n_lines == 1:
        short_fractions = np.asarray([0.5], dtype=np.float64)
    else:
        short_fractions = np.linspace(0.0, 1.0, n_lines, dtype=np.float64)

    world_points = np.empty((3, n_samples, n_lines), dtype=np.float64)
    for line_index, short_fraction in enumerate(short_fractions):
        origin = p00 + short_fraction * short_vec
        for sample_index, long_fraction in enumerate(long_fractions):
            world_points[:, sample_index, line_index] = (
                origin + long_fraction * long_vec
            )

    affine = np.asarray(image.affine, dtype=np.float64)
    inverse_affine = np.linalg.inv(affine)
    homogeneous = np.ones((4, n_samples * n_lines), dtype=np.float64)
    homogeneous[:3, :] = world_points.reshape(3, -1)
    voxel_coords = inverse_affine @ homogeneous
    # nibabel / map_coordinates use (i, j, k) matching array axis order.
    sample_coords = voxel_coords[:3, :].reshape(3, n_samples, n_lines)

    volume = np.asarray(image.dataobj, dtype=np.float64)
    if volume.ndim > _SPATIAL_NDIM:
        volume = volume[..., 0]

    profiles = map_coordinates(
        volume,
        sample_coords,
        order=1,
        mode="nearest",
    )
    return np.asarray(profiles, dtype=np.float64), pixel_size_mm


@dataclass(frozen=True, slots=True)
class MappedWedgeCorners:
    """Atlas wedge corners after inverse mapping into fixed space.

    Attributes:
        atlas_corners: Corner descriptors in processing order.
        fixed_points_mm: Parallel fixed-space coordinates (same order).
        by_label: Fixed-space corners grouped by wedge label.
    """

    atlas_corners: tuple[WedgeCorner, ...]
    fixed_points_mm: tuple[tuple[float, float, float], ...]
    by_label: dict[int, list[list[float]]]


def map_wedge_corners_to_fixed_space(
    *,
    transform_parameter_path: Path,
    moving_image_path: Path,
    output_directory: Path,
) -> MappedWedgeCorners:
    """Map atlas wedge corners into fixed space by inverting transformix ``T``.

    Args:
        transform_parameter_path: Final elastix transform (typically
            ``BSpline_Transform.txt``), chaining earlier stages.
        moving_image_path: Moving image used by transformix for geometry.
        output_directory: Directory for transformix point I/O.

    Returns:
        Mapped corners preserving ``wedge_roi_corners()`` processing order.
    """
    corners = wedge_roi_corners()
    moving_points = [list(corner.point_mm) for corner in corners]
    output_directory.mkdir(parents=True, exist_ok=True)
    save_points(
        points=moving_points,
        output_path=output_directory / "wedge_corners_atlas_mm.txt",
        point_type="point",
    )
    fixed_points = invert_points_through_transformix(
        moving_points=moving_points,
        transform_parameter_path=transform_parameter_path,
        moving_image_path=moving_image_path,
        output_directory=output_directory / "moving_to_fixed",
    )

    by_label: dict[int, list[list[float]]] = {}
    ordered_fixed: list[tuple[float, float, float]] = []
    for corner, fixed_point in zip(corners, fixed_points, strict=True):
        point = (
            float(fixed_point[0]),
            float(fixed_point[1]),
            float(fixed_point[2]),
        )
        ordered_fixed.append(point)
        by_label.setdefault(corner.label, []).append(list(point))
    return MappedWedgeCorners(
        atlas_corners=tuple(corners),
        fixed_points_mm=tuple(ordered_fixed),
        by_label=by_label,
    )


def measure_slice_thickness_from_fixed_wedges(
    *,
    fixed_image_path: Path,
    transform_parameter_path: Path,
    moving_image_path: Path,
    output_directory: Path,
    atlas_image_path: Path | None = None,
    ramp_slope_degrees: float = 15.0,
    n_lines: int = _DEFAULT_PROFILE_LINES,
) -> list[WedgeThicknessResult]:
    """Measure slice thickness for both wedges in a registered fixed image.

    Writes QC artefacts under ``output_directory``, including:

    - ``slice_wedge_mask_atlas.nii.gz`` — labelled wedge ROIs in atlas space
    - ``slice_wedge_corners_atlas.nii.gz`` — corner markers labelled ``1..N`` in
      ``wedge_roi_corners()`` / transformix processing order
    - ``slice_wedge_corners_fixed.nii.gz`` — the same markers after inverse
      mapping into fixed/scan space

    Args:
        fixed_image_path: Scanner (fixed) NIfTI path.
        transform_parameter_path: Final elastix transform parameter file.
        moving_image_path: Moving image path for transformix geometry.
        output_directory: Directory for intermediate point files and QC masks.
        atlas_image_path: Optional atlas/moving NIfTI used to save the
            world-space wedge ROI mask for visual QC. When omitted,
            ``moving_image_path`` is used.
        ramp_slope_degrees: Wedge angle alpha in degrees.
        n_lines: Number of profile lines to average per wedge.

    Returns:
        Per-wedge thickness results.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    fixed_image = nibabel.nifti1.load(filename=str(fixed_image_path))
    mask_source_path = (
        atlas_image_path if atlas_image_path is not None else moving_image_path
    )
    atlas_image = nibabel.nifti1.load(filename=str(mask_source_path))
    save_slice_mask(
        image=atlas_image,
        output_path=output_directory / "slice_wedge_mask_atlas.nii.gz",
    )
    save_wedge_corner_points_mask(
        image=atlas_image,
        output_path=output_directory / "slice_wedge_corners_atlas.nii.gz",
    )

    mapped = map_wedge_corners_to_fixed_space(
        transform_parameter_path=transform_parameter_path,
        moving_image_path=moving_image_path,
        output_directory=output_directory,
    )
    save_world_points_mask(
        image=fixed_image,
        points_mm=mapped.fixed_points_mm,
        output_path=output_directory / "slice_wedge_corners_fixed.nii.gz",
    )

    paired_path = output_directory / "wedge_corners_fixed_mm.txt"
    with paired_path.open("w", encoding="utf-8") as handle:
        handle.write("order label corner_index x_mm y_mm z_mm name\n")
        corner_index_by_label: dict[int, int] = {}
        for order_index, (corner, point) in enumerate(
            zip(mapped.atlas_corners, mapped.fixed_points_mm, strict=True),
            start=1,
        ):
            corner_index = corner_index_by_label.get(corner.label, 0)
            corner_index_by_label[corner.label] = corner_index + 1
            handle.write(
                f"{order_index} {corner.label} {corner_index} "
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {corner.name}\n"
            )

    results: list[WedgeThicknessResult] = []
    for label, corners in sorted(mapped.by_label.items()):
        profiles, pixel_size_mm = sample_rectangle_edge_response(
            image=fixed_image,
            corners_mm=corners,
            n_lines=n_lines,
        )
        thickness_mm = nema_slice_thickness(
            profiles,
            pixel_size_mm,
            ramp_slope_degrees,
        )
        results.append(
            WedgeThicknessResult(
                label=label,
                thickness_mm=float(thickness_mm),
                pixel_size_mm=float(pixel_size_mm),
                n_lines=int(profiles.shape[1]),
                n_samples=int(profiles.shape[0]),
                corners_fixed_mm=tuple(
                    (float(point[0]), float(point[1]), float(point[2]))
                    for point in corners
                ),
            )
        )
    return results

"""Slice-thickness analysis via registered wedge ROIs.

Maps atlas wedge corners into fixed (scan) space by inverting the elastix
transform with transformix, samples edge-response profiles from the fixed
image rectangles, and computes NEMA MS 5-2018 slice thickness.
"""

from __future__ import annotations

import csv
import math
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
from spirit_phantom.core.slice_thickness import (
    calculate_slice_profile,
    nema_slice_thickness,
)
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
_DIAGNOSTIC_PLOT_DPI = 150
# Wedge 1's intensity ramp rises opposite to the default long-axis sample
# direction, so the differentiated slice profile is a trough. Reverse the ERF
# along the sample axis for these labels so FWHM sees a peak. Wedge 2 is left
# unchanged.
_REVERSE_ERF_SAMPLE_DIRECTION_LABELS = frozenset({1})


def discard_wedge_edge_lines(
    *,
    profiles: npt.NDArray[np.float64],
    discard_edge_lines: int,
) -> npt.NDArray[np.float64]:
    """Drop short-axis edge lines that often suffer partial-volume contamination.

    Profile columns are ordered from short-edge fraction ``0`` to ``1``. Dropping
    ``discard_edge_lines`` from each end keeps the interior lines for NEMA
    averaging.

    Args:
        profiles: Edge-response array of shape ``(n_samples, n_lines)``.
        discard_edge_lines: Number of lines to remove from each short-axis end.
            ``0`` leaves ``profiles`` unchanged.

    Returns:
        Trimmed profiles with shape ``(n_samples, n_lines - 2 * discard_edge_lines)``.

    Raises:
        ValueError: If ``discard_edge_lines`` is negative or would leave fewer
            than one line.
    """
    if discard_edge_lines < 0:
        msg = "discard_edge_lines must be >= 0"
        raise ValueError(msg)
    if discard_edge_lines == 0:
        return np.asarray(profiles, dtype=np.float64)

    array = np.asarray(profiles, dtype=np.float64)
    if array.ndim == 1:
        msg = "Cannot discard edge lines from a single profile line"
        raise ValueError(msg)
    n_lines = int(array.shape[1])
    kept = n_lines - 2 * discard_edge_lines
    if kept < 1:
        msg = (
            f"discard_edge_lines={discard_edge_lines} removes all lines "
            f"(n_lines={n_lines}); need n_lines > 2 * discard_edge_lines"
        )
        raise ValueError(msg)
    return np.asarray(
        array[:, discard_edge_lines : n_lines - discard_edge_lines],
        dtype=np.float64,
    )


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


def _half_max_crossings(
    *,
    slice_profile: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Return ``(left_cross, right_cross, half_max)`` in sample-index units.

    Uses the same first-rise / last-fall search as
    :func:`spirit_phantom.core.slice_thickness.full_width_half_maximum`, but
    keeps the linearly interpolated crossings (no integer ceil) for plotting.
    """
    max_value = float(np.max(slice_profile))
    half_max_value = max_value / 2.0
    left_cross: float | None = None
    for index in range(len(slice_profile) - 1):
        y1 = float(slice_profile[index])
        y2 = float(slice_profile[index + 1])
        if y1 < half_max_value <= y2:
            left_cross = (
                float(index + 1)
                if y2 == y1
                else float(index) + (half_max_value - y1) / (y2 - y1)
            )
            break

    right_cross: float | None = None
    for index in range(len(slice_profile) - 1, 0, -1):
        y1 = float(slice_profile[index - 1])
        y2 = float(slice_profile[index])
        if y2 < half_max_value <= y1:
            right_cross = (
                float(index - 1)
                if y2 == y1
                else float(index - 1) + (half_max_value - y1) / (y2 - y1)
            )
            break

    if left_cross is None or right_cross is None:
        msg = "Unable to determine FWHM: cannot find both half-max crossings."
        raise ValueError(msg)
    return left_cross, right_cross, half_max_value


@dataclass(frozen=True, slots=True)
class WedgeProfileDiagnostics:
    """Tabulated ERF / slice-profile curves for one wedge.

    Attributes:
        label: Wedge label.
        pixel_size_mm: Sample spacing along the long edge.
        ramp_slope_degrees: Wedge angle used for thickness scaling.
        thickness_mm: NEMA slice thickness.
        erf_profiles: Shape ``(n_samples, n_lines)`` edge-response samples.
        mean_erf: Mean ERF across lines (display aid).
        slice_profiles: Per-line differentiated profiles.
        mean_slice_profile: Mean slice profile (NEMA averaging step).
        left_cross_index: Interpolated left half-max crossing (samples).
        right_cross_index: Interpolated right half-max crossing (samples).
        half_max: Half of the mean slice-profile peak.
    """

    label: int
    pixel_size_mm: float
    ramp_slope_degrees: float
    thickness_mm: float
    erf_profiles: npt.NDArray[np.float64]
    mean_erf: npt.NDArray[np.float64]
    slice_profiles: npt.NDArray[np.float64]
    mean_slice_profile: npt.NDArray[np.float64]
    left_cross_index: float
    right_cross_index: float
    half_max: float


def build_wedge_profile_diagnostics(
    *,
    label: int,
    erf_profiles: npt.NDArray[np.float64],
    pixel_size_mm: float,
    ramp_slope_degrees: float,
    thickness_mm: float,
) -> WedgeProfileDiagnostics:
    """Build ERF / slice-profile arrays used for CSV and plot exports."""
    if erf_profiles.ndim == 1:
        erf_2d = np.asarray(erf_profiles, dtype=np.float64)[:, None]
    else:
        erf_2d = np.asarray(erf_profiles, dtype=np.float64)

    slice_profile_set = [
        calculate_slice_profile(erf_2d[:, line_index], pixel_size_mm)
        for line_index in range(erf_2d.shape[1])
    ]
    slice_profiles = np.column_stack(slice_profile_set)
    mean_slice_profile = np.mean(slice_profiles, axis=1)
    left_cross, right_cross, half_max = _half_max_crossings(
        slice_profile=mean_slice_profile
    )
    return WedgeProfileDiagnostics(
        label=label,
        pixel_size_mm=float(pixel_size_mm),
        ramp_slope_degrees=float(ramp_slope_degrees),
        thickness_mm=float(thickness_mm),
        erf_profiles=erf_2d,
        mean_erf=np.mean(erf_2d, axis=1),
        slice_profiles=np.asarray(slice_profiles, dtype=np.float64),
        mean_slice_profile=np.asarray(mean_slice_profile, dtype=np.float64),
        left_cross_index=float(left_cross),
        right_cross_index=float(right_cross),
        half_max=float(half_max),
    )


def save_wedge_profile_csv(
    *,
    diagnostics: WedgeProfileDiagnostics,
    output_path: Path,
) -> Path:
    """Write a wide CSV of ERF and slice-profile samples for one wedge."""
    resolved = output_path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(diagnostics.erf_profiles.shape[0])
    n_lines = int(diagnostics.erf_profiles.shape[1])
    n_profile = int(diagnostics.mean_slice_profile.shape[0])
    fieldnames = [
        "wedge_label",
        "sample_index",
        "position_mm",
        "erf_mean",
        *[f"erf_line_{line_index}" for line_index in range(n_lines)],
        "slice_profile_mean",
        *[f"slice_profile_line_{line_index}" for line_index in range(n_lines)],
        "half_max",
        "left_cross_index",
        "right_cross_index",
        "left_cross_mm",
        "right_cross_mm",
        "fwhm_mm",
        "thickness_mm",
        "pixel_size_mm",
        "ramp_slope_degrees",
    ]
    fwhm_mm = (
        diagnostics.right_cross_index - diagnostics.left_cross_index
    ) * diagnostics.pixel_size_mm
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index in range(n_samples):
            row: dict[str, float | int] = {
                "wedge_label": diagnostics.label,
                "sample_index": sample_index,
                "position_mm": sample_index * diagnostics.pixel_size_mm,
                "erf_mean": float(diagnostics.mean_erf[sample_index]),
                "half_max": diagnostics.half_max,
                "left_cross_index": diagnostics.left_cross_index,
                "right_cross_index": diagnostics.right_cross_index,
                "left_cross_mm": (
                    diagnostics.left_cross_index * diagnostics.pixel_size_mm
                ),
                "right_cross_mm": (
                    diagnostics.right_cross_index * diagnostics.pixel_size_mm
                ),
                "fwhm_mm": fwhm_mm,
                "thickness_mm": diagnostics.thickness_mm,
                "pixel_size_mm": diagnostics.pixel_size_mm,
                "ramp_slope_degrees": diagnostics.ramp_slope_degrees,
            }
            for line_index in range(n_lines):
                row[f"erf_line_{line_index}"] = float(
                    diagnostics.erf_profiles[sample_index, line_index]
                )
            # Slice profile is one sample shorter (numerical differentiation).
            if sample_index < n_profile:
                row["slice_profile_mean"] = float(
                    diagnostics.mean_slice_profile[sample_index]
                )
                for line_index in range(n_lines):
                    row[f"slice_profile_line_{line_index}"] = float(
                        diagnostics.slice_profiles[sample_index, line_index]
                    )
            else:
                row["slice_profile_mean"] = math.nan
                for line_index in range(n_lines):
                    row[f"slice_profile_line_{line_index}"] = math.nan
            writer.writerow(row)
    return resolved


def save_wedge_profile_plot(
    *,
    diagnostics: WedgeProfileDiagnostics,
    output_path: Path,
) -> Path:
    """Save a two-panel PNG: edge-response functions and mean slice profile."""
    # Lazy import: keep CLI startup free of matplotlib until diagnostics run.
    import matplotlib.pyplot as plt  # noqa: PLC0415

    resolved = output_path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    n_lines = int(diagnostics.erf_profiles.shape[1])
    erf_x = (
        np.arange(diagnostics.erf_profiles.shape[0], dtype=np.float64)
        * diagnostics.pixel_size_mm
    )
    profile_x = (
        np.arange(diagnostics.mean_slice_profile.shape[0], dtype=np.float64)
        * diagnostics.pixel_size_mm
    )
    left_mm = diagnostics.left_cross_index * diagnostics.pixel_size_mm
    right_mm = diagnostics.right_cross_index * diagnostics.pixel_size_mm
    fwhm_mm = right_mm - left_mm

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(9.0, 7.0), sharex=False)
    erf_axis, profile_axis = axes

    for line_index in range(n_lines):
        erf_axis.plot(
            erf_x,
            diagnostics.erf_profiles[:, line_index],
            color="tab:gray",
            alpha=0.35,
            linewidth=1.0,
            label="line ERF" if line_index == 0 else None,
        )
    erf_axis.plot(
        erf_x,
        diagnostics.mean_erf,
        color="tab:blue",
        linewidth=2.0,
        label="mean ERF",
    )
    erf_axis.set_ylabel("Intensity")
    erf_axis.set_title(f"Wedge {diagnostics.label}: edge response functions")
    erf_axis.grid(visible=True, alpha=0.3)
    erf_axis.legend(loc="best")

    for line_index in range(n_lines):
        profile_axis.plot(
            profile_x,
            diagnostics.slice_profiles[:, line_index],
            color="tab:gray",
            alpha=0.35,
            linewidth=1.0,
            label="line profile" if line_index == 0 else None,
        )
    profile_axis.plot(
        profile_x,
        diagnostics.mean_slice_profile,
        color="tab:orange",
        linewidth=2.0,
        label="mean slice profile",
    )
    profile_axis.axhline(
        diagnostics.half_max,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"half-max ({diagnostics.half_max:.3g})",
    )
    profile_axis.axvline(left_mm, color="tab:green", linestyle=":", linewidth=1.2)
    profile_axis.axvline(
        right_mm,
        color="tab:green",
        linestyle=":",
        linewidth=1.2,
        label=f"FWHM={fwhm_mm:.3f} mm",
    )
    profile_axis.set_xlabel("Position along wedge long axis / mm")
    profile_axis.set_ylabel("dI / dx")
    profile_axis.set_title(
        f"Wedge {diagnostics.label}: slice profile -> thickness "
        f"{diagnostics.thickness_mm:.4f} mm "
        f"(alpha={diagnostics.ramp_slope_degrees:g} deg)"
    )
    profile_axis.grid(visible=True, alpha=0.3)
    profile_axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(resolved, dpi=_DIAGNOSTIC_PLOT_DPI)
    plt.close(figure)
    return resolved


def save_wedge_profile_diagnostics(
    *,
    diagnostics: WedgeProfileDiagnostics,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write CSV + PNG diagnostics for one wedge under ``output_directory``."""
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = save_wedge_profile_csv(
        diagnostics=diagnostics,
        output_path=output_directory / f"wedge_{diagnostics.label}_profiles.csv",
    )
    plot_path = save_wedge_profile_plot(
        diagnostics=diagnostics,
        output_path=output_directory / f"wedge_{diagnostics.label}_profiles.png",
    )
    return csv_path, plot_path


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
    discard_edge_lines: int = 1,
) -> list[WedgeThicknessResult]:
    """Measure slice thickness for both wedges in a registered fixed image.

    Writes QC artefacts under ``output_directory``, including:

    - ``slice_wedge_mask_atlas.nii.gz`` — labelled wedge ROIs in atlas space
    - ``slice_wedge_corners_atlas.nii.gz`` — corner markers labelled ``1..N`` in
      ``wedge_roi_corners()`` / transformix processing order
    - ``slice_wedge_corners_fixed.nii.gz`` — the same markers after inverse
      mapping into fixed/scan space
    - ``wedge_{label}_profiles.csv`` / ``.png`` — ERF and slice-profile curves
      used for the NEMA thickness estimate
    - ``slice_thickness_summary.csv`` — one row per wedge with thickness and
      FWHM diagnostics

    Args:
        fixed_image_path: Scanner (fixed) NIfTI path.
        transform_parameter_path: Final elastix transform parameter file.
        moving_image_path: Moving image path for transformix geometry.
        output_directory: Directory for intermediate point files and QC masks.
        atlas_image_path: Optional atlas/moving NIfTI used to save the
            world-space wedge ROI mask for visual QC. When omitted,
            ``moving_image_path`` is used.
        ramp_slope_degrees: Wedge angle alpha in degrees.
        n_lines: Number of profile lines sampled across the short edge before
            optional edge discarding.
        discard_edge_lines: Number of short-axis edge lines to drop from each
            end before NEMA averaging (reduces partial-volume contamination).
            ``0`` keeps every sampled line.

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
    summary_rows: list[dict[str, float | int]] = []
    for label, corners in sorted(mapped.by_label.items()):
        profiles, pixel_size_mm = sample_rectangle_edge_response(
            image=fixed_image,
            corners_mm=corners,
            n_lines=n_lines,
        )
        profiles = discard_wedge_edge_lines(
            profiles=profiles,
            discard_edge_lines=discard_edge_lines,
        )
        if label in _REVERSE_ERF_SAMPLE_DIRECTION_LABELS:
            # Sample from the opposite end of the long axis so dI/dx is a peak.
            profiles = np.asarray(profiles[::-1, :], dtype=np.float64)
        thickness_mm = nema_slice_thickness(
            profiles,
            pixel_size_mm,
            ramp_slope_degrees,
        )
        diagnostics = build_wedge_profile_diagnostics(
            label=label,
            erf_profiles=profiles,
            pixel_size_mm=pixel_size_mm,
            ramp_slope_degrees=ramp_slope_degrees,
            thickness_mm=float(thickness_mm),
        )
        save_wedge_profile_diagnostics(
            diagnostics=diagnostics,
            output_directory=output_directory,
        )
        fwhm_mm = (
            diagnostics.right_cross_index - diagnostics.left_cross_index
        ) * diagnostics.pixel_size_mm
        summary_rows.append(
            {
                "wedge_label": label,
                "thickness_mm": float(thickness_mm),
                "fwhm_mm": fwhm_mm,
                "left_cross_mm": (
                    diagnostics.left_cross_index * diagnostics.pixel_size_mm
                ),
                "right_cross_mm": (
                    diagnostics.right_cross_index * diagnostics.pixel_size_mm
                ),
                "half_max": diagnostics.half_max,
                "pixel_size_mm": float(pixel_size_mm),
                "n_lines_sampled": int(n_lines),
                "n_lines_used": int(profiles.shape[1]),
                "discard_edge_lines": int(discard_edge_lines),
                "n_samples": int(profiles.shape[0]),
                "ramp_slope_degrees": float(ramp_slope_degrees),
            }
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

    summary_path = output_directory / "slice_thickness_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "wedge_label",
            "thickness_mm",
            "fwhm_mm",
            "left_cross_mm",
            "right_cross_mm",
            "half_max",
            "pixel_size_mm",
            "n_lines_sampled",
            "n_lines_used",
            "discard_edge_lines",
            "n_samples",
            "ramp_slope_degrees",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return results

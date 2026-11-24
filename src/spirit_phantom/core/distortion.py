"""Scanner geometric distortion from elastix affine + B-spline stages.

Assumes an accurate rigid→affine→B-spline registration of the SPIRIT atlas
(moving) to a scanner image (fixed). Rigid pose (Euler rotation and
translation only) is excluded; voxel size / matrix mismatches are handled by
image spacing metadata in physical space, not by the rigid stage. The residual
displacement induced by the affine and B-spline stages at the packaged
fiducial landmarks is reported as an estimate of scanner-introduced
geometric distortion (including global scale and shear).

See the project docs page *Scanner distortion* for method notes and open
doubts.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import nibabel

from spirit_phantom.core.fiducials import Fiducial, load_spirit_fiducials
from spirit_phantom.core.generate_slice_mask import save_world_points_mask
from spirit_phantom.core.registration import (
    ensure_inverse_transform,
    save_transform_parameter_file,
)
from spirit_phantom.core.registration_constants import (
    AFFINE_TRANSFORM_FILENAME,
    BSPLINE_IMAGE_FILENAME,
    BSPLINE_TRANSFORM_FILENAME,
    RIGID_TRANSFORM_FILENAME,
)
from spirit_phantom.io.points import (
    InversionMethod,
    invert_points_through_transformix,
    save_points,
    transform_points_with_transformix,
)
from spirit_phantom.utils.progress import (
    capture_logs_to_file,
    print_cli_stage,
    print_cli_stage_done,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

DISTORTION_LOG_FILENAME = "distortion.log"
DISTORTION_SUMMARY_CSV_FILENAME = "distortion_summary.csv"
DISTORTION_PER_POINT_CSV_FILENAME = "distortion_per_point.csv"
AFFINE_NO_RIGID_TRANSFORM_FILENAME = "Affine_NoRigid_Transform.txt"
BSPLINE_AFFINE_ONLY_TRANSFORM_FILENAME = "BSpline_AffineOnly_Transform.txt"
FIDUCIALS_ATLAS_MM_FILENAME = "fiducials_atlas_mm.txt"
FIDUCIALS_FIXED_MM_FILENAME = "fiducials_fixed_mm.txt"
FIDUCIALS_RIGID_ALIGNED_MM_FILENAME = "fiducials_rigid_aligned_mm.txt"
FIDUCIALS_NONRIGID_ROUNDTRIP_MM_FILENAME = "fiducials_nonrigid_roundtrip_mm.txt"
FIDUCIALS_FIXED_MASK_FILENAME = "fiducials_fixed_mm.nii.gz"

_ANALYSIS_STAGE_COUNT = 4
_SPATIAL_NDIM = 3
_NONRIGID_ROUNDTRIP_WARNING_MM = 0.1


@dataclass(frozen=True, slots=True)
class FiducialDistortion:
    """Per-fiducial scanner-distortion residual after excluding rigid pose.

    Attributes:
        fiducial_id: Packaged fiducial identifier (``F01`` to ``F34``).
        part: Physical grid assembly name.
        layer: Axial layer within its grid.
        atlas_mm: Landmark in atlas / moving world millimetres.
        fixed_mm: Landmark mapped into scanner (fixed) space via the inverse
            of the full registration transform.
        rigid_aligned_mm: Fixed-space landmark after the rigid stage only
            (pose-aligned frame that is the input domain of affine∘B-spline).
        nonrigid_roundtrip_mm: Result of applying the affine+B-spline-only
            chain to ``rigid_aligned_mm``; should recover ``atlas_mm``.
        displacement_mm: ``atlas_mm - rigid_aligned_mm`` — the displacement
            induced by affine + B-spline (scanner geometric residual under
            the method assumptions).
        magnitude_mm: Euclidean length of ``displacement_mm``.
        radius_from_isocentre_mm: Distance of ``atlas_mm`` from the phantom
            centre ``(0, 0, 0)``.
    """

    fiducial_id: str
    part: str
    layer: str
    atlas_mm: tuple[float, float, float]
    fixed_mm: tuple[float, float, float]
    rigid_aligned_mm: tuple[float, float, float]
    nonrigid_roundtrip_mm: tuple[float, float, float]
    displacement_mm: tuple[float, float, float]
    magnitude_mm: float
    radius_from_isocentre_mm: float


@dataclass(frozen=True, slots=True)
class DistortionSummary:
    """Aggregate statistics over all evaluated fiducials.

    Attributes:
        n_fiducials: Number of landmarks evaluated.
        mean_magnitude_mm: Mean displacement magnitude.
        max_magnitude_mm: Largest displacement magnitude.
        rms_magnitude_mm: Root-mean-square displacement magnitude.
        max_nonrigid_roundtrip_residual_mm: Worst absolute coordinate residual
            of the affine+B-spline round-trip check
            (``nonrigid_roundtrip_mm`` vs ``atlas_mm``).
    """

    n_fiducials: int
    mean_magnitude_mm: float
    max_magnitude_mm: float
    rms_magnitude_mm: float
    max_nonrigid_roundtrip_residual_mm: float


@dataclass(frozen=True, slots=True)
class DistortionResult:
    """Full scanner-distortion evaluation for one registration.

    Attributes:
        points: Per-fiducial residuals.
        summary: Aggregate statistics.
        affine_bspline_only_transform_path: Written B-spline head of the
            affine+B-spline chain with rigid unlinked.
    """

    points: tuple[FiducialDistortion, ...]
    summary: DistortionSummary
    affine_bspline_only_transform_path: Path


def _as_xyz(
    *, point: list[float] | tuple[float, float, float]
) -> tuple[float, float, float]:
    """Coerce a length-3 sequence to a float triple.

    Args:
        point: Point coordinates.

    Returns:
        ``(x, y, z)`` float triple.

    Raises:
        ValueError: If ``point`` is not length 3.
    """
    if len(point) != _SPATIAL_NDIM:
        msg = f"Expected a 3-D point, got {len(point)} coordinates: {point!r}"
        raise ValueError(msg)
    return (float(point[0]), float(point[1]), float(point[2]))


def _vector_difference(
    *,
    minuend: tuple[float, float, float],
    subtrahend: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return ``minuend - subtrahend`` component-wise.

    Args:
        minuend: Left-hand point.
        subtrahend: Right-hand point.

    Returns:
        Difference vector.
    """
    return (
        minuend[0] - subtrahend[0],
        minuend[1] - subtrahend[1],
        minuend[2] - subtrahend[2],
    )


def _euclidean_norm(*, vector: tuple[float, float, float]) -> float:
    """Return the Euclidean length of a 3-D vector.

    Args:
        vector: Displacement components in millimetres.

    Returns:
        Magnitude in millimetres.
    """
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def _max_abs_residual_mm(
    *,
    mapped: list[list[float]],
    targets: list[tuple[float, float, float]],
) -> float:
    """Largest absolute per-axis residual between two parallel point sets.

    Args:
        mapped: Points produced by a transform.
        targets: Expected points.

    Returns:
        Maximum absolute coordinate difference in millimetres.
    """
    return max(
        abs(mapped_point[dim] - target[dim])
        for mapped_point, target in zip(mapped, targets, strict=True)
        for dim in range(_SPATIAL_NDIM)
    )


def build_affine_bspline_only_transform(
    *,
    affine_transform_path: Path,
    bspline_transform_path: Path,
    output_directory: Path,
) -> Path:
    """Write an affine→B-spline chain with the rigid stage unlinked.

    Copies the registration affine parameters with
    ``InitialTransformParameterFileName = NoInitialTransform``, then writes a
    B-spline file that chains that standalone affine. The resulting head maps
    the post-rigid intermediate space to moving/atlas space:

    ``T_ab = T_bspline ∘ T_affine``.

    Args:
        affine_transform_path: Registration ``Affine_Transform.txt`` (normally
            still chained to rigid on disk).
        bspline_transform_path: Registration ``BSpline_Transform.txt``.
        output_directory: Directory for the rewritten parameter files.

    Returns:
        Path to the written B-spline head
        (``BSpline_AffineOnly_Transform.txt``).

    Raises:
        FileNotFoundError: If either input transform file is missing.
    """
    # Delayed: ITK is heavy and unused by pure residual helpers / CSV writers.
    import itk  # noqa: PLC0415

    if not affine_transform_path.is_file():
        msg = f"Affine transform file not found: {affine_transform_path}"
        raise FileNotFoundError(msg)
    if not bspline_transform_path.is_file():
        msg = f"B-spline transform file not found: {bspline_transform_path}"
        raise FileNotFoundError(msg)

    output_directory.mkdir(parents=True, exist_ok=True)

    affine_params = itk.ParameterObject.New()
    affine_params.ReadParameterFile(str(affine_transform_path))
    standalone_affine_path = output_directory / AFFINE_NO_RIGID_TRANSFORM_FILENAME
    save_transform_parameter_file(
        transform_params=affine_params,
        filename=standalone_affine_path,
        initial_transform_file=None,
    )

    bspline_params = itk.ParameterObject.New()
    bspline_params.ReadParameterFile(str(bspline_transform_path))
    standalone_bspline_path = output_directory / BSPLINE_AFFINE_ONLY_TRANSFORM_FILENAME
    save_transform_parameter_file(
        transform_params=bspline_params,
        filename=standalone_bspline_path,
        initial_transform_file=standalone_affine_path,
    )
    return standalone_bspline_path


def compute_fiducial_distortions(
    *,
    fiducials: tuple[Fiducial, ...],
    atlas_mm: list[tuple[float, float, float]],
    fixed_mm: list[list[float]],
    rigid_aligned_mm: list[list[float]],
    nonrigid_roundtrip_mm: list[list[float]],
) -> tuple[FiducialDistortion, ...]:
    """Build per-fiducial distortion records from parallel point sets.

    Args:
        fiducials: Fiducial metadata in the same order as the point lists.
        atlas_mm: Atlas-space landmark coordinates.
        fixed_mm: Fixed-space coordinates from the full inverse.
        rigid_aligned_mm: Post-rigid intermediate coordinates.
        nonrigid_roundtrip_mm: Affine+B-spline applied to ``rigid_aligned_mm``.

    Returns:
        Immutable per-fiducial distortion records.

    Raises:
        ValueError: If the input lengths disagree.
    """
    n_points = len(fiducials)
    if not (
        len(atlas_mm)
        == len(fixed_mm)
        == len(rigid_aligned_mm)
        == len(nonrigid_roundtrip_mm)
        == n_points
    ):
        msg = (
            "Fiducial and point-set lengths must match: "
            f"fiducials={n_points}, atlas={len(atlas_mm)}, fixed={len(fixed_mm)}, "
            f"rigid={len(rigid_aligned_mm)}, roundtrip={len(nonrigid_roundtrip_mm)}"
        )
        raise ValueError(msg)

    records: list[FiducialDistortion] = []
    for fiducial, atlas, fixed, rigid, roundtrip in zip(
        fiducials,
        atlas_mm,
        fixed_mm,
        rigid_aligned_mm,
        nonrigid_roundtrip_mm,
        strict=True,
    ):
        atlas_xyz = _as_xyz(point=atlas)
        rigid_xyz = _as_xyz(point=rigid)
        displacement = _vector_difference(minuend=atlas_xyz, subtrahend=rigid_xyz)
        records.append(
            FiducialDistortion(
                fiducial_id=fiducial.id,
                part=fiducial.part,
                layer=fiducial.layer,
                atlas_mm=atlas_xyz,
                fixed_mm=_as_xyz(point=fixed),
                rigid_aligned_mm=rigid_xyz,
                nonrigid_roundtrip_mm=_as_xyz(point=roundtrip),
                displacement_mm=displacement,
                magnitude_mm=_euclidean_norm(vector=displacement),
                radius_from_isocentre_mm=_euclidean_norm(vector=atlas_xyz),
            )
        )
    return tuple(records)


def summarise_distortions(
    *,
    points: tuple[FiducialDistortion, ...],
) -> DistortionSummary:
    """Aggregate per-fiducial residuals into summary statistics.

    Args:
        points: Per-fiducial distortion records.

    Returns:
        Summary statistics.

    Raises:
        ValueError: If ``points`` is empty.
    """
    if not points:
        msg = "Cannot summarise an empty distortion point set."
        raise ValueError(msg)

    magnitudes = [point.magnitude_mm for point in points]
    mean_magnitude = sum(magnitudes) / len(magnitudes)
    max_magnitude = max(magnitudes)
    rms_magnitude = math.sqrt(
        sum(value * value for value in magnitudes) / len(magnitudes)
    )
    max_roundtrip = max(
        abs(coord)
        for point in points
        for coord in _vector_difference(
            minuend=point.nonrigid_roundtrip_mm,
            subtrahend=point.atlas_mm,
        )
    )
    return DistortionSummary(
        n_fiducials=len(points),
        mean_magnitude_mm=mean_magnitude,
        max_magnitude_mm=max_magnitude,
        rms_magnitude_mm=rms_magnitude,
        max_nonrigid_roundtrip_residual_mm=max_roundtrip,
    )


def write_distortion_per_point_csv(
    *,
    points: tuple[FiducialDistortion, ...],
    output_path: Path,
) -> Path:
    """Write one CSV row per fiducial with coordinates and residuals.

    Args:
        points: Per-fiducial distortion records.
        output_path: Destination CSV path.

    Returns:
        ``output_path``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fiducial_id",
        "part",
        "layer",
        "atlas_x_mm",
        "atlas_y_mm",
        "atlas_z_mm",
        "fixed_x_mm",
        "fixed_y_mm",
        "fixed_z_mm",
        "rigid_aligned_x_mm",
        "rigid_aligned_y_mm",
        "rigid_aligned_z_mm",
        "nonrigid_roundtrip_x_mm",
        "nonrigid_roundtrip_y_mm",
        "nonrigid_roundtrip_z_mm",
        "displacement_x_mm",
        "displacement_y_mm",
        "displacement_z_mm",
        "magnitude_mm",
        "radius_from_isocentre_mm",
    ]
    with output_path.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "fiducial_id": point.fiducial_id,
                    "part": point.part,
                    "layer": point.layer,
                    "atlas_x_mm": point.atlas_mm[0],
                    "atlas_y_mm": point.atlas_mm[1],
                    "atlas_z_mm": point.atlas_mm[2],
                    "fixed_x_mm": point.fixed_mm[0],
                    "fixed_y_mm": point.fixed_mm[1],
                    "fixed_z_mm": point.fixed_mm[2],
                    "rigid_aligned_x_mm": point.rigid_aligned_mm[0],
                    "rigid_aligned_y_mm": point.rigid_aligned_mm[1],
                    "rigid_aligned_z_mm": point.rigid_aligned_mm[2],
                    "nonrigid_roundtrip_x_mm": point.nonrigid_roundtrip_mm[0],
                    "nonrigid_roundtrip_y_mm": point.nonrigid_roundtrip_mm[1],
                    "nonrigid_roundtrip_z_mm": point.nonrigid_roundtrip_mm[2],
                    "displacement_x_mm": point.displacement_mm[0],
                    "displacement_y_mm": point.displacement_mm[1],
                    "displacement_z_mm": point.displacement_mm[2],
                    "magnitude_mm": point.magnitude_mm,
                    "radius_from_isocentre_mm": point.radius_from_isocentre_mm,
                }
            )
    return output_path


def write_distortion_summary_csv(
    *,
    summary: DistortionSummary,
    output_path: Path,
) -> Path:
    """Write a one-row CSV of aggregate distortion statistics.

    Args:
        summary: Aggregate statistics.
        output_path: Destination CSV path.

    Returns:
        ``output_path``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n_fiducials",
        "mean_magnitude_mm",
        "max_magnitude_mm",
        "rms_magnitude_mm",
        "max_nonrigid_roundtrip_residual_mm",
    ]
    with output_path.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "n_fiducials": summary.n_fiducials,
                "mean_magnitude_mm": summary.mean_magnitude_mm,
                "max_magnitude_mm": summary.max_magnitude_mm,
                "rms_magnitude_mm": summary.rms_magnitude_mm,
                "max_nonrigid_roundtrip_residual_mm": (
                    summary.max_nonrigid_roundtrip_residual_mm
                ),
            }
        )
    return output_path


def evaluate_scanner_distortion(  # noqa: PLR0915
    *,
    registration_directory: Path,
    fixed_image_path: Path,
    moving_image_path: Path | None = None,
    output_directory: Path,
    fiducials_config_path: Path | None = None,
    inverse_transform_path: Path | None = None,
    inversion_method: InversionMethod = InversionMethod.ELASTIX,
    cli_user: bool = False,
) -> DistortionResult:
    """Estimate scanner distortion at SPIRIT fiducials from a registration.

    Steps:

    1. Map packaged ``atlas_mm`` fiducials into fixed/scan space with the
       shared inverse of the full ``BSpline_Transform.txt`` chain
       (rigid∘affine∘B-spline), created once via
       :func:`~spirit_phantom.core.registration.ensure_inverse_transform`.
       These coordinates should land on the physical distortion markers in
       the scan when registration is accurate.
    2. Apply the rigid stage alone to those fixed-space points to obtain the
       post-rigid / pose-aligned intermediate frame.
    3. Build an affine+B-spline-only transform (rigid unlinked) and apply it
       with transformix to the post-rigid points. The result should recover
       ``atlas_mm``; the residual is logged as a consistency check.
    4. Report ``displacement = atlas_mm - rigid_aligned_mm`` as the
       deformation induced by affine + B-spline (the scanner geometric
       residual under the method assumptions).

    Args:
        registration_directory: Existing ``register`` output directory
            containing ``Rigid_Transform.txt``, ``Affine_Transform.txt`` and
            ``BSpline_Transform.txt``.
        fixed_image_path: Scanner (fixed) image used as the registration
            target.
        moving_image_path: Optional atlas/moving image for transformix
            geometry. When omitted, prefers ``Bspline_Image.nii.gz`` in the
            registration directory.
        output_directory: Directory for transformix I/O, rewritten transforms,
            CSVs and QC masks.
        fiducials_config_path: Optional override for the packaged fiducials
            JSON.
        inverse_transform_path: Optional precomputed inverse. When omitted,
            :func:`~spirit_phantom.core.registration.ensure_inverse_transform`
            is called on ``registration_directory``.
        inversion_method: Method used to invert the full forward transform.
        cli_user: When True, print numbered stage progress with timings.

    Returns:
        Per-fiducial residuals and aggregate summary.

    Raises:
        FileNotFoundError: If a required registration transform is missing.
        ValueError: If no fiducials are available.
    """
    rigid_transform_path = registration_directory / RIGID_TRANSFORM_FILENAME
    affine_transform_path = registration_directory / AFFINE_TRANSFORM_FILENAME
    bspline_transform_path = registration_directory / BSPLINE_TRANSFORM_FILENAME
    for path in (rigid_transform_path, affine_transform_path, bspline_transform_path):
        if not path.is_file():
            msg = f"Transform file not found: {path}"
            raise FileNotFoundError(msg)

    if moving_image_path is None:
        bspline_image = registration_directory / BSPLINE_IMAGE_FILENAME
        if not bspline_image.is_file():
            msg = (
                "moving_image_path was not provided and "
                f"{BSPLINE_IMAGE_FILENAME} is missing from {registration_directory}."
            )
            raise FileNotFoundError(msg)
        resolved_moving_image_path = bspline_image
    else:
        resolved_moving_image_path = moving_image_path

    fiducials = load_spirit_fiducials(config_path=fiducials_config_path)
    if not fiducials:
        msg = "No SPIRIT fiducials available for distortion evaluation."
        raise ValueError(msg)
    atlas_mm = [fiducial.atlas_mm for fiducial in fiducials]

    output_directory.mkdir(parents=True, exist_ok=True)
    with capture_logs_to_file(log_path=output_directory / DISTORTION_LOG_FILENAME):
        save_points(
            points=[list(point) for point in atlas_mm],
            output_path=output_directory / FIDUCIALS_ATLAS_MM_FILENAME,
            point_type="point",
        )

        stage_label = "Inverse registration"
        started_at = print_cli_stage(
            stage=1, total=_ANALYSIS_STAGE_COUNT, label=stage_label, enabled=cli_user
        )
        resolved_inverse = (
            inverse_transform_path
            if inverse_transform_path is not None
            else ensure_inverse_transform(
                registration_directory=registration_directory,
                fixed_image_path=fixed_image_path,
                forward_transform_path=bspline_transform_path,
            )
        )
        fixed_mm = invert_points_through_transformix(
            moving_points=[list(point) for point in atlas_mm],
            transform_parameter_path=bspline_transform_path,
            fixed_image_path=fixed_image_path,
            moving_image_path=resolved_moving_image_path,
            output_directory=output_directory / "moving_to_fixed",
            method=inversion_method,
            inverse_transform_path=resolved_inverse,
        )
        save_points(
            points=fixed_mm,
            output_path=output_directory / FIDUCIALS_FIXED_MM_FILENAME,
            point_type="point",
        )
        print_cli_stage_done(
            stage=1,
            total=_ANALYSIS_STAGE_COUNT,
            label=stage_label,
            started_at=started_at,
            enabled=cli_user,
        )

        stage_label = "Rigid-only transform"
        started_at = print_cli_stage(
            stage=2, total=_ANALYSIS_STAGE_COUNT, label=stage_label, enabled=cli_user
        )
        rigid_aligned_mm = transform_points_with_transformix(
            points=fixed_mm,
            transform_parameter_path=rigid_transform_path,
            moving_image_path=resolved_moving_image_path,
            output_directory=output_directory / "rigid_only",
        )
        save_points(
            points=rigid_aligned_mm,
            output_path=output_directory / FIDUCIALS_RIGID_ALIGNED_MM_FILENAME,
            point_type="point",
        )
        print_cli_stage_done(
            stage=2,
            total=_ANALYSIS_STAGE_COUNT,
            label=stage_label,
            started_at=started_at,
            enabled=cli_user,
        )

        stage_label = "Affine and B-spline transform"
        started_at = print_cli_stage(
            stage=3, total=_ANALYSIS_STAGE_COUNT, label=stage_label, enabled=cli_user
        )
        affine_bspline_only_path = build_affine_bspline_only_transform(
            affine_transform_path=affine_transform_path,
            bspline_transform_path=bspline_transform_path,
            output_directory=output_directory / "affine_bspline_only",
        )
        nonrigid_roundtrip_mm = transform_points_with_transformix(
            points=rigid_aligned_mm,
            transform_parameter_path=affine_bspline_only_path,
            moving_image_path=resolved_moving_image_path,
            output_directory=output_directory / "affine_bspline_points",
        )
        save_points(
            points=nonrigid_roundtrip_mm,
            output_path=output_directory / FIDUCIALS_NONRIGID_ROUNDTRIP_MM_FILENAME,
            point_type="point",
        )
        roundtrip_residual = _max_abs_residual_mm(
            mapped=nonrigid_roundtrip_mm,
            targets=atlas_mm,
        )
        if roundtrip_residual > _NONRIGID_ROUNDTRIP_WARNING_MM:
            logger.warning(
                "Affine+B-spline round-trip residual %.4g mm exceeds %.4g mm.",
                roundtrip_residual,
                _NONRIGID_ROUNDTRIP_WARNING_MM,
            )
        else:
            logger.info(
                "Affine+B-spline round-trip residual: %.4g mm.",
                roundtrip_residual,
            )
        print_cli_stage_done(
            stage=3,
            total=_ANALYSIS_STAGE_COUNT,
            label=stage_label,
            started_at=started_at,
            enabled=cli_user,
        )

        stage_label = "Writing distortion outputs"
        started_at = print_cli_stage(
            stage=4, total=_ANALYSIS_STAGE_COUNT, label=stage_label, enabled=cli_user
        )
        points = compute_fiducial_distortions(
            fiducials=fiducials,
            atlas_mm=atlas_mm,
            fixed_mm=fixed_mm,
            rigid_aligned_mm=rigid_aligned_mm,
            nonrigid_roundtrip_mm=nonrigid_roundtrip_mm,
        )
        summary = summarise_distortions(points=points)
        write_distortion_per_point_csv(
            points=points,
            output_path=output_directory / DISTORTION_PER_POINT_CSV_FILENAME,
        )
        write_distortion_summary_csv(
            summary=summary,
            output_path=output_directory / DISTORTION_SUMMARY_CSV_FILENAME,
        )
        fixed_image = nibabel.nifti1.load(filename=str(fixed_image_path))
        save_world_points_mask(
            image=fixed_image,
            points_mm=[point.fixed_mm for point in points],
            output_path=output_directory / FIDUCIALS_FIXED_MASK_FILENAME,
            description="SPIRIT fiducials in fixed space (1-based F01.. order)",
        )
        logger.info(
            "Distortion summary: n=%d mean=%.4g mm max=%.4g mm rms=%.4g mm "
            "nonrigid_roundtrip_max=%.4g mm",
            summary.n_fiducials,
            summary.mean_magnitude_mm,
            summary.max_magnitude_mm,
            summary.rms_magnitude_mm,
            summary.max_nonrigid_roundtrip_residual_mm,
        )
        print_cli_stage_done(
            stage=4,
            total=_ANALYSIS_STAGE_COUNT,
            label=stage_label,
            started_at=started_at,
            enabled=cli_user,
        )

    return DistortionResult(
        points=points,
        summary=summary,
        affine_bspline_only_transform_path=affine_bspline_only_path,
    )

"""Tests for scanner-distortion evaluation helpers and transform chaining."""

from __future__ import annotations

from typing import TYPE_CHECKING

import itk
import nibabel
import numpy as np
import pytest

from spirit_phantom.core.distortion import (
    AFFINE_NO_RIGID_TRANSFORM_FILENAME,
    BSPLINE_AFFINE_ONLY_TRANSFORM_FILENAME,
    FiducialDistortion,
    build_affine_bspline_only_transform,
    compute_fiducial_distortions,
    evaluate_scanner_distortion,
    summarise_distortions,
    write_distortion_per_point_csv,
    write_distortion_summary_csv,
)
from spirit_phantom.core.fiducials import Fiducial
from spirit_phantom.io.points import transform_points_with_transformix

if TYPE_CHECKING:
    from pathlib import Path


def _fiducial(
    *,
    fiducial_id: str,
    atlas_mm: tuple[float, float, float],
) -> Fiducial:
    """Build a minimal fiducial for unit tests."""
    return Fiducial(
        id=fiducial_id,
        part="lower_fiducial_grid",
        confluence_address="C-5-t",
        component_label=23,
        layer="inner",
        atlas_mm=atlas_mm,
    )


def test_compute_fiducial_distortions_reports_atlas_minus_rigid() -> None:
    """Displacement must be atlas minus rigid-aligned, with magnitude and radius."""
    fiducials = (
        _fiducial(fiducial_id="F01", atlas_mm=(3.0, 4.0, 0.0)),
        _fiducial(fiducial_id="F02", atlas_mm=(0.0, 0.0, 5.0)),
    )
    points = compute_fiducial_distortions(
        fiducials=fiducials,
        atlas_mm=[fiducial.atlas_mm for fiducial in fiducials],
        fixed_mm=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        rigid_aligned_mm=[[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        nonrigid_roundtrip_mm=[[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
    )

    assert points[0].displacement_mm == pytest.approx((2.0, 2.0, 0.0))
    assert points[0].magnitude_mm == pytest.approx((2.0**2 + 2.0**2) ** 0.5)
    assert points[0].radius_from_isocentre_mm == pytest.approx(5.0)
    assert points[1].displacement_mm == pytest.approx((0.0, 0.0, 4.0))
    assert points[1].magnitude_mm == pytest.approx(4.0)


def test_summarise_distortions_aggregates_magnitudes() -> None:
    """Summary should report mean, max, RMS and round-trip residual."""
    points = (
        FiducialDistortion(
            fiducial_id="F01",
            part="lower_fiducial_grid",
            layer="inner",
            atlas_mm=(3.0, 0.0, 0.0),
            fixed_mm=(0.0, 0.0, 0.0),
            rigid_aligned_mm=(0.0, 0.0, 0.0),
            nonrigid_roundtrip_mm=(3.1, 0.0, 0.0),
            displacement_mm=(3.0, 0.0, 0.0),
            magnitude_mm=3.0,
            radius_from_isocentre_mm=3.0,
        ),
        FiducialDistortion(
            fiducial_id="F02",
            part="upper_fiducial_grid",
            layer="outer",
            atlas_mm=(0.0, 4.0, 0.0),
            fixed_mm=(0.0, 0.0, 0.0),
            rigid_aligned_mm=(0.0, 0.0, 0.0),
            nonrigid_roundtrip_mm=(0.0, 4.0, 0.0),
            displacement_mm=(0.0, 4.0, 0.0),
            magnitude_mm=4.0,
            radius_from_isocentre_mm=4.0,
        ),
    )
    summary = summarise_distortions(points=points)
    assert summary.n_fiducials == 2
    assert summary.mean_magnitude_mm == pytest.approx(3.5)
    assert summary.max_magnitude_mm == pytest.approx(4.0)
    assert summary.rms_magnitude_mm == pytest.approx(((9.0 + 16.0) / 2.0) ** 0.5)
    assert summary.max_nonrigid_roundtrip_residual_mm == pytest.approx(0.1)


def test_write_distortion_csvs(tmp_path: Path) -> None:
    """CSV writers should create headered files with expected columns."""
    points = (
        FiducialDistortion(
            fiducial_id="F01",
            part="lower_fiducial_grid",
            layer="inner",
            atlas_mm=(1.0, 2.0, 3.0),
            fixed_mm=(4.0, 5.0, 6.0),
            rigid_aligned_mm=(0.5, 1.5, 2.5),
            nonrigid_roundtrip_mm=(1.0, 2.0, 3.0),
            displacement_mm=(0.5, 0.5, 0.5),
            magnitude_mm=(0.75) ** 0.5,
            radius_from_isocentre_mm=(14.0) ** 0.5,
        ),
    )
    summary = summarise_distortions(points=points)
    per_point = write_distortion_per_point_csv(
        points=points,
        output_path=tmp_path / "per_point.csv",
    )
    summary_path = write_distortion_summary_csv(
        summary=summary,
        output_path=tmp_path / "summary.csv",
    )
    assert "fiducial_id" in per_point.read_text(encoding="utf-8")
    assert "mean_magnitude_mm" in summary_path.read_text(encoding="utf-8")


def _write_centred_image(*, image_path: Path) -> None:
    """Write a 32 mm cube of 1 mm voxels centred on the world origin."""
    data = np.zeros((32, 32, 32), dtype=np.float32)
    affine = np.eye(4)
    affine[:3, 3] = [-15.5, -15.5, -15.5]
    nibabel.save(nibabel.Nifti1Image(dataobj=data, affine=affine), str(image_path))


def _write_euler_transform(
    *,
    image_path: Path,
    transform_path: Path,
    parameters: tuple[float, float, float, float, float, float],
    initial_transform_file: Path | None = None,
) -> None:
    """Write an Euler TransformParameters file (rotations rad, translations mm)."""
    itk_image = itk.imread(str(image_path), itk.F)
    origin = [float(value) for value in itk.origin(itk_image)]
    spacing = [float(value) for value in itk.spacing(itk_image)]
    direction = np.asarray(
        itk.array_from_matrix(itk_image.GetDirection()), dtype=np.float64
    ).ravel()
    size = [int(value) for value in itk.size(itk_image)]
    initial = (
        str(initial_transform_file)
        if initial_transform_file is not None
        else "NoInitialTransform"
    )
    lines = [
        '(Transform "EulerTransform")',
        "(NumberOfParameters 6)",
        f"(TransformParameters {' '.join(str(value) for value in parameters)})",
        f'(InitialTransformParameterFileName "{initial}")',
        '(HowToCombineTransforms "Compose")',
        "(FixedImageDimension 3)",
        "(MovingImageDimension 3)",
        '(FixedInternalImagePixelType "float")',
        '(MovingInternalImagePixelType "float")',
        f"(Size {' '.join(str(value) for value in size)})",
        "(Index 0 0 0)",
        f"(Spacing {' '.join(str(value) for value in spacing)})",
        f"(Origin {' '.join(str(value) for value in origin)})",
        f"(Direction {' '.join(str(value) for value in direction)})",
        "(CenterOfRotationPoint 0 0 0)",
        '(ComputeZYX "false")',
        '(ResampleInterpolator "FinalNearestNeighborInterpolator")',
        '(Resampler "DefaultResampler")',
        "(DefaultPixelValue 0)",
        '(ResultImageFormat "nii.gz")',
        '(ResultImagePixelType "float")',
        '(CompressResultImage "true")',
    ]
    transform_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scale_affine_transform(
    *,
    image_path: Path,
    transform_path: Path,
    scale: float,
    initial_transform_file: Path | None = None,
) -> None:
    """Write a diagonal affine (isotropic scale, no shear/translation)."""
    itk_image = itk.imread(str(image_path), itk.F)
    origin = [float(value) for value in itk.origin(itk_image)]
    spacing = [float(value) for value in itk.spacing(itk_image)]
    direction = np.asarray(
        itk.array_from_matrix(itk_image.GetDirection()), dtype=np.float64
    ).ravel()
    size = [int(value) for value in itk.size(itk_image)]
    initial = (
        str(initial_transform_file)
        if initial_transform_file is not None
        else "NoInitialTransform"
    )
    # Affine matrix stored row-major 3x3 then translation.
    parameters = (
        scale,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
    )
    lines = [
        '(Transform "AffineTransform")',
        "(NumberOfParameters 12)",
        f"(TransformParameters {' '.join(str(value) for value in parameters)})",
        f'(InitialTransformParameterFileName "{initial}")',
        '(HowToCombineTransforms "Compose")',
        "(FixedImageDimension 3)",
        "(MovingImageDimension 3)",
        '(FixedInternalImagePixelType "float")',
        '(MovingInternalImagePixelType "float")',
        f"(Size {' '.join(str(value) for value in size)})",
        "(Index 0 0 0)",
        f"(Spacing {' '.join(str(value) for value in spacing)})",
        f"(Origin {' '.join(str(value) for value in origin)})",
        f"(Direction {' '.join(str(value) for value in direction)})",
        "(CenterOfRotationPoint 0 0 0)",
        '(ResampleInterpolator "FinalNearestNeighborInterpolator")',
        '(Resampler "DefaultResampler")',
        "(DefaultPixelValue 0)",
        '(ResultImageFormat "nii.gz")',
        '(ResultImagePixelType "float")',
        '(CompressResultImage "true")',
    ]
    transform_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_affine_bspline_only_transform_unlinks_rigid(tmp_path: Path) -> None:
    """Rewritten chain must drop the rigid initial transform reference."""
    image_path = tmp_path / "image.nii.gz"
    _write_centred_image(image_path=image_path)
    rigid_path = tmp_path / "Rigid_Transform.txt"
    affine_path = tmp_path / "Affine_Transform.txt"
    bspline_path = tmp_path / "BSpline_Transform.txt"
    _write_euler_transform(
        image_path=image_path,
        transform_path=rigid_path,
        parameters=(0.0, 0.0, 0.1, 1.0, 0.0, 0.0),
    )
    _write_scale_affine_transform(
        image_path=image_path,
        transform_path=affine_path,
        scale=1.05,
        initial_transform_file=rigid_path,
    )
    # Stand in for B-spline with a second Euler stage chained to affine.
    _write_euler_transform(
        image_path=image_path,
        transform_path=bspline_path,
        parameters=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        initial_transform_file=affine_path,
    )

    head = build_affine_bspline_only_transform(
        affine_transform_path=affine_path,
        bspline_transform_path=bspline_path,
        output_directory=tmp_path / "out",
    )
    assert head.name == BSPLINE_AFFINE_ONLY_TRANSFORM_FILENAME
    affine_only = tmp_path / "out" / AFFINE_NO_RIGID_TRANSFORM_FILENAME
    assert affine_only.is_file()
    affine_text = affine_only.read_text(encoding="utf-8")
    assert "NoInitialTransform" in affine_text
    assert str(rigid_path) not in affine_text
    head_text = head.read_text(encoding="utf-8")
    assert (
        str(affine_only) in head_text or AFFINE_NO_RIGID_TRANSFORM_FILENAME in head_text
    )


def test_affine_only_chain_applies_scale_without_rigid(tmp_path: Path) -> None:
    """Unlinked affine scale must act on post-rigid points, not fixed points."""
    image_path = tmp_path / "image.nii.gz"
    _write_centred_image(image_path=image_path)
    rigid_path = tmp_path / "Rigid_Transform.txt"
    affine_path = tmp_path / "Affine_Transform.txt"
    bspline_path = tmp_path / "BSpline_Transform.txt"
    _write_euler_transform(
        image_path=image_path,
        transform_path=rigid_path,
        parameters=(0.0, 0.0, 0.0, 2.0, 0.0, 0.0),
    )
    _write_scale_affine_transform(
        image_path=image_path,
        transform_path=affine_path,
        scale=1.1,
        initial_transform_file=rigid_path,
    )
    _write_euler_transform(
        image_path=image_path,
        transform_path=bspline_path,
        parameters=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        initial_transform_file=affine_path,
    )

    head = build_affine_bspline_only_transform(
        affine_transform_path=affine_path,
        bspline_transform_path=bspline_path,
        output_directory=tmp_path / "chain",
    )
    # Post-rigid point (5, 0, 0) should scale to (5.5, 0, 0).
    mapped = transform_points_with_transformix(
        points=[[5.0, 0.0, 0.0]],
        transform_parameter_path=head,
        moving_image_path=image_path,
        output_directory=tmp_path / "tx",
    )
    assert mapped[0] == pytest.approx([5.5, 0.0, 0.0], abs=1e-5)


def test_evaluate_scanner_distortion_with_synthetic_chain(tmp_path: Path) -> None:
    """End-to-end on a known rigid-translation + scale chain using one fiducial."""
    image_path = tmp_path / "image.nii.gz"
    _write_centred_image(image_path=image_path)
    registration_directory = tmp_path / "reg"
    registration_directory.mkdir()
    rigid_path = registration_directory / "Rigid_Transform.txt"
    affine_path = registration_directory / "Affine_Transform.txt"
    bspline_path = registration_directory / "BSpline_Transform.txt"
    # Rigid translates +2 mm in x; affine scales by 1.1; B-spline identity.
    _write_euler_transform(
        image_path=image_path,
        transform_path=rigid_path,
        parameters=(0.0, 0.0, 0.0, 2.0, 0.0, 0.0),
    )
    _write_scale_affine_transform(
        image_path=image_path,
        transform_path=affine_path,
        scale=1.1,
        initial_transform_file=rigid_path,
    )
    _write_euler_transform(
        image_path=image_path,
        transform_path=bspline_path,
        parameters=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        initial_transform_file=affine_path,
    )

    # Choose atlas point so that full T^{-1} is well defined under scale+translate:
    # T(x) = 1.1 * (x + (2,0,0)) = atlas ⇒ x = atlas/1.1 - (2,0,0).
    # Use a custom one-fiducial config so inversion stays cheap.
    fiducials_json = tmp_path / "fiducials.json"
    fiducials_json.write_text(
        """
{
  "phantom": {"name": "SPIRIT", "part_number": "A-3011-0003", "issue": "1.0"},
  "fiducial_geometry": {
    "shape": "sphere",
    "diameter_mm": 8.0,
    "grid_spacing_mm": 34.0
  },
  "coordinate_frame": "atlas_mm",
  "fiducials": [
    {
      "id": "F01",
      "part": "lower_fiducial_grid",
      "confluence_address": "C-5-t",
      "component_label": 23,
      "layer": "inner",
      "atlas_mm": [5.5, 0.0, 0.0]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    result = evaluate_scanner_distortion(
        registration_directory=registration_directory,
        fixed_image_path=image_path,
        moving_image_path=image_path,
        output_directory=tmp_path / "distortion",
        fiducials_config_path=fiducials_json,
    )

    assert result.summary.n_fiducials == 1
    point = result.points[0]
    # Elastix applies the chain in ITK LPS physical space. For an X-only
    # translation+scale that means the NIfTI/RAS fixed coordinate that maps to
    # atlas 5.5 is +7 (not the RAS-naive 5.5/1.1 - 2 = 3):
    #   fixed_itk=-7 -> rigid (+2) -> -5 -> affine (*1.1) -> -5.5 -> nifti atlas 5.5.
    # Post-rigid nifti is therefore 5.0 and displacement is 0.5 mm.
    assert point.fixed_mm == pytest.approx((7.0, 0.0, 0.0), abs=0.05)
    assert point.rigid_aligned_mm == pytest.approx((5.0, 0.0, 0.0), abs=0.05)
    assert point.displacement_mm == pytest.approx((0.5, 0.0, 0.0), abs=0.05)
    assert point.magnitude_mm == pytest.approx(0.5, abs=0.05)
    assert point.nonrigid_roundtrip_mm == pytest.approx((5.5, 0.0, 0.0), abs=0.05)
    assert (tmp_path / "distortion" / "distortion_per_point.csv").is_file()
    assert (tmp_path / "distortion" / "distortion_summary.csv").is_file()

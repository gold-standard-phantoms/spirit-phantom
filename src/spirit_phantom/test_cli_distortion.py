"""Tests for the scanner-distortion CLI entry points."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from spirit_phantom import cli
from spirit_phantom.core import (
    distortion,
    guarded_registration,
    registration,
    slice_thickness_analysis,
)

if TYPE_CHECKING:
    import pytest


def _fake_result() -> distortion.DistortionResult:
    point = distortion.FiducialDistortion(
        fiducial_id="F01",
        part="lower_fiducial_grid",
        layer="inner",
        atlas_mm=(1.0, 0.0, 0.0),
        fixed_mm=(0.0, 0.0, 0.0),
        rigid_aligned_mm=(0.5, 0.0, 0.0),
        nonrigid_roundtrip_mm=(1.0, 0.0, 0.0),
        displacement_mm=(0.5, 0.0, 0.0),
        magnitude_mm=0.5,
        radius_from_isocentre_mm=1.0,
    )
    summary = distortion.DistortionSummary(
        n_fiducials=1,
        mean_magnitude_mm=0.5,
        max_magnitude_mm=0.5,
        rms_magnitude_mm=0.5,
        max_nonrigid_roundtrip_residual_mm=0.0,
    )
    return distortion.DistortionResult(
        points=(point,),
        summary=summary,
        affine_bspline_only_transform_path=Path("unused"),
    )


def _write_registration_outputs(
    output_directory: Path,
) -> registration.RegistrationResult:
    output_directory.mkdir(parents=True, exist_ok=True)
    component_atlas = output_directory / "transformed_component_atlas.nii.gz"
    registered = output_directory / "Bspline_Image.nii.gz"
    rigid = output_directory / "Rigid_Transform.txt"
    affine = output_directory / "Affine_Transform.txt"
    transform = output_directory / "BSpline_Transform.txt"
    component_atlas.write_bytes(b"c")
    registered.write_bytes(b"r")
    rigid.write_text("rigid", encoding="utf-8")
    affine.write_text("affine", encoding="utf-8")
    transform.write_text("transform", encoding="utf-8")
    return registration.RegistrationResult(
        rigid_image_path=output_directory / "Rigid_Image.nii.gz",
        affine_image_path=output_directory / "Affine_Image.nii.gz",
        bspline_image_path=registered,
        rigid_parameters_path=output_directory / "Rigid_Parameters_In.txt",
        affine_parameters_path=output_directory / "Affine_Parameters_In.txt",
        bspline_parameters_path=output_directory / "BSpline_Parameters_In.txt",
        rigid_transform_path=rigid,
        affine_transform_path=affine,
        bspline_transform_path=transform,
        registered_image_path=registered,
        registration_transform_path=transform,
        transformed_component_atlas_path=component_atlas,
    )


def test_analyse_distortion_requires_existing_registration(tmp_path: Path) -> None:
    """Standalone analysis should refuse a directory without transforms."""
    fixed_image = tmp_path / "fixed.nii.gz"
    moving_image = tmp_path / "moving.nii.gz"
    fixed_image.write_bytes(b"fixed")
    moving_image.write_bytes(b"moving")

    result = CliRunner().invoke(
        cli.app,
        [
            "analyse",
            "distortion",
            str(tmp_path),
            str(fixed_image),
            str(moving_image),
        ],
    )

    assert result.exit_code != 0
    assert "Transform file not found" in " ".join(result.output.split())


def test_analyse_distortion_uses_registration_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Standalone analysis should read transforms from the registration dir."""
    registration_directory = tmp_path / "reg"
    _write_registration_outputs(registration_directory)
    fixed_image = tmp_path / "fixed.nii.gz"
    moving_image = tmp_path / "moving.nii.gz"
    fixed_image.write_bytes(b"fixed")
    moving_image.write_bytes(b"moving")
    captured: dict[str, object] = {}

    def _fake_evaluate(**kwargs: object) -> distortion.DistortionResult:
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(distortion, "evaluate_scanner_distortion", _fake_evaluate)

    result = CliRunner().invoke(
        cli.app,
        [
            "analyse",
            "distortion",
            str(registration_directory),
            str(fixed_image),
            str(moving_image),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["registration_directory"] == registration_directory
    assert captured["moving_image_path"] == (
        registration_directory / "Bspline_Image.nii.gz"
    )
    assert captured["output_directory"] == registration_directory / "distortion"
    assert captured["cli_user"] is True
    assert "0.5000" in result.output


def test_register_analyse_distortion_runs_after_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``register --analyse distortion`` should chain into the analysis."""
    fixed_image = tmp_path / "fixed.nii.gz"
    moving_image = tmp_path / "moving.nii.gz"
    fixed_image.write_bytes(b"fixed")
    moving_image.write_bytes(b"moving")
    output_directory = tmp_path / "out"
    captured: dict[str, object] = {}
    inverse_path = output_directory / "Inverse_BSpline_Transform.txt"

    def _fake_register(**kwargs: object) -> registration.RegistrationResult:
        resolved = Path(str(kwargs["output_directory"]))
        return _write_registration_outputs(resolved)

    def _fake_ensure_inverse(**kwargs: object) -> Path:
        path = Path(str(kwargs["registration_directory"])) / (
            "Inverse_BSpline_Transform.txt"
        )
        path.write_text("inverse", encoding="utf-8")
        return path

    def _fake_evaluate(**kwargs: object) -> distortion.DistortionResult:
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        guarded_registration, "run_registration_isolated", _fake_register
    )
    monkeypatch.setattr(registration, "ensure_inverse_transform", _fake_ensure_inverse)
    monkeypatch.setattr(distortion, "evaluate_scanner_distortion", _fake_evaluate)

    result = CliRunner().invoke(
        cli.app,
        [
            "register",
            str(fixed_image),
            str(moving_image),
            "--output-directory",
            str(output_directory),
            "--analyse",
            "distortion",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["registration_directory"] == output_directory
    assert captured["output_directory"] == output_directory / "distortion"
    assert captured["inverse_transform_path"] == inverse_path
    assert captured["cli_user"] is False


def test_register_analyse_multiple_ensures_inverse_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Repeated ``--analyse`` should share one inverse and run both analyses."""
    fixed_image = tmp_path / "fixed.nii.gz"
    moving_image = tmp_path / "moving.nii.gz"
    fixed_image.write_bytes(b"fixed")
    moving_image.write_bytes(b"moving")
    output_directory = tmp_path / "out"
    ensure_calls: list[Path] = []
    thickness_captured: dict[str, object] = {}
    distortion_captured: dict[str, object] = {}

    def _fake_register(**kwargs: object) -> registration.RegistrationResult:
        resolved = Path(str(kwargs["output_directory"]))
        return _write_registration_outputs(resolved)

    def _fake_ensure_inverse(**kwargs: object) -> Path:
        resolved = Path(str(kwargs["registration_directory"]))
        ensure_calls.append(resolved)
        path = resolved / "Inverse_BSpline_Transform.txt"
        path.write_text("inverse", encoding="utf-8")
        return path

    def _fake_measure(
        **kwargs: object,
    ) -> list[slice_thickness_analysis.WedgeThicknessResult]:
        thickness_captured.update(kwargs)
        return [
            slice_thickness_analysis.WedgeThicknessResult(
                label=1,
                thickness_mm=5.0,
                pixel_size_mm=0.5,
                n_lines=3,
                n_samples=64,
                corners_fixed_mm=(),
            )
        ]

    def _fake_evaluate(**kwargs: object) -> distortion.DistortionResult:
        distortion_captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(
        guarded_registration, "run_registration_isolated", _fake_register
    )
    monkeypatch.setattr(registration, "ensure_inverse_transform", _fake_ensure_inverse)
    monkeypatch.setattr(
        slice_thickness_analysis,
        "measure_slice_thickness_from_fixed_wedges",
        _fake_measure,
    )
    monkeypatch.setattr(distortion, "evaluate_scanner_distortion", _fake_evaluate)

    result = CliRunner().invoke(
        cli.app,
        [
            "register",
            str(fixed_image),
            str(moving_image),
            "--output-directory",
            str(output_directory),
            "--analyse",
            "slice-thickness",
            "--analyse",
            "distortion",
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert ensure_calls == [output_directory]
    shared_inverse = output_directory / "Inverse_BSpline_Transform.txt"
    assert thickness_captured["inverse_transform_path"] == shared_inverse
    assert distortion_captured["inverse_transform_path"] == shared_inverse
    assert "5.0000" in result.output
    assert "0.5000" in result.output

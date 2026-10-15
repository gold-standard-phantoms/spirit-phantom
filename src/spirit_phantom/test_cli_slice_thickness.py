"""Tests for the slice-thickness CLI entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from spirit_phantom import cli
from spirit_phantom.core import (
    guarded_registration,
    registration,
    slice_thickness_analysis,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _fake_results() -> list[slice_thickness_analysis.WedgeThicknessResult]:
    return [
        slice_thickness_analysis.WedgeThicknessResult(
            label=label,
            thickness_mm=5.0,
            pixel_size_mm=0.5,
            n_lines=3,
            n_samples=64,
            corners_fixed_mm=(),
        )
        for label in (1, 2)
    ]


def _write_registration_outputs(
    output_directory: Path,
) -> registration.RegistrationResult:
    output_directory.mkdir(parents=True, exist_ok=True)
    component_atlas = output_directory / "transformed_component_atlas.nii.gz"
    registered = output_directory / "Bspline_Image.nii.gz"
    transform = output_directory / "BSpline_Transform.txt"
    component_atlas.write_bytes(b"c")
    registered.write_bytes(b"r")
    transform.write_text("transform", encoding="utf-8")
    return registration.RegistrationResult(
        rigid_image_path=output_directory / "Rigid_Image.nii.gz",
        affine_image_path=output_directory / "Affine_Image.nii.gz",
        bspline_image_path=registered,
        rigid_parameters_path=output_directory / "Rigid_Parameters_In.txt",
        affine_parameters_path=output_directory / "Affine_Parameters_In.txt",
        bspline_parameters_path=output_directory / "BSpline_Parameters_In.txt",
        rigid_transform_path=output_directory / "Rigid_Transform.txt",
        affine_transform_path=output_directory / "Affine_Transform.txt",
        bspline_transform_path=transform,
        registered_image_path=registered,
        registration_transform_path=transform,
        transformed_component_atlas_path=component_atlas,
    )


def test_analyse_slice_thickness_requires_existing_registration(
    tmp_path: Path,
) -> None:
    """Standalone analysis should refuse a directory without a transform."""
    fixed_image = tmp_path / "fixed.nii.gz"
    moving_image = tmp_path / "moving.nii.gz"
    fixed_image.write_bytes(b"fixed")
    moving_image.write_bytes(b"moving")

    result = CliRunner().invoke(
        cli.app,
        [
            "analyse",
            "slice-thickness",
            str(tmp_path),
            str(fixed_image),
            str(moving_image),
        ],
    )

    assert result.exit_code != 0
    assert "Transform file not found" in " ".join(result.output.split())
    assert "Mapping wedge corners" not in result.output


def test_analyse_slice_thickness_uses_registration_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Standalone analysis should read the transform from the registration dir."""
    registration_directory = tmp_path / "reg"
    _write_registration_outputs(registration_directory)
    fixed_image = tmp_path / "fixed.nii.gz"
    moving_image = tmp_path / "moving.nii.gz"
    fixed_image.write_bytes(b"fixed")
    moving_image.write_bytes(b"moving")
    captured: dict[str, object] = {}

    def _fake_measure(
        **kwargs: object,
    ) -> list[slice_thickness_analysis.WedgeThicknessResult]:
        captured.update(kwargs)
        return _fake_results()

    monkeypatch.setattr(
        slice_thickness_analysis,
        "measure_slice_thickness_from_fixed_wedges",
        _fake_measure,
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "analyse",
            "slice-thickness",
            str(registration_directory),
            str(fixed_image),
            str(moving_image),
            "--discard-edge-lines",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["transform_parameter_path"] == (
        registration_directory / "BSpline_Transform.txt"
    )
    assert captured["moving_image_path"] == (
        registration_directory / "Bspline_Image.nii.gz"
    )
    assert captured["atlas_image_path"] == moving_image
    assert captured["output_directory"] == registration_directory / "slice_thickness"
    assert captured["discard_edge_lines"] == 0
    assert "5.0000" in result.output


def test_register_analyse_slice_thickness_runs_after_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``register --analyse slice-thickness`` should chain into the analysis."""
    fixed_image = tmp_path / "fixed.nii.gz"
    moving_image = tmp_path / "moving.nii.gz"
    fixed_image.write_bytes(b"fixed")
    moving_image.write_bytes(b"moving")
    output_directory = tmp_path / "out"
    captured: dict[str, object] = {}

    def _fake_run_registration_isolated(
        **_: object,
    ) -> registration.RegistrationResult:
        return _write_registration_outputs(output_directory)

    def _fake_measure(
        **kwargs: object,
    ) -> list[slice_thickness_analysis.WedgeThicknessResult]:
        captured.update(kwargs)
        return _fake_results()

    monkeypatch.setattr(
        guarded_registration,
        "run_registration_isolated",
        _fake_run_registration_isolated,
    )
    monkeypatch.setattr(
        slice_thickness_analysis,
        "measure_slice_thickness_from_fixed_wedges",
        _fake_measure,
    )

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
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["transform_parameter_path"] == (
        output_directory / "BSpline_Transform.txt"
    )
    assert captured["output_directory"] == output_directory / "slice_thickness"
    assert captured["discard_edge_lines"] == 1

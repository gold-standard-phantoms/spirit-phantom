"""Tests for shared inverse-transform lookup in the registration directory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from spirit_phantom.core import registration
from spirit_phantom.core.registration_constants import (
    INVERSE_AFFINE_TRANSFORM_FILENAME,
    INVERSE_BSPLINE_TRANSFORM_FILENAME,
)

if TYPE_CHECKING:
    import pytest


def test_ensure_inverse_transform_reuses_existing_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing inverse files should be returned without recomputing."""
    registration_directory = tmp_path / "reg"
    registration_directory.mkdir()
    inverse_affine = registration_directory / INVERSE_AFFINE_TRANSFORM_FILENAME
    inverse_bspline = registration_directory / INVERSE_BSPLINE_TRANSFORM_FILENAME
    inverse_affine.write_text("affine", encoding="utf-8")
    inverse_bspline.write_text("bspline", encoding="utf-8")
    fixed_image = tmp_path / "fixed.nii.gz"
    fixed_image.write_bytes(b"fixed")
    compute_calls: list[Path] = []

    def _fake_compute(**kwargs: object) -> Path:
        compute_calls.append(Path(str(kwargs["output_directory"])))
        return (
            Path(str(kwargs["output_directory"])) / INVERSE_BSPLINE_TRANSFORM_FILENAME
        )

    monkeypatch.setattr(registration, "compute_inverse_transform", _fake_compute)

    result = registration.ensure_inverse_transform(
        registration_directory=registration_directory,
        fixed_image_path=fixed_image,
    )

    assert result == inverse_bspline
    assert compute_calls == []


def test_ensure_inverse_transform_computes_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing inverse files should trigger a compute into the registration dir."""
    registration_directory = tmp_path / "reg"
    registration_directory.mkdir()
    forward = registration_directory / "BSpline_Transform.txt"
    forward.write_text("forward", encoding="utf-8")
    fixed_image = tmp_path / "fixed.nii.gz"
    fixed_image.write_bytes(b"fixed")
    compute_calls: list[dict[str, object]] = []

    def _fake_compute(**kwargs: object) -> Path:
        compute_calls.append(kwargs)
        path = (
            Path(str(kwargs["output_directory"])) / INVERSE_BSPLINE_TRANSFORM_FILENAME
        )
        path.write_text("inverse", encoding="utf-8")
        (
            Path(str(kwargs["output_directory"])) / INVERSE_AFFINE_TRANSFORM_FILENAME
        ).write_text("affine", encoding="utf-8")
        return path

    monkeypatch.setattr(registration, "compute_inverse_transform", _fake_compute)

    result = registration.ensure_inverse_transform(
        registration_directory=registration_directory,
        fixed_image_path=fixed_image,
    )

    assert result == registration_directory / INVERSE_BSPLINE_TRANSFORM_FILENAME
    assert len(compute_calls) == 1
    assert compute_calls[0]["output_directory"] == registration_directory
    assert compute_calls[0]["transform_parameter_path"] == forward
    assert compute_calls[0]["fixed_image_path"] == fixed_image


def test_ensure_inverse_transform_force_recomputes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``force=True`` should recompute even when inverse files already exist."""
    registration_directory = tmp_path / "reg"
    registration_directory.mkdir()
    forward = registration_directory / "BSpline_Transform.txt"
    forward.write_text("forward", encoding="utf-8")
    (registration_directory / INVERSE_AFFINE_TRANSFORM_FILENAME).write_text(
        "affine", encoding="utf-8"
    )
    (registration_directory / INVERSE_BSPLINE_TRANSFORM_FILENAME).write_text(
        "bspline", encoding="utf-8"
    )
    fixed_image = tmp_path / "fixed.nii.gz"
    fixed_image.write_bytes(b"fixed")
    compute_calls: list[Path] = []

    def _fake_compute(**kwargs: object) -> Path:
        compute_calls.append(Path(str(kwargs["output_directory"])))
        return (
            Path(str(kwargs["output_directory"])) / INVERSE_BSPLINE_TRANSFORM_FILENAME
        )

    monkeypatch.setattr(registration, "compute_inverse_transform", _fake_compute)

    result = registration.ensure_inverse_transform(
        registration_directory=registration_directory,
        fixed_image_path=fixed_image,
        force=True,
    )

    assert result == registration_directory / INVERSE_BSPLINE_TRANSFORM_FILENAME
    assert compute_calls == [registration_directory]

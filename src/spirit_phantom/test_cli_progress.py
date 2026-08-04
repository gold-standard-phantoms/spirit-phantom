"""Tests for CLI registration progress UX."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from spirit_phantom import cli
from spirit_phantom.core import registration

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_format_duration_seconds_and_minutes() -> None:
    """Duration helper should format short and long intervals clearly."""
    assert registration._format_duration(seconds=12.34) == "12.3s"
    assert registration._format_duration(seconds=125.0) == "2m 05s"


def test_register_rejects_quiet_and_verbose_together() -> None:
    """``register`` should reject combining ``--quiet`` and ``--verbose``."""
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "register",
            "fixed.nii.gz",
            "--quiet",
            "--verbose",
        ],
    )

    assert result.exit_code != 0
    assert "either --quiet or --verbose" in result.output


def test_register_help_lists_progress_options() -> None:
    """``register --help`` should document quiet and verbose flags."""
    runner = CliRunner()
    result = runner.invoke(cli.app, ["register", "--help"])

    assert result.exit_code == 0
    assert "--quiet" in result.output
    assert "--verbose" in result.output


def test_register_quiet_suppresses_progress_messages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Quiet mode should avoid CLI progress chatter around a mocked registration."""
    fixed_image = tmp_path / "fixed.nii.gz"
    moving_image = tmp_path / "moving.nii.gz"
    fixed_image.write_bytes(b"fixed")
    moving_image.write_bytes(b"moving")
    output_directory = tmp_path / "out"

    def _fake_register_atlas(**_: object) -> registration.RegistrationResult:
        component_atlas = output_directory / "transformed_component_atlas.nii.gz"
        registered = output_directory / "Bspline_Image.nii.gz"
        transform = output_directory / "BSpline_Transform.txt"
        output_directory.mkdir(parents=True, exist_ok=True)
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

    monkeypatch.setattr(registration, "register_atlas", _fake_register_atlas)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        [
            "register",
            str(fixed_image),
            str(moving_image),
            "--output-directory",
            str(output_directory),
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""


def test_resolve_default_atlases_prefetches_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default atlas resolution should warm both signal and component caches."""
    signal_path = tmp_path / "signal.nii.gz"
    component_path = tmp_path / "components.nii.gz"
    signal_path.write_bytes(b"s")
    component_path.write_bytes(b"c")
    calls = {"signal": 0, "component": 0}

    def _fake_signal() -> Path:
        calls["signal"] += 1
        return signal_path

    def _fake_component() -> Path:
        calls["component"] += 1
        return component_path

    monkeypatch.setattr(cli, "get_default_register_moving_image_path", _fake_signal)
    monkeypatch.setattr(cli, "get_default_component_atlas_image_path", _fake_component)

    resolved = cli._resolve_default_atlases(quiet=True)

    assert resolved == signal_path
    assert calls == {"signal": 1, "component": 1}


def test_print_cli_stage_messages(capsys: pytest.CaptureFixture[str]) -> None:
    """Numbered stage helpers should print only when enabled."""
    started_at = registration._print_cli_stage(
        stage=1,
        label="Rigid registration",
        enabled=True,
    )
    registration._print_cli_stage_done(
        stage=1,
        label="Rigid registration",
        started_at=started_at,
        enabled=True,
    )
    registration._print_cli_stage(stage=2, label="Affine registration", enabled=False)

    captured = capsys.readouterr().out
    assert "[1/4] Rigid registration..." in captured
    assert "[1/4] Rigid registration done" in captured
    assert "[2/4]" not in captured

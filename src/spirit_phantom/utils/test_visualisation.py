"""Tests for registration visualisation helpers."""

from pathlib import Path

import itk
import numpy as np
import pytest

from spirit_phantom.utils import visualisation


def _patch_checkerboard_inputs(
    *,
    monkeypatch: pytest.MonkeyPatch,
    written_paths: list[Path],
) -> None:
    """Patch image loading and writing for checkerboard path tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        written_paths: List populated with image output paths passed to ITK.
    """

    def fake_load_nifti_as_array(*, image_path: Path) -> np.ndarray:
        del image_path
        data = np.zeros((4, 4, 3), dtype=np.float32)
        data[:, :, 1] = 1.0
        return data

    def fake_imwrite(image: object, filename: str) -> None:
        del image
        written_paths.append(Path(filename))

    def fake_checker_board_image_filter(
        fixed_image: object,
        registered_image: object,
        *,
        checker_pattern: list[int],
    ) -> object:
        del registered_image, checker_pattern
        return fixed_image

    monkeypatch.setattr(
        visualisation,
        "_load_nifti_as_array",
        fake_load_nifti_as_array,
    )
    monkeypatch.setattr(
        itk,
        "checker_board_image_filter",
        fake_checker_board_image_filter,
    )
    monkeypatch.setattr(itk, "imwrite", fake_imwrite)


def test_visualise_checkerboard_defaults_to_registered_image_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test checkerboard images are saved beside the registered image by default."""
    written_paths: list[Path] = []
    _patch_checkerboard_inputs(
        monkeypatch=monkeypatch,
        written_paths=written_paths,
    )

    registered_image_path = tmp_path / "registration" / "registered.nii.gz"
    output_paths = visualisation.visualise_checkerboard(
        fixed_image_path=tmp_path / "fixed.nii.gz",
        registered_image_path=registered_image_path,
        slice_indices=[1],
    )

    expected_path = registered_image_path.parent / "checkerboard_itk_slice_001.png"
    assert output_paths == [expected_path]
    assert written_paths == [expected_path]


def test_visualise_checkerboard_uses_explicit_save_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test checkerboard images are saved to an explicit directory when provided."""
    written_paths: list[Path] = []
    _patch_checkerboard_inputs(
        monkeypatch=monkeypatch,
        written_paths=written_paths,
    )

    save_directory = tmp_path / "checkerboards"
    output_paths = visualisation.visualise_checkerboard(
        fixed_image_path=tmp_path / "fixed.nii.gz",
        registered_image_path=tmp_path / "registration" / "registered.nii.gz",
        slice_indices=[1],
        save_directory=save_directory,
    )

    expected_path = save_directory / "checkerboard_itk_slice_001.png"
    assert output_paths == [expected_path]
    assert written_paths == [expected_path]

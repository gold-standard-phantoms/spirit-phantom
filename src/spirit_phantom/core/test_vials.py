"""Tests for vial statistics extraction from registered atlas images."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest

from spirit_phantom.core.vials import (
    compute_vial_statistics,
    format_vial_statistics_table,
    print_vial_statistics_table,
    save_vial_statistics_table,
)

ATLAS_FIXTURE_PATH = (
    Path(__file__).parent / "test_data" / "spirit_issue1.0_vx0.25_95d7fe0.nii.gz"
)


def _save_nifti(
    *,
    image_data: np.ndarray,
    output_path: Path,
    affine: np.ndarray | None = None,
) -> None:
    """Save NumPy array as a NIfTI image for test setup.

    Args:
        image_data: Array to save.
        output_path: NIfTI output path.
        affine: Optional 4x4 affine transform. Identity is used by default.
    """
    transform = np.eye(4) if affine is None else affine
    nifti_image = nib.Nifti1Image(image_data, transform)
    nib.save(nifti_image, str(output_path))


def test_compute_vial_statistics_with_synthetic_nifti(tmp_path: Path) -> None:
    """Check ROI behaviour with and without erosion."""
    atlas = np.zeros((7, 7, 7), dtype=np.int16)
    atlas[1:6, 1:6, 1:6] = 1
    atlas[0:1, 6:7, 0:1] = 2

    mri = np.zeros((7, 7, 7), dtype=np.float32)
    mri[atlas == 1] = 10.0
    mri[atlas == 2] = 50.0

    atlas_path = tmp_path / "atlas.nii.gz"
    mri_path = tmp_path / "mri.nii.gz"
    _save_nifti(image_data=atlas, output_path=atlas_path)
    _save_nifti(image_data=mri, output_path=mri_path)

    stats_without_erosion = compute_vial_statistics(
        registered_atlas_image_path=atlas_path,
        mri_scan_image_path=mri_path,
        labels=(1, 2),
        erosion_voxels=0,
    )

    label_1_stats_no_erosion = stats_without_erosion[0]
    assert label_1_stats_no_erosion[0] == 1
    assert label_1_stats_no_erosion[1] == pytest.approx(10.0)
    assert label_1_stats_no_erosion[2] == pytest.approx(0.0)
    assert label_1_stats_no_erosion[3] == 125

    label_2_stats_no_erosion = stats_without_erosion[1]
    assert label_2_stats_no_erosion[0] == 2
    assert label_2_stats_no_erosion[1] == pytest.approx(50.0)
    assert label_2_stats_no_erosion[2] == pytest.approx(0.0)
    assert label_2_stats_no_erosion[3] == 1

    stats_with_erosion = compute_vial_statistics(
        registered_atlas_image_path=atlas_path,
        mri_scan_image_path=mri_path,
        labels=(1, 2),
        erosion_voxels=1,
    )

    label_1_stats_with_erosion = stats_with_erosion[0]
    assert label_1_stats_with_erosion[0] == 1
    assert label_1_stats_with_erosion[1] == pytest.approx(10.0)
    assert label_1_stats_with_erosion[2] == pytest.approx(0.0)
    assert label_1_stats_with_erosion[3] == 27

    label_2_stats_with_erosion = stats_with_erosion[1]
    assert label_2_stats_with_erosion[0] == 2
    assert np.isnan(label_2_stats_with_erosion[1])
    assert np.isnan(label_2_stats_with_erosion[2])
    assert label_2_stats_with_erosion[3] == 0


def test_compute_vial_statistics_uses_real_atlas_data(tmp_path: Path) -> None:
    """Load the project atlas NIfTI and verify known deterministic behaviour."""
    atlas_image = cast("nib.Nifti1Image", nib.load(str(ATLAS_FIXTURE_PATH)))
    atlas_data = np.rint(np.asanyarray(atlas_image.dataobj)).astype(np.int16)

    present_labels = tuple(sorted(int(value) for value in np.unique(atlas_data) if value > 0))
    assert present_labels, "Expected positive atlas labels in fixture."
    labels_to_test = present_labels[: min(3, len(present_labels))]

    mri_data = atlas_data.astype(np.float32) * 10.0
    mri_path = tmp_path / "scanner_mock.nii.gz"
    nib.save(nib.Nifti1Image(mri_data, atlas_image.affine), str(mri_path))

    stats = compute_vial_statistics(
        registered_atlas_image_path=ATLAS_FIXTURE_PATH,
        mri_scan_image_path=mri_path,
        labels=labels_to_test,
        erosion_voxels=0,
    )

    for row in stats:
        label = int(row[0])
        mean_intensity = float(row[1])
        std_intensity = float(row[2])
        voxel_count = int(row[3])

        assert label in labels_to_test
        assert mean_intensity == pytest.approx(label * 10.0)
        assert std_intensity == pytest.approx(0.0)
        assert voxel_count > 0


def test_format_vial_statistics_table_renders_expected_layout() -> None:
    """Format vial rows into a deterministic table string."""
    rows = [
        [1, 10.25, 0.5, 125],
        [2, float("nan"), float("nan"), 0],
    ]

    table = format_vial_statistics_table(rows=rows)
    expected = "\n".join(
        [
            "Label | Mean intensity | Std intensity | Voxel count",
            "------+----------------+---------------+------------",
            "1     | 10.250000      | 0.500000      | 125        ",
            "2     | nan            | nan           | 0          ",
        ]
    )

    assert table
    assert table == expected


def test_format_vial_statistics_table_returns_empty_message() -> None:
    """Return a clear message when no rows are provided."""
    assert format_vial_statistics_table(rows=[]) == "No vial statistics were returned."


def test_save_vial_statistics_table_writes_expected_content(tmp_path: Path) -> None:
    """Save the vial statistics table to disk and return the output path."""
    rows = [
        [1, 12.0, 0.0, 10],
        [2, 20.5, 1.25, 42],
    ]
    output_path = tmp_path / "reports" / "vials.txt"

    returned_path = save_vial_statistics_table(rows=rows, output_path=output_path)

    assert returned_path == output_path
    assert output_path.exists()
    expected_content = f"{format_vial_statistics_table(rows=rows)}\n"
    assert output_path.read_text(encoding="utf-8") == expected_content


def test_print_vial_statistics_table_writes_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print a formatted table to stdout."""
    rows = [[1, 12.0, 0.0, 10]]
    print_vial_statistics_table(rows=rows)
    captured = capsys.readouterr()
    written = captured.out

    assert written.endswith("\n")
    assert "Label" in written
    assert "12.000000" in written

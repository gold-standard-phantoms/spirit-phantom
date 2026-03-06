"""Tests for vial statistics extraction from registered atlas images."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from spirit_phantom.core import vials as vials_module
from spirit_phantom.core.vials import (
    _compute_vial_statistics,
    compute_vial_statistics_details,
    format_vial_statistics_details_table,
    format_vial_statistics_table,
    print_vial_statistics_details_table,
    print_vial_statistics_table,
    save_vial_statistics_details_table,
    save_vial_statistics_table,
)

if TYPE_CHECKING:
    from pathlib import Path


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

    stats_without_erosion = _compute_vial_statistics(
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

    stats_with_erosion = _compute_vial_statistics(
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


def test_compute_vial_statistics_details_returns_expected_rows(tmp_path: Path) -> None:
    """Return detailed vial rows in A-to-T order with metadata columns."""
    atlas = np.zeros((5, 5, 5), dtype=np.int16)
    atlas[1:4, 1:4, 1:4] = 17
    atlas[0:1, 0:2, 0:2] = 18
    atlas[4:5, 3:5, 3:5] = 20
    atlas[2:3, 4:5, 0:1] = 14
    atlas[4:5, 0:1, 4:5] = 13

    mri = np.zeros((5, 5, 5), dtype=np.float32)
    mri[atlas == 17] = 101.0
    mri[atlas == 18] = 202.0
    mri[atlas == 20] = 303.0
    mri[atlas == 14] = 404.0
    mri[atlas == 13] = 505.0

    atlas_path = tmp_path / "atlas_detailed.nii.gz"
    mri_path = tmp_path / "mri_detailed.nii.gz"
    _save_nifti(image_data=atlas, output_path=atlas_path)
    _save_nifti(image_data=mri, output_path=mri_path)

    rows = compute_vial_statistics_details(
        registered_atlas_image_path=atlas_path,
        mri_scan_image_path=mri_path,
        erosion_voxels=0,
    )

    assert len(rows) == 20
    row_ids = [row.vial_id for row in rows]
    assert row_ids == [chr(code) for code in range(ord("A"), ord("T") + 1)]

    row_a = rows[0]
    row_b = rows[1]
    row_c = rows[2]
    row_f = rows[5]
    row_g = rows[6]

    assert row_a.product_code == "MNCL-0320"
    assert row_a.mean_intensity == pytest.approx(101.0)
    assert row_b.product_code == "MNCL-0159"
    assert row_b.mean_intensity == pytest.approx(202.0)
    assert row_c.product_code == "MNCL-0110"
    assert row_c.mean_intensity == pytest.approx(303.0)
    assert row_f.product_code == "PVP-0100"
    assert row_f.mean_intensity == pytest.approx(404.0)
    assert row_g.product_code == "PVP-0150"
    assert row_g.mean_intensity == pytest.approx(505.0)


def test_format_vial_statistics_details_table_renders_expected_layout() -> None:
    """Format detailed rows into a deterministic table string."""
    rows = [
        vials_module.DetailedVialStatistic(
            vial_id="A",
            product_code="MNCL-0320",
            description="0.320mM Aqueous MnCl2",
            mean_intensity=12.5,
            stdev=0.25,
            number_of_voxels=125,
        ),
        vials_module.DetailedVialStatistic(
            vial_id="B",
            product_code="MNCL-0159",
            description="0.159mM Aqueous MnCl2",
            mean_intensity=float("nan"),
            stdev=float("nan"),
            number_of_voxels=0,
        ),
    ]

    table = format_vial_statistics_details_table(rows=rows)
    assert table
    assert "Vial ID" in table
    assert "MNCL-0320" in table
    assert "12.500000" in table
    assert "nan" in table


def test_save_vial_statistics_details_table_writes_expected_content(
    tmp_path: Path,
) -> None:
    """Save the detailed vial statistics table and return output path."""
    rows = [
        vials_module.DetailedVialStatistic(
            vial_id="A",
            product_code="MNCL-0320",
            description="0.320mM Aqueous MnCl2",
            mean_intensity=12.0,
            stdev=0.0,
            number_of_voxels=10,
        )
    ]
    output_path = tmp_path / "reports" / "vials_detailed.txt"

    returned_path = save_vial_statistics_details_table(
        rows=rows, output_path=output_path
    )

    assert returned_path == output_path
    assert output_path.exists()
    expected_content = f"{format_vial_statistics_details_table(rows=rows)}\n"
    assert output_path.read_text(encoding="utf-8") == expected_content


def test_print_vial_statistics_details_table_writes_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print a formatted detailed table to stdout."""
    rows = [
        vials_module.DetailedVialStatistic(
            vial_id="A",
            product_code="MNCL-0320",
            description="0.320mM Aqueous MnCl2",
            mean_intensity=12.0,
            stdev=0.0,
            number_of_voxels=10,
        )
    ]
    print_vial_statistics_details_table(rows=rows)
    captured = capsys.readouterr()
    written = captured.out

    assert written.endswith("\n")
    assert "Vial ID" in written
    assert "MNCL-0320" in written

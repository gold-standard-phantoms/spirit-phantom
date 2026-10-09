"""Tests for fixed-space wedge sampling used by slice-thickness analysis."""

from __future__ import annotations

import ast
import math
from importlib.util import find_spec
from pathlib import Path
from typing import cast

import nibabel
import numpy as np
import pytest

from spirit_phantom.core.generate_slice_mask import (
    save_slice_mask,
    save_wedge_corner_points_mask,
    wedge_roi_corners,
)
from spirit_phantom.core.slice_thickness import nema_slice_thickness
from spirit_phantom.core.slice_thickness_analysis import (
    build_wedge_profile_diagnostics,
    sample_rectangle_edge_response,
    save_wedge_profile_diagnostics,
)


def test_save_slice_mask_writes_labelled_nifti(tmp_path: Path) -> None:
    """World-space wedge mask should be saved with labels 1 and 2."""
    shape = (60, 20, 5)
    data = np.zeros(shape, dtype=np.float32)
    affine = np.eye(4)
    affine[0, 3] = -30.0
    affine[1, 3] = -10.0
    affine[2, 3] = -2.0
    image = nibabel.Nifti1Image(dataobj=data, affine=affine)

    output_path = save_slice_mask(
        image=image,
        output_path=tmp_path / "slice_wedge_mask_atlas.nii.gz",
    )
    assert output_path.is_file()
    saved_image = cast("nibabel.nifti1.Nifti1Image", nibabel.load(str(output_path)))
    saved = np.asarray(saved_image.get_fdata(), dtype=np.uint8)
    assert set(np.unique(saved)) == {0, 1, 2}
    # Wedges sit on world Z = 0; with origin z=-2 and 1 mm voxels that is index 2.
    assert set(np.unique(saved[:, :, 2])) == {0, 1, 2}
    assert np.all(saved[:, :, 0] == 0)
    assert np.all(saved[:, :, 1] == 0)


def test_save_wedge_corner_points_mask_uses_processing_order(
    tmp_path: Path,
) -> None:
    """Corner NIfTI labels should be 1..N in wedge_roi_corners() order."""
    shape = (60, 20, 5)
    data = np.zeros(shape, dtype=np.float32)
    affine = np.eye(4)
    affine[0, 3] = -30.0
    affine[1, 3] = -10.0
    affine[2, 3] = -2.0
    image = nibabel.Nifti1Image(dataobj=data, affine=affine)

    output_path = save_wedge_corner_points_mask(
        image=image,
        output_path=tmp_path / "slice_wedge_corners_atlas.nii.gz",
        mark_radius_voxels=0,
    )
    saved_image = cast("nibabel.nifti1.Nifti1Image", nibabel.load(str(output_path)))
    saved = np.asarray(saved_image.get_fdata(), dtype=np.uint8)
    corners = wedge_roi_corners()
    assert set(np.unique(saved)) == {0, *range(1, len(corners) + 1)}

    inverse_affine = np.linalg.inv(np.asarray(image.affine, dtype=np.float64))
    for order_index, corner in enumerate(corners, start=1):
        voxel = inverse_affine @ np.array([*corner.point_mm, 1.0], dtype=np.float64)
        i, j, k = np.rint(voxel[:3]).astype(int)
        assert saved[i, j, k] == order_index


def test_isolation_helpers_have_no_module_level_itk_import() -> None:
    """Parent-side isolation modules must not import ITK at module level."""
    for module_name in (
        "spirit_phantom.core.registration_constants",
        "spirit_phantom.core.guarded_registration",
    ):
        spec = find_spec(module_name)
        assert spec is not None
        assert spec.origin is not None
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
                assert "itk" not in names
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module != "itk"
                assert not node.module.startswith("itk.")
                assert node.module != "spirit_phantom.core.registration"


def test_wedge_roi_corners_has_two_rectangles() -> None:
    """Atlas wedge corners should provide four points for each of two wedges."""
    corners = wedge_roi_corners()
    assert len(corners) == 8
    assert {corner.label for corner in corners} == {1, 2}
    wedge_1 = [corner for corner in corners if corner.label == 1]
    assert wedge_1[0].point_mm == (-25.0, 0.0, 0.0)
    assert wedge_1[1].point_mm == (25.0, 0.0, 0.0)


def test_save_wedge_profile_diagnostics_writes_csv_and_png(tmp_path: Path) -> None:
    """Diagnostic export should write CSV curve data and a PNG plot."""
    n_samples = 41
    n_lines = 3
    erf = np.zeros((n_samples, n_lines), dtype=np.float64)
    erf[0:10, :] = 0.0
    for index in range(10, 30):
        erf[index, :] = float(index - 10)
    erf[30:, :] = 20.0
    # Mild line-to-line noise so per-line columns are distinct.
    erf[:, 1] += 0.5
    erf[:, 2] -= 0.5

    pixel_size_mm = 1.0
    thickness_mm = nema_slice_thickness(erf, pixel_size_mm, 15.0)
    diagnostics = build_wedge_profile_diagnostics(
        label=1,
        erf_profiles=erf,
        pixel_size_mm=pixel_size_mm,
        ramp_slope_degrees=15.0,
        thickness_mm=float(thickness_mm),
    )
    csv_path, plot_path = save_wedge_profile_diagnostics(
        diagnostics=diagnostics,
        output_directory=tmp_path,
    )
    assert csv_path.is_file()
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0

    text = csv_path.read_text(encoding="utf-8")
    assert "erf_mean" in text
    assert "slice_profile_mean" in text
    assert "thickness_mm" in text
    assert text.count("\n") >= n_samples  # header + rows


def test_sample_rectangle_edge_response_recovers_known_thickness() -> None:
    """A synthetic ramp along X should yield the expected NEMA thickness."""
    # 1 mm isotropic identity affine: voxel index == world mm.
    shape = (81, 11, 5)
    data = np.zeros(shape, dtype=np.float64)
    # Plateau / ramp / plateau along x at y in [3, 7], z = 2.
    data[0:20, 3:8, 2] = 0.0
    for x in range(20, 41):
        data[x, 3:8, 2] = float(x - 20)
    data[41:81, 3:8, 2] = 20.0
    affine = np.eye(4)
    image = nibabel.Nifti1Image(dataobj=data, affine=affine)

    corners = (
        (10.0, 3.0, 2.0),
        (50.0, 3.0, 2.0),
        (50.0, 7.0, 2.0),
        (10.0, 7.0, 2.0),
    )
    profiles, pixel_size = sample_rectangle_edge_response(
        image=image,
        corners_mm=corners,
        n_lines=5,
        samples_per_line=41,
    )
    assert profiles.shape == (41, 5)
    assert pixel_size == pytest.approx(1.0)

    # Differentiating the unit ramp yields ~20 samples at half-max crossings.
    thickness = nema_slice_thickness(profiles, pixel_size, 15.0)
    expected = 20.0 * math.tan(math.radians(15.0))
    assert thickness == pytest.approx(expected, rel=0.2)

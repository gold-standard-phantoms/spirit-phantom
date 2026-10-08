"""Tests for fixed-space wedge sampling used by slice-thickness analysis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import nibabel
import numpy as np
import pytest

from spirit_phantom.core.generate_slice_mask import (
    save_slice_mask,
    wedge_roi_corners,
)
from spirit_phantom.core.slice_thickness import nema_slice_thickness
from spirit_phantom.core.slice_thickness_analysis import sample_rectangle_edge_response

if TYPE_CHECKING:
    from pathlib import Path


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


def test_wedge_roi_corners_has_two_rectangles() -> None:
    """Atlas wedge corners should provide four points for each of two wedges."""
    corners = wedge_roi_corners()
    assert len(corners) == 8
    assert {corner.label for corner in corners} == {1, 2}
    wedge_1 = [corner for corner in corners if corner.label == 1]
    assert wedge_1[0].point_mm == (-25.0, 0.0, 0.0)
    assert wedge_1[1].point_mm == (25.0, 0.0, 0.0)


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

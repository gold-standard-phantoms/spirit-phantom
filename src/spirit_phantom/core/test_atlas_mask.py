"""Tests for full-label atlas-to-scan mask mapping."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import nibabel
import numpy as np

from spirit_phantom.core.atlas_mask import (
    coordinate_mapped_atlas_mask,
    default_mapped_atlas_mask_path,
    nifti_path_stem,
)


def test_nifti_path_stem_strips_compound_suffix() -> None:
    """``.nii.gz`` should be treated as a single suffix."""
    assert nifti_path_stem(path=Path("transformed_component_atlas.nii.gz")) == (
        "transformed_component_atlas"
    )
    assert nifti_path_stem(path=Path("scan.nii")) == "scan"


def test_default_mapped_atlas_mask_path_uses_registration_dir_and_both_stems(
    tmp_path: Path,
) -> None:
    """Default output should sit beside the atlas and name both inputs."""
    registration_dir = tmp_path / "registered_data" / "20261006_194157"
    atlas_path = registration_dir / "transformed_component_atlas.nii.gz"
    scan_name = (
        "SPIRIT_32ch_sess1_am_cold_12_FLASH_1p0mm_withGrappa_2_NoseInFeetDir_"
        "20251103083536_FLASH_1p0mm_withGrappa_2_NoseInFeetDir.nii.gz"
    )
    scan_path = tmp_path / "mri_data" / scan_name

    expected = registration_dir / (
        "mapped_atlas_mask__transformed_component_atlas__"
        "SPIRIT_32ch_sess1_am_cold_12_FLASH_1p0mm_withGrappa_2_NoseInFeetDir_"
        "20251103083536_FLASH_1p0mm_withGrappa_2_NoseInFeetDir.nii.gz"
    )
    assert (
        default_mapped_atlas_mask_path(
            registered_component_atlas_image_path=atlas_path,
            scan_image_path=scan_path,
        )
        == expected
    )


def test_coordinate_mapped_atlas_mask_preserves_all_labels(
    tmp_path: Path,
) -> None:
    """Mapped atlas mask should preserve all labels."""
    atlas_data = np.zeros(shape=(5, 5, 1), dtype=np.uint8)
    atlas_data[1, 1, 0] = np.uint8(1)
    atlas_data[2, 2, 0] = np.uint8(7)
    atlas_data[3, 3, 0] = np.uint8(21)
    atlas_data[4, 4, 0] = np.uint8(22)
    atlas_image = nibabel.Nifti1Image(dataobj=atlas_data, affine=np.eye(4))
    atlas_path = tmp_path / "atlas.nii.gz"
    nibabel.save(atlas_image, str(atlas_path))

    scan_data = np.zeros(shape=(5, 5, 1), dtype=np.float32)
    scan_image = nibabel.Nifti1Image(dataobj=scan_data, affine=np.eye(4))
    scan_path = tmp_path / "scan.nii.gz"
    nibabel.save(scan_image, str(scan_path))

    output_mask_path = tmp_path / "mapped_mask.nii.gz"
    saved_mask_path = coordinate_mapped_atlas_mask(
        registered_component_atlas_image_path=atlas_path,
        scan_image_path=scan_path,
        output_mask_image_path=output_mask_path,
    )

    saved_mask_image = cast(
        "nibabel.nifti1.Nifti1Image", nibabel.load(str(saved_mask_path))
    )
    saved_mask = np.asarray(saved_mask_image.get_fdata(), dtype=np.uint8)
    assert set(np.unique(saved_mask).tolist()) == {0, 1, 7, 21, 22}
    assert int(saved_mask[1, 1, 0]) == 1
    assert int(saved_mask[2, 2, 0]) == 7
    assert int(saved_mask[3, 3, 0]) == 21
    assert int(saved_mask[4, 4, 0]) == 22

    unexpected_plot_path = tmp_path / "mapped_mask_sad_filter_plot.png"
    unexpected_intermediate_path = tmp_path / "mapped_mask_mapped_label_mask.nii.gz"
    assert not unexpected_plot_path.exists()
    assert not unexpected_intermediate_path.exists()


def test_coordinate_mapped_atlas_mask_default_path_beside_atlas(
    tmp_path: Path,
) -> None:
    """When no output path is given, write beside the registered atlas."""
    registration_dir = tmp_path / "registered_data" / "20261006_194157"
    registration_dir.mkdir(parents=True)
    mri_dir = tmp_path / "mri_data"
    mri_dir.mkdir()

    atlas_data = np.zeros(shape=(5, 5, 1), dtype=np.uint8)
    atlas_data[1, 1, 0] = np.uint8(1)
    atlas_image = nibabel.Nifti1Image(dataobj=atlas_data, affine=np.eye(4))
    atlas_path = registration_dir / "transformed_component_atlas.nii.gz"
    nibabel.save(atlas_image, str(atlas_path))

    scan_data = np.zeros(shape=(5, 5, 1), dtype=np.float32)
    scan_image = nibabel.Nifti1Image(dataobj=scan_data, affine=np.eye(4))
    scan_path = mri_dir / "clinical_scan.nii.gz"
    nibabel.save(scan_image, str(scan_path))

    saved_mask_path = coordinate_mapped_atlas_mask(
        registered_component_atlas_image_path=atlas_path,
        scan_image_path=scan_path,
    )

    expected_path = (
        registration_dir
        / "mapped_atlas_mask__transformed_component_atlas__clinical_scan.nii.gz"
    )
    assert saved_mask_path == expected_path
    assert saved_mask_path.exists()
    assert not (
        mri_dir / "mapped_atlas_mask__transformed_component_atlas__clinical_scan.nii.gz"
    ).exists()

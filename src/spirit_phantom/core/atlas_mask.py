"""Full-label atlas-to-scan mask mapping (no thermometry-specific processing)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import nibabel

from spirit_phantom.io.atlas_transfer import (
    save_labelled_mask_image,
    transfer_atlas_labels_to_image_space,
)

if TYPE_CHECKING:
    from pathlib import Path


def coordinate_mapped_atlas_mask(
    *,
    registered_component_atlas_image_path: Path,
    scan_image_path: Path,
    output_mask_image_path: Path | None = None,
) -> Path:
    """Map all atlas labels onto a scan image and save the mapped mask.

    Args:
        registered_component_atlas_image_path: Path to the registered component atlas NIfTI.
        scan_image_path: Path to the target scan NIfTI image.
        output_mask_image_path: Optional explicit output mask path.

    Returns:
        Path to the saved segmentation mask NIfTI file.
    """
    registered_atlas_image = nibabel.nifti1.load(
        filename=str(registered_component_atlas_image_path)
    )
    scan_image = nibabel.nifti1.load(filename=str(scan_image_path))

    mapped_labelled_mask = transfer_atlas_labels_to_image_space(
        atlas_image=registered_atlas_image,
        target_image=scan_image,
    )

    if output_mask_image_path is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_mask_image_path = (
            scan_image_path.parent / f"mapped_atlas_mask_{timestamp}.nii.gz"
        )
    output_mask_image_path.parent.mkdir(parents=True, exist_ok=True)
    save_labelled_mask_image(
        mask=mapped_labelled_mask,
        reference_image=scan_image,
        output_image_path=output_mask_image_path,
        description="Mapped atlas labels",
    )
    return output_mask_image_path

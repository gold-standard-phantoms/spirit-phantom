"""Full-label atlas-to-scan mask mapping (no thermometry-specific processing)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel

from spirit_phantom.io.atlas_transfer import (
    save_labelled_mask_image,
    transfer_atlas_labels_to_image_space,
)

if TYPE_CHECKING:
    from pathlib import Path


def nifti_path_stem(*, path: Path) -> str:
    """Return the filename stem, treating ``.nii.gz`` as a single suffix.

    Args:
        path: Path to a NIfTI (or other) file.

    Returns:
        Filename without the final ``.nii.gz`` or single-suffix extension.
    """
    if path.name.endswith(".nii.gz"):
        return path.name[: -len(".nii.gz")]
    return path.stem


def default_mapped_atlas_mask_path(
    *,
    registered_component_atlas_image_path: Path,
    scan_image_path: Path,
) -> Path:
    """Build the default mapped-mask path beside the registration atlas.

    The output sits in the same directory as
    ``registered_component_atlas_image_path`` (typically the registration
    output folder) and encodes both input stems:

    ``mapped_atlas_mask__{atlas_stem}__{scan_stem}.nii.gz``

    Args:
        registered_component_atlas_image_path: Registered component atlas path.
        scan_image_path: Target scan path whose grid receives the labels.

    Returns:
        Default output path for the mapped atlas mask.
    """
    atlas_stem = nifti_path_stem(path=registered_component_atlas_image_path)
    scan_stem = nifti_path_stem(path=scan_image_path)
    filename = f"mapped_atlas_mask__{atlas_stem}__{scan_stem}.nii.gz"
    return registered_component_atlas_image_path.parent / filename


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
        output_mask_image_path: Optional explicit output mask path. When omitted,
            the mask is written beside the registered atlas using
            :func:`default_mapped_atlas_mask_path`.

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
        output_mask_image_path = default_mapped_atlas_mask_path(
            registered_component_atlas_image_path=registered_component_atlas_image_path,
            scan_image_path=scan_image_path,
        )
    output_mask_image_path.parent.mkdir(parents=True, exist_ok=True)
    save_labelled_mask_image(
        mask=mapped_labelled_mask,
        reference_image=scan_image,
        output_image_path=output_mask_image_path,
        description="Mapped atlas labels",
    )
    return output_mask_image_path

"""Validate default SPIRIT atlas downloads and NIfTI integrity.

This script downloads (or reuses cached) default SPIRIT atlas resources and
performs lightweight integrity checks using nibabel.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, cast

import nibabel as nib
import numpy as np

from spirit_phantom import (
    get_default_component_atlas_image_path,
    get_default_register_moving_image_path,
)

if TYPE_CHECKING:
    from pathlib import Path


NiftiImage = nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image


def _validate_nifti_file(*, atlas_path: Path) -> bool:
    """Load and validate a single NIfTI atlas file.

    Args:
        atlas_path: Path to the atlas file on disk.

    Returns:
        `True` when the image looks valid, otherwise `False`.
    """
    is_valid = atlas_path.exists()
    if not is_valid:
        return is_valid

    try:
        loaded_image = nib.load(str(atlas_path))
        is_nifti = isinstance(
            loaded_image, nib.nifti1.Nifti1Image | nib.nifti2.Nifti2Image
        )
        if is_nifti:
            image = cast("NiftiImage", loaded_image)
            data = image.get_fdata(dtype=np.float32)
            affine = np.asarray(image.affine)
            is_valid = (
                data.size > 0
                and data.ndim >= 3
                and bool(np.all(np.isfinite(data)))
                and affine.shape == (4, 4)
                and bool(np.all(np.isfinite(affine)))
            )
        else:
            is_valid = False
    except (OSError, TypeError, ValueError):
        is_valid = False

    return is_valid


def main() -> None:
    """Download/cache the default atlas resources and validate both files."""
    signal_atlas_path = get_default_register_moving_image_path()
    component_atlas_path = get_default_component_atlas_image_path()

    signal_ok = _validate_nifti_file(atlas_path=signal_atlas_path)
    component_ok = _validate_nifti_file(atlas_path=component_atlas_path)

    sys.stdout.write(f"signal atlas: {'OK' if signal_ok else 'FAIL'}\n")
    sys.stdout.write(f"component atlas: {'OK' if component_ok else 'FAIL'}\n")
    if not signal_ok or not component_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

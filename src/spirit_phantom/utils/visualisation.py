"""Module for visualising registration results."""

from pathlib import Path
from typing import Final

import itk
import numpy as np

from spirit_phantom.io.points import load_points

NUM_COORDS: Final[int] = 3


def _load_nifti_as_array(image_path: Path) -> np.ndarray:
    """Load a 3D NIfTI image using ITK and return data as (x, y, z) array.

    The ITK image is converted to a NumPy array with axes reordered so that the
    returned array has shape (x, y, z), matching the voxel coordinate convention
    used for the fiducial points in this module.
    """
    image_itk = itk.imread(str(image_path))
    image_array_itk: np.ndarray = itk.array_view_from_image(image_itk)

    if image_array_itk.ndim != NUM_COORDS:
        msg = (
            "Expected a 3D image when loading "
            f"{image_path}, but received an array with "
            f"{image_array_itk.ndim} dimensions"
        )
        raise ValueError(msg)

    # ITK NumPy views are ordered as (z, y, x); transpose to (x, y, z) so that
    # axial slices can be accessed via array[:, :, z_index].
    image_array_xyz: np.ndarray = np.transpose(image_array_itk, axes=(2, 1, 0))
    return image_array_xyz.astype(np.float32, copy=False)


def _normalise_slice(slice_data: np.ndarray) -> np.ndarray:
    """Normalise a 2D slice to the [0, 1] range."""
    slice_min: float = float(np.nanmin(slice_data))
    slice_max: float = float(np.nanmax(slice_data))
    if slice_max <= slice_min:
        return np.zeros_like(slice_data, dtype=np.float32)
    normalised: np.ndarray = (slice_data - slice_min) / (slice_max - slice_min)
    return normalised.astype(np.float32)


def visualise_checkerboard(
    fixed_image_path: Path,
    registered_image_path: Path,
    slice_indices: list[int],
) -> None:
    """Create and save checkerboard visualisations for fixed and registered images.

    Generates checkerboard overlays of the fixed and registered images on the
    specified axial slice indices using ITK and saves each slice as a PNG image
    in the directory of the registered image.

    Args:
        fixed_image_path: Path to the fixed (reference) NIfTI image.
        registered_image_path: Path to the registered moving image in the
            fixed image space.
        slice_indices: Axial slice indices (z indices) to visualise.
    """
    # Load NIfTI images via ITK.
    fixed_data: np.ndarray = _load_nifti_as_array(image_path=fixed_image_path)
    registered_data: np.ndarray = _load_nifti_as_array(
        image_path=registered_image_path,
    )

    if fixed_data.shape != registered_data.shape:
        msg = (
            "Fixed and registered images must have the same shape for visual "
            f"comparison; got {fixed_data.shape} and {registered_data.shape}"
        )
        raise ValueError(msg)

    n_slices_k: int = fixed_data.shape[2]
    valid_slices: list[int] = [int(k) for k in slice_indices if 0 <= k < n_slices_k]

    if not valid_slices:
        msg = "No fiducial slice indices fall within the fixed image volume."
        raise ValueError(msg)

    output_dir: Path = registered_image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    max_panels: Final[int] = min(5, len(valid_slices))

    # Use the ITK checkerboard filter on the selected slices and save per-slice images.
    for slice_idx in valid_slices[:max_panels]:
        fixed_slice_norm: np.ndarray = _normalise_slice(
            slice_data=fixed_data[:, :, slice_idx],
        )
        registered_slice_norm: np.ndarray = _normalise_slice(
            slice_data=registered_data[:, :, slice_idx],
        )

        # Convert to 8-bit for cleaner image file output.
        fixed_slice_uint8: np.ndarray = (fixed_slice_norm * 255.0).astype(np.uint8)
        registered_slice_uint8: np.ndarray = (registered_slice_norm * 255.0).astype(
            np.uint8
        )

        fixed_slice_itk = itk.image_view_from_array(arr=fixed_slice_uint8)
        registered_slice_itk = itk.image_view_from_array(arr=registered_slice_uint8)

        itk_checkerboard = itk.checker_board_image_filter(
            fixed_slice_itk,
            registered_slice_itk,
            checker_pattern=[8, 8],
        )

        itk_output_path: Path = output_dir / (
            f"checkerboard_itk_slice_{slice_idx:03d}.png"
        )
        itk.imwrite(
            itk_checkerboard,
            str(itk_output_path),
        )


def visualise_checkerboard_tranformix(
    fixed_image_path: Path,
    registered_image_path: Path,
    atlas_points_path: Path,
) -> None:
    """Create and save checkerboard visualisations using transformix output points.

    Uses fiducial centre locations in voxel coordinates to derive unique axial
    slice indices (z) and then calls ``visualise_checkerboard`` to generate
    checkerboard overlays of the fixed and registered images using ITK.

    Args:
        fixed_image_path: Path to the fixed (reference) NIfTI image.
        registered_image_path: Path to the registered moving image in the
            fixed image space.
        atlas_points_path: Path to the text file containing fiducial centre
            coordinates in voxels (x y z per line).

    Raises:
        FileNotFoundError: If any of the provided paths do not exist.
        ValueError: If no valid fiducial points can be parsed from the atlas
            points file.
    """
    if not fixed_image_path.is_file():
        msg = f"Fixed image not found: {fixed_image_path}"
        raise FileNotFoundError(msg)
    if not registered_image_path.is_file():
        msg = f"Registered image not found: {registered_image_path}"
        raise FileNotFoundError(msg)
    if not atlas_points_path.is_file():
        msg = f"Atlas points file not found: {atlas_points_path}"
        raise FileNotFoundError(msg)

    # Load fiducial centres in voxels and identify unique z (slice) indices.
    fiducial_points_list: list[list[float]] = load_points(points_path=atlas_points_path)
    fiducial_points_vx: np.ndarray = np.asarray(
        fiducial_points_list,
        dtype=np.float64,
    )
    z_vx: np.ndarray = fiducial_points_vx[:, 2]
    slice_indices_np: np.ndarray = np.unique(np.round(z_vx).astype(np.intp))
    slice_indices_np.sort()
    slice_indices: list[int] = [int(k) for k in slice_indices_np]
    visualise_checkerboard(
        fixed_image_path=fixed_image_path,
        registered_image_path=registered_image_path,
        slice_indices=slice_indices,
    )

"""Tool for generating a 3d digital object to evaluate the registration."""

import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


def create_synthetic_volume(
    shape: tuple[int, int, int] = (100, 100, 100),
    rectangle_object_shape: tuple[int, int, int] = (50, 40, 30),
) -> np.ndarray:
    """Generate a 3D numpy array with a rectangular feature.

    Creates a volume with a rectangular feature (50x40x30) positioned towards
    the centre of the volume.

    Args:
        shape: The volume created
        rectangle_object_shape: the volume of the object inside the volume
    Returns:
        A 3D numpy array with a rectangular feature in the center.
    """
    volume = np.zeros(shape, dtype=np.float32)
    x_rectangle = (
        (shape[0] - rectangle_object_shape[0]) // 2,
        (shape[0] - rectangle_object_shape[0]) // 2 + rectangle_object_shape[0],
    )
    y_rectangle = (
        (shape[1] - rectangle_object_shape[1]) // 2,
        (shape[1] - rectangle_object_shape[1]) // 2 + rectangle_object_shape[1],
    )
    z_rectangle = (
        (shape[2] - rectangle_object_shape[2]) // 2,
        (shape[2] - rectangle_object_shape[2]) // 2 + rectangle_object_shape[2],
    )
    volume[
        x_rectangle[0] : x_rectangle[1],
        y_rectangle[0] : y_rectangle[1],
        z_rectangle[0] : z_rectangle[1],
    ] = 1.0
    return volume


def apply_translation(
    volume: np.ndarray, translation: tuple[int, int, int]
) -> np.ndarray:
    """Apply matrix-space translation to a numpy array.

    Translates the volume by the specified amounts in each cartesian direction.
    The translation is applied in matrix space (pixel coordinates) using
    scipy.ndimage affine transformation to ensure proper resampling.

    Args:
        volume: The input 3D volume array.
        translation: A tuple of (tx, ty, tz) translation amounts in pixels.

    Returns:
        A translated 3D numpy array with the same shape as the input.
    """
    # Build a homogenous coordinate for the translation
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    # Apply affine transformation
    return affine_transform(
        volume, np.linalg.inv(translation_matrix), output_shape=volume.shape, order=1
    )


def apply_translation_old(
    volume: np.ndarray, translation: tuple[int, int, int]
) -> np.ndarray:
    """Apply matrix-space translation to a numpy array.

    Translates the volume by the specified amounts in each cartesian direction.
    The translation is applied in matrix space (pixel coordinates) using
    scipy.ndimage affine transformation to ensure proper resampling.

    Args:
        volume: The input 3D volume array.
        translation: A tuple of (tx, ty, tz) translation amounts in pixels.

    Returns:
        A translated 3D numpy array with the same shape as the input.
    """
    tx, ty, tz = translation
    # Create translation affine matrix
    # For scipy.ndimage.affine_transform, we need the inverse transform
    # We want: output = input + translation
    # Which is: input = output - translation
    # For affine_transform: input = matrix @ output + offset
    # So: matrix = identity, offset = -translation
    translation_matrix = np.eye(3)
    offset = np.array([-tx, -ty, -tz], dtype=float)

    # Apply affine transformation
    return affine_transform(
        volume, translation_matrix, offset=offset, output_shape=volume.shape, order=1
    )


def apply_rotation(
    volume: np.ndarray, rotation_angles_deg: tuple[float, float, float]
) -> np.ndarray:
    """Apply rotations around each axis and resample the volume.

    Applies rotations around x, y, and z axes in degrees, then resamples
    the volume using affine transformation.

    Args:
        volume: The input 3D volume array.
        rotation_angles_deg: A tuple of (rx, ry, rz) rotation angles in degrees.

    Returns:
        A rotated and resampled 3D numpy array with the same shape as the input.
    """
    rx_deg, ry_deg, rz_deg = rotation_angles_deg
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    # Create transforms to get the image centered at the origin prior to rotating
    centre_x, centre_y, centre_z = [dimension // 2 for dimension in volume.shape]

    translate_to_origin = np.array(
        [
            [1, 0, 0, -centre_x],
            [0, 1, 0, -centre_y],
            [0, 0, 1, -centre_z],
            [0, 0, 0, 1],
        ]
    )

    translate_from_origin = np.array(
        [
            [1, 0, 0, centre_x],
            [0, 1, 0, centre_y],
            [0, 0, 1, centre_z],
            [0, 0, 0, 1],
        ]
    )
    # Create rotation matrices for each axis
    # Rotation around x-axis
    rot_x = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1],
        ]
    )
    # Rotation around y-axis
    rot_y = np.array(
        [
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1],
        ]
    )
    # Rotation around z-axis
    rot_z = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Combined rotation matrix (apply z, then y, then x)
    rotation_matrix = (
        translate_from_origin @ rot_x @ rot_y @ rot_z @ translate_to_origin
    )

    # Create affine transformation matrix
    # For scipy.ndimage.affine_transform, we need the inverse transform
    rotation_matrix_inv = np.linalg.inv(rotation_matrix)

    # Apply affine transformation
    return affine_transform(
        volume, rotation_matrix_inv, output_shape=volume.shape, order=1
    )


def save_nifti_from_array(
    volume: np.ndarray,
    file_path: Path,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Convert numpy array to nibabel NIfTI image and save as NIfTI file.

    Creates a NIfTI image with default affine matrix (identity with appropriate
    spacing) and saves it to the specified path. Optionally applies a translation
    by modifying the affine matrix.

    Args:
        volume: The 3D numpy array to save.
        file_path: The path where the NIfTI file should be saved.
        translation: Optional translation to apply via affine matrix (tx, ty, tz).
    """
    # Create default affine matrix (identity with unit spacing)
    affine = np.eye(4)
    # Apply translation in the affine matrix (last column is translation)
    affine[:3, 3] = translation

    nifti_image = nib.Nifti1Image(volume, affine)
    nib.save(nifti_image, file_path)

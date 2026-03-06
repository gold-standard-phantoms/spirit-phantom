"""Utilities for extracting vial intensity statistics from registered atlases.

The expected flow is:

1. Register an atlas to a scanner image.
2. Use the registered atlas labels as ROIs.
3. Erode each ROI by a configurable number of voxels.
4. Compute mean and standard deviation of the scanner values inside each ROI.

Example:
    rows = compute_vial_statistics(
        registered_atlas_image_path=registered_atlas_path,
        mri_scan_image_path=mri_scan_path,
    )
    print_vial_statistics_table(rows=rows)
    save_vial_statistics_table(rows=rows, output_path=output_path)
"""

from __future__ import annotations

from math import isnan
from typing import TYPE_CHECKING, Any, TextIO, TypeAlias, cast

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

DEFAULT_VIAL_LABELS: tuple[int, ...] = tuple(range(1, 21))
DEFAULT_EROSION_VOXELS = 2
EXPECTED_MASK_DIMENSIONS = 3

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

BoolArray: TypeAlias = NDArray[np.bool_]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int32]
VialStatisticRow: TypeAlias = list[float | int]


def _load_nifti_data(*, image_path: Path) -> NDArray[np.generic]:
    """Load a NIfTI image and return its voxel data array.

    Args:
        image_path: Path to the NIfTI file.

    Returns:
        The voxel data as a NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not image_path.exists():
        msg = f"NIfTI file not found: {image_path}"
        raise FileNotFoundError(msg)

    nifti_image = cast("nib.Nifti1Image", nib.load(str(image_path)))
    return np.asanyarray(nifti_image.dataobj)


def _erode_binary_mask(*, binary_mask: BoolArray, iterations: int) -> BoolArray:
    """Erode a 3D binary mask using a 3x3x3 structuring element.

    Args:
        binary_mask: 3D binary mask to erode.
        iterations: Number of erosion iterations.

    Returns:
        Eroded binary mask.

    Raises:
        ValueError: If the mask is not 3D or iterations is negative.
    """
    if binary_mask.ndim != EXPECTED_MASK_DIMENSIONS:
        msg = (
            f"Mask must be {EXPECTED_MASK_DIMENSIONS}D, "
            f"got {binary_mask.ndim}D."
        )
        raise ValueError(msg)
    if iterations < 0:
        msg = "iterations must be greater than or equal to 0."
        raise ValueError(msg)
    if iterations == 0:
        return binary_mask

    eroded: BoolArray = binary_mask.copy()
    z_size, y_size, x_size = eroded.shape

    for _ in range(iterations):
        if not np.any(eroded):
            break

        padded = np.pad(eroded, pad_width=1, mode="constant", constant_values=False)
        neighbourhood: list[BoolArray] = []
        for z_offset in (-1, 0, 1):
            for y_offset in (-1, 0, 1):
                for x_offset in (-1, 0, 1):
                    z_start = z_offset + 1
                    y_start = y_offset + 1
                    x_start = x_offset + 1
                    neighbourhood.append(
                        padded[
                            z_start : z_start + z_size,
                            y_start : y_start + y_size,
                            x_start : x_start + x_size,
                        ]
                    )

        stacked_neighbourhood = np.stack(neighbourhood, axis=0)
        eroded = np.all(stacked_neighbourhood, axis=0)

    return eroded


def compute_vial_statistics(
    *,
    registered_atlas_image_path: Path,
    mri_scan_image_path: Path,
    labels: tuple[int, ...] = DEFAULT_VIAL_LABELS,
    erosion_voxels: int = DEFAULT_EROSION_VOXELS,
) -> list[VialStatisticRow]:
    """Compute vial statistics from a registered atlas image and MRI scan.

    For each label, a binary ROI mask is created from the registered atlas,
    optionally eroded, and used to extract scanner voxel values.

    Args:
        registered_atlas_image_path: Path to the registered atlas NIfTI image.
        mri_scan_image_path: Path to the MRI scanner NIfTI image.
        labels: Label values to analyse. Defaults to labels 1..20.
        erosion_voxels: Number of binary erosion iterations to apply to each ROI.

    Returns:
        A list of rows where each row is:
        ``[label, mean_intensity, std_intensity, voxel_count]``.

        If a label has no voxels after erosion, mean and standard deviation are
        returned as ``nan`` and voxel count is ``0``.

    Raises:
        ValueError: If image dimensions mismatch or erosion parameters are invalid.
        FileNotFoundError: If either input NIfTI path does not exist.
    """
    atlas_raw = _load_nifti_data(image_path=registered_atlas_image_path)
    mri_raw = _load_nifti_data(image_path=mri_scan_image_path)

    atlas_data: IntArray = np.rint(atlas_raw).astype(np.int32)
    mri_data: FloatArray = np.asarray(mri_raw, dtype=np.float64)

    if atlas_data.shape != mri_data.shape:
        msg = (
            "Registered atlas and MRI scan must have the same shape. "
            f"Got atlas shape {atlas_data.shape} and MRI shape {mri_data.shape}."
        )
        raise ValueError(msg)
    if erosion_voxels < 0:
        msg = "erosion_voxels must be greater than or equal to 0."
        raise ValueError(msg)

    rows: list[VialStatisticRow] = []
    for label in labels:
        roi_mask = atlas_data == label
        eroded_mask = _erode_binary_mask(binary_mask=roi_mask, iterations=erosion_voxels)
        voxel_count = int(np.count_nonzero(eroded_mask))

        if voxel_count == 0:
            rows.append([label, float("nan"), float("nan"), 0])
            continue

        roi_values = mri_data[eroded_mask]
        rows.append(
            [
                label,
                float(np.mean(roi_values)),
                float(np.std(roi_values, ddof=0)),
                voxel_count,
            ]
        )

    return rows


def _unpack_vial_row(*, row: Sequence[Any]) -> tuple[int, float, float, int]:
    """Unpack a vial statistics row into strongly typed values.

    Args:
        row: Single statistics row in the form
            ``[label, mean_intensity, std_intensity, voxel_count]``.

    Returns:
        A tuple containing label, mean intensity, standard deviation, and voxel count.
    """
    return int(row[0]), float(row[1]), float(row[2]), int(row[3])


def _format_float_cell(*, value: float, decimals: int = 6) -> str:
    """Format a floating-point cell for display.

    Args:
        value: Floating-point value to format.
        decimals: Number of decimal places.

    Returns:
        Formatted string for table output.
    """
    if isnan(value):
        return "nan"
    return f"{value:.{decimals}f}"


def format_vial_statistics_table(*, rows: Sequence[Any]) -> str:
    """Format vial statistics as a neat table-like string.

    Args:
        rows: Vial statistics rows returned by ``compute_vial_statistics``.

    Returns:
        Multi-line table string suitable for CLI output.
    """
    headers = ("Label", "Mean intensity", "Std intensity", "Voxel count")
    table_rows = [
        (
            str(label),
            _format_float_cell(value=mean_intensity),
            _format_float_cell(value=std_intensity),
            str(voxel_count),
        )
        for (label, mean_intensity, std_intensity, voxel_count) in (
            _unpack_vial_row(row=row) for row in rows
        )
    ]

    if not table_rows:
        return "No vial statistics were returned."

    column_widths: list[int] = []
    for index in range(len(headers)):
        header_width = len(headers[index])
        row_widths = (len(row[index]) for row in table_rows)
        column_widths.append(max(header_width, *row_widths))

    def _format_line(*, cells: tuple[str, str, str, str]) -> str:
        return " | ".join(
            cells[index].ljust(column_widths[index]) for index in range(len(cells))
        )

    separator = "-+-".join("-" * width for width in column_widths)
    lines = [_format_line(cells=headers), separator]
    lines.extend(_format_line(cells=row) for row in table_rows)
    return "\n".join(lines)


def save_vial_statistics_table(*, rows: Sequence[Any], output_path: Path) -> Path:
    """Save formatted vial statistics as UTF-8 text.

    Args:
        rows: Vial statistics rows returned by ``compute_vial_statistics``.
        output_path: Path to the output text file.

    Returns:
        The same ``output_path`` for convenient chaining.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_text = format_vial_statistics_table(rows=rows)
    output_path.write_text(f"{table_text}\n", encoding="utf-8")
    return output_path


def print_vial_statistics_table(
    *,
    rows: Sequence[Any],
    file: TextIO | None = None,
) -> None:
    """Print vial statistics in a neat table-like console layout.

    Args:
        rows: Vial statistics rows returned by ``compute_vial_statistics``.
        file: Optional text stream. Defaults to standard output when ``None``.
    """
    print(format_vial_statistics_table(rows=rows), file=file)

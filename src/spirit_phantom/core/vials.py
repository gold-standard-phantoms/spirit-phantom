"""Utilities for extracting vial intensity statistics from registered atlases.

The expected flow is:

1. Register an atlas to a scanner image.
2. Use the registered atlas labels as ROIs.
3. Erode each ROI by a configurable number of voxels (default: 0).
    Erosion is generally recommended because it reduces edge artefacts and small
    registration inaccuracies near vial boundaries. The best erosion depth depends
    on image resolution and analysis goals, so the final choice is left to the user.
4. Compute mean and standard deviation of the scanner values inside each ROI.

Example of the post-registration steps to produce a detailed table of vial statistics:
    detailed_rows = compute_vial_statistics_details(
        registered_atlas_image_path=registered_atlas_path,
        mri_scan_image_path=mri_scan_path,
        erosion_voxels=0,
    )
    print_vial_statistics_details_table(rows=detailed_rows)
    save_vial_statistics_details_table(
        rows=detailed_rows,
        output_path=output_path,
    )


Note, an internal function `_compute_vial_statistics` is used to compute the vial statistics
indexed using values that are convenient for modelling and design. These are re-mapped to the
vial order as shown in the Instructions for Use PD-3011-0007(1.0) Section Device Description.
"""

from __future__ import annotations

import json
from math import isnan
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import itk
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

DEFAULT_VIAL_LABELS: tuple[int, ...] = tuple(range(1, 21))
DEFAULT_EROSION_VOXELS = 0
EXPECTED_MASK_DIMENSIONS = 3
VIAL_CONFIG_PATH = Path(__file__).parent / "configuration" / "vial-configurations.json"
EXPECTED_VIAL_COUNT = len(DEFAULT_VIAL_LABELS)

if TYPE_CHECKING:
    from collections.abc import Sequence

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
VialStatisticRow = list[float | int]


class DetailedVialStatistic(BaseModel):
    """Detailed per-vial statistics with metadata from configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vial_id: str
    product_code: str
    description: str
    mean_intensity: float
    stdev: float
    number_of_voxels: int


class VialConfiguration(BaseModel):
    """Container for a single vial configuration record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vial_id: str
    type: str
    product_code: str
    description: str
    vial_size: str
    si_traceable_characterisation_available: list[str]
    segment_index: int


class _VialConfigurationPayload(BaseModel):
    """Payload model for vial configuration JSON structure."""

    model_config = ConfigDict(extra="forbid")

    vials: list[VialConfiguration]


def _validate_vial_configurations(
    *, configurations: list[VialConfiguration]
) -> list[VialConfiguration]:
    """Validate vial configuration uniqueness constraints.

    Args:
        configurations: Parsed vial configuration rows.

    Returns:
        The original ``configurations`` list when validation succeeds.

    Raises:
        ValueError: If vial IDs or segment indices are duplicated.
    """
    seen_vial_ids: set[str] = set()
    seen_segment_indices: set[int] = set()
    for configuration in configurations:
        vial_id = configuration.vial_id
        segment_index = int(configuration.segment_index)

        if vial_id in seen_vial_ids:
            msg = f"Duplicate vial_id '{vial_id}' found in vial configuration."
            raise ValueError(msg)
        if segment_index in seen_segment_indices:
            msg = (
                "Duplicate segment_index "
                f"'{segment_index}' found in vial configuration."
            )
            raise ValueError(msg)

        seen_vial_ids.add(vial_id)
        seen_segment_indices.add(segment_index)

    return configurations


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

    image_itk = itk.imread(str(image_path))
    image_array: np.ndarray = itk.array_view_from_image(image_itk)
    return np.asanyarray(image_array)


def _load_vial_configurations() -> list[VialConfiguration]:
    """Load vial configuration records from JSON.

    Args:
    Returns:
        Parsed list of vial configuration records.

    Raises:
        TypeError: If the JSON structure has invalid types.
        ValueError: If required fields are missing or duplicate keys are present.
        FileNotFoundError: If the configuration file does not exist.
    """
    raw_payload: Any = json.loads(VIAL_CONFIG_PATH.read_text(encoding="utf-8"))
    try:
        payload = _VialConfigurationPayload.model_validate(raw_payload)
    except ValidationError as exc:
        msg = "Invalid vial configuration JSON structure."
        raise ValueError(msg) from exc

    return _validate_vial_configurations(configurations=payload.vials)


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
        msg = f"Mask must be {EXPECTED_MASK_DIMENSIONS}D, got {binary_mask.ndim}D."
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


def _compute_vial_statistics(
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
        erosion_voxels: Number of binary erosion iterations to apply to each ROI. Defaults to 0.

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
        eroded_mask = _erode_binary_mask(
            binary_mask=roi_mask, iterations=erosion_voxels
        )
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


def compute_vial_statistics_details(
    *,
    registered_atlas_image_path: Path,
    mri_scan_image_path: Path,
    erosion_voxels: int = DEFAULT_EROSION_VOXELS,
) -> list[DetailedVialStatistic]:
    """Compute vial statistics with configuration details in an ordered table.

    Erosion is typically beneficial because it reduces partial-volume and boundary
    effects, but the appropriate level depends on resolution and use case.

    Args:
        registered_atlas_image_path: Path to the registered atlas NIfTI image.
        mri_scan_image_path: Path to the MRI scanner NIfTI image.
        erosion_voxels: Number of binary erosion iterations to apply to each ROI.
            Defaults to ``0`` (no erosion). A non-zero value is often preferred to
            reduce boundary artefacts, but should be chosen by the user.

    Returns:
        Detailed rows ordered by vial ID (A to T).
    """
    configurations = _validate_vial_configurations(
        configurations=_load_vial_configurations()
    )
    ordered_configurations = sorted(configurations, key=lambda item: item.vial_id)
    labels = tuple(
        sorted(
            int(configuration.segment_index) for configuration in ordered_configurations
        )
    )

    raw_rows = _compute_vial_statistics(
        registered_atlas_image_path=registered_atlas_image_path,
        mri_scan_image_path=mri_scan_image_path,
        labels=labels,
        erosion_voxels=erosion_voxels,
    )
    label_to_statistics: dict[int, tuple[float, float, int]] = {
        int(row[0]): (float(row[1]), float(row[2]), int(row[3])) for row in raw_rows
    }

    detailed_rows: list[DetailedVialStatistic] = []
    for configuration in ordered_configurations:
        mean_intensity, std_intensity, voxel_count = label_to_statistics.get(
            int(configuration.segment_index),
            (float("nan"), float("nan"), 0),
        )
        detailed_rows.append(
            DetailedVialStatistic(
                vial_id=configuration.vial_id,
                product_code=configuration.product_code,
                description=configuration.description,
                mean_intensity=mean_intensity,
                stdev=std_intensity,
                number_of_voxels=voxel_count,
            )
        )

    return detailed_rows


def format_vial_statistics_details_table(
    *, rows: Sequence[DetailedVialStatistic]
) -> str:
    """Format detailed vial statistics as a neat table-like string.

    Args:
        rows: Detailed vial statistics from ``compute_vial_statistics_details``.

    Returns:
        Multi-line table string suitable for CLI output.
    """
    headers = (
        "Vial ID",
        "Product Code",
        "Description",
        "Mean Intensity",
        "Stdev",
        "Number of Voxels",
    )
    table_rows = [
        (
            row.vial_id,
            row.product_code,
            row.description,
            _format_float_cell(value=row.mean_intensity),
            _format_float_cell(value=row.stdev),
            str(row.number_of_voxels),
        )
        for row in rows
    ]

    if not table_rows:
        return "No vial statistics were returned."

    column_widths: list[int] = []
    for index in range(len(headers)):
        header_width = len(headers[index])
        row_widths = (len(row[index]) for row in table_rows)
        column_widths.append(max(header_width, *row_widths))

    def _format_line(*, cells: tuple[str, str, str, str, str, str]) -> str:
        return " | ".join(
            cells[index].ljust(column_widths[index]) for index in range(len(cells))
        )

    separator = "-+-".join("-" * width for width in column_widths)
    lines = [_format_line(cells=headers), separator]
    lines.extend(_format_line(cells=row) for row in table_rows)
    return "\n".join(lines)


def save_vial_statistics_details_table(
    *, rows: Sequence[DetailedVialStatistic], output_path: Path
) -> Path:
    """Save formatted detailed vial statistics as UTF-8 text.

    Args:
        rows: Detailed vial statistics from ``compute_vial_statistics_details``.
        output_path: Path to the output text file.

    Returns:
        The same ``output_path`` for convenient chaining.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_text = format_vial_statistics_details_table(rows=rows)
    output_path.write_text(f"{table_text}\n", encoding="utf-8")
    return output_path


def print_vial_statistics_details_table(
    *,
    rows: Sequence[DetailedVialStatistic],
    file: TextIO | None = None,
) -> None:
    """Print detailed vial statistics in a neat table-like console layout.

    Args:
        rows: Detailed vial statistics from ``compute_vial_statistics_details``.
        file: Optional text stream. Defaults to standard output when ``None``.
    """
    print(format_vial_statistics_details_table(rows=rows), file=file)


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


def generate_dice_score_table(
    *,
    manual_segmentation_image_path: Path,
    registered_atlas_image_path: Path,
) -> list[dict[str, int | str | float]]:
    """Generate per-vial Dice scores using manual-to-atlas vial remapping.

    The manual segmentation labels are expected to use vial order A..T as
    intensities 1..20. The registered atlas uses segment indices from vial
    configuration, so each manual label is remapped via ``vial_id`` before Dice
    calculation.

    Args:
        manual_segmentation_image_path: Path to manual segmentation NIfTI.
        registered_atlas_image_path: Path to registered atlas NIfTI.

    Returns:
        Per-vial rows containing vial metadata and Dice components:
        ``vial_id``, ``manual_label``, ``atlas_label``, ``dice_score``,
        ``manual_voxels``, ``atlas_voxels``, and ``intersection_voxels``.

    Raises:
        ValueError: If image dimensions mismatch or vial mapping is invalid.
        FileNotFoundError: If either input NIfTI path does not exist.
    """
    manual_raw = _load_nifti_data(image_path=manual_segmentation_image_path)
    atlas_raw = _load_nifti_data(image_path=registered_atlas_image_path)

    manual_data: IntArray = np.rint(manual_raw).astype(np.int32)
    atlas_data: IntArray = np.rint(atlas_raw).astype(np.int32)

    if manual_data.shape != atlas_data.shape:
        msg = (
            "Manual segmentation and registered atlas must have the same shape. "
            f"Got manual shape {manual_data.shape} and atlas shape {atlas_data.shape}."
        )
        raise ValueError(msg)

    configurations = _validate_vial_configurations(
        configurations=_load_vial_configurations()
    )
    ordered_configurations = sorted(configurations, key=lambda item: item.vial_id)

    if len(ordered_configurations) < EXPECTED_VIAL_COUNT:
        msg = (
            "Vial configuration must include at least "
            f"{EXPECTED_VIAL_COUNT} rows to map manual labels "
            f"{DEFAULT_VIAL_LABELS[0]}..{DEFAULT_VIAL_LABELS[-1]} to atlas segment "
            "indices."
        )
        raise ValueError(msg)

    rows: list[dict[str, int | str | float]] = []
    for manual_label in DEFAULT_VIAL_LABELS:
        configuration = ordered_configurations[manual_label - 1]
        atlas_label = int(configuration.segment_index)

        manual_mask = manual_data == manual_label
        atlas_mask = atlas_data == atlas_label

        manual_voxels = int(np.count_nonzero(manual_mask))
        atlas_voxels = int(np.count_nonzero(atlas_mask))
        intersection_voxels = int(np.count_nonzero(manual_mask & atlas_mask))
        denominator = manual_voxels + atlas_voxels
        dice_score = (
            float("nan")
            if denominator == 0
            else float((2.0 * intersection_voxels) / denominator)
        )

        rows.append(
            {
                "vial_id": configuration.vial_id,
                "manual_label": manual_label,
                "atlas_label": atlas_label,
                "dice_score": dice_score,
                "manual_voxels": manual_voxels,
                "atlas_voxels": atlas_voxels,
                "intersection_voxels": intersection_voxels,
            }
        )

    return rows

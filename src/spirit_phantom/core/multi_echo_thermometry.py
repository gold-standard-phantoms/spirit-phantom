"""Multi-echo ethylene glycol thermometry.

Mask generation maps atlas labels 21 and 22 onto a multi-echo GRE scan and
keeps voxels whose echo train has a large sum of absolute differences:

    SAD = sum(abs(S[n] - S[n - 1])) for n in 1..N

Temperature estimation packages that mask with user-supplied echo-time files
and fits the dual-resonance model in qmri. Every series must share the first
series' spatial shape and affine, which is the grid qmri fits as one echo train.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import nibabel
import numpy as np
from pydantic import BaseModel, ConfigDict

from spirit_phantom.io.atlas_transfer import transfer_atlas_labels_to_image_space
from spirit_phantom.utils.binary_morphology import binary_dilation_connectivity_one

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qmri.thermometry.multiecho import RegionThermometryResult

ThermometryAnalysisMethod = Literal[
    "regionwise",
    "voxelwise",
    "regionwise_bootstrap",
]

ETHYLENE_GLYCOL_LABELS: tuple[int, int] = (21, 22)
_MIN_THERMOMETRY_NDIM_FOR_TIME_AXIS: int = 4
_TEMPERATURE_MAP_NDIM: int = 3
_EG_MASK_DILATION_ITERATIONS: int = 1
_EG_MASK_MIN_SAD_COUNTS: float = 1000.0
_DIAGNOSTIC_PLOT_DPI: int = 160
_SIGNAL_CURVE_SAMPLES: int = 2000
_ECHO_TIME_MATCH_DECIMALS: int = 9
_SECONDS_TO_MILLISECONDS: float = 1000.0
_TRAIN_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")
_VIAL_PLOT_COLOURS: dict[str, str] = {"U": "#cc4530", "V": "#30a0cc"}
_DEFAULT_VIAL_COLOUR: str = "#7f7f7f"
DEFAULT_MAGNETIC_FIELD_TESLA: float = 3.0
_TEMPERATURE_OUTPUT_PREFIX: str = "eg_thermometry"
_TEMPERATURE_MEASUREMENTS_FILENAME: str = "temperature_measurements.txt"
ETHYLENE_GLYCOL_MASK_FILENAME: str = "ethylene_glycol_mask.nii.gz"
_FREQUENCY_DIFFERENCE_PARAMETER_INDEX: int = 4
_DUAL_RESONANCE_PARAMETER_COUNT: int = 6
_FITTED_PARAMETER_NDIM: int = 2
_BOOTSTRAP_SAMPLES: int = 100
_THERMOMETRY_ANALYSIS_METHODS: frozenset[str] = frozenset(
    ("regionwise", "voxelwise", "regionwise_bootstrap")
)


def _get_mask_output_stem(*, output_mask_image_path: Path) -> str:
    """Return the output mask filename stem with ``.nii.gz`` handled."""
    if output_mask_image_path.name.endswith(".nii.gz"):
        return output_mask_image_path.name[: -len(".nii.gz")]
    return output_mask_image_path.stem


def dilate_segmentation_mask(
    mask: np.ndarray,
    *,
    iterations: int,
) -> np.ndarray:
    """Apply binary dilation to a label or mask array.

    Args:
        mask: Non-zero entries treated as foreground (2D or 3D).
        iterations: Number of dilation steps (voxel layers added per step).
            A value of ``0`` leaves the original binary mask unchanged.

    Returns:
        ``uint8`` array of the same shape as ``mask`` with values 0 or 1.
    """
    if iterations < 0:
        msg = "iterations must be greater than or equal to 0."
        raise ValueError(msg)

    foreground = np.asarray(mask) != 0
    if iterations == 0:
        return np.asarray(foreground, dtype=np.uint8)

    dilated = binary_dilation_connectivity_one(
        mask=foreground,
        iterations=iterations,
    )
    return np.asarray(dilated, dtype=np.uint8)


def _get_mask_output_plot_path(*, output_mask_image_path: Path) -> Path:
    """Build the diagnostic plot path alongside the saved mask image."""
    mask_stem = _get_mask_output_stem(output_mask_image_path=output_mask_image_path)
    return output_mask_image_path.parent / f"{mask_stem}_sad_filter_plot.png"


def _get_output_mapped_label_mask_path(*, output_mask_image_path: Path) -> Path:
    """Build the mapped-label mask path alongside the saved final mask image."""
    mask_stem = _get_mask_output_stem(output_mask_image_path=output_mask_image_path)
    return output_mask_image_path.parent / f"{mask_stem}_mapped_label_mask.nii.gz"


def _save_labelled_mask_image(
    *,
    mask: np.ndarray,
    reference_image: nibabel.nifti1.Nifti1Image,
    output_image_path: Path,
    description: str,
) -> None:
    """Save a labelled mask NIfTI with copied affine metadata.

    Args:
        mask: Labelled mask to save.
        reference_image: Image whose affine and header metadata are reused.
        output_image_path: Output NIfTI path.
        description: Short NIfTI header description.
    """
    mask_header = reference_image.header.copy()
    mask_header.set_data_dtype(np.uint8)
    mask_header["descrip"] = np.array(description, dtype="|S80")
    mask_image = nibabel.Nifti1Image(
        dataobj=np.asarray(mask, dtype=np.uint8),
        affine=reference_image.affine,
        header=mask_header,
    )
    mask_image.set_qform(reference_image.affine, code=1)
    mask_image.set_sform(reference_image.affine, code=1)
    nibabel.save(mask_image, str(output_image_path))


def _dilate_labelled_mask(*, labelled_mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate each label independently while preserving label identities."""
    dilated_labelled_mask = np.asarray(labelled_mask, dtype=np.uint8).copy()
    if iterations == 0:
        return dilated_labelled_mask

    unique_labels = np.unique(labelled_mask)
    for label in unique_labels[unique_labels != 0]:
        label_mask = np.asarray(labelled_mask == label, dtype=np.uint8)
        dilated_label_mask = dilate_segmentation_mask(
            mask=label_mask,
            iterations=iterations,
        )
        addable_voxels = np.logical_and(
            dilated_label_mask != 0,
            dilated_labelled_mask == 0,
        )
        dilated_labelled_mask[addable_voxels] = np.uint8(label)
    return dilated_labelled_mask


def _save_sad_filter_plot(
    *,
    included_traces: Sequence[np.ndarray],
    excluded_traces: Sequence[np.ndarray],
    output_plot_path: Path,
    minimum_sad_counts: float,
) -> None:
    """Save a Sum of Absolute Differences (SAD) diagnostic trace plot."""
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    _, axis = plt.subplots(figsize=(10, 6))

    if included_traces:
        for trace in included_traces:
            axis.plot(
                trace,
                color="tab:green",
                alpha=0.20,
                linewidth=1.0,
            )
    if excluded_traces:
        for trace in excluded_traces:
            axis.plot(
                trace,
                color="tab:red",
                alpha=0.08,
                linewidth=0.8,
            )

    axis.set_xlabel("Time / echo index")
    axis.set_ylabel("Signal intensity")
    axis.set_title(
        "EG mask SAD filter diagnostic "
        f"(included={len(included_traces)}, excluded={len(excluded_traces)})"
    )
    axis.grid(visible=True, alpha=0.3)
    axis.text(
        x=0.01,
        y=0.98,
        s=f"Threshold: SAD >= {minimum_sad_counts:.1f} counts",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=_DIAGNOSTIC_PLOT_DPI)
    plt.close()


def _filter_mask_by_sad_threshold(
    *,
    thermometry_image: nibabel.nifti1.Nifti1Image,
    labelled_mask: np.ndarray,
    minimum_sad_counts: float,
    output_plot_path: Path | None,
) -> tuple[np.ndarray, int, int]:
    """Keep only mask voxels with sufficient Sum of Absolute Differences (SAD).

    Args:
        thermometry_image: Multi-echo thermometry image.
        labelled_mask: Labelled mask in thermometry voxel space.
        minimum_sad_counts: Minimum Sum of Absolute Differences value required
            to retain a voxel.
        output_plot_path: Path where the diagnostic plot should be written. If
            ``None``, no diagnostic plot is generated.

    Returns:
        Tuple of ``(filtered_mask, included_voxel_count, excluded_voxel_count)``.
    """
    thermometry_data = np.asarray(thermometry_image.get_fdata(), dtype=np.float64)
    if thermometry_data.ndim < _MIN_THERMOMETRY_NDIM_FOR_TIME_AXIS:
        msg = (
            f"Expected thermometry image to have at least "
            f"{_MIN_THERMOMETRY_NDIM_FOR_TIME_AXIS} dimensions; "
            f"got shape {thermometry_data.shape}."
        )
        raise ValueError(msg)

    candidate_voxels = np.argwhere(np.asarray(labelled_mask) != 0)
    filtered_mask = np.zeros(shape=labelled_mask.shape, dtype=np.uint8)
    included_traces: list[np.ndarray] = []
    excluded_traces: list[np.ndarray] = []

    for candidate_voxel in candidate_voxels:
        x_index, y_index, z_index = (
            int(candidate_voxel[0]),
            int(candidate_voxel[1]),
            int(candidate_voxel[2]),
        )
        voxel_trace = np.asarray(
            thermometry_data[x_index, y_index, z_index, :],
            dtype=np.float64,
        ).ravel()
        sad_counts = float(np.sum(np.abs(np.diff(voxel_trace))))
        if sad_counts >= minimum_sad_counts:
            filtered_mask[x_index, y_index, z_index] = np.uint8(
                labelled_mask[x_index, y_index, z_index]
            )
            included_traces.append(voxel_trace)
        else:
            excluded_traces.append(voxel_trace)

    if output_plot_path is not None:
        _save_sad_filter_plot(
            included_traces=included_traces,
            excluded_traces=excluded_traces,
            output_plot_path=output_plot_path,
            minimum_sad_counts=minimum_sad_counts,
        )
    return filtered_mask, len(included_traces), len(excluded_traces)


def coordinate_ethylene_glycol_vial_segmentation(
    *,
    registered_component_atlas_image_path: Path,
    multi_echo_gradient_echo_scan_image_path: Path,
    output_mask_image_path: Path | None = None,
    minimum_sad_counts: float = _EG_MASK_MIN_SAD_COUNTS,
    dilation_iterations: int = _EG_MASK_DILATION_ITERATIONS,
    generate_sad_visualisation: bool = False,
) -> Path:
    """Create an ethylene glycol segmentation mask from atlas and multi-echo data.

    Only EG labels are transferred from the atlas, so dilation and SAD filtering
    operate exclusively on EG voxels.

    Args:
        registered_component_atlas_image_path: Path to the registered component atlas NIfTI.
        multi_echo_gradient_echo_scan_image_path: Path to the multi-echo GRE NIfTI.
        output_mask_image_path: Optional explicit output mask path.
        minimum_sad_counts: SAD inclusion threshold in counts.
        dilation_iterations: Number of binary-dilation iterations.
        generate_sad_visualisation: Whether to save the SAD diagnostic plot.

    Returns:
        Path to the saved segmentation mask NIfTI file.
    """
    if minimum_sad_counts < 0.0:
        msg = "minimum_sad_counts must be greater than or equal to 0."
        raise ValueError(msg)
    if dilation_iterations < 0:
        msg = "dilation_iterations must be greater than or equal to 0."
        raise ValueError(msg)

    registered_atlas_image = nibabel.nifti1.load(
        filename=str(registered_component_atlas_image_path)
    )
    eg_thermometry_image = nibabel.nifti1.load(
        filename=str(multi_echo_gradient_echo_scan_image_path)
    )

    mapped_labelled_mask = transfer_atlas_labels_to_image_space(
        atlas_image=registered_atlas_image,
        target_image=eg_thermometry_image,
        labels=list(ETHYLENE_GLYCOL_LABELS),
    )

    if output_mask_image_path is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_mask_image_path = (
            multi_echo_gradient_echo_scan_image_path.parent
            / f"ethylene_glycol_mask_{timestamp}.nii.gz"
        )
    output_mask_image_path.parent.mkdir(parents=True, exist_ok=True)
    mapped_label_mask_output_path = _get_output_mapped_label_mask_path(
        output_mask_image_path=output_mask_image_path
    )
    _save_labelled_mask_image(
        mask=mapped_labelled_mask,
        reference_image=eg_thermometry_image,
        output_image_path=mapped_label_mask_output_path,
        description="Mapped EG labels from component atlas before dilation/SAD",
    )

    dilated_labelled_mask = _dilate_labelled_mask(
        labelled_mask=mapped_labelled_mask,
        iterations=dilation_iterations,
    )

    output_plot_path = (
        _get_mask_output_plot_path(output_mask_image_path=output_mask_image_path)
        if generate_sad_visualisation
        else None
    )

    filtered_mask, _, _ = _filter_mask_by_sad_threshold(
        thermometry_image=eg_thermometry_image,
        labelled_mask=dilated_labelled_mask,
        minimum_sad_counts=minimum_sad_counts,
        output_plot_path=output_plot_path,
    )
    _save_labelled_mask_image(
        mask=filtered_mask,
        reference_image=eg_thermometry_image,
        output_image_path=output_mask_image_path,
        description="Mapped EG labels, dilated and SAD-filtered",
    )
    return output_mask_image_path


class EchoTrain(BaseModel):
    """One multi-echo GRE series and the echo-time file qmri should read.

    Attributes:
        image_path: 4D magnitude NIfTI with echoes on the last axis.
        echo_times_file: Text file of echo times in seconds, one value per echo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_path: Path
    echo_times_file: Path


class ThermometryPackage(BaseModel):
    """Segmentation, echo trains, and field strength handed to qmri.

    Attributes:
        segmentation_path: SAD-filtered ethylene glycol mask on the first series' grid.
        trains: Multi-echo series, in the same order as their echo-time files.
        magnetic_field_tesla: Field strength used for the ethylene glycol calibration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    segmentation_path: Path
    trains: tuple[EchoTrain, ...]
    magnetic_field_tesla: float = DEFAULT_MAGNETIC_FIELD_TESLA


class VialTemperature(BaseModel):
    """Temperature estimate for one ethylene glycol vial.

    Attributes:
        vial_id: Vial letter from the component atlas configuration.
        segment_index: Atlas label fitted by qmri.
        temperature_celsius: Representative region temperature in degrees Celsius.
        temperature_uncertainty_celsius: Standard uncertainty (k=1) in degrees Celsius.
        frequency_difference_hz: Fitted CH2-OH frequency difference in hertz.
        r_squared: Mean coefficient of determination across the region's fits.
        n_voxels: Number of voxels in the region.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    vial_id: str
    segment_index: int
    temperature_celsius: float
    temperature_uncertainty_celsius: float
    frequency_difference_hz: float
    r_squared: float
    n_voxels: int


class EthyleneGlycolThermometryResult(BaseModel):
    """Outputs of one ethylene glycol temperature estimation.

    Attributes:
        temperature_map_path: qmri temperature map NIfTI, in degrees Celsius.
        report_path: qmri JSON report path.
        measurements_path: Vial temperature table written by spirit-phantom.
        method: qmri region analysis method that was run.
        magnetic_field_tesla: Field strength passed to the calibration.
        vials: One row per fitted thermometry vial.
        visualisation_paths: Diagnostic figures written when visualisation is requested.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature_map_path: Path
    report_path: Path
    measurements_path: Path
    method: ThermometryAnalysisMethod
    magnetic_field_tesla: float
    vials: tuple[VialTemperature, ...]
    visualisation_paths: tuple[Path, ...] = ()


def load_echo_times_seconds(*, echo_times_file: Path) -> np.ndarray:
    """Load a 1-D list of echo times in seconds.

    This is the text layout qmri reads: one echo time per line, in seconds.

    Args:
        echo_times_file: Path to the echo-time text file.

    Returns:
        Echo times in seconds, shape ``(n_echoes,)``.

    Raises:
        ValueError: If the file is not a 1-D list of numbers.
    """
    try:
        values = np.atleast_1d(np.loadtxt(fname=echo_times_file, dtype=np.float64))
    except ValueError as error:
        msg = (
            "Echo-times file must contain a 1D list of echo times in seconds: "
            f"{echo_times_file}"
        )
        raise ValueError(msg) from error
    if values.ndim != 1 or values.size == 0:
        msg = (
            "Echo-times file must contain a 1D list of echo times in seconds: "
            f"{echo_times_file}"
        )
        raise ValueError(msg)
    return values


def format_ethylene_glycol_temperature_table(
    *,
    vials: Sequence[VialTemperature],
    magnetic_field_tesla: float,
    method: str,
) -> str:
    """Format vial temperatures as a plain-text table.

    Args:
        vials: Temperature rows, one per fitted vial.
        magnetic_field_tesla: Field strength used for the calibration.
        method: qmri analysis method name.

    Returns:
        The table, including a trailing newline.
    """
    header = (
        f"{'vial':<6}{'segment':>9}{'temperature_C':>16}{'uncertainty_C':>16}"
        f"{'df_Hz':>12}{'r_squared':>12}{'n_voxels':>10}"
    )
    lines = [
        "Ethylene glycol thermometry",
        f"Magnetic field: {magnetic_field_tesla:g} T",
        f"Method: {method}",
        "Uncertainties are standard uncertainties (k=1).",
        "",
        header,
    ]
    for vial in vials:
        lines.append(
            f"{vial.vial_id:<6}{vial.segment_index:>9}"
            f"{vial.temperature_celsius:>16.2f}"
            f"{vial.temperature_uncertainty_celsius:>16.2f}"
            f"{vial.frequency_difference_hz:>12.1f}"
            f"{vial.r_squared:>12.3f}"
            f"{vial.n_voxels:>10d}"
        )
    return "\n".join(lines) + "\n"


def _read_multiecho_grid(
    *,
    image_path: Path,
) -> tuple[tuple[int, int, int], np.ndarray, int]:
    """Read the spatial shape, affine, and echo count of a multi-echo NIfTI.

    Args:
        image_path: Path to a multi-echo magnitude image.

    Returns:
        Spatial shape ``(x, y, z)``, affine, and the number of echoes.

    Raises:
        ValueError: If the image is not 4-D with echoes on the last axis.
    """
    image = nibabel.nifti1.load(filename=str(image_path))
    shape = tuple(int(size) for size in image.shape)
    if len(shape) != _MIN_THERMOMETRY_NDIM_FOR_TIME_AXIS:
        msg = (
            "Multi-echo image must be 4D with echoes on the last axis: "
            f"{image_path} has shape {shape}."
        )
        raise ValueError(msg)
    spatial_shape = (shape[0], shape[1], shape[2])
    n_echoes = shape[3]
    affine = np.asarray(image.affine, dtype=np.float64)
    return spatial_shape, affine, n_echoes


def _assert_path_exists(*, path: Path, description: str) -> None:
    """Raise if a required input file is missing.

    Args:
        path: File that must already exist.
        description: What the file is, used in the error message.

    Raises:
        ValueError: If ``path`` does not exist.
    """
    if not path.exists():
        msg = f"{description} not found: {path}"
        raise ValueError(msg)


def _assert_shared_multiecho_grid(*, trains: Sequence[EchoTrain]) -> None:
    """Require every series to share the first series' spatial shape and affine.

    qmri concatenates echoes onto the first image's voxel grid and raises if a
    later series differs in ``shape[:3]`` or fails ``numpy.allclose`` on the
    affine. This check uses those same comparisons and names the files.

    Args:
        trains: Multi-echo series to compare. The first series is the reference.

    Raises:
        ValueError: If a series is not 4-D, or does not share the reference grid.
    """
    reference = trains[0]
    reference_shape, reference_affine, _n_echoes = _read_multiecho_grid(
        image_path=reference.image_path
    )
    for train in trains[1:]:
        spatial_shape, affine, _n_echoes = _read_multiecho_grid(
            image_path=train.image_path
        )
        if spatial_shape != reference_shape:
            msg = (
                f"{train.image_path} has spatial shape {spatial_shape}, but "
                f"{reference.image_path} has spatial shape {reference_shape}. "
                "qmri requires every multi-echo series to share the same spatial shape."
            )
            raise ValueError(msg)
        if not np.allclose(affine, reference_affine):
            msg = (
                f"{train.image_path} does not share an affine with "
                f"{reference.image_path}. qmri requires every multi-echo series "
                "to share the same affine."
            )
            raise ValueError(msg)


def _assert_echo_times_match_images(*, trains: Sequence[EchoTrain]) -> None:
    """Require each echo-time file to have one value per echo in its image.

    Args:
        trains: Series whose echo-time files are checked against the NIfTI.

    Raises:
        ValueError: If a file is not a 1-D list, or its length differs from the
            image's last axis.
    """
    for train in trains:
        echo_times = load_echo_times_seconds(echo_times_file=train.echo_times_file)
        _spatial_shape, _affine, n_echoes = _read_multiecho_grid(
            image_path=train.image_path
        )
        if echo_times.size != n_echoes:
            msg = (
                f"{train.echo_times_file} has {echo_times.size} echo times but "
                f"{train.image_path} has {n_echoes} echoes."
            )
            raise ValueError(msg)


def _assert_thermometry_inputs(
    *,
    registered_component_atlas_image_path: Path,
    trains: Sequence[EchoTrain],
    magnetic_field_tesla: float,
    method: str,
    minimum_sad_counts: float,
    dilation_iterations: int,
) -> None:
    """Validate thermometry inputs before segmentation or fitting.

    Args:
        registered_component_atlas_image_path: Registered component atlas NIfTI.
        trains: Multi-echo series paired with echo-time files.
        magnetic_field_tesla: Field strength in tesla.
        method: Requested qmri analysis method.
        minimum_sad_counts: SAD inclusion threshold in counts.
        dilation_iterations: Binary-dilation iterations for the EG mask.

    Raises:
        ValueError: If a required file is missing, the method or field strength
            is invalid, or the series do not share a grid.
    """
    if not trains:
        msg = "At least one multi-echo image and echo-time file must be provided."
        raise ValueError(msg)
    if magnetic_field_tesla <= 0.0:
        msg = "magnetic_field_tesla must be greater than 0."
        raise ValueError(msg)
    if method not in _THERMOMETRY_ANALYSIS_METHODS:
        msg = (
            f"Unknown thermometry method: {method!r}. Use 'regionwise', "
            "'voxelwise', or 'regionwise_bootstrap'."
        )
        raise ValueError(msg)
    if minimum_sad_counts < 0.0:
        msg = "minimum_sad_counts must be greater than or equal to 0."
        raise ValueError(msg)
    if dilation_iterations < 0:
        msg = "dilation_iterations must be greater than or equal to 0."
        raise ValueError(msg)

    _assert_path_exists(
        path=registered_component_atlas_image_path,
        description="Registered component atlas file",
    )
    for train in trains:
        _assert_path_exists(
            path=train.image_path,
            description="Multi-echo gradient echo file",
        )
        _assert_path_exists(
            path=train.echo_times_file,
            description="Echo-times file",
        )
    _assert_shared_multiecho_grid(trains=trains)
    _assert_echo_times_match_images(trains=trains)


def _assert_mask_has_ethylene_glycol(*, mask_path: Path) -> None:
    """Raise when SAD filtering removed every ethylene glycol voxel.

    Args:
        mask_path: Saved EG mask NIfTI.

    Raises:
        ValueError: If the mask has no non-zero voxels.
    """
    mask_image = nibabel.nifti1.load(filename=str(mask_path))
    mask = np.asarray(mask_image.dataobj)
    if not np.any(mask):
        msg = f"Ethylene glycol mask contains no labelled voxels: {mask_path}"
        raise ValueError(msg)


def _mean_r_squared(*, r_squared: np.ndarray) -> float:
    """Return the mean R², or NaN when a region has no fits.

    Args:
        r_squared: Per-fit coefficients of determination.

    Returns:
        Mean of the finite values, or NaN if there are none.
    """
    if r_squared.size == 0:
        return float("nan")
    return float(np.nanmean(r_squared))


def _vial_temperatures_from_report(
    *,
    regions: Sequence[RegionThermometryResult],
) -> tuple[VialTemperature, ...]:
    """Map qmri region results onto configured vial letters.

    Args:
        regions: ``RegionThermometryResult`` rows from the qmri report.

    Returns:
        Vial temperatures sorted by vial letter.

    Raises:
        ValueError: If a fitted label is not a configured vial.
    """
    from spirit_phantom.core.vials import (  # noqa: PLC0415
        vial_id_for_segment_index,
    )

    vials: list[VialTemperature] = []
    for region in regions:
        segment_index = int(region.region_id)
        mean_parameters = np.asarray(region.mean_fitted_params, dtype=np.float64)
        vials.append(
            VialTemperature(
                vial_id=vial_id_for_segment_index(segment_index=segment_index),
                segment_index=segment_index,
                temperature_celsius=float(region.temperature),
                temperature_uncertainty_celsius=float(region.temperature_uncertainty),
                frequency_difference_hz=float(
                    mean_parameters[_FREQUENCY_DIFFERENCE_PARAMETER_INDEX]
                ),
                r_squared=_mean_r_squared(
                    r_squared=np.asarray(region.r_squared, dtype=np.float64)
                ),
                n_voxels=int(region.region_size),
            )
        )
    return tuple(sorted(vials, key=lambda vial: vial.vial_id))


def _vial_plot_colour(*, vial_id: str) -> str:
    """Return the plot colour for a thermometry vial.

    Args:
        vial_id: Vial letter, for example ``"U"``.

    Returns:
        A hex colour. Vials other than U and V share one fallback colour.
    """
    return _VIAL_PLOT_COLOURS.get(vial_id, _DEFAULT_VIAL_COLOUR)


def _train_echo_masks(
    *,
    echo_times_seconds: np.ndarray,
    trains: Sequence[EchoTrain],
) -> tuple[tuple[str, str, np.ndarray], ...]:
    """Assign each combined echo to the series whose echo times contain it.

    Args:
        echo_times_seconds: Echo times after qmri has sorted the combined train.
        trains: Series in the order they were supplied, each with an echo-time file.

    Returns:
        One ``(series filename, marker, echo mask)`` tuple per series. An echo
        that appears in more than one file is kept on the earliest series.
    """
    rounded_echoes = np.round(echo_times_seconds, decimals=_ECHO_TIME_MATCH_DECIMALS)
    claimed = np.zeros(shape=rounded_echoes.shape, dtype=bool)
    membership: list[tuple[str, str, np.ndarray]] = []
    for index, train in enumerate(trains):
        train_times = load_echo_times_seconds(echo_times_file=train.echo_times_file)
        in_train = np.isin(
            rounded_echoes,
            np.round(train_times, decimals=_ECHO_TIME_MATCH_DECIMALS),
        )
        exclusive = in_train & ~claimed
        claimed |= exclusive
        marker = _TRAIN_MARKERS[index % len(_TRAIN_MARKERS)]
        membership.append((train.image_path.name, marker, exclusive))
    return tuple(membership)


def _finite_parameter_vector(*, parameters: np.ndarray) -> np.ndarray | None:
    """Return a finite six-parameter vector, or ``None``.

    Args:
        parameters: Candidate dual-resonance parameters.

    Returns:
        A one-dimensional vector of six finite values, or ``None``.
    """
    vector = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if vector.size != _DUAL_RESONANCE_PARAMETER_COUNT:
        return None
    if not bool(np.all(np.isfinite(vector))):
        return None
    return vector


def _fit_curve_parameters(*, region: RegionThermometryResult) -> np.ndarray | None:
    """Return a finite dual-resonance parameter vector for plotting.

    Args:
        region: qmri region result.

    Returns:
        Six fitted parameters, preferring the mean of fits at or above the R²
        threshold. ``None`` when no finite parameter vector is available.
    """
    mean_parameters = _finite_parameter_vector(
        parameters=np.asarray(region.mean_fitted_params, dtype=np.float64),
    )
    if mean_parameters is not None:
        return mean_parameters
    fitted_parameters = np.asarray(region.fitted_params, dtype=np.float64)
    if fitted_parameters.ndim != _FITTED_PARAMETER_NDIM or fitted_parameters.size == 0:
        return None
    return _finite_parameter_vector(parameters=np.asarray(fitted_parameters[0]))


def _dual_resonance_signal(
    *,
    echo_times_seconds: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    """Evaluate the dual-resonance model at the supplied echo times.

    Args:
        echo_times_seconds: Echo times in seconds.
        parameters: Six fitted parameters, in qmri order.

    Returns:
        Model signal at each echo time.
    """
    from qmri.thermometry.multiecho import (  # noqa: PLC0415
        thermometry_signal_model,
    )

    amplitude_1, amplitude_2, r2star_1, r2star_2, frequency_difference, phase = (
        float(value) for value in parameters
    )
    return thermometry_signal_model(
        t=echo_times_seconds,
        amplitude_1=amplitude_1,
        amplitude_2=amplitude_2,
        r2star_1=r2star_1,
        r2star_2=r2star_2,
        df=frequency_difference,
        dphi_deg=phase,
    )


def _save_signal_fit_plot(
    *,
    region: RegionThermometryResult,
    vial_id: str,
    echo_times_seconds: np.ndarray,
    trains: Sequence[EchoTrain],
    output_directory: Path,
) -> Path:
    """Save one vial's echo trains and the dual-resonance curve fitted to them.

    Measured samples are drawn faintly, with a different marker for each GRE
    series. The residual panel is the mean of those samples minus the curve.

    Args:
        region: qmri region result for this vial.
        vial_id: Vial letter used in the title and filename.
        echo_times_seconds: Combined echo times in seconds, sorted as qmri fitted them.
        trains: Series used to label markers.
        output_directory: Directory for ``vial_{id}_signal_fit.png``.

    Returns:
        Path of the saved figure.
    """
    colour = _vial_plot_colour(vial_id=vial_id)
    echo_times_ms = echo_times_seconds * _SECONDS_TO_MILLISECONDS
    signals = np.atleast_2d(np.asarray(region.signal_values, dtype=np.float64))
    parameters = _fit_curve_parameters(region=region)
    membership = _train_echo_masks(echo_times_seconds=echo_times_seconds, trains=trains)

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    signal_axis = axes[0]
    residual_axis = axes[1]
    labelled_series: set[str] = set()
    for sample in signals:
        for series_name, marker, echo_mask in membership:
            if not np.any(echo_mask):
                continue
            label = series_name if series_name not in labelled_series else None
            labelled_series.add(series_name)
            signal_axis.scatter(
                echo_times_ms[echo_mask],
                sample[echo_mask],
                marker=marker,
                s=18,
                color=colour,
                alpha=0.35,
                edgecolors="black",
                linewidths=0.4,
                label=label,
            )

    if parameters is not None:
        fine_times = np.linspace(
            0.0,
            float(echo_times_seconds[-1]),
            _SIGNAL_CURVE_SAMPLES,
        )
        signal_axis.plot(
            fine_times * _SECONDS_TO_MILLISECONDS,
            _dual_resonance_signal(
                echo_times_seconds=fine_times,
                parameters=parameters,
            ),
            color=colour,
            linewidth=1.5,
            label="Dual-resonance fit",
        )
        residual = np.mean(signals, axis=0) - _dual_resonance_signal(
            echo_times_seconds=echo_times_seconds,
            parameters=parameters,
        )
        residual_axis.plot(
            echo_times_ms,
            residual,
            marker=".",
            color=colour,
            linestyle="none",
        )

    residual_axis.axhline(y=0.0, color="black", linewidth=0.8)
    signal_axis.set_ylabel("Signal magnitude (a.u.)")
    signal_axis.set_title(
        f"Vial {vial_id}: {region.temperature:.2f} °C, "
        f"R² = {_mean_r_squared(r_squared=np.asarray(region.r_squared)):.3f}"
    )
    signal_axis.legend(loc="upper right", fontsize=8)
    signal_axis.grid(visible=True, alpha=0.3)
    residual_axis.set_xlabel("Echo time (ms)")
    residual_axis.set_ylabel("Residual")
    residual_axis.grid(visible=True, alpha=0.3)
    figure.tight_layout()
    output_path = output_directory / f"vial_{vial_id}_signal_fit.png"
    figure.savefig(output_path, dpi=_DIAGNOSTIC_PLOT_DPI)
    plt.close(figure)
    return output_path


def _save_thermometry_visualisations(
    *,
    regions: Sequence[RegionThermometryResult],
    echo_times_seconds: Sequence[float],
    trains: Sequence[EchoTrain],
    output_directory: Path,
) -> tuple[Path, ...]:
    """Save one signal-and-fit figure per vial.

    Args:
        regions: qmri region results, one per fitted label.
        echo_times_seconds: Combined echo times in seconds.
        trains: Series used to mark which echoes belong to which image.
        output_directory: Directory for the figures.

    Returns:
        One signal-fit path per vial, in letter order.
    """
    from spirit_phantom.core.vials import (  # noqa: PLC0415
        vial_id_for_segment_index,
    )

    echo_times = np.asarray(echo_times_seconds, dtype=np.float64)
    labelled_regions = tuple(
        sorted(
            (
                (
                    vial_id_for_segment_index(segment_index=int(region.region_id)),
                    region,
                )
                for region in regions
            ),
            key=lambda item: item[0],
        )
    )
    return tuple(
        _save_signal_fit_plot(
            region=region,
            vial_id=vial_id,
            echo_times_seconds=echo_times,
            trains=trains,
            output_directory=output_directory,
        )
        for vial_id, region in labelled_regions
    )


def _fit_ethylene_glycol_temperature(
    *,
    package: ThermometryPackage,
    output_directory: Path,
    method: ThermometryAnalysisMethod,
) -> tuple[
    Path,
    Path,
    tuple[VialTemperature, ...],
    tuple[RegionThermometryResult, ...],
    tuple[float, ...],
]:
    """Run qmri on a prepared thermometry package.

    Args:
        package: Mask, echo trains, and field strength.
        output_directory: Directory for the qmri temperature map and report.
        method: qmri region analysis method.

    Returns:
        Temperature-map path, report path, vial temperatures, region results,
        and the combined echo times in seconds.

    Raises:
        ValueError: If qmri does not write a 3-D temperature map.
    """
    from qmri.pipelines.thermometry import (  # noqa: PLC0415
        run_multiecho_thermometry,
    )

    temperature_image, report = run_multiecho_thermometry(
        multiecho_files=[train.image_path for train in package.trains],
        segmentation_file=package.segmentation_path,
        echo_times_files=[train.echo_times_file for train in package.trains],
        method=method,
        n_bootstrap=_BOOTSTRAP_SAMPLES,
        magnetic_field_tesla=package.magnetic_field_tesla,
        output_dir=output_directory,
        output_prefix=_TEMPERATURE_OUTPUT_PREFIX,
    )
    temperature_map = np.asarray(temperature_image.data)
    if report.output_file is None or temperature_map.ndim != _TEMPERATURE_MAP_NDIM:
        msg = "qmri did not write a 3D temperature map."
        raise ValueError(msg)
    temperature_map_path = Path(report.output_file)
    report_path = output_directory / f"{_TEMPERATURE_OUTPUT_PREFIX}_report.json"
    vials = _vial_temperatures_from_report(regions=report.regions)
    return (
        temperature_map_path,
        report_path,
        vials,
        tuple(report.regions),
        tuple(report.echo_times),
    )


def coordinate_ethylene_glycol_thermometry(
    *,
    registered_component_atlas_image_path: Path,
    trains: Sequence[EchoTrain],
    output_directory: Path,
    magnetic_field_tesla: float = DEFAULT_MAGNETIC_FIELD_TESLA,
    method: ThermometryAnalysisMethod = "regionwise",
    minimum_sad_counts: float = _EG_MASK_MIN_SAD_COUNTS,
    dilation_iterations: int = _EG_MASK_DILATION_ITERATIONS,
    generate_visualisations: bool = False,
) -> EthyleneGlycolThermometryResult:
    """Estimate ethylene glycol vial temperatures from a registered atlas.

    The ethylene glycol mask is built on the first series. Later series must
    share that series' spatial shape and affine. Echo-time files are the qmri
    seconds files supplied by the caller; this function does not derive them.

    Args:
        registered_component_atlas_image_path: Registered component atlas NIfTI.
        trains: Multi-echo GRE series, each with an echo-time file in seconds.
        output_directory: Directory for the mask, qmri outputs, and vial table.
        magnetic_field_tesla: Field strength in tesla. Defaults to 3, the field
            SPIRIT ethylene glycol materials are calibrated at.
        method: qmri analysis method. Defaults to ``regionwise``.
        minimum_sad_counts: SAD inclusion threshold in counts.
        dilation_iterations: Binary-dilation iterations for the EG mask.
        generate_visualisations: Whether to save the SAD mask plot and a
            signal-and-fit figure per vial.

    Returns:
        Paths to the temperature map, qmri report, and vial table, plus the
        per-vial temperatures. Diagnostic figure paths are included when
        visualisation is requested.

    Raises:
        ValueError: If inputs are missing, the series do not share a grid, an
            echo-time file does not match its image, or the mask is empty.
    """
    _assert_thermometry_inputs(
        registered_component_atlas_image_path=registered_component_atlas_image_path,
        trains=trains,
        magnetic_field_tesla=magnetic_field_tesla,
        method=method,
        minimum_sad_counts=minimum_sad_counts,
        dilation_iterations=dilation_iterations,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    reference_image_path = trains[0].image_path
    segmentation_path = coordinate_ethylene_glycol_vial_segmentation(
        registered_component_atlas_image_path=registered_component_atlas_image_path,
        multi_echo_gradient_echo_scan_image_path=reference_image_path,
        output_mask_image_path=output_directory / ETHYLENE_GLYCOL_MASK_FILENAME,
        minimum_sad_counts=minimum_sad_counts,
        dilation_iterations=dilation_iterations,
        generate_sad_visualisation=generate_visualisations,
    )
    _assert_mask_has_ethylene_glycol(mask_path=segmentation_path)
    package = ThermometryPackage(
        segmentation_path=segmentation_path,
        trains=tuple(trains),
        magnetic_field_tesla=magnetic_field_tesla,
    )
    (
        temperature_map_path,
        report_path,
        vials,
        regions,
        echo_times,
    ) = _fit_ethylene_glycol_temperature(
        package=package,
        output_directory=output_directory,
        method=method,
    )
    visualisation_paths: tuple[Path, ...] = ()
    if generate_visualisations:
        visualisation_paths = (
            _get_mask_output_plot_path(output_mask_image_path=segmentation_path),
            *_save_thermometry_visualisations(
                regions=regions,
                echo_times_seconds=echo_times,
                trains=trains,
                output_directory=output_directory,
            ),
        )
    measurements_path = output_directory / _TEMPERATURE_MEASUREMENTS_FILENAME
    measurements_path.write_text(
        format_ethylene_glycol_temperature_table(
            vials=vials,
            magnetic_field_tesla=magnetic_field_tesla,
            method=method,
        ),
        encoding="utf-8",
    )
    return EthyleneGlycolThermometryResult(
        temperature_map_path=temperature_map_path,
        report_path=report_path,
        measurements_path=measurements_path,
        method=method,
        magnetic_field_tesla=magnetic_field_tesla,
        vials=vials,
        visualisation_paths=visualisation_paths,
    )

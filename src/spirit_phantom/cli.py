"""Command line interface for spirit-phantom workflows.

The registration command defaults to a SPIRIT atlas image downloaded with
`pooch` when a moving image is not provided explicitly.

The ``analyse`` command group includes:

- ``vial-measurements`` for per-vial intensity statistics.
- ``dice`` for per-vial overlap scores between manual and atlas labels.
- ``vials`` for per-vial segmentation accuracy metrics (manual vs atlas).
- ``eg-mask`` for ethylene glycol vial mask generation from multi-echo data.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from spirit_phantom import (
    get_default_component_atlas_image_path,
    get_default_register_moving_image_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

app = typer.Typer(help="SPIRIT phantom command line tools.")
analyse_app = typer.Typer(help="Run analyses on registered phantom data.")
app.add_typer(analyse_app, name="analyse")

# Keep EG-mask CLI defaults local so importing the CLI does not pull
# multi-echo thermometry (matplotlib/nibabel) until eg-mask runs.
_EG_MASK_MIN_SAD_COUNTS_DEFAULT = 1000.0
_EG_MASK_DILATION_ITERATIONS_DEFAULT = 1


class AnalysisMethod(StrEnum):
    """Supported analysis methods for CLI workflows."""

    VIAL_MEASUREMENTS = "vial-measurements"


def _configure_verbose_logging() -> None:
    """Surface library INFO progress messages for verbose CLI runs.

    Library stages already emit ``logging`` messages. This configures a simple
    console handler so interactive ``--verbose`` users can see them without
    affecting library importers.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)


def _emit_cli_message(message: str, *, quiet: bool) -> None:
    """Print a CLI message unless quiet mode is enabled.

    Args:
        message: Text to print.
        quiet: When True, suppress the message.
    """
    if not quiet:
        print(message)


def _resolve_default_atlases(*, quiet: bool) -> Path:
    """Download or reuse cached default atlases, returning the signal atlas path.

    Prefetches the component atlas as well so registration does not pause for a
    second silent download after the B-spline stage.

    Args:
        quiet: When True, suppress status messages around atlas preparation.

    Returns:
        Local path to the cached signal atlas image.
    """
    _emit_cli_message("Preparing default SPIRIT atlases...", quiet=quiet)
    signal_atlas_path = get_default_register_moving_image_path()
    get_default_component_atlas_image_path()
    _emit_cli_message(f"Signal atlas: {signal_atlas_path}", quiet=quiet)
    return signal_atlas_path


def _build_timestamped_output_directory() -> Path:
    """Create a timestamped output directory under the current working directory.

    Returns:
        Path to the created directory.
    """
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_directory = Path.cwd() / "registered_data" / timestamp
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def _validate_erosion_voxels(*, erosion_voxels: int) -> None:
    """Validate the requested erosion voxel count.

    Args:
        erosion_voxels: Number of erosion voxels requested by the user.

    Raises:
        typer.BadParameter: If the value is negative.
    """
    if erosion_voxels < 0:
        msg = "--erosion-voxels must be greater than or equal to 0."
        raise typer.BadParameter(msg)


def _run_vial_measurements(
    *,
    transformed_component_atlas_image_path: Path,
    mri_scan_image_path: Path,
    erosion_voxels: int,
    output_directory: Path | None,
    save_results: bool,
) -> None:
    """Run vial measurement analysis and optionally save detailed results.

    Args:
        transformed_component_atlas_image_path: Path to the transformed component atlas image.
        mri_scan_image_path: Path to the scanner image.
        erosion_voxels: Number of voxels to erode each vial mask.
        output_directory: Directory used for output files when saving is enabled.
        save_results: Whether detailed results should be written to disk.
    """
    # Delay heavy analysis imports so `--help` stays responsive.
    from spirit_phantom.core.vials import (  # noqa: PLC0415
        compute_vial_statistics_details,
        print_vial_statistics_details_table,
        save_vial_statistics_details_table,
    )

    _validate_erosion_voxels(erosion_voxels=erosion_voxels)
    details_rows = compute_vial_statistics_details(
        registered_atlas_image_path=transformed_component_atlas_image_path,
        mri_scan_image_path=mri_scan_image_path,
        erosion_voxels=erosion_voxels,
    )
    print_vial_statistics_details_table(rows=details_rows)

    if save_results:
        if output_directory is None:
            msg = "output_directory is required when save_results is True."
            raise ValueError(msg)
        output_path = save_vial_statistics_details_table(
            rows=details_rows,
            output_path=output_directory / "vial_statistics_details.txt",
        )
        print(f"Saved detailed vial statistics: {output_path}")


def _build_checkerboard_slice_indices(*, fixed_image_path: Path) -> list[int]:
    """Build evenly distributed axial slice indices for checkerboard output.

    Args:
        fixed_image_path: Path to the fixed image used in registration.

    Returns:
        A list of slice indices suitable for checkerboard visualisation.
    """
    # Delay ITK import until checkerboards are explicitly requested.
    import itk  # noqa: PLC0415

    fixed_image = itk.imread(str(fixed_image_path))
    fixed_array = itk.array_view_from_image(fixed_image)

    minimum_spatial_dimensions = 3
    if fixed_array.ndim < minimum_spatial_dimensions:
        return [0]

    n_slices = int(fixed_array.shape[0])
    if n_slices <= 1:
        return [0]

    max_slices = 5
    slice_indices: list[int] = []
    for slice_number in range(1, max_slices + 1):
        candidate_index = round((slice_number * (n_slices - 1)) / (max_slices + 1))
        if candidate_index not in slice_indices:
            slice_indices.append(candidate_index)

    return slice_indices


def _generate_checkerboard_images(
    *,
    fixed_image_path: Path,
    transformed_component_atlas_image_path: Path,
) -> None:
    """Generate checkerboard images for visual registration quality checks.

    Args:
        fixed_image_path: Path to the fixed image used as registration target.
        transformed_component_atlas_image_path: Path to the transformed component atlas image output.
    """
    from spirit_phantom.utils.visualisation import (  # noqa: PLC0415
        visualise_checkerboard,
    )

    slice_indices = _build_checkerboard_slice_indices(fixed_image_path=fixed_image_path)
    visualise_checkerboard(
        fixed_image_path=fixed_image_path,
        registered_image_path=transformed_component_atlas_image_path,
        slice_indices=slice_indices,
    )
    print(
        f"Saved checkerboard images in: {transformed_component_atlas_image_path.parent}"
    )


def _format_dice_score_rows_table(
    *, rows: Sequence[dict[str, int | str | float]]
) -> str:
    """Format per-vial Dice score rows into a readable text table.

    Args:
        rows: Rows returned by ``generate_dice_score_table`` containing vial ID,
            manual/atlas labels, Dice score, and voxel count fields.

    Returns:
        Table string suitable for command-line output.
    """
    headers = [
        "vial_id",
        "manual_label",
        "atlas_label",
        "dice_score",
        "manual_voxels",
        "atlas_voxels",
        "intersection_voxels",
    ]

    def _cell_to_text(*, value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    table_rows = [
        [_cell_to_text(value=row.get(header, "")) for header in headers] for row in rows
    ]
    all_rows = [headers, *table_rows]
    column_widths = [
        max(len(row[column_index]) for row in all_rows)
        for column_index in range(len(headers))
    ]

    def _format_line(*, cells: list[str]) -> str:
        return " | ".join(
            cell.ljust(column_width)
            for cell, column_width in zip(cells, column_widths, strict=True)
        )

    separator = "-+-".join("-" * width for width in column_widths)
    lines = [_format_line(cells=headers), separator]
    lines.extend(_format_line(cells=row) for row in table_rows)
    return "\n".join(lines)


def _run_vial_segmentation_accuracy_analysis(
    *,
    manual_segmentation_image_path: Path,
    registered_atlas_image_path: Path,
) -> None:
    """Load segmentations, compute per-vial metrics, and print the results table.

    Args:
        manual_segmentation_image_path: Path to the manual labelled segmentation.
        registered_atlas_image_path: Path to the registered atlas segmentation.

    Raises:
        typer.BadParameter: If paths are missing or validation fails.
    """
    from spirit_phantom.core.vials import (  # noqa: PLC0415
        format_vial_segmentation_accuracy_table,
        generate_vial_segmentation_accuracy_table,
    )

    if not manual_segmentation_image_path.exists():
        msg = f"Manual segmentation file not found: {manual_segmentation_image_path}"
        raise typer.BadParameter(msg)
    if not registered_atlas_image_path.exists():
        msg = f"Registered atlas file not found: {registered_atlas_image_path}"
        raise typer.BadParameter(msg)

    try:
        rows = generate_vial_segmentation_accuracy_table(
            manual_segmentation_image_path=manual_segmentation_image_path,
            registered_atlas_image_path=registered_atlas_image_path,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    print(format_vial_segmentation_accuracy_table(rows=rows))


@app.command()
def register(  # noqa: PLR0913
    fixed_image: Annotated[
        Path,
        typer.Argument(help="Path to the fixed (scanner) image."),
    ],
    moving_image: Annotated[
        Path | None,
        typer.Argument(
            help=(
                "Path to the moving (atlas) image. If omitted, the default SPIRIT "
                "atlas is downloaded and cached with pooch."
            )
        ),
    ] = None,
    *,
    output_directory: Annotated[
        Path | None,
        typer.Option(
            "--output-directory",
            "-o",
            help="Directory for registration outputs. Defaults to a timestamped path.",
        ),
    ] = None,
    analyse: Annotated[
        AnalysisMethod | None,
        typer.Option(
            "--analyse",
            help="Optionally run an analysis immediately after registration.",
        ),
    ] = None,
    erosion_voxels: Annotated[
        int,
        typer.Option(
            "--erosion-voxels",
            help="Erosion voxels for vial measurement analysis.",
        ),
    ] = 0,
    generate_checkerboards: Annotated[
        bool,
        typer.Option(
            "--generate-checkerboards/--no-generate-checkerboards",
            help="Generate checkerboard overlay PNGs after registration.",
        ),
    ] = False,
    phantom_inverted: Annotated[
        bool,
        typer.Option(
            "--phantom-inverted/--no-phantom-inverted",
            help=(
                "Apply an initial 180-degree Y-rotation before registration to "
                "correct for an inverted phantom scan."
            ),
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress progress messages.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show extra progress detail, including library INFO logs.",
        ),
    ] = False,
) -> None:
    """Register an atlas image, with optional follow-up analysis.

    Args:
        fixed_image: Path to the fixed (scanner) image.
        moving_image: Path to the moving (atlas) image. Uses a downloaded,
            cached SPIRIT atlas by default.
        output_directory: Directory for registration outputs.
        analyse: Optional analysis method to run after registration.
        erosion_voxels: Erosion voxels for vial measurement analysis.
        generate_checkerboards: Whether checkerboard images should be generated.
        phantom_inverted: Whether an initial orientation correction should be
            applied for an inverted phantom scan.
        quiet: Suppress progress messages.
        verbose: Show additional progress detail and library INFO logs.
    """
    if quiet and verbose:
        msg = "Use either --quiet or --verbose, not both."
        raise typer.BadParameter(msg)

    if verbose:
        _configure_verbose_logging()

    if moving_image is not None:
        resolved_moving_image = moving_image
        _emit_cli_message(
            f"Using moving image: {resolved_moving_image}",
            quiet=quiet,
        )
    else:
        resolved_moving_image = _resolve_default_atlases(quiet=quiet)

    resolved_output_directory = (
        output_directory
        if output_directory is not None
        else _build_timestamped_output_directory()
    )
    if phantom_inverted:
        _emit_cli_message(
            "Phantom inverted: applying initial 180-degree Y-rotation.",
            quiet=quiet,
        )

    # Import ITK/elastix only after atlas prep so startup chatter is not blocked.
    _emit_cli_message("Loading registration engine (ITK)...", quiet=quiet)
    from spirit_phantom.core.registration import register_atlas  # noqa: PLC0415

    registration_result = register_atlas(
        moving_image=resolved_moving_image,
        fixed_image=fixed_image,
        output_directory=resolved_output_directory,
        cli_user=not quiet,
        phantom_inverted=phantom_inverted,
    )

    if registration_result.transformed_component_atlas_path is None:
        msg = "Registration did not produce a transformed component atlas."
        raise RuntimeError(msg)

    _emit_cli_message(f"Output directory: {resolved_output_directory}", quiet=quiet)
    _emit_cli_message(
        f"Registered atlas image: {registration_result.registered_image_path}",
        quiet=quiet,
    )
    _emit_cli_message(
        "Transformed component atlas: "
        f"{registration_result.transformed_component_atlas_path}",
        quiet=quiet,
    )
    _emit_cli_message(
        "Final registration transform: "
        f"{registration_result.registration_transform_path}",
        quiet=quiet,
    )

    if generate_checkerboards:
        _emit_cli_message(
            "Generating checkerboard visualisations based on registered signal atlas",
            quiet=quiet,
        )
        _generate_checkerboard_images(
            fixed_image_path=fixed_image,
            transformed_component_atlas_image_path=registration_result.transformed_component_atlas_path,
        )

    if analyse == AnalysisMethod.VIAL_MEASUREMENTS:
        _emit_cli_message("Running analysis: vial-measurements", quiet=quiet)
        _run_vial_measurements(
            transformed_component_atlas_image_path=registration_result.transformed_component_atlas_path,
            mri_scan_image_path=fixed_image,
            erosion_voxels=erosion_voxels,
            output_directory=resolved_output_directory,
            save_results=True,
        )


@analyse_app.command("vial-measurements")
def analyse_vial_measurements(
    transformed_component_atlas_image_path: Annotated[
        Path,
        typer.Argument(help="Path to the transformed component atlas image."),
    ],
    mri_scan_image_path: Annotated[
        Path,
        typer.Argument(help="Path to the scanner image used for intensity sampling."),
    ],
    erosion_voxels: Annotated[
        int,
        typer.Option(
            "--erosion-voxels",
            help="Number of voxels to erode each vial ROI.",
        ),
    ] = 0,
    output_directory: Annotated[
        Path | None,
        typer.Option(
            "--output-directory",
            "-o",
            help="Optional output directory. Saves vial_statistics_details.txt when set.",
        ),
    ] = None,
) -> None:
    """Run vial measurement analysis on an already registered atlas.

    Args:
        transformed_component_atlas_image_path: Path to the transformed component atlas image.
        mri_scan_image_path: Path to the scanner image.
        erosion_voxels: Number of voxels to erode each vial ROI.
        output_directory: Optional output directory for saved detailed results.
    """
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
    _run_vial_measurements(
        transformed_component_atlas_image_path=transformed_component_atlas_image_path,
        mri_scan_image_path=mri_scan_image_path,
        erosion_voxels=erosion_voxels,
        output_directory=output_directory,
        save_results=output_directory is not None,
    )


@analyse_app.command("dice")
def analyse_dice(
    manual_segmentation_image_path: Annotated[
        Path,
        typer.Argument(
            help=(
                "Manual vial segmentation (labels 1..20 for vials A..T) in "
                "scanner (fixed) image space."
            ),
        ),
    ],
    registered_atlas_image_path: Annotated[
        Path,
        typer.Argument(
            help=(
                "Registered component atlas labels in the same scanner space, "
                "typically registration output transformed_component_atlas.nii.gz."
            ),
        ),
    ],
) -> None:
    """Compute and print a per-vial Dice score table.

    Args:
        manual_segmentation_image_path: Manual segmentation on the scanner image
            where labels 1..20 map to vials A..T.
        registered_atlas_image_path: Labelled component atlas warped into scanner
            space (configured atlas segment indices per vial).
    """
    from spirit_phantom.core.vials import generate_dice_score_table  # noqa: PLC0415

    if not manual_segmentation_image_path.exists():
        msg = f"Manual segmentation file not found: {manual_segmentation_image_path}"
        raise typer.BadParameter(msg)
    if not registered_atlas_image_path.exists():
        msg = f"Registered atlas file not found: {registered_atlas_image_path}"
        raise typer.BadParameter(msg)

    try:
        rows = generate_dice_score_table(
            manual_segmentation_image_path=manual_segmentation_image_path,
            registered_atlas_image_path=registered_atlas_image_path,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    print(_format_dice_score_rows_table(rows=rows))


@analyse_app.command("vials")
def analyse_vials(
    manual_segmentation_image_path: Annotated[
        Path,
        typer.Argument(
            help=(
                "Manual vial segmentation (labels 1..20 for vials A..T) in "
                "scanner (fixed) image space."
            ),
        ),
    ],
    registered_atlas_image_path: Annotated[
        Path,
        typer.Argument(
            help=(
                "Registered component atlas labels in the same scanner space, "
                "typically registration output transformed_component_atlas.nii.gz."
            ),
        ),
    ],
) -> None:
    """Compute and print per-vial segmentation accuracy metrics.

    Same inputs as ``analyse dice``. Compares manual segmentation (ground truth) to
    the registered component atlas per vial. The table includes FPR, FNR, overlap
    counts, confusion counts (TP/FP/FN/TN), sensitivity, and specificity.

    Args:
        manual_segmentation_image_path: Manual segmentation on the scanner image
            where labels 1..20 map to vials A..T.
        registered_atlas_image_path: Labelled component atlas warped into scanner
            space (configured atlas segment indices per vial).
    """
    _run_vial_segmentation_accuracy_analysis(
        manual_segmentation_image_path=manual_segmentation_image_path,
        registered_atlas_image_path=registered_atlas_image_path,
    )


@analyse_app.command("eg-mask")
def analyse_eg_mask(
    registered_component_atlas: Annotated[
        Path,
        typer.Argument(
            help="Path to the registered component atlas labelled segmentation NIfTI image."
        ),
    ],
    multi_echo_scan: Annotated[
        Path,
        typer.Argument(help="Path to the multi-echo gradient echo NIfTI image."),
    ],
    *,
    output_mask_image_path: Annotated[
        Path | None,
        typer.Option(
            "--output-mask-image-path",
            "-o",
            help=(
                "Optional output path for the saved mask NIfTI. Defaults to "
                "ethylene_glycol_mask_<timestamp>.nii.gz in the multi-echo parent directory."
            ),
        ),
    ] = None,
    vis: Annotated[
        bool,
        typer.Option(
            "--vis",
            help="Save a Sum of Absolute Differences diagnostic plot for EG mask filtering.",
        ),
    ] = False,
    minimum_sad_counts: Annotated[
        float,
        typer.Option(
            "--minimum-sad-counts",
            help=(
                "Sum of Absolute Differences threshold in counts for EG mask filtering."
            ),
        ),
    ] = _EG_MASK_MIN_SAD_COUNTS_DEFAULT,
    dilation_iterations: Annotated[
        int,
        typer.Option(
            "--dilation-iterations",
            help=("Number of binary dilation iterations for the EG mask."),
        ),
    ] = _EG_MASK_DILATION_ITERATIONS_DEFAULT,
) -> None:
    """Create and save an ethylene glycol vial mask NIfTI file.

    Args:
        registered_component_atlas: Path to the registered component atlas.
        multi_echo_scan: Path to the multi-echo GRE image.
        output_mask_image_path: Optional explicit output path for the saved mask.
        vis: Whether to save a SAD diagnostic visualisation.
        minimum_sad_counts: Sum of Absolute Differences threshold.
        dilation_iterations: Binary dilation iteration count.
    """
    from spirit_phantom.core.multi_echo_thermometry import (  # noqa: PLC0415
        coordinate_ethylene_glycol_vial_segmentation,
    )

    if not registered_component_atlas.exists():
        msg = f"Registered component atlas file not found: {registered_component_atlas}"
        raise typer.BadParameter(msg)
    if not multi_echo_scan.exists():
        msg = f"Multi-echo gradient echo file not found: {multi_echo_scan}"
        raise typer.BadParameter(msg)
    if minimum_sad_counts < 0.0:
        msg = "--minimum-sad-counts must be greater than or equal to 0."
        raise typer.BadParameter(msg)
    if dilation_iterations < 0:
        msg = "--dilation-iterations must be greater than or equal to 0."
        raise typer.BadParameter(msg)

    try:
        saved_path = coordinate_ethylene_glycol_vial_segmentation(
            registered_component_atlas_image_path=registered_component_atlas,
            multi_echo_gradient_echo_scan_image_path=multi_echo_scan,
            output_mask_image_path=output_mask_image_path,
            minimum_sad_counts=minimum_sad_counts,
            dilation_iterations=dilation_iterations,
            generate_sad_visualisation=vis,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    print(f"Saved ethylene glycol mask: {saved_path}")
    if vis:
        if saved_path.name.endswith(".nii.gz"):
            mask_stem = saved_path.name[: -len(".nii.gz")]
        else:
            mask_stem = saved_path.stem
        print(
            "Saved EG mask Sum of Absolute Differences diagnostic plot: "
            f"{saved_path.parent / f'{mask_stem}_sad_filter_plot.png'}"
        )


def main() -> None:
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    main()

"""Command line interface for spirit-phantom workflows.

The registration command defaults to a SPIRIT atlas image downloaded with
`pooch` when a moving image is not provided explicitly.

The ``analyse`` command group includes:

- ``vial-measurements`` for per-vial intensity statistics.
- ``dice`` for per-vial overlap scores between manual and atlas labels.
"""

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from spirit_phantom import get_default_register_moving_image_path

app = typer.Typer(help="SPIRIT phantom command line tools.")
analyse_app = typer.Typer(help="Run analyses on registered phantom data.")
app.add_typer(analyse_app, name="analyse")


class AnalysisMethod(StrEnum):
    """Supported analysis methods for CLI workflows."""

    VIAL_MEASUREMENTS = "vial-measurements"


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
    registered_atlas_image_path: Path,
    mri_scan_image_path: Path,
    erosion_voxels: int,
    output_directory: Path | None,
    save_results: bool,
) -> None:
    """Run vial measurement analysis and optionally save detailed results.

    Args:
        registered_atlas_image_path: Path to the registered atlas image.
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
        registered_atlas_image_path=registered_atlas_image_path,
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
    registered_image_path: Path,
) -> None:
    """Generate checkerboard images for visual registration quality checks.

    Args:
        fixed_image_path: Path to the fixed image used as registration target.
        registered_image_path: Path to the registered moving image output.
    """
    from spirit_phantom.utils.visualisation import (  # noqa: PLC0415
        visualise_checkerboard,
    )

    slice_indices = _build_checkerboard_slice_indices(fixed_image_path=fixed_image_path)
    visualise_checkerboard(
        fixed_image_path=fixed_image_path,
        registered_image_path=registered_image_path,
        slice_indices=slice_indices,
    )
    print(f"Saved checkerboard images in: {registered_image_path.parent}")


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


@app.command()
def register(
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
    """
    # Delay heavy registration imports so `--help` stays responsive.
    from spirit_phantom.core.registration import register_atlas  # noqa: PLC0415

    resolved_moving_image = (
        moving_image
        if moving_image is not None
        else get_default_register_moving_image_path()
    )
    resolved_output_directory = (
        output_directory
        if output_directory is not None
        else _build_timestamped_output_directory()
    )
    print(
        "registration taking place using moving image located at: ",
        str(resolved_moving_image),
    )
    if phantom_inverted:
        print("Phantom inverted: applying initial 180-degree Y-rotation.")
    registration_result = register_atlas(
        moving_image=resolved_moving_image,
        fixed_image=fixed_image,
        output_directory=resolved_output_directory,
        cli_user=True,
        phantom_inverted=phantom_inverted,
    )

    print("Registration completed.")
    print(f"Output directory: {resolved_output_directory}")
    print(f"Registered atlas image: {registration_result.registered_image_path}")
    print(
        "Final registration transform: "
        f"{registration_result.registration_transform_path}"
    )

    if generate_checkerboards:
        print("Generating checkerboard visualisations")
        _generate_checkerboard_images(
            fixed_image_path=fixed_image,
            registered_image_path=registration_result.registered_image_path,
        )

    if analyse == AnalysisMethod.VIAL_MEASUREMENTS:
        print("Running analysis: vial-measurements")
        _run_vial_measurements(
            registered_atlas_image_path=registration_result.registered_image_path,
            mri_scan_image_path=fixed_image,
            erosion_voxels=erosion_voxels,
            output_directory=resolved_output_directory,
            save_results=True,
        )


@analyse_app.command("vial-measurements")
def analyse_vial_measurements(
    registered_atlas_image_path: Annotated[
        Path,
        typer.Argument(help="Path to the registered atlas image."),
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
        registered_atlas_image_path: Path to the registered atlas image.
        mri_scan_image_path: Path to the scanner image.
        erosion_voxels: Number of voxels to erode each vial ROI.
        output_directory: Optional output directory for saved detailed results.
    """
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
    _run_vial_measurements(
        registered_atlas_image_path=registered_atlas_image_path,
        mri_scan_image_path=mri_scan_image_path,
        erosion_voxels=erosion_voxels,
        output_directory=output_directory,
        save_results=output_directory is not None,
    )


@analyse_app.command("dice")
def analyse_dice(
    manual_segmentation_image_path: Annotated[
        Path,
        typer.Argument(help="Path to the manual labelled segmentation image."),
    ],
    registered_atlas_image_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the registered atlas labelled segmentation image."
        ),
    ],
) -> None:
    """Compute and print a per-vial Dice score table.

    Args:
        manual_segmentation_image_path: Path to the manual labelled
            segmentation where labels 1..20 map to vials A..T.
        registered_atlas_image_path: Path to the registered atlas labelled
            segmentation using configured atlas segment indices.
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


def main() -> None:
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    main()

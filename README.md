## Spirit Phantom library

Tools for analysing Gold Standard Phantoms (GSP) SPIRIT phantom data.

## Installation

### Python Version

We recommend using the latest supported version of Python. `spirit-phantom` currently supports Python 3.11–3.13.

### Dependencies

You must ensure that the following software is available on your system:

- **uv** (for environment and package management)
- **Python** (installed automatically by uv)
- **NumPy** (installed automatically as a project dependency)

Additional tools are required only for development (testing, linting, and documentation)
but are also automatically installed by uv:

- **pytest** and **pytest-cov**
- **ruff**
- **mypy**
- **tox-uv** (for replicating the checks as done by CI/CD)
- **mkdocs**, **mkdocs-material**, **mkdocstrings[python]**, **mkdocs-material** and **markdown-include**
- **scipy** and **scipy-stubs**

All of these are installed when you run `uv sync` in a development environment.

### Install spirit-phantom

If the package is available in your environment index, you can install it into an existing environment with `uv`:

```bash
uv add spirit-phantom
```

Or using `pip`:

```bash
pip install spirit-phantom
```

If you are working directly from a clone of this repository and want to install in editable mode with `pip`:

```bash
pip install -e .
```

## Usage

### Command Line Interface

The CLI supports both atomic and combined workflows:

- Register once, then run one or more analysis commands.
- Register and immediately run analysis in one command.
- Run standalone analysis commands such as per-vial Dice scoring.

Atomic registration:

```bash
uv run spirit-phantom register \
  path/to/scanner_image.nii.gz \
  --output-directory path/to/registration_output
```

Registration outputs are saved to `path/to/registration_output` as:

- `Rigid_Image.nii.gz`
- `Affine_Image.nii.gz`
- `Bspline_Image.nii.gz`

`Bspline_Image.nii.gz` is the final registration result and should be used for vial measurements.

By default, `register` downloads the default SPIRIT atlas image and caches it locally using `pooch` (cache namespace: `spirit-phantom`).
The pinned download URL and expected SHA-256 are configured in `src/spirit_phantom/cli.py`:

- URL: `https://raw.githubusercontent.com/gold-standard-phantoms/public-data/121591075cd927c0cb49cdffe91c3292c4882ffc/phantoms/SPIRIT/atlas/spirit_issue1.0_vx0.25_sub1.nii.gz`
- SHA-256: `77f027524325c4ad4d2d23ee8c224dfd04080216531c309d93e9fe41d686739d`
The first run may require network access; subsequent runs use the cached atlas.

To override the default atlas, pass a moving image path as the second argument:

```bash
uv run spirit-phantom register \
  path/to/scanner_image.nii.gz \
  path/to/atlas.nii.gz \
  --output-directory path/to/registration_output
```

Atomic vial measurement analysis (prints detailed table; saves only when output directory is provided):

```bash
uv run spirit-phantom analyse vial-measurements \
  path/to/registration_output/Bspline_Image.nii.gz \
  path/to/scanner_image.nii.gz \
  --erosion-voxels 0 \
  --output-directory path/to/analysis_output
```

Atomic Dice analysis (prints a per-vial table with label mapping and voxel overlap):

```bash
uv run spirit-phantom analyse dice \
  path/to/manual_segmentation.nii.gz \
  path/to/registration_output/Bspline_Image.nii.gz
```

Example command (Windows relative paths):

```powershell
uv run spirit-phantom analyse dice `
  \path\to\manually\segmented\vials `
  \path\to\registered\atlas `
```

Example output:

```text
vial_id | manual_label | atlas_label | dice_score | manual_voxels | atlas_voxels | intersection_voxels
--------+--------------+-------------+------------+---------------+--------------+--------------------
A       | 1            | 17          | 0.953306   | 71937         | 68809        | 67087
B       | 2            | 18          | 0.935174   | 72022         | 67798        | 65378
C       | 3            | 20          | 0.883508   | 83733         | 66706        | 66457
D       | 4            | 19          | 0.912901   | 79834         | 67952        | 67457
E       | 5            | 11          | 0.793889   | 31786         | 21884        | 21304
F       | 6            | 14          | 0.745040   | 28619         | 23099        | 19266
G       | 7            | 13          | 0.850703   | 26287         | 22696        | 20835
H       | 8            | 10          | 0.805932   | 26067         | 22751        | 19672
I       | 9            | 6           | 0.871800   | 25312         | 22582        | 20877
J       | 10           | 3           | 0.868149   | 21747         | 23577        | 19674
K       | 11           | 4           | 0.874160   | 28492         | 23153        | 22573
L       | 12           | 7           | 0.779448   | 25042         | 23128        | 18773
M       | 13           | 12          | 0.922296   | 21461         | 23067        | 20534
N       | 14           | 16          | 0.886580   | 25631         | 23064        | 21586
O       | 15           | 15          | 0.889272   | 24369         | 22656        | 20909
P       | 16           | 9           | 0.785631   | 27937         | 21529        | 19431
Q       | 17           | 5           | 0.841078   | 27870         | 21607        | 20807
R       | 18           | 1           | 0.886068   | 24611         | 22110        | 20699
S       | 19           | 2           | 0.857560   | 28164         | 22510        | 21728
T       | 20           | 8           | 0.883388   | 25756         | 21752        | 20984
```

The `analyse dice` command expects:

- A manual segmentation where labels `1..20` represent vials `A..T`.
- A registered atlas segmentation aligned to the same shape.

The output table includes:

- `vial_id`, `manual_label`, `atlas_label`
- `dice_score`
- `manual_voxels`, `atlas_voxels`, `intersection_voxels`

Interpretation notes:

- `dice_score` ranges from `0` (no overlap) to `1` (perfect overlap).
- `manual_label` and `atlas_label` show which connected components were matched for each vial.
- `intersection_voxels` is the overlap used in the Dice calculation.
- A lower Dice score with large voxel count differences can indicate local misregistration or segmentation mismatch.

If the two images have different shapes, the command exits with a clear validation error.

Combined workflow (register and then run vial measurements):

```bash
uv run spirit-phantom register \
  path/to/scanner_image.nii.gz \
  --output-directory path/to/registration_output \
  --analyse vial-measurements \
  --erosion-voxels 0
```

In the combined case, detailed vial statistics are saved automatically to:

`path/to/registration_output/vial_statistics_details.txt`

### Slice Thickness

The NEMA MS-5 2018 slice thickness function will be used as an example. 
The spirit-phantom function expects a numpy array for the wedge ROI ordered 
such that the rows each contain one edge transfer function (line up the edge)
with the signal increasing in the direction of the axis. 
An array with multiple rows will be used to input more than one edge transfer 
function.

```
from spirit_phantom.core import slice_thickness
spirit_slice_thickness = slice_thickness.nema_slice_thickness(
        volume_data_for_wedge_increasing_signal,
        pixel_size=pixel_size_mm,
    )
```

Calculation of slice thickness for the two wedges, checking for tilt along
the y-axis (NEMA MS-5 2018: Equation 6) and calculation of the mean is left
to the caller.

### Image Registration Function

The registration module provides functions for registering phantom images and
transforming point coordinates. Registration is performed using a multi-stage
approach: first a rigid (Euler) transform, then an affine transform, followed
by a B-spline transform.

To register a moving image to a fixed image:

```
from pathlib import Path
from spirit_phantom.core.registration import register_atlas

result = register_atlas(
    moving_image=Path("moving_image.nii.gz"),
    fixed_image=Path("fixed_image.nii.gz"),
    output_directory=Path("registration_output"),
)

# Access final registered image and transform (convenience aliases)
print(result.registered_image_path)       # Final registered image
print(result.registration_transform_path) # Final composed transform

# Access intermediate images
print(result.rigid_image_path)    # Rigid-registered image
print(result.affine_image_path)   # Affine-registered image
print(result.bspline_image_path)  # B-spline registered (same as registered_image_path)

# Access input parameters used for each stage
print(result.rigid_parameters_path)
print(result.affine_parameters_path)
print(result.bspline_parameters_path)

# Access output transforms for each stage
print(result.rigid_transform_path)
print(result.affine_transform_path)
print(result.bspline_transform_path)  # Same as registration_transform_path
```

The `register_atlas` function returns a `RegistrationResult` containing paths to all output files. All outputs are saved in the `output_directory`.

### Vial Statistics

The vial statistics module provides a method for extracting SPIRIT vial values
from a registered atlas and presenting a table of vial metadata and measurement
statistics.

```
from pathlib import Path

from spirit_phantom.core.vials import (
    compute_vial_statistics_details,
    print_vial_statistics_details_table,
    save_vial_statistics_details_table,
)

detailed_rows = compute_vial_statistics_details(
    registered_atlas_image_path=Path("registered_atlas.nii.gz"),
    mri_scan_image_path=Path("scanner_image.nii.gz"),
    erosion_voxels=0,
)
print_vial_statistics_details_table(rows=detailed_rows)
save_vial_statistics_details_table(
    rows=detailed_rows,
    output_path=Path("vial_statistics_detailed.txt"),
)
```

Eroding vial ROIs is usually beneficial because it reduces edge artefacts and
registration boundary effects. The best value depends on your image resolution
and analysis goal, so choose `erosion_voxels` based on your data.



An example of the output is with no erosion:

| Vial ID | Product Code | Description           | Mean Intensity | Stdev     | Number of Voxels |
|---------|--------------|-----------------------|----------------|-----------|------------------|
| A       | MNCL-0320    | 0.320mM Aqueous MnCl2 | 383.614261     | 77.880100 | 71875            |
| B       | MNCL-0159    | 0.159mM Aqueous MnCl2 | 279.192702     | 57.907651 | 71748            |
| C       | MNCL-0110    | 0.110mM Aqueous MnCl2 | 192.972366     | 40.317790 | 72267            |
| D       | MNCL-0039    | 0.039mM Aqueous MnCl2 | 369.782298     | 63.996589 | 71290            |
| E       | PVP-0050     | 5% Aqueous PVP NiCl2  | 198.488653     | 42.202034 | 22738            |
| F       | PVP-0100     | 10% Aqueous PVP NiCl2 | 200.131383     | 37.887874 | 23009            |
| G       | PVP-0150     | 15% Aqueous PVP NiCl2 | 368.268869     | 61.760893 | 23160            |
| H       | PVP-0200     | 20% Aqueous PVP NiCl2 | 362.967385     | 64.678591 | 23149            |
| I       | PVP-0250     | 25% Aqueous PVP NiCl2 | 380.053787     | 66.807876 | 22812            |
| J       | PVP-0300     | 30% Aqueous PVP NiCl2 | 371.809641     | 66.445551 | 22799            |
| K       | PVP-0400     | 40% Aqueous PVP NiCl2 | 277.656956     | 47.257150 | 22621            |
| L       | PVP-0500     | 50% Aqueous PVP       | 282.064033     | 51.086179 | 22879            |
| M       | MNCL-0078    | 0.078mM Aqueous MnCl2 | 341.567342     | 67.871481 | 24123            |
| N       | MNCL-0110    | 0.110mM Aqueous MnCl2 | 308.490297     | 59.404350 | 23447            |
| O       | MNCL-0480    | 0.480mM Aqueous MnCl2 | 397.452877     | 77.854880 | 23916            |
| P       | MNCL-0039    | 0.039mM Aqueous MnCl2 | 195.410605     | 34.654564 | 22669            |
| Q       | MNCL-0630    | 0.630mM Aqueous MnCl2 | 274.209387     | 51.286216 | 22456            |
| R       | MNCL-0320    | 0.320mM Aqueous MnCl2 | 341.248839     | 57.818041 | 22826            |
| S       | MNCL-0017    | 0.017mM Aqueous MnCl2 | 238.341952     | 45.232980 | 23012            |
| T       | MNCL-0159    | 0.159mM Aqueous MnCl2 | 359.478665     | 73.337451 | 23881            |

An example of the output is with `erosion_voxels=1`:

| Vial ID | Product Code | Description           | Mean Intensity | Stdev     | Number of Voxels |
|---------|--------------|-----------------------|----------------|-----------|------------------|
| A       | MNCL-0320    | 0.320mM Aqueous MnCl2 | 407.986083     | 13.173449 | 42251            |
| B       | MNCL-0159    | 0.159mM Aqueous MnCl2 | 297.407621     | 10.259056 | 42044            |
| C       | MNCL-0110    | 0.110mM Aqueous MnCl2 | 205.565967     | 9.084624  | 42347            |
| D       | MNCL-0039    | 0.039mM Aqueous MnCl2 | 388.539982     | 11.497023 | 42031            |
| E       | PVP-0050     | 5% Aqueous PVP NiCl2  | 204.181276     | 22.335345 | 11579            |
| F       | PVP-0100     | 10% Aqueous PVP NiCl2 | 204.625617     | 22.757468 | 11758            |
| G       | PVP-0150     | 15% Aqueous PVP NiCl2 | 376.756788     | 37.117770 | 11969            |
| H       | PVP-0200     | 20% Aqueous PVP NiCl2 | 372.564816     | 36.220253 | 11926            |
| I       | PVP-0250     | 25% Aqueous PVP NiCl2 | 388.671666     | 40.846955 | 11668            |
| J       | PVP-0300     | 30% Aqueous PVP NiCl2 | 381.078283     | 40.157710 | 11599            |
| K       | PVP-0400     | 40% Aqueous PVP NiCl2 | 281.891059     | 31.866884 | 11520            |
| L       | PVP-0500     | 50% Aqueous PVP       | 289.681632     | 27.082627 | 11738            |
| M       | MNCL-0078    | 0.078mM Aqueous MnCl2 | 355.301416     | 28.396466 | 12574            |
| N       | MNCL-0110    | 0.110mM Aqueous MnCl2 | 319.399306     | 32.319146 | 12096            |
| O       | MNCL-0480    | 0.480mM Aqueous MnCl2 | 413.484241     | 36.169857 | 12469            |
| P       | MNCL-0039    | 0.039mM Aqueous MnCl2 | 200.654553     | 18.833725 | 11530            |
| Q       | MNCL-0630    | 0.630mM Aqueous MnCl2 | 281.663908     | 28.163795 | 11360            |
| R       | MNCL-0320    | 0.320mM Aqueous MnCl2 | 351.518779     | 23.100912 | 11662            |
| S       | MNCL-0017    | 0.017mM Aqueous MnCl2 | 244.778041     | 22.960382 | 11804            |
| T       | MNCL-0159    | 0.159mM Aqueous MnCl2 | 373.729220     | 31.781529 | 12464            |

### Checkerboard visualisation

After registration, checkerboard images can be generated to visually inspect the
alignment between the fixed and registered images on selected axial slices:

```
from pathlib import Path

from spirit_phantom.utils.visualisation import visualise_checkerboard

visualise_checkerboard(
    fixed_image_path=Path("fixed_image.nii.gz"),
    registered_image_path=Path("registration_output/registered_image.nii.gz"),
    slice_indices=[180, 240, 300],
)
```

The `slice_indices` argument specifies the axial slice indices (z indices) at which
checkerboard images are generated, allowing visual assessment of registration
quality on slices of interest.

## Development

Steps to set up your environment for development on `spirit-phantom`:

- **Clone the repository** from GitHub.
- **Set up an environment** and install the development dependencies:

```bash
uv sync
```

To set up with a specific Python version (for example 3.11):

```bash
uv sync --python=3.11
```

Commit messages should follow the Conventional Commits specification:

`https://www.conventionalcommits.org`

To run the test suite:

```bash
uv run pytest
```

To check code quality:

```bash
uv run ruff check
uv run ruff format
uv run mypy src
```

### Documentation

Project documentation is built using MkDocs and `mkdocstrings`. To serve the documentation locally in a browser, run:

```bash
uv run mkdocs serve
```

Then open `http://127.0.0.1:8000/` in your browser.
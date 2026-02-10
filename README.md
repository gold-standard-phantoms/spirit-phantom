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

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

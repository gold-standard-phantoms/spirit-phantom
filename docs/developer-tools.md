# Developer tools

Quality-control helpers, local documentation, and the development environment.
These are not the main analysis products; use them to check registration,
segmentation overlap, and the docs site.

## Local documentation

From a clone of this repository:

```bash
uv sync
uv run mkdocs serve
```

Then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). MkDocs is a development
dependency, so a plain `uv pip install` from git is not enough.

Build a static site (used by GitHub Pages CI):

```bash
uv run mkdocs build --strict
```

## Development environment

```bash
uv sync
```

To pin a Python version (for example 3.11):

```bash
uv sync --python=3.11
```

Commit messages should follow [Conventional Commits](https://www.conventionalcommits.org).

Run the test suite:

```bash
uv run pytest
```

Check code quality:

```bash
uv run ruff check
uv run ruff format
uv run mypy src
```

## Dice score checker

Compare a manual vial segmentation with the registered component atlas.

CLI:

```bash
uv run spirit-phantom analyse dice \
  path/to/manual_segmentation.nii.gz \
  path/to/registration_output/transformed_component_atlas.nii.gz
```

Python:

```python
from pathlib import Path

from spirit_phantom.core.vials import generate_dice_score_table

rows = generate_dice_score_table(
    manual_segmentation_image_path=Path("manual_segmentation.nii.gz"),
    registered_atlas_image_path=Path("transformed_component_atlas.nii.gz"),
)
```

Inputs must share a voxel grid:

- Manual segmentation in scanner (fixed) image space, labels `1..22` for vials `A..V`.
- Registered component atlas (`transformed_component_atlas.nii.gz` from `register`),
  not the signal-only `Bspline_Image.nii.gz`.

Example columns:

```text
vial_id | manual_label | atlas_label | dice_score | manual_voxels | atlas_voxels | intersection_voxels
```

The full table has 22 rows (vials `A..V`). Thermometry vials `U` and `V` use
manual labels and atlas segment indices `21` and `22`.

Interpretation:

- `dice_score` ranges from `0` (no overlap) to `1` (perfect overlap).
- `manual_label` and `atlas_label` show which connected components were matched.
- `intersection_voxels` is the overlap used in the Dice calculation.
- A lower Dice score with large voxel-count differences can indicate local
  misregistration or segmentation mismatch.

If the two images have different shapes, the command exits with a validation error.

## Vial segmentation accuracy

Same inputs as the Dice checker, with extra confusion metrics. Manual
segmentation is treated as ground truth.

```bash
uv run spirit-phantom analyse vials \
  path/to/manual_segmentation.nii.gz \
  path/to/registration_output/transformed_component_atlas.nii.gz
```

Python equivalent: `generate_vial_segmentation_accuracy_table` and
`format_vial_segmentation_accuracy_table` in `spirit_phantom.core.vials`.

The table includes `dice_score`, `fpr`, `fnr`, voxel counts, `tp_voxels` /
`fp_voxels` / `fn_voxels` / `tn_voxels`, `sensitivity`, and `specificity`.

- `fpr` is `fp / (fp + tn)`; `fnr` is `fn / manual_voxels` (equal to `1 - sensitivity`).
- `intersection_voxels` and `tp_voxels` are identical.
- High `specificity` (for example `0.99`) means atlas false positives for that
  vial are rare outside the manual ROI.
- A higher `fnr` means a larger fraction of manual vial voxels were not captured
  by the registered atlas mask, even when `dice_score` remains moderate.

## Checkerboard quality control

After `register`, pass `--generate-checkerboards` to write overlay PNGs on evenly
spaced axial slices of the fixed image.

From Python, `visualise_checkerboard` in `spirit_phantom.utils.visualisation`
takes explicit slice indices. `visualise_checkerboard_tranformix` derives slices
from a transformix-format fiducial points file.

## Atlas integrity check

Download (or reuse) the cached default SPIRIT atlases and check that both NIfTI
files load:

```bash
uv run python -m spirit_phantom.test_atlas_resources
```

Prints `OK` or `FAIL` for the signal atlas and the component atlas.

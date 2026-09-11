# CLI usage

The `spirit-phantom` command is a [Typer](https://typer.tiangolo.com/) application. After install, start with `--help`:

```bash
uv run spirit-phantom --help
uv run spirit-phantom register --help
uv run spirit-phantom analyse --help
```

The CLI supports both atomic and combined workflows:

- Register once, then run one or more analysis commands.
- Register and immediately run vial measurements in one command.
- Run standalone analysis commands such as per-vial Dice scoring and ethylene glycol mask generation.

## Registration

Atomic registration (outputs are saved to a timestamped directory by default):

```bash
uv run spirit-phantom register \
  path/to/scanner_image.nii.gz \
  --output-directory path/to/registration_output
```

Registration outputs are saved to `path/to/registration_output` as:

- `Rigid_Image.nii.gz`
- `Affine_Image.nii.gz`
- `Bspline_Image.nii.gz`
- `transformed_component_atlas.nii.gz`

`Bspline_Image.nii.gz` is the final registered signal atlas image.
`transformed_component_atlas.nii.gz` is produced by applying the final transform to the
default component atlas and should be used for vial measurements.

By default, `register` uses the default SPIRIT atlas set and caches files locally using
`pooch` (cache namespace: `spirit-phantom`).
The pinned download URLs and expected SHA-256 values are configured in
`src/spirit_phantom/__init__.py`.

The first run may require network access; both the signal and component atlases are
prefetched before registration so the late component download does not look like a hang.
Subsequent runs reuse the cache.

If registration fails with a memory / RAM error, download a lower-resolution atlas
(for example `vx0.5` or `vx1.0` instead of the default `vx0.25`) from
[phantoms/SPIRIT/atlas](https://github.com/gold-standard-phantoms/public-data/tree/main/phantoms/SPIRIT/atlas),
pass it as the moving image, and use the matching `*_components.nii.gz` for analyses.
Also try closing other applications, or running on a machine with more RAM.

Registration prints numbered stage progress with elapsed times:

```text
[1/4] Rigid registration...
[1/4] Rigid registration done (18.2s)
[2/4] Affine registration...
...
[4/4] Transforming component atlas done (12.1s)
Registration complete (3m 16s)
```

Progress controls:

- `--quiet` / `-q`: suppress progress messages
- `--verbose` / `-v`: show additional detail, including library INFO logs

Do not pass `--quiet` and `--verbose` together.

Heavy dependencies are loaded only when needed: importing the CLI does not pull
multi-echo thermometry (matplotlib/nibabel), and ITK/elastix is imported after
atlas preparation when `register` runs.

```bash
uv run spirit-phantom register path/to/scanner_image.nii.gz --quiet
uv run spirit-phantom register path/to/scanner_image.nii.gz --verbose
```

To override the default atlas, pass a moving image path as the second argument:

```bash
uv run spirit-phantom register \
  path/to/scanner_image.nii.gz \
  path/to/atlas.nii.gz \
  --output-directory path/to/registration_output
```

If the phantom was scanned inverted (upside down), provide an initial orientation
correction before rigid registration:

```bash
uv run spirit-phantom register \
  path/to/scanner_image.nii.gz \
  --phantom-inverted \
  --output-directory path/to/registration_output
```

## Combined workflow

Register, analyse vial measurements, and generate checkerboard quality-control images in one step:

```bash
uv run spirit-phantom register \
  path/to/scanner_image.nii.gz \
  --analyse vial-measurements \
  --generate-checkerboards
```

Additional options can be combined:

```bash
uv run spirit-phantom register \
  path/to/scanner_image.nii.gz \
  --output-directory path/to/registration_output \
  --analyse vial-measurements \
  --erosion-voxels 1 \
  --generate-checkerboards
```

In the combined case, detailed vial statistics are saved automatically to:

`path/to/registration_output/vial_statistics_details.txt`

`--analyse` currently accepts only `vial-measurements`. Other analyses are standalone `analyse` subcommands.

## Standalone analyses

### `analyse vial-measurements`

Prints a detailed table. Saves `vial_statistics_details.txt` only when an output directory is provided:

```bash
uv run spirit-phantom analyse vial-measurements \
  path/to/registration_output/transformed_component_atlas.nii.gz \
  path/to/scanner_image.nii.gz \
  --erosion-voxels 0 \
  --output-directory path/to/analysis_output
```

Eroding vial ROIs is usually beneficial because it reduces edge artefacts and
registration boundary effects. Choose `--erosion-voxels` based on image resolution
and analysis goal.

Vial measurement tables list all 22 configured vials (`A..V`), including thermometry vials `U` and `V`.

### `analyse dice`

Per-vial Dice overlap between a manual segmentation and the registered component atlas.
Use a manual mask drawn on the scanner image and `transformed_component_atlas.nii.gz`
(not the signal-only `Bspline_Image.nii.gz`).

```bash
uv run spirit-phantom analyse dice \
  path/to/manual_segmentation.nii.gz \
  path/to/registration_output/transformed_component_atlas.nii.gz
```

The command expects:

- Manual segmentation in scanner (fixed) image space where labels `1..22` represent vials `A..V`.
- Registered component atlas segmentation on the same voxel grid.

The output table includes `vial_id`, `manual_label`, `atlas_label`, `dice_score`,
`manual_voxels`, `atlas_voxels`, and `intersection_voxels`.

Thermometry vials `U` and `V` use manual labels and atlas segment indices `21` and `22`.
If the two images have different shapes, the command exits with a validation error.

See [Developer tools](developer-tools.md) for how to interpret Dice scores.

### `analyse vials`

Same inputs as `analyse dice`. Prints per-vial confusion metrics with the manual
segmentation as ground truth (false positive/negative rates, sensitivity, specificity).

```bash
uv run spirit-phantom analyse vials \
  path/to/manual_segmentation.nii.gz \
  path/to/registration_output/transformed_component_atlas.nii.gz
```

See [Developer tools](developer-tools.md) for column meanings.

### `analyse eg-mask`

Map ethylene glycol atlas labels onto a multi-echo GRE scan and filter voxels with
a sum-of-absolute-differences (SAD) threshold:

```bash
uv run spirit-phantom analyse eg-mask \
  path/to/registered_component_atlas.nii.gz \
  path/to/multiecho_scan.nii.gz \
  --output-mask-image-path path/to/output_mask.nii.gz \
  --minimum-sad-counts 1000 \
  --dilation-iterations 1 \
  --vis
```

If `--output-mask-image-path` is omitted, the output defaults to:

`<parent of multiecho_scan>/ethylene_glycol_mask_<timestamp>.nii.gz`

This is the transfer step in the README workflow figure: the registered component
atlas is mapped onto a clinical (multi-echo) grid. It does not compute temperature;
full thermometry analysis is still in development.

When `--vis` is enabled, a diagnostic plot is saved next to the mask as
`<mask_stem>_sad_filter_plot.png`.

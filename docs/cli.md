# CLI usage

The `spirit-phantom` command is a [Typer](https://typer.tiangolo.com/) application. After install, start with `--help`:

```bash
uv run spirit-phantom --help
uv run spirit-phantom register --help
uv run spirit-phantom analyse --help
uv run spirit-phantom developer --help
```

The CLI has three top-level groups:

- `register` — atlas-to-scanner registration
- `analyse` — product analyses (`vial-measurements`, `slice-thickness`, `distortion`, `map-mask`, `eg-thermometry`)
- `developer` — quality-control helpers (`dice`, `vials`); see [Developer tools](developer-tools.md)

The CLI supports both atomic and combined workflows:

- Register once, then run one or more analysis commands.
- Register and immediately run vial measurements or slice thickness in one command.
- Run standalone analysis commands such as atlas label transfer (`map-mask`) and ethylene glycol thermometry (`eg-thermometry`).

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

Those checkerboard PNGs are registration quality-control overlays. See
[Interpreting outputs](interpreting-outputs.md) for how to read them.

In the combined case, detailed vial statistics are saved automatically to:

`path/to/registration_output/vial_statistics_details.txt`

`--analyse` accepts `vial-measurements`, `slice-thickness`, or `distortion`. To
register and measure slice thickness in one step:

```bash
uv run spirit-phantom register \
  path/to/scanner_image.nii.gz \
  --output-directory path/to/registration_output \
  --analyse slice-thickness
```

Slice-thickness outputs are written to
`path/to/registration_output/slice_thickness/`. `--ramp-slope-degrees` and
`--discard-edge-lines` behave as in `analyse slice-thickness` below, and only
take effect with `--analyse slice-thickness`.

Other analyses are standalone `analyse` subcommands.

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

### `analyse map-mask`

Map all non-zero atlas labels onto a scan that may sit on a different voxel grid.
This is the transfer step in the README workflow figure.

```bash
uv run spirit-phantom analyse map-mask \
  path/to/registered_component_atlas.nii.gz \
  path/to/clinical_scan.nii.gz \
  --output-mask-image-path path/to/mapped_atlas_mask.nii.gz
```

If `--output-mask-image-path` is omitted, the output defaults to:

`mapped_atlas_mask_<scan_stem>.nii.gz` beside the registered component atlas
(typically the registration output directory).

Skip this command when the clinical image is already the registration target:
`register` writes `transformed_component_atlas.nii.gz` on that grid. Use
`map-mask` when the scan of interest is a different grid (for example a
multi-echo series). The mapping uses nearest-neighbour pull resampling so
discrete labels are preserved and every target voxel is assigned.

### `analyse eg-thermometry`

Estimate temperature in the ethylene glycol NMR vials (atlas labels 21 and 22,
vials U and V). The command maps the registered component atlas onto the first
multi-echo GRE series, SAD-filters those labels, and fits the series with
[qmri](https://github.com/gold-standard-phantoms/qmri) multi-echo thermometry.

Every series must share the first image's spatial shape and affine. qmri
concatenates the echoes onto that one voxel grid and rejects a series that
differs. Pass one echo-time text file per series, in the same order as the
images. Each file is echo times in seconds, one value per echo.

`--field-strength` defaults to 3 T, the field SPIRIT ethylene glycol materials
are calibrated at. The value is passed to qmri directly, so a missing JSON
sidecar is not an error.

```bash
uv run spirit-phantom analyse eg-thermometry \
  path/to/registration_output/transformed_component_atlas.nii.gz \
  path/to/gre_series_1.nii.gz \
  path/to/gre_series_2.nii.gz \
  --echo-times path/to/series_1_te_s.txt \
  --echo-times path/to/series_2_te_s.txt \
  --output-directory path/to/eg_thermometry
```

`--method` accepts `regionwise` (default), `voxelwise`, or
`regionwise_bootstrap`. `--minimum-sad-counts` and `--dilation-iterations`
control the ethylene glycol mask. `--vis` saves diagnostic figures in the
output directory:

- `ethylene_glycol_mask_sad_filter_plot.png` — Sum of Absolute Differences
  traces kept and rejected by the mask filter
- `vial_U_signal_fit.png` and `vial_V_signal_fit.png` — measured echoes for
  each vial, with a separate marker for each GRE series, the dual-resonance
  fit, and the residual of the mean signal

If `--output-directory` is omitted, outputs are written to `eg_thermometry`
beside the first GRE image:

- `ethylene_glycol_mask.nii.gz`
- `eg_thermometry_temperature_map.nii.gz`
- `eg_thermometry_report.json`
- `temperature_measurements.txt`

### `analyse slice-thickness`

Measure NEMA wedge slice thickness from a previous `register` run. The
registration directory must contain `BSpline_Transform.txt`; this command does
not register.

```bash
uv run spirit-phantom analyse slice-thickness \
  path/to/registration_output \
  path/to/scanner_image.nii.gz \
  --ramp-slope-degrees 15 \
  --discard-edge-lines 1
```

Pass the moving (atlas) image as a third argument if registration used a
non-default atlas. Outputs default to
`path/to/registration_output/slice_thickness/` (override with
`--output-directory`).

Options:

- `--ramp-slope-degrees`: wedge angle α (15° for SPIRIT).
- `--discard-edge-lines`: profile lines dropped from each side of a wedge
  before averaging, to reduce partial-volume contamination (default 1; 0 keeps
  all five lines).

The terminal shows numbered stage progress and a per-wedge thickness table:

```text
[1/3] Inverse registration...
[1/3] Inverse registration done (36.0s)
[2/3] Saving wedge QC masks...
[2/3] Saving wedge QC masks done (11.8s)
[3/3] Measuring slice thickness...
[3/3] Measuring slice thickness done (1.4s)
wedge_label | thickness_mm | pixel_size_mm | n_lines | n_samples
1 | 1.3377 | 0.9984 | 3 | 51
2 | 1.3381 | 0.9988 | 3 | 51
Slice-thickness outputs: path/to/registration_output/slice_thickness
```

The output directory holds `slice_thickness_summary.csv`, per-wedge profile
CSVs and plots, wedge ROI and corner QC NIfTIs, and `slice_thickness.log` with
detailed diagnostics (inverse-transform residual, half-maximum crossings,
slice profiles). See
[Interpreting outputs](interpreting-outputs.md#slice-thickness) for what each
file shows and the checks to make before trusting the result.

### `analyse distortion`

Estimate scanner geometric distortion at the packaged SPIRIT fiducials from a
previous `register` run. The registration directory must contain
`Rigid_Transform.txt`, `Affine_Transform.txt` and `BSpline_Transform.txt`.

```bash
uv run spirit-phantom analyse distortion \
  path/to/registration_output \
  path/to/scanner_image.nii.gz
```

Pass the moving (atlas) image as a third argument if registration used a
non-default atlas. Outputs default to
`path/to/registration_output/distortion/` (override with `--output-directory`).

The terminal shows numbered stage progress (same inverse-registration banner as
slice thickness) and aggregate magnitudes:

```text
[1/4] Inverse registration...
[1/4] Inverse registration done (36.0s)
[2/4] Rigid-only transform...
[2/4] Rigid-only transform done (0.4s)
[3/4] Affine and B-spline transform...
[3/4] Affine and B-spline transform done (0.5s)
[4/4] Writing distortion outputs...
[4/4] Writing distortion outputs done (0.3s)
n_fiducials | mean_magnitude_mm | max_magnitude_mm | rms_magnitude_mm | max_nonrigid_roundtrip_residual_mm
34 | 0.4120 | 1.2030 | 0.5010 | 0.0120
Distortion outputs: path/to/registration_output/distortion
```

Method, open doubts, and file-by-file interpretation are in
[Scanner distortion](scanner-distortion.md) and
[Interpreting outputs](interpreting-outputs.md#scanner-distortion).

You can also chain it after registration:

```bash
uv run spirit-phantom register \
  path/to/scanner_image.nii.gz \
  --analyse distortion
```

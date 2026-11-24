# Python API usage

Import `spirit_phantom` from other Python programs. The snippets below cover the
main workflows. Function signatures and remaining helpers are in the MkDocs
[API reference](api/index.md) (those pages need `uv run mkdocs serve` or the hosted site).

All public functions that take paths or configuration use keyword arguments.

## Registration

Registration uses a multi-stage pipeline: rigid (Euler), then affine, then B-spline.

```python
from pathlib import Path

from spirit_phantom.core.registration import register_atlas

result = register_atlas(
    moving_image=Path("moving_image.nii.gz"),
    fixed_image=Path("fixed_image.nii.gz"),
    output_directory=Path("registration_output"),
    phantom_inverted=False,
)

print(result.registered_image_path)
print(result.registration_transform_path)
print(result.transformed_component_atlas_path)
print(result.rigid_image_path)
print(result.affine_image_path)
print(result.bspline_image_path)
```

`register_atlas` returns a `RegistrationResult` containing paths to all output files.
All outputs are saved in `output_directory`.

For CLI-equivalent isolation from elastix out-of-memory kills, call
`run_registration_isolated` from `spirit_phantom.core.guarded_registration` with the
same keyword arguments.

To apply an inverted-phantom correction yourself, use
`write_initial_flip_transform` in `spirit_phantom.core.initial_transform`.
The CLI `--phantom-inverted` flag does this before rigid registration.

Default atlas files can be downloaded and cached with:

```python
from spirit_phantom import (
    get_default_component_atlas_image_path,
    get_default_register_moving_image_path,
)

signal_atlas = get_default_register_moving_image_path()
component_atlas = get_default_component_atlas_image_path()
```

## Vial statistics

Extract SPIRIT vial values from a transformed component atlas:

```python
from pathlib import Path

from spirit_phantom.core.vials import (
    compute_vial_statistics_details,
    print_vial_statistics_details_table,
    save_vial_statistics_details_table,
)

detailed_rows = compute_vial_statistics_details(
    registered_atlas_image_path=Path("transformed_component_atlas.nii.gz"),
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
registration boundary effects. Choose `erosion_voxels` based on your data.
Tables list all 22 configured vials (`A..V`), including thermometry vials `U` and `V`.

## Slice thickness

To measure both SPIRIT wedges from a registration (the Python equivalent of
`analyse slice-thickness`):

```python
from pathlib import Path

from spirit_phantom.core.slice_thickness_analysis import (
    measure_slice_thickness_from_fixed_wedges,
)

results = measure_slice_thickness_from_fixed_wedges(
    fixed_image_path=Path("scanner_image.nii.gz"),
    transform_parameter_path=Path("registration_output/BSpline_Transform.txt"),
    moving_image_path=Path("registration_output/Bspline_Image.nii.gz"),
    atlas_image_path=Path("atlas.nii.gz"),
    output_directory=Path("registration_output/slice_thickness"),
    ramp_slope_degrees=15.0,
    discard_edge_lines=1,
)
for result in results:
    print(result.label, result.thickness_mm)
```

Each `WedgeThicknessResult` holds the thickness, sample spacing, line and
sample counts, and the wedge corners in scan space. The output directory
receives the same files as the CLI; see
[Interpreting outputs](interpreting-outputs.md#slice-thickness).

To compute thickness from your own edge-response data, call
`nema_slice_thickness` directly. It expects a NumPy array of shape
`(n_samples, n_lines)` in which each column is one edge response function (a
line up the edge) with the signal increasing along that axis. Multiple columns
are averaged as separate edge response functions.

```python
from spirit_phantom.core import slice_thickness

spirit_slice_thickness = slice_thickness.nema_slice_thickness(
    volume_data_for_wedge_increasing_signal,
    pixel_size=pixel_size_mm,
)
```

Calculation of slice thickness for the two wedges, checking for tilt along
the y-axis (NEMA MS-5 2018: Equation 6) and calculation of the mean is left
to the caller. Related helpers are `calculate_slice_profile` and
`full_width_half_maximum`.

## Atlas transfer (mapped mask)

Map every non-zero atlas label onto a scan that may sit on a different voxel
grid. This is the Python equivalent of `analyse map-mask`. Use it when the
image you want to analyse was not the registration target (for example a
multi-echo series). Skip this step when the clinical image is already the
fixed image used in `register`.

```python
from pathlib import Path

from spirit_phantom.core.atlas_mask import coordinate_mapped_atlas_mask

mask_path = coordinate_mapped_atlas_mask(
    registered_component_atlas_image_path=Path(
        "registered_data/20261006_194157/transformed_component_atlas.nii.gz"
    ),
    scan_image_path=Path("clinical_scan.nii.gz"),
)
print(mask_path)
```

When `output_mask_image_path` is omitted, the mask is written beside the
registered atlas as `mapped_atlas_mask_<scan_stem>.nii.gz`.
Pass `output_mask_image_path=` to override.

For in-memory transfer (no file write), use
`transfer_atlas_labels_to_image_space` or `build_atlas_in_target_space` in
`spirit_phantom.io.atlas_transfer`. Both return a `uint8` array with shape
`target_image.shape[:3]`. Pass `labels=` to keep a subset of atlas values;
the default keeps all non-zero labels. Nearest-neighbour pull resampling
preserves discrete labels and fills every target voxel.

## Ethylene glycol mask

Map EG atlas labels onto a multi-echo GRE scan and apply SAD filtering.
`analyse eg-thermometry` calls this before the temperature fit. The function
uses the same 3-D atlas transfer as `map-mask`, but only for ethylene glycol
labels, then dilates and SAD-filters. It does not compute temperature.

```python
from pathlib import Path

from spirit_phantom.core.multi_echo_thermometry import (
    coordinate_ethylene_glycol_vial_segmentation,
)

mask_path = coordinate_ethylene_glycol_vial_segmentation(
    registered_component_atlas_image_path=Path("transformed_component_atlas.nii.gz"),
    multi_echo_gradient_echo_scan_image_path=Path("multiecho_scan.nii.gz"),
    output_mask_image_path=Path("ethylene_glycol_mask.nii.gz"),
    minimum_sad_counts=1000.0,
    dilation_iterations=1,
    generate_sad_visualisation=True,
)
print(mask_path)
```

## Ethylene glycol thermometry

Estimate temperature in vials U and V (atlas labels 21 and 22). This is the
Python equivalent of `analyse eg-thermometry`. Supply one qmri echo-time file
per GRE series, in seconds, one value per echo. The series must share the
first image's spatial shape and affine, because qmri fits them as one echo
train on that grid. `magnetic_field_tesla` defaults to 3.

```python
from pathlib import Path

from spirit_phantom.core.multi_echo_thermometry import (
    EchoTrain,
    coordinate_ethylene_glycol_thermometry,
)

result = coordinate_ethylene_glycol_thermometry(
    registered_component_atlas_image_path=Path("transformed_component_atlas.nii.gz"),
    trains=(
        EchoTrain(
            image_path=Path("gre_series_1.nii.gz"),
            echo_times_file=Path("series_1_te_s.txt"),
        ),
        EchoTrain(
            image_path=Path("gre_series_2.nii.gz"),
            echo_times_file=Path("series_2_te_s.txt"),
        ),
    ),
    output_directory=Path("eg_thermometry"),
)
for vial in result.vials:
    print(
        f"{vial.vial_id}: {vial.temperature_celsius:.2f} °C "
        f"(± {vial.temperature_uncertainty_celsius:.2f})"
    )
```

`generate_visualisations=True` also writes the SAD mask plot and one
signal-and-fit figure per vial (`vial_U_signal_fit.png`,
`vial_V_signal_fit.png`). Each signal figure marks the GRE series separately
and shows the residual of the mean signal. The paths are on
`result.visualisation_paths`. See
[Interpreting outputs](interpreting-outputs.md#ethylene-glycol-thermometry)
for how to read the table and the figures.

## Checkerboard visualisation

After registration, checkerboard images can be generated to inspect alignment
on selected axial slices. See [Interpreting outputs](interpreting-outputs.md)
for how to read a successful registration against a failed one.

```python
from pathlib import Path

from spirit_phantom.utils.visualisation import visualise_checkerboard

checkerboard_paths = visualise_checkerboard(
    fixed_image_path=Path("fixed_image.nii.gz"),
    registered_image_path=Path("registration_output/registered_image.nii.gz"),
    slice_indices=[180, 240, 300],
    save_directory=Path("registration_output/checkerboards"),
)

for checkerboard_path in checkerboard_paths:
    print(checkerboard_path)
```

`slice_indices` are axial slice indices (z). `visualise_checkerboard` returns a
list of output PNG paths. Each image is saved in `save_directory`, or beside
`registered_image_path` when no save directory is provided.

`visualise_checkerboard_tranformix` picks slices from a transformix-format
fiducial points file instead of an explicit slice list.

## Transformix point I/O

Load and save point sets in elastix/transformix text format with
`spirit_phantom.io.points.load_points`, `save_points`, and
`parse_transformix_output`.

`invert_points_through_transformix` maps atlas (moving) points into scan
(fixed) space. By default it estimates the inverse transform with elastix
(`DisplacementMagnitudePenalty`, via
`spirit_phantom.core.registration.compute_inverse_transform`) and applies it
with transformix. Pass `method=InversionMethod.PICARD` to use per-point
fixed-point iteration instead; this only converges for small rotations and
fails for inverted-phantom registrations.

## Scanner distortion

After registration, estimate geometric distortion at the packaged SPIRIT
fiducials from the affine and B-spline stages (rigid excluded). See
[Scanner distortion](scanner-distortion.md) for the method and open doubts.

```python
from pathlib import Path

from spirit_phantom.core.distortion import evaluate_scanner_distortion
from spirit_phantom.core.fiducials import load_spirit_fiducials

fiducials = load_spirit_fiducials()
assert len(fiducials) == 34

result = evaluate_scanner_distortion(
    registration_directory=Path("registration_output"),
    fixed_image_path=Path("scanner_image.nii.gz"),
    output_directory=Path("registration_output/distortion"),
)
print(result.summary.mean_magnitude_mm, result.summary.max_magnitude_mm)
```

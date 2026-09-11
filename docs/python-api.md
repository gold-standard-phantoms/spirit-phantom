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

NEMA MS-5 2018 slice thickness is Python-only (there is no CLI command).
The function expects a NumPy array for the wedge ROI ordered so that each row
contains one edge transfer function (line up the edge) with the signal increasing
along that axis. Multiple rows average more than one edge transfer function.

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

## Ethylene glycol mask (thermometry precursor)

Map EG atlas labels onto a multi-echo GRE scan and apply SAD filtering.
This is the Python equivalent of `analyse eg-mask`. It does not compute temperature.

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

## Checkerboard visualisation

After registration, checkerboard images can be generated to inspect alignment
on selected axial slices:

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

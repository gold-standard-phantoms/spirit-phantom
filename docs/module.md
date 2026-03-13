## API Reference

### Registration Module

Registration uses a staged pipeline (rigid, affine, then B-spline) and returns a
`RegistrationResult` containing convenient paths to stage outputs and the final
registered image and transform.

::: spirit_phantom.core.registration
    rendering:
      show_if_no_docstring: true
      show_signature_annotations: true
      show_bases: true
      show_source: false

### Vials Module

Vial statistics can be computed as detailed metadata-rich rows using
`compute_vial_statistics_details`. ROI erosion is configurable via
`erosion_voxels`; erosion is generally recommended to reduce boundary artefacts,
but the best value depends on image resolution and analysis goals.

::: spirit_phantom.core.vials
    rendering:
      show_if_no_docstring: true
      show_signature_annotations: true
      show_bases: true
      show_source: false


### Slice Thickness Module

::: spirit_phantom.core.slice_thickness
    rendering:
      show_if_no_docstring: true
      show_signature_annotations: true
      show_bases: true
      show_source: false


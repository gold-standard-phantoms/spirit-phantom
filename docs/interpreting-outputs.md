# Interpreting outputs

How to read common quality-control and analysis products from this package. This page will grow as more outputs are documented; the first example is registration checkerboard images.

## Checkerboard images

After registration, `--generate-checkerboards` writes PNG overlays that compare the registered atlas with the MRI on selected axial slices. The overlay is a [checkerboard pattern](https://examples.itk.org/src/filtering/imagecompare/combinetwoimageswithcheckerboardpattern/documentation): the image is divided into a grid of tiles that alternate between the registered atlas and the scanner image.

Where atlas structures line up with the corresponding MRI structures across tile boundaries, orientation and alignment are good. Breaks, steps, or offsets at those boundaries indicate misregistration.

### Pay attention to the slice wedges

The slice-thickness wedges are especially useful for spotting small translational or rotational errors. Check that their edges continue smoothly from atlas tiles into MRI tiles.

### Good registration

In a successful registration, phantom outlines, vials, and wedges remain continuous across tile edges even though intensity and texture change from tile to tile.

| ![Good registration checkerboard, slice 85](images/good/checkerboard_itk_slice_085.png) | ![Good registration checkerboard, slice 128](images/good/checkerboard_itk_slice_128.png) | ![Good registration checkerboard, slice 170](images/good/checkerboard_itk_slice_170.png) |
| --- | --- | --- |

### Poor registration

When registration fails, the same structures do not meet at tile boundaries: circles look stepped, vial walls jump, and slice wedges are offset between atlas and MRI tiles.

| ![Poor registration checkerboard, slice 85](images/poor/checkerboard_itk_slice_085.png) | ![Poor registration checkerboard, slice 128](images/poor/checkerboard_itk_slice_128.png) | ![Poor registration checkerboard, slice 170](images/poor/checkerboard_itk_slice_170.png) |
| --- | --- | --- |

Registration performed with the phantom and atlas inverted with respect to one another.

Generate these overlays from the CLI with `--generate-checkerboards`, or from Python with `visualise_checkerboard` (see [CLI usage](cli.md) and [Python API usage](python-api.md)).

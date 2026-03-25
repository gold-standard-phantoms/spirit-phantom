"""Tests for transforming points from fixed to moving image space."""

from __future__ import annotations

import json
from pathlib import Path
from time import gmtime, strftime

import itk
import numpy as np

from spirit_phantom.io.point_transforms import transform_points_fixed_to_moving
from spirit_phantom.io.points import save_points

IMAGE_SIZE = (256, 256)
ROTATION_RADIANS = 0.2
TRANSLATION_X = -4.564513
TRANSLATION_Y = -2.091174
ROTATION_CENTRE_X = 128.0
ROTATION_CENTRE_Y = 128.0


def _create_moving_image(*, output_path: Path) -> None:
    """Create and save a synthetic 2D moving image.

    Args:
        output_path: File path where the image is written.
    """
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, IMAGE_SIZE[0], dtype=np.float32),
        np.linspace(0.0, 1.0, IMAGE_SIZE[1], dtype=np.float32),
        indexing="ij",
    )
    image_array = (yy + xx) * 100.0
    moving_image = itk.image_from_array(image_array.astype(np.float32))
    itk.imwrite(moving_image, str(output_path))


def _build_elastix_transform_text() -> str:
    """Build a minimal Elastix Euler transform parameter file."""
    return "\n".join(
        (
            '(Transform "EulerTransform")',
            "(NumberOfParameters 3)",
            (
                "(TransformParameters "
                f"{ROTATION_RADIANS:.6f} {TRANSLATION_X:.6f} {TRANSLATION_Y:.6f})"
            ),
            '(InitialTransformParameterFileName "NoInitialTransform")',
            '(HowToCombineTransforms "Compose")',
            "// Image specific",
            "(FixedImageDimension 2)",
            "(MovingImageDimension 2)",
            '(FixedInternalImagePixelType "float")',
            '(MovingInternalImagePixelType "float")',
            "(Size 256 256)",
            "(Index 0 0)",
            "(Spacing 1.0000000000 1.0000000000)",
            "(Origin 0.0000000000 0.0000000000)",
            "(Direction 1.0000000000 0.0000000000 0.0000000000 1.0000000000)",
            "// EulerTransform specific",
            (
                "(CenterOfRotationPoint "
                f"{ROTATION_CENTRE_X:.10f} {ROTATION_CENTRE_Y:.10f})"
            ),
            "// ResampleInterpolator specific",
            '(ResampleInterpolator "FinalBSplineInterpolator")',
            "(FinalBSplineInterpolationOrder 3)",
            "// Resampler specific",
            '(Resampler "DefaultResampler")',
            "(DefaultPixelValue 0.000000)",
            '(ResultImageFormat "mhd")',
            '(ResultImagePixelType "short")',
            "",
        ),
    )


def _write_transform_file(*, output_path: Path) -> None:
    """Write Elastix Euler transform parameters to disk.

    Args:
        output_path: Destination path of the transform parameter file.
    """
    output_path.write_text(_build_elastix_transform_text(), encoding="utf-8")


def _fixed_domain_points() -> np.ndarray:
    """Return test points in the fixed domain.

    Returns:
        A ``[N, 2]`` array of ``x/y`` points.
    """
    return np.array(
        [
            [32.0, 64.0],
            [128.0, 128.0],
            [240.0, 16.0],
            [200.5, 111.25],
        ],
        dtype=np.float64,
    )


def _expected_points_in_moving_domain(*, fixed_points: np.ndarray) -> np.ndarray:
    """Compute expected moving-domain points for an Euler transform.

    Args:
        fixed_points: Points in fixed domain with shape ``[N, 2]``.

    Returns:
        Points transformed into moving domain with shape ``[N, 2]``.
    """
    rotation_matrix = np.array(
        [
            [np.cos(ROTATION_RADIANS), -np.sin(ROTATION_RADIANS)],
            [np.sin(ROTATION_RADIANS), np.cos(ROTATION_RADIANS)],
        ],
        dtype=np.float64,
    )
    centre = np.array([ROTATION_CENTRE_X, ROTATION_CENTRE_Y], dtype=np.float64)
    translation = np.array([TRANSLATION_X, TRANSLATION_Y], dtype=np.float64)
    centred = fixed_points - centre
    rotated = centred @ rotation_matrix.T
    return rotated + centre + translation


def _run_transform_pipeline(*, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Run a full transformix point transform pipeline.

    Args:
        output_dir: Directory where intermediate and output files are saved.

    Returns:
        Tuple of ``(actual_points, expected_points)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    moving_image_path = output_dir / "moving_image.mha"
    transform_path = output_dir / "transform_parameters.txt"
    points_path = output_dir / "fixed_points.txt"

    _create_moving_image(output_path=moving_image_path)
    _write_transform_file(output_path=transform_path)
    fixed_points = _fixed_domain_points()
    save_points(
        points=fixed_points.tolist(),
        output_path=points_path,
        point_type="point",
    )

    transformed_points = transform_points_fixed_to_moving(
        moving_image_path=moving_image_path,
        registration_transform_path=transform_path,
        points_in_fixed_domain_path=points_path,
        save_path=output_dir,
    )
    expected_points = _expected_points_in_moving_domain(fixed_points=fixed_points)

    save_points(
        points=expected_points.tolist(),
        output_path=output_dir / "expected_points.txt",
        point_type="point",
    )
    return transformed_points, expected_points


def test_transform_points_fixed_to_moving_with_temporary_transform_file(
    tmp_path: Path,
) -> None:
    """Transform fixed-domain points using a generated Euler transform file."""
    transformed_points, expected_points = _run_transform_pipeline(output_dir=tmp_path)

    assert transformed_points.shape == expected_points.shape
    assert np.allclose(transformed_points, expected_points, atol=1e-5)

    assert (tmp_path / "moving_image.mha").exists()
    assert (tmp_path / "transform_parameters.txt").exists()
    assert (tmp_path / "fixed_points.txt").exists()
    assert (tmp_path / "transformed_points.txt").exists()
    assert (tmp_path / "transformix_output_points.txt").exists()
    assert (tmp_path / "expected_points.txt").exists()


def main() -> None:
    """Run the transform pipeline and save timestamped debug artefacts.

    This helps manual validation of transform behaviour and intermediate files
    generated by transformix.
    """
    timestamp = strftime("%Y%m%d_%H%M%S", gmtime())
    output_dir = Path.cwd() / "debug_transform_outputs" / f"transform_{timestamp}"
    transformed_points, expected_points = _run_transform_pipeline(output_dir=output_dir)

    diagnostics = {
        "max_abs_error": float(np.max(np.abs(transformed_points - expected_points))),
        "n_points": int(transformed_points.shape[0]),
        "output_directory": str(output_dir),
        "recommendation": (
            "For a custom registration smoke test, use the atlas, apply a known "
            "voxel translation, and scale image intensities to verify robustness."
        ),
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    (output_dir / "run_complete.txt").write_text(
        f"Saved transform debug artefacts to: {output_dir}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Tests for multi-echo thermometry EG mask generation and temperature fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import nibabel
import numpy as np
import pytest

from spirit_phantom.core.multi_echo_thermometry import (
    DEFAULT_MAGNETIC_FIELD_TESLA,
    EchoTrain,
    _assert_shared_multiecho_grid,
    coordinate_ethylene_glycol_thermometry,
    coordinate_ethylene_glycol_vial_segmentation,
    dilate_segmentation_mask,
    load_echo_times_seconds,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_dilate_segmentation_mask_expands_foreground_voxels() -> None:
    """Binary dilation should grow the initial foreground support."""
    mask = np.zeros(shape=(5, 5, 1), dtype=np.uint8)
    mask[2, 2, 0] = np.uint8(1)

    dilated_mask = dilate_segmentation_mask(mask=mask, iterations=1)

    assert dilated_mask.shape == mask.shape
    assert dilated_mask.dtype == np.uint8
    assert int(np.count_nonzero(dilated_mask)) > int(np.count_nonzero(mask))


def test_dilate_segmentation_mask_zero_iterations_keeps_mask_unchanged() -> None:
    """Zero iterations should return the original binary foreground mask."""
    mask = np.zeros(shape=(5, 5, 1), dtype=np.uint8)
    mask[2, 2, 0] = np.uint8(3)
    mask[1, 3, 0] = np.uint8(1)

    undilated_mask = dilate_segmentation_mask(mask=mask, iterations=0)

    expected_binary_mask = (mask != 0).astype(np.uint8)
    assert undilated_mask.shape == mask.shape
    assert undilated_mask.dtype == np.uint8
    np.testing.assert_array_equal(undilated_mask, expected_binary_mask)


def test_coordinate_eg_mask_applies_sad_filter_and_writes_plot(
    tmp_path: Path,
) -> None:
    """Generated EG mask should retain only voxels above the SAD threshold.

    For a voxel trace ``S`` with 5 time points, the Sum of Absolute
    Differences (SAD) is:

    ``SAD = |S1 - S0| + |S2 - S1| + |S3 - S2| + |S4 - S3|``.

    This test uses:

    - ``[0,  20, 0,  20, 0]``   -> `` 20 +  20 +  20 +  20 =  80``  (excluded)
    - ``[0, 100, 80, 100, 80]`` -> ``100 + 20 + 20 + 20 = 160``     (excluded)
    - ``[0, 260, 0, 260, 0]``   -> ``260 + 260 + 260 + 260 = 1040`` (included)
    - ``[0, 300, 0, 300, 0]``   -> ``300 + 300 + 300 + 300 = 1200`` (included)
    """
    atlas_data = np.zeros(shape=(5, 5, 1), dtype=np.uint8)
    atlas_data[1, 1, 0] = np.uint8(21)
    atlas_data[2, 2, 0] = np.uint8(21)
    atlas_data[3, 3, 0] = np.uint8(22)
    atlas_data[4, 4, 0] = np.uint8(22)
    atlas_image = nibabel.Nifti1Image(dataobj=atlas_data, affine=np.eye(4))
    atlas_path = tmp_path / "atlas.nii.gz"
    nibabel.save(atlas_image, str(atlas_path))

    thermometry_data = np.zeros(shape=(5, 5, 1, 5), dtype=np.float32)
    thermometry_data[1, 1, 0, :] = np.array(
        [0.0, 20.0, 0.0, 20.0, 0.0], dtype=np.float32
    )
    thermometry_data[2, 2, 0, :] = np.array(
        [0.0, 100.0, 80.0, 100.0, 80.0], dtype=np.float32
    )
    thermometry_data[3, 3, 0, :] = np.array(
        [0.0, 260.0, 0.0, 260.0, 0.0], dtype=np.float32
    )
    thermometry_data[4, 4, 0, :] = np.array(
        [0.0, 300.0, 0.0, 300.0, 0.0], dtype=np.float32
    )
    thermometry_image = nibabel.Nifti1Image(dataobj=thermometry_data, affine=np.eye(4))
    thermometry_path = tmp_path / "thermometry.nii.gz"
    nibabel.save(thermometry_image, str(thermometry_path))

    output_mask_path = tmp_path / "ethylene_glycol_mask.nii.gz"
    saved_mask_path = coordinate_ethylene_glycol_vial_segmentation(
        registered_component_atlas_image_path=atlas_path,
        multi_echo_gradient_echo_scan_image_path=thermometry_path,
        minimum_sad_counts=1000.0,
        dilation_iterations=0,
        output_mask_image_path=output_mask_path,
        generate_sad_visualisation=True,
    )

    saved_mask_image = cast(
        "nibabel.nifti1.Nifti1Image", nibabel.load(str(saved_mask_path))
    )
    saved_mask = np.asarray(saved_mask_image.get_fdata(), dtype=np.uint8)
    assert int(np.count_nonzero(saved_mask)) == 2
    assert float(saved_mask[1, 1, 0]) == 0.0  # excluded with SAD < 1000
    assert float(saved_mask[2, 2, 0]) == 0.0  # excluded with SAD < 1000
    assert float(saved_mask[3, 3, 0]) == 22.0  # included with SAD >= 1000
    assert float(saved_mask[4, 4, 0]) == 22.0  # included with SAD >= 1000
    assert set(np.unique(saved_mask).tolist()) == {0, 22}

    expected_plot_path = tmp_path / "ethylene_glycol_mask_sad_filter_plot.png"
    assert expected_plot_path.exists()

    expected_mapped_label_mask_path = (
        tmp_path / "ethylene_glycol_mask_mapped_label_mask.nii.gz"
    )
    assert expected_mapped_label_mask_path.exists()
    mapped_label_mask_image = cast(
        "nibabel.nifti1.Nifti1Image", nibabel.load(str(expected_mapped_label_mask_path))
    )
    mapped_label_mask = np.asarray(mapped_label_mask_image.get_fdata(), dtype=np.uint8)
    assert float(mapped_label_mask[1, 1, 0]) == 21.0
    assert float(mapped_label_mask[2, 2, 0]) == 21.0
    assert float(mapped_label_mask[3, 3, 0]) == 22.0
    assert float(mapped_label_mask[4, 4, 0]) == 22.0


def _save_multiecho(
    *,
    path: Path,
    shape: tuple[int, int, int, int] = (4, 4, 1, 4),
    affine: np.ndarray | None = None,
    data: np.ndarray | None = None,
) -> None:
    """Write a small 4-D NIfTI for thermometry input checks."""
    volume = (
        np.zeros(shape=shape, dtype=np.float32)
        if data is None
        else np.asarray(data, dtype=np.float32)
    )
    image = nibabel.Nifti1Image(
        dataobj=volume,
        affine=np.eye(4) if affine is None else affine,
    )
    nibabel.save(img=image, filename=str(path))


def _echo_train(
    *,
    image_path: Path,
    echo_times_file: Path,
    n_echoes: int,
) -> EchoTrain:
    """Write ``n_echoes`` evenly spaced echo times and pair them with an image."""
    np.savetxt(
        fname=echo_times_file,
        X=np.linspace(0.001, 0.024, n_echoes),
        fmt="%.8f",
    )
    return EchoTrain(image_path=image_path, echo_times_file=echo_times_file)


def test_load_echo_times_seconds_rejects_two_dimensional_file(tmp_path: Path) -> None:
    """An echo-time file must be a 1-D list, matching qmri's loader."""
    echo_times_file = tmp_path / "echo_times.txt"
    echo_times_file.write_text("0.001 0.002\n0.003 0.004\n", encoding="utf-8")

    with pytest.raises(ValueError, match="1D list"):
        load_echo_times_seconds(echo_times_file=echo_times_file)


def test_coordinate_thermometry_rejects_echo_time_length_mismatch(
    tmp_path: Path,
) -> None:
    """The echo-time file length must equal the image's echo count."""
    image_path = tmp_path / "gre.nii.gz"
    _save_multiecho(path=image_path, shape=(4, 4, 1, 4))
    echo_times_file = tmp_path / "echo_times.txt"
    np.savetxt(fname=echo_times_file, X=np.array([0.001, 0.002]), fmt="%.8f")
    atlas_path = tmp_path / "atlas.nii.gz"
    atlas_path.write_bytes(b"unused")

    with pytest.raises(ValueError, match="2 echo times"):
        coordinate_ethylene_glycol_thermometry(
            registered_component_atlas_image_path=atlas_path,
            trains=(EchoTrain(image_path=image_path, echo_times_file=echo_times_file),),
            output_directory=tmp_path / "out",
        )

    assert not (tmp_path / "out").exists()


def test_coordinate_thermometry_rejects_different_spatial_shapes(
    tmp_path: Path,
) -> None:
    """Every series is fitted on the first image's spatial grid."""
    first_path = tmp_path / "first.nii.gz"
    second_path = tmp_path / "second.nii.gz"
    _save_multiecho(path=first_path, shape=(4, 4, 1, 4))
    _save_multiecho(path=second_path, shape=(5, 4, 1, 4))
    atlas_path = tmp_path / "atlas.nii.gz"
    atlas_path.write_bytes(b"unused")
    trains = (
        _echo_train(
            image_path=first_path,
            echo_times_file=tmp_path / "first_te.txt",
            n_echoes=4,
        ),
        _echo_train(
            image_path=second_path,
            echo_times_file=tmp_path / "second_te.txt",
            n_echoes=4,
        ),
    )

    with pytest.raises(ValueError, match="spatial shape"):
        coordinate_ethylene_glycol_thermometry(
            registered_component_atlas_image_path=atlas_path,
            trains=trains,
            output_directory=tmp_path / "out",
        )

    assert not (tmp_path / "out").exists()


def test_coordinate_thermometry_rejects_affine_outside_allclose(
    tmp_path: Path,
) -> None:
    """A translated series is not the same voxel grid qmri can concatenate."""
    first_path = tmp_path / "first.nii.gz"
    second_path = tmp_path / "second.nii.gz"
    shifted_affine = np.eye(4)
    shifted_affine[0, 3] = 10.0
    _save_multiecho(path=first_path)
    _save_multiecho(path=second_path, affine=shifted_affine)
    atlas_path = tmp_path / "atlas.nii.gz"
    atlas_path.write_bytes(b"unused")
    trains = (
        _echo_train(
            image_path=first_path,
            echo_times_file=tmp_path / "first_te.txt",
            n_echoes=4,
        ),
        _echo_train(
            image_path=second_path,
            echo_times_file=tmp_path / "second_te.txt",
            n_echoes=4,
        ),
    )

    with pytest.raises(ValueError, match="affine"):
        coordinate_ethylene_glycol_thermometry(
            registered_component_atlas_image_path=atlas_path,
            trains=trains,
            output_directory=tmp_path / "out",
        )


def test_shared_grid_accepts_affine_within_allclose(tmp_path: Path) -> None:
    """Affine comparison uses numpy.allclose, as qmri does."""
    first_path = tmp_path / "first.nii.gz"
    second_path = tmp_path / "second.nii.gz"
    near_affine = np.eye(4)
    near_affine[0, 3] = 1e-9
    _save_multiecho(path=first_path)
    _save_multiecho(path=second_path, affine=near_affine)
    trains = (
        _echo_train(
            image_path=first_path,
            echo_times_file=tmp_path / "first_te.txt",
            n_echoes=4,
        ),
        _echo_train(
            image_path=second_path,
            echo_times_file=tmp_path / "second_te.txt",
            n_echoes=4,
        ),
    )

    _assert_shared_multiecho_grid(trains=trains)


def test_coordinate_thermometry_defaults_to_3t_and_recovers_vial_temperatures(
    tmp_path: Path,
) -> None:
    """A noiseless dual-resonance volume reports vials U and V at 3 T.

    Field strength is omitted so the coordinator's default of 3 T is what the
    ethylene glycol calibration uses. A different field would not recover the
    planted temperature.
    """
    from qmri.thermometry.multiecho import (  # noqa: PLC0415
        calculate_df_from_temperature,
        thermometry_signal_model,
    )

    planted_temperature_celsius = 22.0
    n_echoes = 24
    echo_times_file = tmp_path / "echo_times.txt"
    np.savetxt(
        fname=echo_times_file,
        X=np.linspace(0.001, 0.024, n_echoes),
        fmt="%.8f",
    )
    echo_times = np.loadtxt(fname=echo_times_file, dtype=np.float64)
    frequency_difference_hz = float(
        calculate_df_from_temperature(
            temperature_celsius=planted_temperature_celsius,
            magnetic_field_tesla=DEFAULT_MAGNETIC_FIELD_TESLA,
        )
    )
    signal = np.asarray(
        thermometry_signal_model(
            t=echo_times,
            amplitude_1=4000.0,
            amplitude_2=2000.0,
            r2star_1=30.0,
            r2star_2=40.0,
            df=frequency_difference_hz,
            dphi_deg=0.0,
        ),
        dtype=np.float32,
    )
    volume = np.zeros(shape=(4, 4, 1, n_echoes), dtype=np.float32)
    volume[1, 1, 0, :] = signal
    volume[2, 2, 0, :] = signal
    image_path = tmp_path / "gre.nii.gz"
    _save_multiecho(path=image_path, data=volume)

    atlas = np.zeros(shape=(4, 4, 1), dtype=np.uint8)
    atlas[1, 1, 0] = np.uint8(21)
    atlas[2, 2, 0] = np.uint8(22)
    atlas_path = tmp_path / "atlas.nii.gz"
    nibabel.save(
        img=nibabel.Nifti1Image(dataobj=atlas, affine=np.eye(4)),
        filename=str(atlas_path),
    )

    result = coordinate_ethylene_glycol_thermometry(
        registered_component_atlas_image_path=atlas_path,
        trains=(EchoTrain(image_path=image_path, echo_times_file=echo_times_file),),
        output_directory=tmp_path / "out",
        minimum_sad_counts=0.0,
        dilation_iterations=0,
        generate_visualisations=True,
    )

    assert result.magnetic_field_tesla == DEFAULT_MAGNETIC_FIELD_TESLA
    assert result.method == "regionwise"
    assert result.temperature_map_path.exists()
    assert result.report_path.exists()
    assert result.measurements_path.exists()
    temperatures = {vial.vial_id: vial for vial in result.vials}
    assert set(temperatures) == {"U", "V"}
    assert temperatures["U"].segment_index == 21
    assert temperatures["V"].segment_index == 22
    assert temperatures["U"].temperature_celsius == pytest.approx(
        planted_temperature_celsius,
        abs=0.5,
    )
    assert temperatures["V"].temperature_celsius == pytest.approx(
        planted_temperature_celsius,
        abs=0.5,
    )
    measurements = result.measurements_path.read_text(encoding="utf-8")
    assert "3 T" in measurements
    assert "U" in measurements
    assert "V" in measurements
    output_directory = tmp_path / "out"
    expected_figures = {
        output_directory / "ethylene_glycol_mask_sad_filter_plot.png",
        output_directory / "vial_U_signal_fit.png",
        output_directory / "vial_V_signal_fit.png",
    }
    assert set(result.visualisation_paths) == expected_figures
    for figure_path in expected_figures:
        assert figure_path.is_file()
        assert figure_path.stat().st_size > 0

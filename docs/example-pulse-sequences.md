# Example pulse sequences

Suggested scanner protocols for common SPIRIT phantom workflows. These are
starting points, not vendor-locked recipes — adjust within the constraints below
for your platform.

## High-resolution scan for atlas registration

Use a high-resolution volumetric scan as the fixed image when registering the
SPIRIT atlas. That registration can then be reused to extract ROIs from other
series acquired in the same session.

| Parameter | Guidance |
| --- | --- |
| Sequence | Fast Spin Echo (example) |
| Voxel size | ≤ 1 mm isotropic (or at most 1 mm in-plane and ≤ 1 mm slice thickness) |
| Coverage | Entire phantom volume |
| Contrast | Choose TE and TR for clear contrast between vial contents, tube walls, and flood fill |

!!! important "Keep the phantom fixed on the couch"

    Do not move the phantom relative to the couch between the registration scan
    and later series that should share that registration. Couch motion itself is
    fine: the scanner records table position, so spatial correspondence remains
    valid. Shifting the phantom on the couch breaks that link and the atlas
    transform cannot be reused for ROI extraction on those other images.

Typical workflow:

1. Acquire this high-resolution volume with the phantom fixed on the couch.
   This enables co-registration with the atlas.
2. Register the SPIRIT atlas to it (`register`).
3. Acquire further series (for example multi-echo thermometry) without moving
   the phantom on the couch.
4. Map atlas labels onto those series (`analyse map-mask`, or `analyse eg-thermometry` for the ethylene glycol vials).

## Ethylene glycol thermometry

Thermometry uses a multi-echo gradient-echo (multi-GRE) acquisition through the
ethylene glycol vials. Ethylene glycol has two chemical species, the CH₂ and OH
groups, which produce a beat pattern in the multi-echo magnitude signal. The
frequency difference between those peaks is related to temperature by:

$$
T[^{\circ}\mathrm{C}] = 193.35 - 1.02 \times 10^{8} \cdot \frac{|\Delta f_{12}|}{\gamma B_0}
$$

where $T$ is the temperature in °C, $|\Delta f_{12}|$ is the frequency difference
between the two peaks in Hz, $\gamma$ is the ¹H gyromagnetic ratio in Hz/T, and
$B_0$ is the main field strength in tesla. At about 20°C that difference is about
108.5 Hz at 1.5 T and 217 Hz at 3.0 T. This will result in a beat frequency of
in-phase and out-of-phase signals which need to be properly sampled. Our own
experience to obtain a proper fit is to have a sampling frequency close to
1 kHz. Thus, if the minimum inter-echo spacing is ~2 ms, one can achieve such a
high sampling frequency by shifting two acquisitions by half the minimum
inter-echo spacing (see below).

| Parameter | Guidance |
| --- | --- |
| Sequence | Multi-GRE (maximum number of echoes; often up to 32) |
| In-plane resolution | < 1 mm (acquired) |
| First echo | As short as possible — typically about 1.5–2 ms (use half-echo / partial Fourier) |
| Echo spacing | Shortest possible; with high bandwidth / minimum water–fat shift (Philips), about 2–3 ms |
| Dual acquisition | Acquire again with the first echo shifted by half the inter-echo spacing |
| Combined sampling | Typically ~64 echoes at ~1–1.5 ms effective sampling |
| Coverage | Single slice is enough |
| Averages | One or two |
| Scan time | Usually about 1–2 minutes |

Shifting the first echo by half the inter-echo spacing interleaves the two echo
trains and roughly doubles the temporal sampling rate (for example from ~2–3 ms
to ~1–1.5 ms).

### Echo-time text files

`analyse eg-thermometry` does not read echo times from the NIfTI. Pass one text
file per series with `--echo-times`, in the same order as the images. A file
such as `series_1_te_s.txt` is a single column of echo times in seconds, one
value per line, with no header. The number of lines must equal the number of
echoes in that series, which is the length of the NIfTI's last axis.

For an equally spaced train, write each echo out as

$$
TE_n = (TE + n \cdot \Delta TE) / 1000
$$

with $TE$ and $\Delta TE$ in milliseconds and $n$ starting at 0. A first echo of
1.45 ms and a spacing of 1.2 ms begins:

```text
0.001450
0.002650
0.003850
0.005050
```

Continue until there is one line for every echo in that image. The second
acquisition, with the first echo shifted by half of 1.2 ms, is a separate file
(`series_2_te_s.txt`) starting at 2.10 ms:

```text
0.002100
0.003300
0.004500
0.005700
```

```bash
uv run spirit-phantom analyse eg-thermometry \
  path/to/registration_output/transformed_component_atlas.nii.gz \
  path/to/gre_series_1.nii.gz \
  path/to/gre_series_2.nii.gz \
  --echo-times path/to/series_1_te_s.txt \
  --echo-times path/to/series_2_te_s.txt \
  --output-directory path/to/eg_thermometry
```

After acquisition, estimate vial temperature with `analyse eg-thermometry`.
That command maps the ethylene glycol atlas labels onto the multi-echo series,
SAD-filters them, and fits the echo trains. Keep the phantom fixed on the couch
relative to the
[high-resolution registration scan](#high-resolution-scan-for-atlas-registration)
so that registration can be reused.
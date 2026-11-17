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
4. Map atlas labels onto those series (`analyse map-mask` or `analyse eg-mask`).

## Ethylene glycol thermometry

Thermometry uses a multi-echo gradient-echo (multi-GRE) acquisition through the
ethylene glycol vials. The principle is based on the fact that this compound has
two visible NMR peaks: one hydroxyl peak and one methylene peak. The frequency
difference between these two peaks is given by the equation:

$$
y = 4.5677 - 0.0097723x
$$

with y = chemical shift difference (in ppm) and x = temperature (in K). For a
typical room temperature of 20C, it gives a difference of y = 1.7ppm, which
corresponds to ~108.5 Hz at 1.5T and 217Hz at 3.0T. This will result in a beat
frequency of in-phase and out-of-phase signals which need to be properly
sampled. Our own experience to obtain a proper fit is to have a sampling
frequency close to 1 kHz. Thus, if the minimum inter-echo spacing is ~2ms, one
can achieve such a high sampling frequency by shifting two acquisitions by half
the minimum inter-echo spacing (see below).

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

After acquisition, map the ethylene glycol atlas labels onto the multi-echo
series with `analyse eg-mask` (mask generation only; temperature analysis is
still in development). Keep the phantom fixed on the couch relative to the
[high-resolution registration scan](#high-resolution-scan-for-atlas-registration)
so that registration can be reused.
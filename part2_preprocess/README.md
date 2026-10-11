# THz LFI Signal Pre-processing Module

A signal pre-processing module for terahertz laser feedback
interferometry (LFI) skin imaging data.

Part of a four-person collaborative THz LFI skin imaging pipeline
developed under the supervision of Dr Karl Bertling, The University of
Queensland.

## Scope

This module sits between the raw data import stage and the image
generation stage of the shared pipeline. Input and output are both
`LFIRawData` objects (amplitude + phase + `z_indices` + `target_type` +
`metadata`), matching the contract published by the data-import module
(teammate A).

The newer 2026 RDM files also contain `RX_LINE_OF_BEST_FIT`, a fitted
184-sample LFI trace per image pixel. It is exposed separately as
`LFIWaveformData` because it is not a raw ADC waveform and does not share
the image-level amplitude/phase contract.

## Operations

| Stage | Methods | Skin | Vegetables | Leaf |
| --- | --- | --- | --- | --- |
| Noise reduction (on `amplitude`) | `median2d`, `gaussian2d`, `savgol_1d`, `bandpass_1d`, `none` | `gaussian2d` (sigma=1) | `gaussian2d` (sigma=1) | `none` |
| Normalisation (on `amplitude`) | `baseline`, `zscore`, `none` | `none` | `none` | `none` |
| Phase unwrapping (on `phase`) | `skimage_2d` (gated, optionally masked), `validated_reference`, `itoh_1d`, `none` | masked, gate 0.001: released | masked, gate 0.001: kept wrapped | saved group reference in the batch script |
| Spatial correction | robust scan-line offsets, Gaussian background, explicit FFT notches | mask, sigma=32 | mask, sigma=32 | no mask, sigma=20 |

### What v0.6 produces (`metadata["spatial_products"]`, exported as NPZ channels)

| Channel | Meaning |
| --- | --- |
| `amplitude_corrected` | Scan-line offsets (and requested notches) removed. **Object contrast is kept.** Default for imaging. |
| `amplitude_detail` | `amplitude_corrected` minus a Gaussian background. A relative local-contrast view; it flattens objects larger than the kernel. NaN outside the signal mask. |
| `phase_corrected` | Unwrapped phase minus Gaussian background and line offsets (relative). NaN when withheld or outside the mask. |
| `phase_background_corrected` | Background removed only; reproduces the group's saved Leaf `U_M`. |
| `*_background`, `*_stripe`, `*_notch_removed` | Every removed component. |
| `signal_mask`, `phase_correction_available` | Where, and whether, phase products may be used. |
| `raw_amplitude`, `raw_phase` | Untouched inputs. |

Amplitude and phase are treated differently on purpose. In amplitude the
large-scale variation is the sample (absorbing material is dark), so
subtracting a smooth background removes the objects. In unwrapped phase the
large-scale variation is the optical path to the surface, which hides
everything else. v0.5 subtracted the background from both; the review in
`docs/review_2026-10-11_v06.md` shows what that did to real images.

Scan-line offsets are estimated per line as the median difference from the
neighbouring lines (window 15), so a real horizontal feature is only
mistaken for an offset if it is thinner than half the window and covers
more than half a line.

### Phase encodings

- **Leaf:** `wrap(L_P*pi/4096 - pi)`, validated against all pixels of the 91
  saved U references at stored indices 0..90 (max modulo error < 2e-14 rad).
  The saved `U_M` equals `U` minus a Gaussian background (sigma=20 px,
  truncate=2, nearest) to < 1.3e-13 rad.
- **2026 Vegetables:** signed 14-bit, `code / 8192 * pi`, range-checked.
  Frame 20 is an anomalous start-up frame.
- **Skin:** signed 14-bit codes stored in a uint16 container,
  `int16(code) / 8192 * pi`. Numerical evidence only (codes span exactly
  [-8192, 8191]; phase wraps become 2*pi steps instead of pi/2 steps);
  `phase_encoding_status = "numerically_supported"` until the supervisor
  confirms it.

### Phase unwrapping gate

The 2D unwrap is deterministic (`rng_seed = 0`). With
`mask = "amplitude_otsu"` the residue fraction is evaluated, and the unwrap
performed, only where there is signal; the released phase is NaN elsewhere.
A slice is released only if at most 0.1% of 2x2 cells in that region carry a
residue. The gate is this strict because an independent unwrap of the Leaf
data (residues 2-5%) matched the group's saved result on only 14-48% of
pixels. Decisions and diagnostics are stored in
`metadata["phase_unwrapping_status"]`.

All parameters live in `config/default.toml`; per-target overrides under
`[targets.<name>...]` select operation families by scan type. No physical
calibration or clinical validation is claimed.

### Displaying the images

Pixels are not square and Leaf arrays are stored long-axis first. Use
`thz_preprocessing.display.oriented` and `pixel_aspect`, and a greyscale
colour map; stretched red-blue panels make real images unrecognisable.
For Vegetables, frames 120-140 are near focus; 21-30 are not.

## Repository layout

```
config/default.toml                 # pipeline parameters
src/thz_preprocessing/
    api.py                          # run_pipeline(data, config_path=None, ...)
    config.py                       # TOML loader + per-target merge
    io.py                           # image-level and fitted-waveform loaders
    shared_loader.py                # adapter to teammate A's loader (vendor/data_import)
    noise_reduction.py              # median2d / gaussian2d / savgol_1d / bandpass_1d
    normalisation.py                # baseline / zscore
    phase_unwrapping.py             # skimage_2d / itoh_1d
    spatial_correction.py           # robust line offsets, background, removed components
    masking.py                      # amplitude-based signal mask
    display.py                      # orientation and pixel aspect for figures
    waveform_processing.py          # conservative fitted-trace cleaning + metrics
    volume_validation.py            # multi-frame signal-preservation metrics
tests/                              # encodings, quality gates, spatial correction, repeatability
examples/
    run_pipeline.py                 # minimal synthetic example
    run_pilot.py                    # end-to-end demo on real skin + leaf
    run_spatial_processing.py       # all 91 matched Leaf frames + Skin + Vegetables
    evaluate.py                     # quantitative evaluation E1-E11 (thesis Chapter 5)
    build_handoff.py                # downstream package from the processed images
    audit_leaf_semantics.py         # reproducible raw/reference correspondence audit
data/                               # local datasets, git-ignored
docs/
    dataset_inventory.md            # Aim 1 inventory note
    ad_hoc_workflow_review.md       # Aim 1 workflow review note
    progress_seminar_slide_outline.md
```

## Installation (development)

```bash
git clone <repo-url>
cd thz-lfi-preprocessing
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,demo]"
```

## Usage

### Minimal

```python
from thz_preprocessing import api, LFIRawData

result = api.run_pipeline(my_data)               # my_data: LFIRawData
print(result.metadata["pipeline"])               # ordered stage record
print(result.metadata["timings_seconds"])        # per-stage seconds
```

### Reading the supplied pilot datasets

```python
from thz_preprocessing import (
    load_skin_hdf5,
    load_leaf_mat,
    load_rdm_hdf5_frame,
    load_rdm_hdf5_volume,
    load_rdm_fitted_waveform,
)

skin = load_skin_hdf5("data/skin/2025-11-28_*.hdf5")
leaf = load_leaf_mat(
    "data/leaf/leaf1.mat",
    spatial_slice=(slice(0, 256), slice(0, 256)),
    z_slice=slice(0, 128),
)
vegetable = load_rdm_hdf5_frame(
    "data/vegetables/frame-21.hdf5",
    phase_encoding="signed_14bit_pi",
)
fitted_trace = load_rdm_fitted_waveform(
    "data/vegetables/frame-20.hdf5",
    spatial_slice=(slice(320, 384), slice(0, 256)),
)
volume = load_rdm_hdf5_volume(
    ["data/vegetables/frame-21.hdf5", "data/vegetables/frame-22.hdf5"],
    phase_encoding="signed_14bit_pi",
    experiment_details_path="data/vegetables/experiment_details.txt",
    z_reference_frame_id=20,
)
```

### End-to-end demo (produces figures for slides)

```bash
python examples/run_pilot.py            # run all available pilots
python examples/run_pilot.py --skin     # skin only
python examples/run_pilot.py --vegetable
python examples/run_pilot.py --waveform
python examples/run_pilot.py --volume
```

Output figures land in `output/`.

### Spatial processing, evaluation and handoff

```bash
PYTHONPATH=src .venv/bin/python -m examples.audit_leaf_semantics      # Leaf raw/reference correspondence
PYTHONPATH=src .venv/bin/python -m examples.run_spatial_processing    # 113 images -> output/spatial_v06/
PYTHONPATH=src .venv/bin/python -m examples.evaluate                  # E1-E11 -> output/evaluation_v06/
PYTHONPATH=src .venv/bin/python -m examples.build_handoff             # downstream package -> output/handoff_v06/
```

`output/spatial_v05/` and `output/processed/` hold superseded batches
(v0.5 flattened amplitude contrast; the older pilot used a withdrawn Leaf
decoding). Do not use them for current work.

## Tests

```bash
pytest tests/ -v
```

The repeatability test (`test_pipeline_repeatability_byte_identical`)
encodes proposal §3.2.4: same input + same config → byte-identical
output.

For 2026 RDM image phase, the loader deliberately does not infer the
fixed-point scale from observed extrema. Complete frames use the confirmed
`signed_14bit_pi` convention and are range-checked. Frame 20 is retained only
for the fitted-waveform pilot because its image phase is an anomalous startup
frame spanning the signed-16 container.

## Author

Haobo Yu (49512977), under supervision of Dr Karl Bertling.

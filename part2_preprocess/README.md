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

| Stage | Methods | Default for skin | Default for leaf |
| --- | --- | --- | --- |
| Noise reduction (on `amplitude`) | `median2d`, `gaussian2d`, `savgol_1d`, `bandpass_1d`, `none` | `gaussian2d` (sigma=1) | `none` |
| Normalisation (on `amplitude`) | `baseline`, `zscore`, `none` | `none` | `none` |
| Phase unwrapping (on `phase`) | `skimage_2d`, `itoh_1d`, `none` | `skimage_2d` | `skimage_2d` |
| Derived spatial correction | Gaussian background subtraction, robust line-offset correction, explicit FFT notches | enabled | enabled |

v0.5 keeps measurement `amplitude`/`phase` separate from relative spatial
products in `metadata["spatial_products"]`. Those products are exported
as separate NPZ channels, including every removed component and raw input.
Corrected amplitude can be negative and is not calibrated reflectivity.
Corrected phase is unavailable (NaN, with a false availability flag) when
spatial unwrapping is withheld. No automatic FFT notch is selected.
Background/line correction is exploratory and can remove real broad
structure; its pixel-scale parameters are not physical calibration.

Leaf decoding now uses `wrap(L_P*pi/4096 - pi)`, validated against all
pixels of the 91 saved U references at matching stored indices 0..90
(maximum modulo error below 2e-14 rad). The old `pi/8192` Leaf conversion
does not reproduce those references. This change is Leaf-specific: the
confirmed 2026 RDM signed-14 convention is unchanged. The physical Leaf
stack coordinate and acquisition timestamps remain unverified, so no
cross-frame filter or unwrap is used by default.

The saved U_M is also reproduced numerically by subtracting a Gaussian
background from U (sigma=20 px, truncate=2, nearest boundary), checked
over all 91 full-resolution references to <1.3e-13 rad. The new batch
reuses the modulo-validated saved U as its explicitly tagged unwrapped
input. `phase_background_corrected` reproduces U_M, while
`phase_corrected` additionally applies line correction. This is not
independent validation of the historical unwrapping or physical accuracy.

All parameters live in `config/default.toml`; per-target overrides
under `[targets.<name>...]` select operation families by scan type.
The 2D method is deterministic (`rng_seed = 0`) and applies a pre-flight
phase-residue quality gate. The base residue limit is 0.06 (raised from 0.01 after evaluation E8); Vegetables keeps 0.01 because its frames unwrap with visible branch errors at 0.06 (see `docs/evaluation_v05.md`).
A high-residue slice stays wrapped instead of
being silently released as a physically valid unwrapped image; the decision
and diagnostics are stored in `metadata["phase_unwrapping_status"]`. A slice
that passes the gate but needs no `2*pi` correction is labelled
`already_continuous` rather than being presented as an improvement.

## Repository layout

```
config/default.toml                 # pipeline parameters
src/thz_preprocessing/
    api.py                          # run_pipeline(data, config_path=None, ...)
    config.py                       # TOML loader + per-target merge
    io.py                           # image-level and fitted-waveform loaders
    noise_reduction.py              # median2d / gaussian2d / savgol_1d / bandpass_1d
    normalisation.py                # baseline / zscore
    phase_unwrapping.py             # skimage_2d / itoh_1d
    spatial_correction.py           # separate spatial residuals + removed components
    waveform_processing.py          # conservative fitted-trace cleaning + metrics
    volume_validation.py            # multi-frame signal-preservation metrics
tests/                              # encodings, quality gates, spatial correction, repeatability
examples/
    run_pipeline.py                 # minimal synthetic example
    run_pilot.py                    # end-to-end demo on real skin + leaf
    run_spatial_processing.py       # all 91 matched Leaf frames + Skin + Vegetables
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

### Spatial processing and reference audit

```bash
PYTHONPATH=src .venv/bin/python -m examples.audit_leaf_semantics
PYTHONPATH=src .venv/bin/python -m examples.run_spatial_processing
```

New full-resolution products and a manifest land in `output/spatial_v05/`.
Old `output/processed/leaf_processed_pilot_slice.npz` used the superseded
Leaf decoding/axis assumptions; do not use it for current validation.

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

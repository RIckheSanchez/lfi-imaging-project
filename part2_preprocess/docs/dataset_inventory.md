# Dataset inventory (Aim 1 — characterise)

> Aim 1 success criterion (proposal §3.1):
> *"written dataset inventory and a short technical note describing the
> group's existing processing conventions."*

This document is the dataset inventory side. The current-workflow side
is in `docs/ad_hoc_workflow_review.md`.

Both notes are based on the data supplied via
`Proj_LFI_Thesis-A17709/02 Data Sets/` on UQ RDM, with `target_type`
strings matching teammate A's `LFIRawData` contract.

## Summary

Revision 2026-10-11: Skin phase is decoded as signed 14-bit codes in a uint16 container
(numerical evidence, see `review_2026-10-11_v06.md`); the "scale remains unconfirmed; v0.5
withholds" statements below are superseded. Images must be shown at the recorded aspect
ratio (1.965 for the 400x1300 raster).

Revision 2026-10-05: the numerical audit pairs saved Leaf U indices 0..90
with the same raw indices and supports radians, using
`wrap(L_P*pi/4096 - pi)` (max all-pixel error 1.813e-14 rad). The physical
stack coordinate remains unresolved. Earlier "sweep-axis", "masked" and
Leaf `pi/8192` labels are withdrawn. See `spatial_processing_v05.md`.

| Target | RDM path | File(s) | Size | Native shape | After loader → `LFIRawData` | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **Skin** (P0, primary) | `02 Data Sets/Sound/Skin/` (the only Skin scan on the RDM, checked 2026-10-11) | `2025-11-28_10-14-26.594507929.hdf5` | 3 MB | per channel: `(400, 1300)` uint16; 3 channels: `IMAGE_AMPLITUDE`, `IMAGE_PHASE`, `PEAK_TO_PEAK` | `amplitude (1, 400, 1300) float64`, `phase (1, 400, 1300) float64` rad | Image-level (demodulated). Single frame `frames/40` under `QCL0/Left`. |
| **Leaf** (P1, secondary) | `02 Data Sets/Leaf/` | `leaf1.mat` (+ `SuperLeaf1z.mp4` ref. video) | 1.35 GB | raw `L_A` / `L_P`: `(1200, 400, 475)`; references `L_P_U` / `L_P_U_M`: `(1200, 400, 91)`; float64 | `amplitude (475, 1200, 400) float64`, `phase (475, 1200, 400) float64` after reference-consistent Leaf conversion | Stored images; same-index reference pairing 0..90 validated; physical time/depth remains unconfirmed. |
| **Vegetables 2026** (new RDM source) | `02 Data Sets/2026 Semester 2/2026_Vegetables/` | `experiment_details.txt`; many `frame-XX.hdf5` / `frame-XX.cbor` pairs. RDM holds frames 20-155 (136 frames). Local since 2026-10-11: frames 20-30 and 40-140 in steps of 10 (sizes verified against RDM). | RDM folder ≈ 130.1 GB; local HDF5 subset ≈ 6.9 GB | image datasets `(400, 1300)`; fitted `RX_LINE_OF_BEST_FIT` traces `(400, 1300, 184)` | tested volume: `amplitude/phase (2, 400, 1300)` for frames 21-22, using an explicit and range-checked signed-14 phase convention | Multi-frame depth scan from 30.05 mm to 44.67 mm in 0.1 mm increments. Frames 21-22 form the first quantitative depth-volume pilot. |
| **Tomography** (P2, optional) | `02 Data Sets/Tomography/` | not pulled | RDM folder ≈ 56.6 GB | unknown | unknown | Optional; not needed for Progress Seminar. |

## Skin pilot — characterised values

Inspected directly from
`data/skin/2025-11-28_10-14-26.594507929.hdf5`:

```
/frames/40/Image Data/QCL0/Left/
    IMAGE_AMPLITUDE   (400, 1300) uint16   min=0     max=10021  mean=2399.9  std=2123.9
    IMAGE_PHASE       (400, 1300) uint16   min=0     max=65535  mean=37751.3 std=28829.7
    PEAK_TO_PEAK      (400, 1300) uint16   min=937   max=11025  mean=3118.5  std=1625.9
```

Observations relevant to pre-processing:

* **Bimodal amplitude histogram** (background pixels around ≈ 500–1500
  + tissue pixels around ≈ 3000–5000), supporting percentile-baseline
  normalisation as the safer default for skin over z-score (which
  would be skewed by the bright tissue mode).
* **Stored phase reaches both ends of the uint16 container.** Reinterpreting
  those words as int16 gives `[-8192,8191]`. The old full-uint16 mapping
  compresses the values into approximately `[-pi/4,pi/4]`, but that does
  not prove the physical phase is continuous. The scale remains unconfirmed;
  v0.5 withholds unwrapping and derived phase correction for this file.
* **No HDF5 root attributes** — all acquisition metadata (drive
  current, modulation frequency, dwell time, FOV) is **absent** from
  this file; open question logged below.

## Leaf — characterised values

Inspected from `data/leaf/leaf1.mat` (`scipy.io.whosmat`):

| Variable | Shape | Dtype | Apparent role |
| --- | --- | --- | --- |
| `L_A` | `(1200, 400, 475)` | float64 | Demodulated amplitude image stack; third axis unconfirmed. |
| `L_P` | `(1200, 400, 475)` | float64 | Encoded phase image stack; reference-consistent decoding `wrap(L_P*pi/4096 - pi)`. |
| `L_P_U` | `(1200, 400, 91)` | float64 | Saved radian phase reference, numerically matched to raw indices 0..90. |
| `L_P_U_M` | `(1200, 400, 91)` | float64 | Modified reference; difference from `U` supports smooth spatial background correction, not a simple validity mask. |

Observations relevant to pre-processing:

* **475 is a stack length, not proof of waveform samples.** Kashyap 2023
  describes time-lapse imaging. Our stack-axis Leaf operators remain
  exploratory pending metadata; the paper uses spatial 2D unwrapping.
* **Leaf reference consistency is numerically resolved.** The loader uses
  `wrap(L_P*pi/4096 - pi)`. This is distinct from 2026 RDM signed-14
  decoding and is not absolute phase calibration. Stack-axis Itoh is no
  longer the Leaf default.

## 2026 Vegetables — characterised values

Inspected from local frames
`data/vegetables/frame-20.hdf5` through `frame-22.hdf5`, downloaded from
`02 Data Sets/2026 Semester 2/2026_Vegetables/`:

```
/data/IMAGE_AMPLITUDE/
    frame_20_qcl_1_left    (400, 1300) uint16   min=0       max=65535
    frame_20_qcl_1_right   (400, 1300) uint16
/data/IMAGE_PHASE/
    frame_20_qcl_1_left    (400, 1300) int16    min=-32768  max=32755
    frame_20_qcl_1_right   (400, 1300) int16
/data/PEAK_TO_PEAK/
    frame_20_qcl_1_left    (400, 1300) uint16
    frame_20_qcl_1_right   (400, 1300) uint16
/data/RX_LINE_OF_BEST_FIT/
    frame_20_left          (400, 1300, 184) int16
    frame_20_right         (400, 1300, 184) int16
```

`experiment_details.txt` says:

```
scan of vegetables
start z position (mm): 30.05
stop z position (mm): 44.67
imcrements (mm): 0.1
notes: vegetables will dry out over course of experiment. performed in open air.
```

Observations relevant to pre-processing:

* The HDF5 layout is different from the legacy Skin pilot. It uses
  `/data/IMAGE_*` datasets rather than `/frames/{id}/Image Data/...`.
  The loader for this schema is `io.load_rdm_hdf5_frame`.
* The file includes both left and right channels. Current local output uses
  `qcl=1, side="left"` to keep one reproducible pilot path.
* `RX_LINE_OF_BEST_FIT` provides a **fitted LFI trace** per pixel with
  184 samples. It is not the raw ADC waveform. This is supported by the
  dataset name and acquisition flags: `enLobf=1`, `logLobf=1`, while
  `logRxRaw=0`, `logRxAvg=0`, `logRxDec=0`, and `logRxSub=0`.
* `IMAGE_PHASE` is logged FFT phase (`logFftPhase=1`) stored in an int16
  container. Complete frames 21 and 22 use exactly `[-8192, 8191]`, so
  `phase_rad = phase_code / 8192 * pi`. Frame 20 spans almost the full
  int16 range and is treated as anomalous startup data, not automatically
  reinterpreted as a second valid encoding. The loader requires the phase
  convention explicitly and rejects codes outside its valid interval.
* The fitted-trace pilot uses a light 5-point, third-order
  Savitzky-Golay filter after per-trace DC removal. On the tested crop it
  gives median input/output correlation 0.987, median peak-to-peak
  retention 0.922, and roughness ratio 0.483. These are signal-
  preservation checks, not clinical validation.

### Multi-frame depth pilot

Frames 21 and 22 were loaded with `io.load_rdm_hdf5_volume` as a
`(2, 400, 1300)` volume. Using frame 20 as the scan-position reference,
their derived z positions are 30.15 mm and 30.25 mm. Frame 20 was kept
out of this validation because only 20.8% of its fitted traces are
non-empty, compared with 100% in frames 21 and 22; it appears to be a
scan-start/partial-acquisition frame.

The v0.4 pilot produced:

| Metric | Result | Interpretation |
| --- | ---: | --- |
| Median raw/processed amplitude Pearson correlation | 0.933 | Main spatial structure is retained after median filtering and baseline normalisation. |
| Maximum normalised amplitude RMSE | 0.027 | Scale-independent image change is small. |
| Minimum phase-modulo consistency | 1.000 | Rewrapping reproduces the input phase. This is an integrity check, not proof that an unwrap is physically correct. |
| Spatial phase-residue fraction | 0.048 / 0.049 | Both frames exceed the configured 0.01 project guardrail, so 2D unwrapping is withheld and the wrapped phase is retained. |
| Adjacent processed amplitude correlation | 0.891 | Consecutive 0.1 mm depth slices remain structurally continuous. |
| Total pipeline runtime | ≈0.27 s | Two full image-level frames remain well below the 30 s practicality target. |

## Resolved data conventions

| Item | Confirmed interpretation | Implementation |
| --- | --- | --- |
| Leaf `L_P` | `wrap(L_P*pi/4096 - pi)` matches all 91 U references at the same indices, <2e-14 rad | `io.load_leaf_mat`; dataset-specific reference equivalence |
| 2026 `IMAGE_PHASE`, complete frames | signed 14-bit FFT phase in an int16 container; `[-8192, 8191] -> [-pi, pi)` | explicitly selected and range-checked by `io.load_rdm_hdf5_frame` |
| 2026 frame 20 `IMAGE_PHASE` | anomalous/startup frame spanning the signed-16 container; excluded from image-phase deliverables | rejected by the default signed-14 loader; explicit override is diagnostic only |
| 2026 `RX_LINE_OF_BEST_FIT` | fitted LFI trace per pixel; last axis is 184 trace samples; not raw ADC | `io.load_rdm_fitted_waveform` |

## Open questions (already in flight to supervisor / teammate A)

| # | Question | Blocks |
| --- | --- | --- |
| 1 | Skin `IMAGE_PHASE` uint16 encoding: `[0, 2π)` or `[-π, π)`? | Choice of `phase_encoding` in `io.load_skin_hdf5`. |
| 2 | Confirm meaning of `frames/{id}` for Skin (frame index in a longer sequence, or simply a session id)? | Whether multi-frame skin scans should be aggregated into `(n_z, H, W)` rather than `(1, H, W)`. |
| 3 | Leaf physical third-axis coordinate, timestamps, calibration and `_U_M` processing script? | Physical-coordinate/absolute calibration; same-index reference pairing 0..90 is now numerically validated. |

These do not block amplitude and fitted-waveform processing, but they do
block claims of physically validated absolute phase reconstruction.

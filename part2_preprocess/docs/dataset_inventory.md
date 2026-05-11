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

| Target | RDM path | File(s) | Size | Native shape | After loader → `LFIRawData` | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| **Skin** (P0, primary) | `02 Data Sets/Skin/` | `2025-11-28_10-14-26.594507929.hdf5` | 3 MB | per channel: `(400, 1300)` uint16; 3 channels: `IMAGE_AMPLITUDE`, `IMAGE_PHASE`, `PEAK_TO_PEAK` | `amplitude (1, 400, 1300) float64`, `phase (1, 400, 1300) float64` rad | Image-level (demodulated). Single frame `frames/40` under `QCL0/Left`. |
| **Leaf** (P1, secondary) | `02 Data Sets/Leaf/` | `leaf1.mat` (+ `SuperLeaf1z.mp4` ref. video) | 1.35 GB | `(1200, 400, 475)` float64 per variable; 4 variables: `L_A`, `L_P`, `L_P_U`, `L_P_U_M` | `amplitude (475, 1200, 400) float64`, `phase (475, 1200, 400) float64` (units **not** rad — see open question) | Sweep-axis volumetric. `L_P_U` / `L_P_U_M` are the group's existing ad-hoc unwrapped phase outputs (see workflow review). |
| **Tomography** (P2, optional) | `02 Data Sets/Tomography/` | not pulled | 1.7 GB | unknown | unknown | Optional; not needed for Progress Seminar. |

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
* **Wrapped phase spans the full uint16 range**, but after `[0, 2π) →
  [-π, π)` conversion the dominant in-tissue range concentrates within
  `[-π/4, +π/4]`. This is the structure that becomes interpretable
  after 2D quality-guided unwrapping (see `output/skin_pilot.png` for
  the before/after).
* **No HDF5 root attributes** — all acquisition metadata (drive
  current, modulation frequency, dwell time, FOV) is **absent** from
  this file; open question logged below.

## Leaf — characterised values

Inspected from `data/leaf/leaf1.mat` (`scipy.io.whosmat`):

| Variable | Shape | Dtype | Apparent role |
| --- | --- | --- | --- |
| `L_A` | `(1200, 400, 475)` | float64 | Amplitude, raw demodulated; third axis = per-pixel sweep. |
| `L_P` | `(1200, 400, 475)` | float64 | Wrapped phase, raw; third axis = per-pixel sweep; **values in `[-8192, +8191]` (i.e. an integer-style 14-bit fixed-point encoding, _not_ radians)** — see open question. |
| `L_P_U` | `(1200, 400, 91)` | float64 | Unwrapped phase, group's ad-hoc workflow output. Third axis shrunk 475 → 91; semantics inferred — see `docs/ad_hoc_workflow_review.md`. |
| `L_P_U_M` | `(1200, 400, 91)` | float64 | Same as `L_P_U` with a validity mask applied. |

Observations relevant to pre-processing:

* **Sweep axis (length 475) is the natural target for 1D operators**
  (bandpass, Savitzky–Golay, Itoh-1D unwrap, per proposal §3.2.2).
* **Phase unit on `L_P` is not radians.** Itoh-1D unwrap via
  `numpy.unwrap` assumes a `2π` period; until the encoding is
  confirmed, the unwrap step on leaf is structurally correct but
  numerically scaled wrong. The fix is a single multiplier in
  `io.load_leaf_mat`, blocked on confirmation from supervisor / team-
  mate A.

## Open questions (already in flight to supervisor / teammate A)

| # | Question | Blocks |
| --- | --- | --- |
| 1 | Skin `IMAGE_PHASE` uint16 encoding: `[0, 2π)` or `[-π, π)`? | Choice of `phase_encoding` in `io.load_skin_hdf5`. |
| 2 | Leaf `L_P` numerical encoding: what value corresponds to `2π`? Is it `16384` (i.e. signed 14-bit) or another scale? | Correct numerical unwrap on leaf in `phase_unwrapping.itoh_1d`. |
| 3 | Are raw waveform datasets stored anywhere (acquisition PC, separate share)? | Whether the proposal's bandpass-1D-vs-savgol-1D evaluation should also run on raw waveforms in Semester 2. |
| 4 | Confirm meaning of `frames/{id}` for Skin (frame index in a longer sequence, or simply a session id)? | Whether multi-frame skin scans should be aggregated into `(n_z, H, W)` rather than `(1, H, W)`. |

These do not block the Progress Seminar — they are the planned mid-year
pivot discussion points (proposal §3.2.5).

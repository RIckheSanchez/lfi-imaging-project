# Demonstrating Part 3 (image generation and reconstruction)

A 3-5 minute demonstration, from the repository root. Paths assume the Part 2
handoff is unpacked next to the repository and the raw Vegetables frames are in
`data/vegetables/` (both are outside Git).

## 0. Before the session

```powershell
python -m pytest -q tests/test_imaging.py tests/test_volume.py tests/test_handoff.py tests/test_pipeline.py tests/test_gui_backend.py
python -m part3_reconstruction handoff ..\THz_LFI_handoff_v061\handoff_v061 --output outputs\handoff_v061 --raw data\vegetables --nominal-pixel-mm 0.066 --napari
```

The second command takes about a minute and writes `outputs/handoff_v061/report.html`
(one file, every figure and table). Keep it open in a browser as a fallback, and
record a short screen capture of steps 3-4 in case the live demo fails.

## 1. The problem (30 s)

Pre-processed THz LFI scans arrive as amplitude and phase with unclear phase units,
non-square pixels and an irregular depth series (0.1 mm steps, then 1 mm).
Part 3 turns them into figures and volumes without guessing calibrations.

## 2. Synthetic validation (30 s)

```powershell
python -m part3_reconstruction demo --output outputs\synthetic
```

Point at `validation.json`: linear phase error ~1e-6 rad after resampling, sphere
Dice values, and the explanation that coarse depth sampling, not the algorithm,
limits geometry.

## 3. Real data report (1 min)

Open `outputs/handoff_v061/report.html`:

- File audit: 113 files read, Part 2's identities hold to 1e-12.
- Amplitude channels: `amplitude_corrected` lowers line striping by about a third.
- Phase displays: released Leaf/Skin phase; HSL fusion keeps phase colour in bright areas.
- Pixel geometry: mirror settings give the 1.965 frame ratio exactly.
- Depth: sections, profile, NIfTI checked with SimpleITK, napari 3D view.

## 4. Inside the team desktop viewer (1-2 min)

```powershell
python -m part3_reconstruction.viewer ..\THz_LFI_handoff_v061\handoff_v061\vegetables_volume_21frames.npz
```

Open the 3D panel, render once without and once with **Reconstruct** ticked. With
it, the 21 irregular slices become a uniform 0.1 mm focus-depth grid (120 slices,
30.15-42.05 mm). For a folder of raw `frame-*.hdf5` files add `--vegetables-2026`
(frame 20 = 30.05 mm, 0.1 mm per frame).

## 5. Part 1 -> 2 -> 3 in memory (30 s, optional)

```python
from part3_reconstruction.pipeline import run_part1_to_part3
from part3_reconstruction import ImageGenerator
data = run_part1_to_part3("data/vegetables/frame-21.hdf5")
ImageGenerator(data).generate_comparison(export="outputs/frame21_parts123.png")
```

## 6. Limitations to state

- Lateral pixel size is a nominal estimate (~0.066 mm from the ~5 cm window).
- z is the beam focus position; depth sections are a focal series, not internal tomography.
- Skin pixel aspect is not confirmed for the 2025 scan.

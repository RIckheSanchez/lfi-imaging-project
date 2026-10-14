# Person 3: image generation and volume reconstruction

For the latest shared-drive findings and batch-audit command, see
[data readiness, 11 October 2026](data_readiness_2026-10-11.md).

Rebuilt 4 October 2026 against Mingxuan Tang's thesis plan. This implementation
is new work, not recovery of the lost seminar code. The validation results below
must not be backdated or substituted for the seminar's historical test claims.

## Run on Windows

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m part3_reconstruction demo
.\.venv\Scripts\python.exe -m part3_reconstruction inspect data/vegetables/frame-21.hdf5
.\.venv\Scripts\python.exe -m part3_reconstruction uq-preview data/vegetables/frame-21.hdf5 --side left
```

The tested local interpreter is Python 3.13.11. Actual dependency versions are
recorded in `requirements-part3-tested.txt` (a reproducibility snapshot, not a
replacement for the team's full requirements).

## Contract for Person 2 and Person 4

Use `ImagingData(amplitude, phase, metadata)` with matching 2D `(y,x)` or 3D
`(z,y,x)` real arrays. Arrays are copied to float32. Missing and infinite pixels
are represented as NaN and masked on display. Entirely missing inputs fail.
For float32-incompatible magnitudes, upstream scaling is required.

Person 2 supplies phase in radians, with `phase_state='unwrapped'` when that has
actually been established. Cyclic display folds radians to `[-pi,pi)` on a
fixed colour scale. Sequential display preserves continuous phase ordering.
Raw acquisition integer counts have `phase_unit='raw_counts'` and can only
be previewed sequentially until calibrated. HSL always uses wrapped-radian hue.

`DataLoader.from_arrays(..., axis_order='zxy')` explicitly converts legacy axes.
NPY pairs, NPZ, selected HDF5 datasets and the observed UQ frame schema are
supported. The schema inspector reads dataset descriptions without loading the
large waveform arrays. No general raw-waveform extraction is implemented here;
that remains Person 2's responsibility.

```python
from part3_reconstruction import ImagingData, ImageGenerator, VolumeReconstructor

data = ImagingData(amplitude, phase, {
    'phase_unit': 'rad', 'phase_state': 'unwrapped',
    'spacing_xy': (dx, dy), 'spatial_unit': 'mm',
    'origin_xy': (x0, y0),
})
generator = ImageGenerator(data, scale_bar_length=1.0)  # 2D data only
figure = generator.generate_comparison(phase_mode='sequential', export='outputs/comparison.png')
# The GUI embeds the returned matplotlib.figure.Figure; no pyplot GUI is launched.

# For a separate 3D ImagingData object:
volume = VolumeReconstructor(volume_data, spacing_xy=(dx, dy),
                             z_coordinates=measured_depths,
                             origin_xy=(x0, y0), spatial_unit='mm')
isotropic = volume.resample_isotropic(spacing=target_spacing)
xz = isotropic.get_orthoslice('xz', index=10, channel='amplitude')
isotropic.export_nifti('outputs/amplitude.nii.gz')
isotropic.export_nifti('outputs/phase.nii.gz', channel='phase')
```

PNG/TIFF are rendered figures, not quantitative intensity containers. Their
adjacent JSON files retain source, display settings and metadata. Preserve the
input arrays for analysis. Normalised colour bars are labelled as such; cyclic
phase colour bars are in radians. `limits=(low,high)` on single-channel methods
supports fixed comparisons across frames; per-frame percentile contrast must not
be interpreted as a common quantitative scale.

## Three-dimensional scope and assumptions

- Explicit physical z coordinates are mandatory. Frame IDs are never inferred
  to be depth, and phase is never silently converted into height.
- Slices are sorted by z; duplicate depths fail. The input must represent aligned
  samples of a volume. Different focal positions alone do not establish a
  tomographic reconstruction of internal material structure.
- `register_translation()` is opt-in amplitude-based XY translation only, using
  phase cross correlation. Both channels receive the same transformation, border
  areas become NaN, and shifts are recorded. This assumes common structure and
  must not automatically register unrelated depth-dependent features. Rotation,
  deformation, and masked registration are not implemented.
- Resampling uses SciPy `RegularGridInterpolator` with trilinear interpolation
  instead of `ndimage.zoom`, because explicit coordinates also support irregular
  depth sampling. It requires calibrated, unwrapped phase. It preserves the first
  voxel centre and uses only output centres inside the measured bounds; at most
  one output spacing is omitted at the upper edge. NaNs propagate locally.
- A voxel-count limit is checked before output allocation; interpolation runs
  one plane at a time. Inputs and output arrays still reside in memory. This is
  not a whole-130-GB streaming reconstruction implementation.
- NIfTI export transposes zyx to xyz and writes both qform and sform, origin,
  spacing and units. Coordinates describe an axis-aligned scanner grid, with no
  claim of patient/anatomical registration. Irregular z must first be resampled.
- Optional `view_napari()` is provided via a lazy import. Install `napari[all]`
  separately and call `napari.run()` in a standalone script. It has not been
  interactively validated in this environment and is not required by core tests.

## Team integration changes

The old `ProcessedSignal.clean_signal` contract lacks explicit amplitude, phase,
units and spacing. It cannot safely be guessed into the new contract. Part 2
must construct `ImagingData` and Person 4 must consume the returned Figure or
VolumeReconstructor. `reconstruct_image()` now returns a comparison Figure;
`reconstruct_3d()` returns a VolumeReconstructor. This is an explicit migration,
not backwards-compatible behaviour for old ReconstructedImage consumers.
The old shared dataclasses and teammates' placeholder modules remain untouched.
The full Part 1 -> Part 2 -> Part 3 -> Part 4 pipeline is not yet integrated.

## Validation and real-data status

Tests cover signed/constant/missing data, analytic normalisation, periodic phase,
axis mapping, NPY/NPZ/HDF5 loading, PNG/TIFF and metadata export, fixed cyclic
range, calibration guards, physical scale bars, metrics with known answers,
irregular/reversed depth interpolation, orthogonal slices, NIfTI data and world
coordinate roundtrip, translation recovery, and allocation guards.

Synthetic demo: a linear phase field resampled from z spacing 2 to 1 has maximum
absolute error about `7.63e-7 rad`. A coarsely sampled binary sphere yields Dice
`0.93663`; interpolation cannot restore lost boundary detail. This does not meet
an arbitrary 0.95 criterion and is retained as a limitation, not removed from the
report. A separate continuous sphere field test exceeds Dice 0.98. These are
synthetic engineering checks, not biological/clinical accuracy measurements.

Two downloaded 2026 Vegetables files (`frame-20.hdf5`, `frame-21.hdf5`) have been
processed on both left/right acquisitions. Each pair is 400x1300. No denoising,
phase unwrapping or Person 2 preprocessing has been applied to these previews.
Outputs and per-file hashes/statistics are under `outputs/real/`, ignored by Git.

Important observations:

- The schema is `data/IMAGE_AMPLITUDE/frame_N_qcl_1_SIDE` and the matching
  `data/IMAGE_PHASE` path; `_BMP` arrays are display derivatives and are excluded.
- Frame 20 reaches uint16 amplitude 65535 and almost the full signed-int16 phase
  range. Its preview has large flat/banded regions. Frame 21 amplitude maxima
  are 2689 (left) and 2698 (right), with phase from -8192 to 8191. These observations
  flag acquisition differences; they do not establish sensor faults or units.
- The experiment note says z starts at 30.05 mm, stops at 44.67 mm, increment
  0.1 mm, and vegetables may dry during the open-air experiment. The endpoint
  difference is not an integer multiple of the given step, so a per-frame
  coordinate mapping must be confirmed rather than invented.
- Radian scale/offset, lateral physical spacing, left/right orientation, valid
  frame selection and the relationship between stage z and reconstructed depth
  remain unconfirmed. No real-data NIfTI or physical-depth claim is produced yet.
- Literature figure comparisons and ITK-SNAP visual verification remain pending;
  NIfTI export has been verified by nibabel roundtrip tests.

Research data stay local and Git-ignored. No files were uploaded to GitHub.

## Update 2026-10-11 (evening): GUI integration and performance

- New `integration.py`: viewer-object adapter, RGBA rendering, orthogonal
  views, PNG bytes and `ReconstructedImage` output. See `docs/gui_integration.md`.
- New `benchmark.py` and `python -m part3_reconstruction benchmark [files...]`.
- Test suite: 53 passed.

### First benchmark (Linux VM on the dev laptop, 2 vCPU, Python 3.10, NumPy 2.2)

| Case | Step | Time (s) | Peak traced (MB) |
|---|---|---|---|
| Vegetables frame, 400×1300 (×4 maps) | load one side | 0.02–0.06 | ≈10 |
| Vegetables frame, 400×1300 | comparison figure + PNG encode | ≈0.67 | ≈43 |
| Synthetic 20×256×256 | resample to 1 mm (→39 slices) | 0.68 | 83 |
| Synthetic 40×400×400 | resample to 1 mm (→79 slices) | 3.66 | 404 |
| Synthetic 40×400×400 | NIfTI export (amplitude) | 1.6–5.8 | <3 |

Notes: peak traced memory is a tracemalloc lower bound on process RSS.
Resampling peaks at roughly 8× the float32 output volume, so very large scans
will need chunked interpolation. NIfTI export time is dominated by gzip and by
disk I/O (written to a Windows-mounted folder), so it varies between runs.
Raw results: `outputs/benchmark/benchmark.json`.

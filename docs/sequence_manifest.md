# Reproducible sequence reconstruction

The sequence runner adds an end-to-end local workflow for calibrated 2D HDF5
maps. It is not the team's complete acquisition/preprocessing/GUI integration.

```powershell
.\.venv\Scripts\python.exe -m part3_reconstruction reconstruct manifest.json --output outputs/run_001 --spacing 0.1
```

The JSON manifest must use schema version 1 and specify:

- `sample_id`: explicit sample identity.
- `calibration_source`: where the units/geometry came from.
- `spacing_xy`: `[dx, dy]`; `origin_xy`: `[x0, y0]` (defaults to `[0,0]`).
- `spatial_unit`: `mm`, `micron` or `meter`, used consistently for all axes.
- `axis_order`: `yx` or `xy`, describing the source 2D datasets.
- `phase_unit`: `rad`, `degrees` or `raw_counts`.
- `phase_state`: `wrapped`, `unwrapped` or `unknown`. Interpolation requires
  `unwrapped`; the loader does not unwrap or align phase offsets between slices.
- For `raw_counts`, require a documented `phase_scale_to_rad` and optional
  `phase_offset_rad`, giving radians = counts * scale + offset.
- `frames`: a list of `{path, amplitude_key, phase_key, z}` entries. File paths
  are relative to the manifest. Dataset leaf names must match across channels,
  e.g. `amplitude/0055` and `phase/0055`. UQ Vegetables explicit dataset paths
  also fit this contract. Do not mix left/right scans in a sequence unless the
  acquisition geometry explicitly supports that interpretation.

No real-data manifest with invented calibration is supplied. Prepare one after
the units and physical frame positions are confirmed. Complete files are checked
for compatible shapes, duplicate source pairs and duplicate depths before volume
allocation. Files are read one pair at a time; the assembled arrays and subsequent
processing copies still require memory. `--max-voxels` bounds both input and
resampled output sizes, but is not a measured peak-RAM limit.

Output: amplitude and phase NIfTI, JSON provenance, centre XY comparison, and a
run report with processing duration. Exports are read back and compared to the
in-memory volume. Every loaded float32 pair has a hash in sorted provenance;
this is not a hash of the original HDF5 file. Existing nonempty output folders
are rejected to avoid overwriting earlier runs. Interrupted runs may leave
partial outputs; only a completed `run.json` records successful verification.

Use an explicit output spacing for irregular depth coordinates. Uniform spacing
can be preserved by omitting `--spacing`. This runner does not automatically
register slices, infer depth, or turn focal-plane scans into internal tomography.

Synthetic end-to-end tests independently check expected voxel values and world
coordinates after loading, sorting, degree conversion, irregular-z interpolation,
NIfTI export and readback. Missing skin/calibration data remain separate blockers
to real-data scientific validation.

# Part 3 → Part 4 GUI integration (for Vincent)

`part3_reconstruction.integration` gives the Streamlit viewer plain NumPy
arrays, PNG bytes and small info dicts. It does not import Streamlit, so it is
unit-tested without a browser (`tests/test_integration.py`).

## Minimal use in `lfi_viewer_final.py`

```python
from part3_reconstruction.integration import (
    from_viewer_volume, render_rgb, orthogonal_views, figure_to_png, to_reconstructed_image)
from part3_reconstruction import ImageGenerator, VolumeReconstructor

# st.session_state.data is the object smart_load() returns (.amplitude/.phase, z first)
vol = from_viewer_volume(st.session_state.data,
                         phase_unit="raw_counts",   # say "rad"/"degrees" only when confirmed
                         phase_state="unknown")

# 2D slice: RGBA uint8, missing pixels transparent
k = st.session_state.current_z_idx
rgba, info = render_rgb(vol.slice(k), "amplitude", flip_y=True)
st.image(rgba, caption=f"{info['channel']} [{info['display_min']:.3g}, {info['display_max']:.3g}] {info['unit']}")

# Phase: "sequential" works for raw counts; "cyclic" needs calibrated radians
rgba, info = render_rgb(vol.slice(k), "phase", phase_mode="sequential", flip_y=True)

# Publication figure / download button
png = figure_to_png(ImageGenerator(vol.slice(k)).generate_comparison(phase_mode="sequential"))
st.download_button("Download figure", png, file_name="comparison.png")

# Orthogonal planes (index-based)
planes = orthogonal_views(vol, (k, row, col))   # {"xy", "xz", "yz"}
```

## Physical 3D (only after calibration is confirmed)

```python
recon = VolumeReconstructor(vol, spacing_xy=(dx, dy), z_coordinates=z_mm)
result = to_reconstructed_image(recon)   # shared.data_types.ReconstructedImage
```

## Rules the GUI should respect

- `z_indices` from the loader are **frame numbers**, stored as
  `metadata["frame_indices"]`; they are not depths. Physical z must come from
  the acquisition record.
- Cyclic phase colouring, HSL fusion, phase resampling and phase NIfTI export
  all refuse uncalibrated phase and raise `ValueError` — show the message to the
  user rather than hiding it.
- `render_rgb` row 0 is y index 0. `st.image` draws row 0 at the top, so pass
  `flip_y=True` to match the Matplotlib figures (`origin="lower"`).
- `quality_score` stays 0.0 unless a trusted reference exists.
- Display arrays are for viewing only. Use `ImageGenerator(..., export=...)` or
  `VolumeReconstructor.export_nifti` for files used as evidence (they write a
  JSON sidecar with parameters).

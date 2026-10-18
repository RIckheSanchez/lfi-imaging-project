# LFI Viewer

LFI Viewer is a macOS desktop application for exploring Laser Feedback
Interferometry scans. It uses PySide6 and pyqtgraph, with a shared Python core
for HDF5 loading, processing and export. The project is part of the University
of Queensland imaging thesis supervised by Karl Bertling.

## Install and run from source

Use Python 3.10 or newer in a Conda environment:

```sh
conda create -n lfi-viewer python=3.13
conda activate lfi-viewer
python -m pip install -r requirements.txt
python -m part4_ui.app
```

You can also run `./run.sh` from the repository. The launcher uses `LFI_PYTHON`
when provided, then an active venv or Conda environment, then local Miniconda
or `python3`. For example:

```sh
LFI_PYTHON=/path/to/environment/bin/python ./run.sh
```

## Run the macOS application

The local build is `dist/LFI Viewer.app`. Open it in Finder or run:

```sh
open "dist/LFI Viewer.app"
```

This build is arm64 for Apple Silicon and was tested on macOS 15.5. Its bundled
Qt wheels require macOS 14 or newer. It includes the runtime dependencies and
does not require Python to be installed. Test datasets are not included.
The build is ad-hoc signed, not Developer ID signed or notarized. Distribution
and Gatekeeper approval on another machine have not been verified.

To rebuild, use an **arm64** Python environment:

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m PyInstaller --clean --noconfirm "LFI Viewer.spec"
```

`build/` and `dist/` are generated and ignored. The spec intentionally excludes
Streamlit, Plotly and their frontend dependencies from the desktop application.

## Supported HDF5 structures

Standard scans contain `/amplitude/<z>` and `/phase/<z>` datasets. Each dataset
is a two-dimensional `(height, width)` array; numeric z keys are sorted and
stacked into `(slices, height, width)`.

Frame scans contain these datasets beneath each numeric frame id:

```text
/frames/<id>/Image Data/QCL0/Left/IMAGE_AMPLITUDE
/frames/<id>/Image Data/QCL0/Left/IMAGE_PHASE
```

Both amplitude and phase must be present with compatible dimensions.

## Features

- Native file and frame-folder dialogs, file/folder drag/drop, and five recent paths saved with QSettings.
- Registered path loaders: Part 1 HDF5 import and public load_volume for frame folders; every opening action uses the same dispatch.
- English and Chinese interface switching.
- Amplitude/phase selection, brightness, contrast, and seven colormaps.
- Linked Z slider, zero-based slice index and actual Z value selector; slice position is displayed one-based without physical Z units.
- Aspect-locked 2D image with bounded pan/zoom and **Fit to window** / Cmd+0.
- Colorbar with the selected data range and the effective display color transform.
- Optional Part 2 preprocessing, enabled through its checkbox. JSON configuration overrides are available only with LFI_DEV_PARAMS=1; the default uses {}. Per-channel status reports normalization, units and phase provenance from metadata. Amplitude labels identify preprocessing and dimensionless zscore output; phase units follow conversion metadata.
- Missing samples are ignored in display/histograms, transparent in PNG/TIFF and volumes, and retained as nan in CSV. All-missing channels produce an error.
- Channel-aware value labels, selected-pixel crosshairs, bounded X/Y inputs, arrow-key movement in image/waveform views, explicit Show waveform button and histogram.
- Amplitude and phase waveforms at the selected pixel.
- Calibrated 2D scale bar when a positive pixel size is entered.
- Background refresh on Volume/Isosurface and percentile changes, with an explicit amplitude-channel notice.
- Downsampled amplitude Volume and three uncapped Isosurface meshes; adjustable
  lower percentile (default 90), fixed 99.5th-percentile upper threshold and camera reset.
  Outer-to-inner mesh opacity is 0.12, 0.30 and 0.60. Volume clips its sampled
  values to the 1st–99.5th percentile range before normalization and uses a
  quadratic opacity transfer function. Isosurface controls are hidden in Volume mode.
- Direct File → Export CSV… (Cmd+Shift+E), export content preview, remembered export directory (Desktop initially), and Show in Finder after saving.
- Selected-channel CSV (original measurements when preprocessing is off; processed values when on), colored pixel PNG/JPEG/TIFF and publication PNG
  with axes/colorbar. PNG/TIFF and publication PNG carry 300 DPI.
- Optional Part 3 depth reconstruction: expand **Depth calibration** in the 3D tab, enter scan settings and confirm the start-frame mapping. Fields are blank and cleared on every file load. Valid metadata `depth_mm` is also accepted; no calibration is guessed.
- Part 3 linearly interpolates amplitude onto a uniform depth grid and takes phase from the nearest measured slice; the 2D measured data remain unchanged. The 3D status explicitly identifies sample-index axes.
- Thread-pool loading, slice/histogram preparation, backend processing, 3D
  computation and export, with busy state and readable error dialogs.
- Part 2 is registered by the desktop entrypoint, with preprocessing initially off. Part 3 reconstruction controls require its registered backend and valid per-slice depth metadata or a complete user-entered calibration. Without calibration the v1.3 reconstruction option and hint stay hidden; a collapsed Depth calibration section is available on the 3D tab when Part 3 is registered.

## Tests

Install development dependencies and run the full pytest suite offscreen:

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Tests use the local sample files `Realdata/mushroom.hdf5` and
`Realdata/chip_borad1.hdf5` and `Realdata/pure marble_29.hdf5`. `Realdata/` is ignored by Git; obtain these scan
fixtures separately when cloning the project. GUI tests use pytest-qt and
exercise worker-thread execution, interaction, exports, loader/backend registration, NaN handling and preprocessing state preservation. Root pytest also discovers Part 1, Part 2 and Part 3 tests. Missing real-data fixtures and the absent Part 2 vendored loader cause documented skips. The chip test saves an offscreen screenshot.

To test the retained Streamlit prototype as well:

```sh
python -m pip install -r legacy/requirements-legacy.txt
PYTHONDONTWRITEBYTECODE=1 python tests/verify_viewer.py
```

Qt's offscreen platform does not support OpenGL viewports. Tests validate real
GL objects and computed geometry there. Native packaged-app evidence is in
`docs/screenshots/bundle-validation/`, including Cocoa 2D/3D captures and a JSON
report. The opt-in validation helper uses the real packaged application, not
mocked graphics:

```sh
LFI_VALIDATE_BUNDLE=/tmp/lfi-bundle-check \
LFI_VALIDATE_FILE="$(pwd)/Realdata/chip_borad1.hdf5" \
"dist/LFI Viewer.app/Contents/MacOS/LFI Viewer"
```

## Known limitations

- Pixel size must be entered manually. The supplied scan files contain no
  calibration metadata; no 10 μm step or 100 kHz rate is inferred.
- 3D axes remain sample indices, including the uniform depth-grid index after Part 3 reconstruction; rendering does not establish correct physical XYZ proportions.
- Part 2 is available but optional; its default settings for the supplied three scans do not generate spatial_products, and their phase remains uncalibrated. Derived relative phase and processing quality are not physical calibration.
- Part 3 is registered optionally at startup and disabled safely if its import fails. Reconstruction remains unavailable without valid depth metadata or user-supplied calibration. The chip scan has no confirmed depth calibration. No Vegetables preset is applied.
- Calibration must come from the user or acquisition metadata. Karl supplied start 30.05 mm and step 0.1 mm for Vegetables, but the frame ID at the start depth (20 or 21) is unresolved; confirm it before scientific use.
- Part 3 output spacing is derived from the user step or the smallest interval between distinct metadata depths; the upstream slice-count safety limit still applies. Phase is copied from a nearest measured slice, not interpolated or converted to depth.
- NPZ/handoff loading is not enabled until real data and phase-unit provenance are provided. Pixel/mask orientation and physical 3D calibration need team confirmation.
- The loader retains complete amplitude/phase arrays in memory. Large scans
  require sufficient RAM; downsampling limits 3D display work, not load memory.
- 3D is amplitude-based. Flat/degenerate or single-layer data may not define
  valid isosurfaces. Mode and percentile changes refresh automatically; brightness, contrast and colormap changes require Render / Refresh 3D again.
- The local bundle is arm64 only and has not been tested on other macOS machines.

## Architecture

`part4_ui/lfi_core/` contains loaders, LFIRawData, pure processing functions,
exports and backend protocols/registration. It does not import Qt or Streamlit.
`part4_ui/app.py` owns the desktop window and worker lifecycle;
`volume_panel.py` owns OpenGL scene presentation; `i18n.py` stores UI text.

`legacy/streamlit_app.py` is the retained Streamlit prototype and uses the same
core. The lightweight `streamlit-final` tag identifies its pre-migration
baseline. See `docs/backend-integration.md` for teammate integration and
`docs/gaussian_preprocessor_example.py` for an inactive example. Existing
`part1_data`, `part2_preprocess`, `part3_reconstruction` and `shared` modules
remain available for team development. Part 1 import and the UI-side Part 2 adapter are wired into the desktop; Part 3 has an optional UI adapter with explicit calibration gating.

## Part 3 imaging and reconstruction module

`part3_reconstruction` turns pre-processed amplitude and phase into figures and
volumes: amplitude/phase display with explicit phase units, HSL fusion,
physical-coordinate and depth-only resampling, NIfTI export, a reader for the
Part 2 handoff, and a handoff check that writes a self-contained HTML report.
Its depth backend is connected to the normal app through
`part4_ui/lfi_core/adapters/part3.py`; verified user-supplied depth calibration
is required. Other Part 3 functions, including imaging and NIfTI export, remain
separate from the desktop UI. Details, API and validation:
[docs/part3_rebuild.md](docs/part3_rebuild.md); demo steps:
[docs/demo_part3.md](docs/demo_part3.md).

Desktop integration now goes through the normal app:

```sh
python -m part4_ui.app
```

In the 3D tab, expand **Depth calibration** and enter confirmed scan settings
(or use verified per-slice `depth_mm` metadata). The adapter linearly resamples
amplitude onto a uniform depth grid and takes phase from the nearest measured
slice; rendered axes remain sample indices, not physical XYZ proportions.
No Vegetables calibration preset is applied, and the frame ID corresponding to
the supplied start depth remains unconfirmed.

The older Part 3 launcher is also supported through the UI-owned
`part4_ui.app.smart_load(path)` compatibility entry point. It delegates ordinary
paths to the registered loaders; the launcher explicitly supplies its Part 2
volume NPZ/folder handling. NPZ is not silently enabled in the normal app.

```sh
python -m part3_reconstruction.viewer path/to/scan.hdf5
python -m part3_reconstruction.viewer path/to/frame-folder
python -m part3_reconstruction.viewer path/to/part2-volume.npz
```

The upstream `--vegetables-2026` preset remains unverified:
**frame-to-depth mapping not yet confirmed**. It is not a default of the normal
app and must not be treated as confirmed calibration. The following legacy
invocation is retained only with that qualification:

```sh
# frame-to-depth mapping not yet confirmed; verify acquisition settings first
python -m part3_reconstruction.viewer data/vegetables --vegetables-2026
```

The legacy launcher retains its registered-backend option before loading;
actual reconstruction still requires loaded data and valid depth metadata.
For explicit user-entered calibration, use the normal app's Depth calibration
section. Startup with the real chip scan has been verified offscreen; this does
not establish a physical depth mapping for chip.

`part3_reconstruction.pipeline.run_part1_to_part3(path)` runs Part 1 loading and
Part 2 pre-processing in memory and returns a Part 3 image. It requires the
Part 2 `spatial_products` expected by that pipeline; phase units and calibration
must retain their provenance.

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q tests/test_imaging.py tests/test_volume.py tests/test_handoff.py tests/test_pipeline.py tests/test_gui_backend.py tests/test_gui_backend_qt.py tests/test_sequence.py tests/test_audit.py tests/test_benchmark.py tests/test_integration.py tests/test_shared_types.py
python -m part3_reconstruction handoff <handoff folder> --output outputs/handoff --raw <raw HDF5 folder>
```

Optional: `napari` and `SimpleITK` enable `--napari` and the NIfTI check behind
`--nominal-pixel-mm`. Exact versions used for the reported results are in
`requirements-part3-tested.txt`.

## Team responsibilities

- Zeyu Wang: data import and management (Part 1).
- Haobo Yu: signal preprocessing (Part 2).
- Mingxuan Tang: image generation and reconstruction (Part 3).
- He Zhenyu (Vincent): UI design, integration and usability evaluation (Part 4).
- Karl Bertling: project supervision.

University of Queensland — academic use.

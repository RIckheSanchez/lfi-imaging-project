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

- Native open dialog, file drag/drop and five recent files saved with QSettings.
- English and Chinese interface switching.
- Amplitude/phase selection, brightness, contrast, and seven colormaps.
- Z slider and numeric slice index, with the actual HDF5 z value displayed.
- Aspect-locked 2D image with bounded pan/zoom and **Fit to window** / Cmd+0.
- Colorbar with raw data range and the effective display color transform.
- Raw-value mouse hover, selected-pixel crosshairs, pixel information and histogram.
- Amplitude and phase waveforms at the selected pixel.
- Calibrated 2D scale bar when a positive pixel size is entered.
- Downsampled amplitude Volume and three uncapped Isosurface meshes; adjustable
  lower percentile (default 90), fixed 99.5th-percentile upper threshold and camera reset.
  Outer-to-inner mesh opacity is 0.12, 0.30 and 0.60. Volume clips its sampled
  values to the 1st–99.5th percentile range before normalization and uses a
  quadratic opacity transfer function. Isosurface controls are hidden in Volume mode.
- Raw selected-channel CSV, colored pixel PNG/JPEG/TIFF and publication PNG
  with axes/colorbar. PNG/TIFF and publication PNG carry 300 DPI.
- Thread-pool loading, slice/histogram preparation, backend processing, 3D
  computation and export, with busy state and readable error dialogs.
- Registered preprocessing/reconstruction interfaces; controls stay hidden
  until a real implementation is registered.

## Tests

Install development dependencies and run the full pytest suite offscreen:

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Tests use the local sample files `Realdata/mushroom.hdf5` and
`Realdata/chip_borad1.hdf5`. `Realdata/` is ignored by Git; obtain these scan
fixtures separately when cloning the project. GUI tests use pytest-qt and
exercise worker-thread execution, interaction, exports, registration and
preprocessing state preservation. The chip test saves an offscreen screenshot.

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
- 3D has no physical Z spacing. XYZ coordinates are sensor/sample indices.
- Team preprocessing and reconstruction modules are not integrated by default.
  The inactive Gaussian example illustrates registration and is not a validated
  scientific processing pipeline.
- The loader retains complete amplitude/phase arrays in memory. Large scans
  require sufficient RAM; downsampling limits 3D display work, not load memory.
- 3D is amplitude-based. Flat/degenerate or single-layer data may not define
  valid isosurfaces. Display changes require Render / Refresh 3D again.
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
remain available for team development; they are not automatically wired into
the desktop backend pipeline.

## Team responsibilities

- Zeyu Wang: data import and management (Part 1).
- Mingxuan Tang: signal preprocessing (Part 2).
- Haobo Yu: image generation and reconstruction (Part 3).
- He Zhenyu (Vincent): UI design, integration and usability evaluation (Part 4).
- Karl Bertling: project supervision.

University of Queensland — academic use.

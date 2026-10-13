# Desktop viewer — stages 0–2

Developed and tested in the existing Conda `base` environment. From the repository root:

```sh
conda run -n base python -m part4_ui.app
```

A clean Conda environment can install `requirements-desktop.txt` and
`requirements-test.txt` using its Python. Streamlit and Plotly are only needed
for the retained legacy frontend and its AppTest script; the desktop entry point
and shared core do not import either package.

```sh
QT_QPA_PLATFORM=offscreen python -m pytest -q -s
python tests/verify_viewer.py
streamlit run legacy/streamlit_app.py
```

`streamlit-final` tags the pre-migration baseline. The legacy frontend now uses
`part4_ui.lfi_core`; its cache wrappers remain frontend-specific. The shared
core is uncached, so it cannot reuse data from a previous source. The moved
loader's only edit is a relative import. The data types file is unchanged.

The desktop window supports open/drop/recent files, synchronized Z indices,
channel/brightness/contrast/colormap controls, calibrated scale bars, raw hover
and selected-pixel values, histograms, and the selected pixel's waveforms.
Mouse wheel/drag zoom and pan are preserved when display parameters change.
File loading, optional preprocessing, slice/histogram preparation, and exports
run on a QThreadPool. Superseded load/slice results are ignored. An indeterminate
progress indicator stays visible until queued work completes. Worker failures
are shown in QMessageBox; closing while busy requests waiting for completion.

Pixel scale defaults to zero because the supplied files have no calibration
metadata. The Z slider and number box are slice indices; the actual HDF5 Z value
is shown below them. Recent files and language use QSettings. The parameter
placeholder is disabled because Identity has no parameters.

Backend integration: implement `Preprocessor.preprocess(data, params)` or
`Reconstructor.reconstruct(data, params)` in `lfi_core/backend.py`'s protocol.
Inject a preprocessor with `MainWindow(preprocessor=your_backend)`; the window
calls it off the GUI thread. Identity implementations return their input object.
Desktop 3D rendering/reconstruction is intentionally a placeholder at this
stage-2 checkpoint. The real implementation belongs to subsequent stages.

Tests exercise actual mouse clicks and raw hover coordinates, language,
phase/adjustment/zoom preservation, scale enable/disable, recent-file persistence,
worker exports, readable load failures, preprocessing thread identity, and the
real chip loader. The chip test uses a 10 ms QTimer to confirm GUI events continue
through `smart_load`, and saves `docs/screenshots/chip_desktop.png` using QWidget.grab.
Native file dialogs are mocked in offscreen tests; drag/drop handler behavior is
covered. These tests do not establish native macOS dialog appearance or OpenGL
3D behavior. Qt's offscreen `propagateSizeHints` messages are expected warnings.

Test logs for each stage are in `tests/stage*-output.txt` or
`tests/stage*-pytest-output.txt` / `tests/stage*-legacy-output.txt`.

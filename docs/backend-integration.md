# Backend integration

The shared core declares structural `Preprocessor` and `Reconstructor` protocols
in `part4_ui/lfi_core/backend.py`. Implement respectively:

```python
preprocess(data: LFIRawData, params: dict) -> LFIRawData
reconstruct(data: LFIRawData, params: dict) -> LFIRawData
```

Return LFIRawData with matching amplitude/phase shapes and one z index per
slice. Raise a readable exception for invalid input. Do not create Qt widgets
or mutate the source data: implementations run on QThreadPool worker threads,
and existing views may still refer to the source arrays.

Register instances before `MainWindow` is created. No plugin is enabled by
default. Registering None or an Identity instance disables that integration.
The corresponding controls are hidden until a real implementation is registered.

```python
from part4_ui.lfi_core.backend import register_preprocessor, register_reconstructor
from your_team_module import MyPreprocessor, MyReconstructor
from part4_ui.app import main

register_preprocessor(MyPreprocessor(), params={"sigma": 1.0})
register_reconstructor(MyReconstructor(), params={"iterations": 5})
main()
```

Registration copies the default parameter dictionary. Set LFI_DEV_PARAMS=1 before
launch to show the editable JSON parameters; without it, the controls are hidden
and preprocessing always receives an empty override dictionary {}. In developer
mode, the pre-processing panel shows an editable JSON object. Committing a valid JSON edit updates the
parameters; if processing is enabled the current scan is reprocessed.
Reconstruction parameters are passed from its registration dictionary.
Each worker receives an independent deep copy so later UI edits cannot change
an in-flight operation. Checkbox changes enable/disable the selected backend.

See `gaussian_preprocessor_example.py` for an inactive example. It uses only
NumPy (no SciPy dependency), does not import/register automatically, and is for
illustration rather than a validated scientific processing pipeline.


## Part 2 preprocessing

The desktop entrypoint registers Part2Preprocessor before creating MainWindow.
It calls public preprocess_part1 with an absolute path to the repository TOML.
The JSON parameters are nested config overrides. The selected amplitude product
is recorded in metadata.ui_preprocessing; the full Part 2 metadata is retained
under part2_preprocessing. Radian labels require upstream phase_converted=True
and Part 2 phase_units="radians". Missing data remain NaN; display calculations
ignore them, image/volume missing pixels are transparent, CSV retains nan, and
all-missing channels raise an error. No measured values are filled with zero.

## Path loader registration (future NPZ integration)

Use part4_ui.lfi_core.loaders.Loader and register_loader before creating windows.
Each loader declares a unique name, path_types=("file",) or ("dir",), extensions,
and implements load(path)->UI LFIRawData and make_data_key(path)->hashable tuple.
get_loaders returns a snapshot. Open, Recent, drag/drop and Open Folder use the
same load_path dispatch on a worker. HDF5 and Part 1 frame-folder loaders are
registered by default; neither is replaced through a module monkeypatch.

Mingxuan's future NPZ loader should have name="part3-handoff-npz", file type and
extensions=(".npz",). It must validate schema/version, axis order, array and mask
shapes, frame IDs, z/depth calibration and phase conversion provenance. Do not
set phase_converted=True solely because a field is named raw_phase or the file
has an NPZ suffix. Preserve metadata and the source path through an explicit
UI LFIRawData mapping. Include source file mtime_ns/size, relevant sidecars,
loader version and processing parameters in make_data_key. Then call
register_loader(instance); never replace part4_ui.app.smart_load.

NPZ is deliberately not registered in this release: a real handoff dataset and
reliable phase-unit evidence are required first. Folder identities contain the
resolved paths, mtime_ns and sizes of every data file and experiment_details.txt.
The registry preserves insertion order, so a plugin must not overlap the default
HDF5 or directory claims without a future explicit selection UI. Unsupported
files raise readable errors instead of silently falling back.

Reconstruction UI is hidden without a non-Identity registration. With a backend
registered, matching depth_mm enables the checkbox; missing/invalid depth shows
the calibration explanation after a scan is loaded.

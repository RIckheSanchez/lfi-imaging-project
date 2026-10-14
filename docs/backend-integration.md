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

Registration copies the default parameter dictionary. The pre-processing panel
shows an editable JSON object. Committing a valid JSON edit updates the
parameters; if processing is enabled the current scan is reprocessed.
Reconstruction parameters are passed from its registration dictionary.
Each worker receives an independent deep copy so later UI edits cannot change
an in-flight operation. Checkbox changes enable/disable the selected backend.

See `gaussian_preprocessor_example.py` for an inactive example. It uses only
NumPy (no SciPy dependency), does not import/register automatically, and is for
illustration rather than a validated scientific processing pipeline.

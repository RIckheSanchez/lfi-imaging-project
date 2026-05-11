# THz LFI Signal Pre-processing Module

A signal pre-processing module for terahertz laser feedback
interferometry (LFI) skin imaging data.

Part of a four-person collaborative THz LFI skin imaging pipeline
developed under the supervision of Dr Karl Bertling, The University of
Queensland.

## Scope

This module sits between the raw data import stage and the image
generation stage of the shared pipeline. Input and output are both
`LFIRawData` objects (amplitude + phase + `z_indices` + `target_type` +
`metadata`), matching the contract published by the data-import module
(teammate A).

## Operations

| Stage | Methods | Default for skin | Default for leaf |
| --- | --- | --- | --- |
| Noise reduction (on `amplitude`) | `median2d`, `gaussian2d`, `savgol_1d`, `bandpass_1d`, `none` | `median2d` | `savgol_1d` |
| Normalisation (on `amplitude`) | `baseline`, `zscore`, `none` | `baseline` | `zscore` (robust) |
| Phase unwrapping (on `phase`) | `skimage_2d`, `itoh_1d`, `none` | `skimage_2d` | `itoh_1d` |

All parameters live in `config/default.toml`; per-target overrides
under `[targets.<name>...]` select operation families by scan type.

## Repository layout

```
config/default.toml                 # pipeline parameters
src/thz_preprocessing/
    api.py                          # run_pipeline(data, config_path=None, ...)
    config.py                       # TOML loader + per-target merge
    io.py                           # LFIRawData + temporary skin/leaf loaders
    noise_reduction.py              # median2d / gaussian2d / savgol_1d / bandpass_1d
    normalisation.py                # baseline / zscore
    phase_unwrapping.py             # skimage_2d / itoh_1d
tests/test_api.py                   # 9 tests inc. repeatability + real-data smoke
examples/
    run_pipeline.py                 # minimal synthetic example
    run_pilot.py                    # end-to-end demo on real skin + leaf
data/                               # local datasets, git-ignored
docs/
    dataset_inventory.md            # Aim 1 inventory note
    ad_hoc_workflow_review.md       # Aim 1 workflow review note
    progress_seminar_slide_outline.md
```

## Installation (development)

```bash
git clone <repo-url>
cd thz-lfi-preprocessing
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,demo]"
```

## Usage

### Minimal

```python
from thz_preprocessing import api, LFIRawData

result = api.run_pipeline(my_data)               # my_data: LFIRawData
print(result.metadata["pipeline"])               # ordered stage record
print(result.metadata["timings_seconds"])        # per-stage seconds
```

### Reading the supplied pilot datasets

```python
from thz_preprocessing import load_skin_hdf5, load_leaf_mat

skin = load_skin_hdf5("data/skin/2025-11-28_*.hdf5")
leaf = load_leaf_mat(
    "data/leaf/leaf1.mat",
    spatial_slice=(slice(0, 256), slice(0, 256)),
    z_slice=slice(0, 128),
)
```

### End-to-end demo (produces figures for slides)

```bash
python examples/run_pilot.py            # both skin and leaf
python examples/run_pilot.py --skin     # skin only
```

Output figures land in `output/`.

## Tests

```bash
pytest tests/ -v
```

The repeatability test (`test_pipeline_repeatability_byte_identical`)
encodes proposal §3.2.4: same input + same config → byte-identical
output.

## Author

Haobo Yu (49512977), under supervision of Dr Karl Bertling.

# THz LFI Signal Pre-processing Module

A signal pre-processing module for terahertz laser feedback interferometry (LFI) skin imaging data.

Part of a four-person collaborative THz LFI skin imaging pipeline developed under the supervision of Dr Karl Bertling, The University of Queensland.

## Scope

This module sits between the raw data import stage and the image generation stage of the shared pipeline. It accepts imported LFI measurements and produces cleaned, normalised, and (where required) phase-unwrapped signals ready for downstream image reconstruction.

## Planned operations

- **Noise reduction** — band-pass filtering / Savitzky–Golay / wavelet denoising
- **Signal normalisation** — per-scan baseline subtraction / z-score / reference-channel division
- **Phase unwrapping** — 1D Itoh / quality-guided 2D
- **Structured parameter selection** — all operations driven by a named, documented config file

## Status

Skeleton API in place. Implementation in progress — see `src/thz_preprocessing/` for module stubs.

## Installation (development)

```bash
git clone <repo-url>
cd thz-lfi-preprocessing
pip install -e .
```

## Usage (planned)

```python
from thz_preprocessing import api

result = api.run_pipeline(
    raw_signal,
    config_path="config/default.toml",
)
```

## Author

Haobo Yu (49512977), under supervision of Dr Karl Bertling.

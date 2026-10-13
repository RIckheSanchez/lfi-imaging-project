# REIT4841 - LFI Data Loader

Laser Feedback Interferometry (LFI) data loader and output interface for UQ REIT4841.  
Downstream code should use **`output_interface` only** - do not call underscore-prefixed functions in `loader`.

## Install

```bash
pip install -r requirements.txt
# Recommended environment with h5py already installed, e.g.:
# ~/miniconda3/bin/python3 -m pip install -r requirements.txt
```

## Quick start

```python
from output_interface import load_single_scan, load_volume, get_phase_radians

scan = load_single_scan("data/2026_Vegetables/frame-21.hdf5")
phase = get_phase_radians(scan)          # radians; raises if encoding unconfirmed

vol = load_volume("data/2026_Vegetables")  # stack multi-file frames in numeric order
# Place experiment_details.txt in the same folder → vol.metadata["depth_mm"]
```

See [`INTEGRATION.md`](INTEGRATION.md) for contracts and phase rules.

## Tests

```bash
pytest -q
# Legacy entry points still work:
python3 test_loader.py
python3 test_output_interface.py
```

## How to add a new format / dataset

1. **Identity:** add a folder keyword in `loader._TARGET_ALIASES` → canonical name (e.g. `"sound": "Sound"`).  
2. **Structure:** new HDF5 layout → `_detect_format` + `_read_hdf5_*`; `.mat` → `MAT_LAYOUTS` variable names.  
3. **Phase:** add an entry in `phase_encoding.PHASE_ENCODINGS`; if unconfirmed, use `_UnconfirmedEncoding`, then map it in `TARGET_PHASE_ENCODING`.  
   **Never** choose a formula from observed min/max.  
4. **Tests:** add cases under `tests/` for identity, shape, encoding name, and `phase_converted`.  
5. **Docs:** update `RDM_DATASETS.md` and `INTEGRATION.md`.

## Key files

| File | Role |
|---|---|
| `output_interface.py` | Public API |
| `loader.py` | I/O implementation |
| `phase_encoding.py` | Phase lookup table |
| `RDM_DATASETS.md` | Real shapes / formats inventory |
| `METADATA_FIELDS.md` | Sidecar field reference |
| `master-checklist.md` | Delivery checklist (19 items) |
| `PROVENANCE.md` | frame-1 / frame-3 provenance TBD |
| `sus/` | SUS materials (coordinate with Person 4 / He Zhenyu) |

## Known limits (for the thesis report)

- Chip / Marble / Skin / PCB phase formulas are unconfirmed → raw codes retained.  
- Sound `.mat` variable names need a real RDM file check.  
- Leaf v5 partial loads still peak high in RAM; v7.3 can stream.  
- Public Skin data: contact reply said no more for public use; verify against RDM.

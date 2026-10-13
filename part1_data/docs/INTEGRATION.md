# LFI data loading - integration guide (downstream pipeline)

Depend only on `output_interface.py` and `LFIRawData`. Do not call underscore-prefixed helpers in `loader.py`.

## Minimal example

```python
from output_interface import (
    load_single_scan, load_volume, iter_scans,
    get_phase_radians, PhaseNotConvertedError,
)

scan = load_single_scan("data/2026_Vegetables/frame-21.hdf5")
amp = scan.amplitude                 # (n, H, W) float64
phase = get_phase_radians(scan)      # (n, H, W) float64, radians

# Vegetables: one depth scan split into many frame-XX.hdf5 → one volume
# Directory walk sorts by numeric frame id (21 before 100), not lexical order
vol = load_volume("data/2026_Vegetables")   # add experiment_details.txt for depth_mm
phase = get_phase_radians(vol)              # (n_frames, H, W)

errors = []
for scan in iter_scans("data/2026_Vegetables", errors=errors):
    try:
        phase = get_phase_radians(scan)
    except PhaseNotConvertedError:
        continue                     # unconfirmed formula or anomalous frame
    ...
print(errors)                        # [(path, message), ...] skipped files
```

## Supported files

| `metadata['file_format']` | Structure | Datasets |
|---|---|---|
| `hdf5_zstack` | `/amplitude/<z>`, `/phase/<z>` | Chip, Marble |
| `hdf5_frames` | `/frames/<id>/Image Data/<QCL>/<Side>/IMAGE_*` | Skin sample, `frame-3.hdf5` |
| `hdf5_data_v2026` | `/data/IMAGE_*/frame_<id>_qcl_<n>_<left\|right>` | 2026 Vegetables |
| `volume_from_frames` | many single-frame files stacked | `load_volume()` |
| `mat` | MATLAB v5 / v7 / v7.3; names in `loader.MAT_LAYOUTS` | Leaf (`scan_mode=time_series`); Sound (phase unconfirmed) |

## `LFIRawData` contract

| Field | Meaning |
|---|---|
| `amplitude` | `(n, H, W)`, `float64`, raw amplitude |
| `phase` | `(n, H, W)`; radians `float64` if `phase_converted`, else raw codes |
| `z_indices` | length n; z keys, frame ids, or mat stack indices |
| `waveform` | optional `(n, H, W, 184)` when `include_waveform=True` |
| `target_type` | from **folder name** identity, never from value range |
| `metadata` | see below |

| key | Meaning |
|---|---|
| `file_format` | table above |
| `phase_encoding_used` | key in `phase_encoding.PHASE_ENCODINGS` |
| `phase_converted` | whether radians conversion ran |
| `original_dtype` | on-disk phase dtype |
| `frame_ids` / `anomalous_frame_ids` | loaded / anomalous frames |
| `depth_mm` / `z_start_mm` / `z_step_mm` | physical depth if `experiment_details.txt` present |
| `laser_channel` | e.g. `QCL1/Left` |
| `scan_identifier` | scan id (may be empty) |
| `acquisition` | `thz/info` parameters when present |
| `rdm_source_path` | RDM path (caller-supplied) |

## Phase rules

Lookup only by dataset identity (`TARGET_PHASE_ENCODING`):

| Dataset | Encoding | Converted? |
|---|---|---|
| Chip / Marble | `zstack_unconfirmed` | **No** (range [-180, 180); likely degrees) |
| Vegetables | `vegetables_int16`: code/8192·π, int16 in [-8192, 8191] | Yes |
| Leaf | `leaf_wrap_4096`: wrap(L_P·π/4096 − π), int16 in [-8192, 8191] | Yes |
| Skin | `skin_unconfirmed` | **No** |
| PCB / Megabugs | `pcb_unconfirmed` | **No** |
| Sound | `sound_unconfirmed` | **No** |

Vegetables frame 20 is anomalous: excluded by default; with `exclude_anomalous=False` it loads raw and is flagged.  
`get_phase_radians()` raises `PhaseNotConvertedError` on raw codes.

## Common parameters

| Parameter | Use |
|---|---|
| `side='Right'` / `qcl='QCL1'` | channel select |
| `frame_ids=[21, 22]` | subset (HDF5 frame ids; `.mat` 0-based indices) |
| `exclude_anomalous=False` | include anomalous frames |
| `include_waveform=True` | load `RX_LINE_OF_BEST_FIT` (~190 MB/frame) |
| `mat_vars={...}` | override `.mat` variable names |

### `load_volume(data_dir, ...)`

Stacks `frame-XX.hdf5` in **numeric** order into `(n, H, W)`.  
If `experiment_details.txt` is present:

```
depth_mm[i] = z_start_mm + (frame_id - first_frame_id) * z_step_mm
```

## Export

```bash
python3 output_interface.py data/2026_Vegetables --out exported
python3 output_interface.py data/2026_Vegetables --out exported --arrays
python3 output_interface.py data/Marble --out exported --sidecar --hdf5-attrs
```

`schema_version` is currently `1.1` (still reads `1.0`).

## Known limits

- Leaf third axis is a **time series** (Karl / contact reply ~2026-10).  
- Leaf v5 partial load still peaks high in RAM; prefer MATLAB v7.3 for true streaming.  
- Sound `.mat` layout skeleton ready; confirm variable names on a real RDM file.  
- Skin / PCB / Chip / Marble formulas unconfirmed.  
- See `PROVENANCE.md` for local `frame-1` / `frame-3` identity.

## Tests

```bash
python3 -m pytest -q tests/
python3 test_loader.py
python3 test_output_interface.py
cd sus && python3 test_sus_score.py
```

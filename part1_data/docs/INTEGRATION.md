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
| Sound | `sound_unconfirmed` | **No** (requires `mat_vars=`; layout unverified) |

Vegetables frame 20 is anomalous: excluded by default.  
`load_volume` **skips** `frame-20.hdf5` before calling `load_single_scan` so a full Vegetables folder does not crash.  
With `exclude_anomalous=False` anomalous frames load as raw codes and are flagged.  
`get_phase_radians()` raises `PhaseNotConvertedError` on raw codes.

## Common parameters

| Parameter | Use |
|---|---|
| `side='Right'` / `qcl='QCL1'` | channel select (frames / v2026 only) |
| `frame_ids=[21, 22]` | subset (HDF5 frame ids; `.mat` 0-based indices); **not** for z-stack |
| `exclude_anomalous=False` | include anomalous frames |
| `include_waveform=True` | load `RX_LINE_OF_BEST_FIT` (~190 MB/frame) |
| `mat_vars={...}` | **required for Sound**; overrides Leaf names if needed |

Default channel when `qcl`/`side` omitted (frames + v2026): **lowest-numbered QCL + Left**.

### `load_volume(data_dir, ...)`

Stacks `frame-XX.hdf5` in **numeric** order into `(n, H, W)`.  
Anomalous frame files (Vegetables 20) are skipped by default and listed in `metadata['skipped_anomalous_frame_ids']`.  
If `experiment_details.txt` is present:

```
depth_mm[i] = z_start_mm + (frame_id - reference) * z_step_mm
```

Accepted keys include `start z position (mm)` and `imcrements (mm)` (typo for increments).  
`reference` = `first_frame_id` from the file when present; otherwise the **minimum frame id on disk** (including anomalous files such as Vegetables 20 that `load_volume` skips). Do not use the min of the loaded batch alone — that under-shifts depth by one step.

## Export

```bash
python3 output_interface.py data/2026_Vegetables --out exported
python3 output_interface.py data/2026_Vegetables --out exported --arrays
python3 output_interface.py data/Marble --out exported --sidecar --hdf5-attrs
```

`schema_version` is currently `1.1` (still reads `1.0`).

## Known limits

- Leaf third axis is a **time series** (Karl, email, 2026-10-14); Leaf does **not** attach `depth_mm`.  
- Leaf v5 partial load still peaks high in RAM; prefer MATLAB v7.3 for true streaming.  
- Sound `.mat` is unverified — always pass `mat_vars=`; do not guess names.  
- Skin / PCB / Chip / Marble formulas unconfirmed (await Karl).  
- `write_metadata_sidecar` requires `path=` (does not write beside RDM sources by default).  
- See `PROVENANCE.md` for local `frame-1` / `frame-3` identity.  
- Team `shared/data_types.py` is **not** the live UI contract (missing `z_indices` / `metadata` / `get_slice`).  
  Part 4 currently uses a copy under `part4_ui/lfi_core/` (same core fields as Part 1 `LFIRawData`, without `waveform`).  
  Prefer `part1_data` / `output_interface` as the source of truth once the UI adapter lands on `main`.

## Integration status (Part 4 UI — He Zhenyu, 2026-10)

| Claim | Status |
|---|---|
| Byte-level match vs UI’s previous reader (`load_single_scan` defaults) on chip / mushroom / marble samples | **Verified** |
| Part 1 pytest suite on UI side | **Passed** |
| UI read path switched to Part 1 `load_single_scan` via adapter | **In progress** (not on `main` yet) |
| `load_volume` used by UI | **Not used** |
| Phase labels in UI | Was wrongly always “rad”; will follow Part 1 `phase_converted` / raw-code status |
| SUS with desktop UI | Coordinate **one** participant session for Part 1 materials + Part 4 UI |

**Poster / demo wording:** “Data consistent with UI reader; UI wiring in progress — not merged to `main`.”  
Do **not** write “fully integrated with Part 4” until Vincent confirms the push.

## Tests

```bash
python3 -m pytest -q tests/
python3 test_loader.py
python3 test_output_interface.py
cd sus && python3 test_sus_score.py
```

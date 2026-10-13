# Part 1 — LFI data loader

Laser Feedback Interferometry (LFI) data import for UQ REIT4841.  
Downstream code should use **`output_interface` only** — do not call underscore-prefixed helpers in `loader.py`.

## Install / import

```bash
pip install -r requirements.txt
# From the team repo root (preferred for teammates):
#   from part1_data import load_single_scan, load_volume, get_phase_radians
# Or cd into this folder and use flat imports (pytest / scripts).
```

```python
from part1_data import load_single_scan, load_volume, get_phase_radians
# equivalent when cwd is part1_data/:
# from output_interface import load_single_scan, load_volume, get_phase_radians

scan = load_single_scan("data/2026_Vegetables/frame-21.hdf5")
phase = get_phase_radians(scan)  # radians; raises if encoding unconfirmed

vol = load_volume("data/2026_Vegetables")  # skips Vegetables frame 20 by default
# Place experiment_details.txt beside Vegetables frames → vol.metadata["depth_mm"]
```

See [`docs/INTEGRATION.md`](docs/INTEGRATION.md) for contracts and phase rules.

## Tests

```bash
cd part1_data
python3 -m pytest -v tests/
```

## Default laser channel

For **frames** and **v2026** layouts, when `qcl` / `side` are omitted:

- **QCL:** lowest-numbered QCL present (e.g. QCL0 before QCL1)
- **Side:** `Left` when available

Pass `qcl=` / `side=` explicitly to override.  
**z-stack** (Chip / Marble) does not support `frame_ids` / `qcl` / `side` — passing them raises `ValueError`.

## Sound (.mat)

Sound variable names are **not verified** on a real RDM file. Do not guess.  
Always pass `mat_vars={"amplitude": "...", "phase": "..."}`.

## Metadata sidecar

`write_metadata_sidecar(data, path=...)` requires an explicit path by default  
(RDM mounts are often read-only). Use `--sidecar` with `output_interface.py` to write under `--out`.

## Key files

| File | Role |
|---|---|
| `output_interface.py` | Public API |
| `loader.py` | I/O implementation |
| `phase_encoding.py` | Phase lookup table (identity only) |
| `docs/RDM_DATASETS.md` | Real shapes / formats inventory |
| `docs/METADATA_FIELDS.md` | Sidecar field reference |
| `docs/PROVENANCE.md` | frame-1 / frame-3 provenance TBD |
| `sus/` | SUS materials (coordinate with Person 4 / He Zhenyu) |

## Known limits

- Chip / Marble / Skin / PCB phase formulas unconfirmed → raw codes retained (await Karl).  
- Sound `.mat` requires `mat_vars`; layout not verified.  
- Leaf third axis is a **time series** (Karl, email, 2026-10-14) — no `depth_mm`.  
- Leaf v5 partial loads still peak high in RAM; v7.3 can stream.  
- Team `shared/data_types.py` may differ from `LFIRawData` here — confirm the Part 2 contract with Haobo before claiming integration.

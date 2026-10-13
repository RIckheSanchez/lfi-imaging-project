# Metadata field reference

Thesis Plan 4.1: metadata **saved alongside the data** (JSON sidecar + optional HDF5 attributes).

## How to write sidecars

```python
from output_interface import load_single_scan, write_metadata_sidecar, write_metadata_hdf5_attrs

scan = load_single_scan("data/2026_Vegetables/frame-21.hdf5")
write_metadata_sidecar(scan)                          # → frame-21.meta.json next to source
write_metadata_hdf5_attrs(scan, "exported/frame-21_meta.h5")  # attrs-only HDF5
```

CLI:

```bash
python3 output_interface.py data/Marble --out exported --sidecar --hdf5-attrs
```

## Fields

| Field | Type | Meaning | Source |
|---|---|---|---|
| `schema_version` | str | export format version (now `1.1`) | output_interface |
| `target_type` | str | Chip / Leaf / Skin / ... | path identity |
| `file_format` | str | hdf5_zstack / frames / v2026 / mat / volume_from_frames | structure detect |
| `scan_mode` | str | z_stack / frames / time_series / volume | format + type |
| `shape` | [n,H,W] | amplitude shape | load result |
| `frame_ids` / `z_indices` | list | slice or frame ids | file |
| `original_dtype` | str | on-disk phase dtype | file |
| `phase_encoding_used` | str | encoding table key | phase_encoding |
| `phase_converted` | bool | whether radians conversion ran | convert result |
| `phase_units` | str | `rad` or `raw_code` | record |
| `anomalous_frame_ids` | list | e.g. Vegetables 20 | encoding table |
| `laser_channel` | str | e.g. QCL1/Left | HDF5 path |
| `scan_identifier` | str | scan id (may be empty) | file dataset |
| `acquisition` | dict | thz/info etc. | v2026 / frames |
| `depth_mm` / `z_start_mm` / `z_step_mm` | list/float | physical depth | experiment_details.txt |
| `rdm_source_path` | str | RDM path | caller |
| `notes` | str | Karl confirmations etc. | human / encoding |
| `has_waveform` | bool | waveform loaded | include_waveform |

Schema class: `metadata_schema.LFIMetadata`.

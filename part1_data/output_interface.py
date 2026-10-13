"""
output_interface.py
Output interface layer - downstream depends on this layer only; do not call loader internals

Provides:
    load_single_scan()    load one file (re-exported from loader; downstream need not import loader)
    load_volume()         stack multi-file frames by numeric id (Vegetables depth scan)
    iter_scans()          lazily yield LFIRawData one file at a time
    scan_to_record()      LFIRawData → JSON-serializable dict (no large arrays)
    get_phase_radians()   return radian phase; error if not converted
    export_scan()         export one scan: <name>.json (+ optional <name>.npz)
    load_exported()       reload LFIRawData from .npz
    batch_export()        batch-export a folder + manifest.json + metadata_index.csv

CLI:
    python3 output_interface.py data/2026_Vegetables --out exported
    python3 output_interface.py data/2026_Vegetables --out exported --arrays
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from data_types import LFIRawData
from loader import (
    _infer_target_type,
    find_data_files,
    load_single_scan,
    load_volume,
)

__all__ = [
    "SCHEMA_VERSION", "PhaseNotConvertedError", "LFIRawData",
    "load_single_scan", "load_volume", "iter_scans", "scan_to_record",
    "get_phase_radians", "export_scan", "load_exported", "batch_export",
    "write_metadata_sidecar", "write_metadata_hdf5_attrs",
]

# Bump when export format breaks compatibility; downstream can gate on this
SCHEMA_VERSION = "1.1"

INDEX_COLUMNS = [
    "source", "status", "target_type", "file_format", "shape",
    "frame_ids", "anomalous_frame_ids", "original_dtype",
    "phase_encoding_used", "phase_converted", "phase_units",
    "laser_channel", "scan_identifier", "depth_mm", "json", "npz", "error",
]


class PhaseNotConvertedError(RuntimeError):
    """Phase is still raw codes (formula unconfirmed or anomalous frames); cannot treat as radians."""


# ── Internal helpers ────────────────────────────────────────────

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode(errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _find_data_files(data_dir: Path) -> list[Path]:
    """Sort by numeric frame id (frame-21 before frame-100); never use pure lexical sorted()."""
    return find_data_files(data_dir, recursive=True)


def _export_name(source: Path) -> str:
    """parent folder + stem to avoid colliding frame-XX names across datasets."""
    return f"{source.parent.name}_{source.stem}"


# ── Public API ──────────────────────────────────────────────────

def iter_scans(
    data_dir: str,
    target_type: Optional[str] = None,
    *,
    on_error: str = "skip",
    errors: Optional[list] = None,
    **load_kwargs,
) -> Iterator[LFIRawData]:
    """
    Yield scans under a folder one by one.

    Args:
        data_dir:    data root (recursive .hdf5 / .h5 / .mat)
        target_type: only this target type; None = all
        on_error:    "skip" skip failed files and warn; "raise" raise
        errors:      if a list is passed, append (path, message) failures here
        **load_kwargs: forwarded to load_single_scan (e.g. side='Right')
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    for fp in _find_data_files(root):
        # filter by path identity first to avoid reading multi-GB files just to discard them
        if (target_type is not None and "target_type" not in load_kwargs
                and _infer_target_type(str(fp)) != target_type):
            continue
        try:
            data = load_single_scan(str(fp), **load_kwargs)
        except Exception as e:
            if on_error == "raise":
                raise
            msg = f"{type(e).__name__}: {e}"
            warnings.warn(f"[iter_scans] skip {fp}: {msg}", stacklevel=2)
            if errors is not None:
                errors.append((str(fp), msg))
            continue
        if target_type is None or data.target_type == target_type:
            yield data


def scan_to_record(data: LFIRawData) -> dict:
    """LFIRawData → JSON-serializable record (shape/units/metadata only; no arrays)."""
    converted = bool(data.metadata.get("phase_converted", False))
    # drop non-JSON-able leftovers from metadata; keep serializable fields only
    meta = {
        k: v for k, v in data.metadata.items()
        if not isinstance(v, np.ndarray)
    }
    return _to_jsonable({
        "schema_version": SCHEMA_VERSION,
        "target_type": data.target_type,
        "file_path": data.file_path,
        "shape": list(data.amplitude.shape),
        "z_indices": list(data.z_indices),
        "amplitude_dtype": str(data.amplitude.dtype),
        "phase_dtype": str(data.phase.dtype),
        "phase_units": "rad" if converted else "raw_code",
        "has_waveform": data.waveform is not None,
        "depth_mm": meta.get("depth_mm"),
        "metadata": meta,
    })


def get_phase_radians(data: LFIRawData) -> np.ndarray:
    """
    Return phase in radians.

    Raises:
        PhaseNotConvertedError: phase is still raw codes
    """
    m = data.metadata
    if m.get("phase_converted"):
        return data.phase
    if m.get("anomalous_frame_ids"):
        reason = f"contains anomalous frames {m['anomalous_frame_ids']}"
    else:
        reason = f"phase encoding '{m.get('phase_encoding_used')}' unconfirmed"
    raise PhaseNotConvertedError(
        f"{data.file_path}: phase is raw codes({reason}), cannot treat as radians."
    )


def export_scan(data: LFIRawData, out_dir: str, *, arrays: bool = False,
                name: Optional[str] = None) -> dict:
    """
    Write one scan into out_dir.

    Returns:
        {"json": path, "npz": path or None}
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = name or _export_name(Path(data.file_path))
    record = scan_to_record(data)

    json_path = out / f"{name}.json"
    json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    npz_path = None
    if arrays:
        npz_path = out / f"{name}.npz"
        payload = {
            "amplitude": data.amplitude,
            "phase": data.phase,
            "z_indices": np.asarray(data.z_indices),
            "record_json": np.asarray(json.dumps(record, ensure_ascii=False)),
        }
        if data.waveform is not None:
            payload["waveform"] = data.waveform
        depth = data.metadata.get("depth_mm")
        if depth is not None:
            payload["depth_mm"] = np.asarray(depth, dtype=np.float64)
        np.savez_compressed(npz_path, **payload)
    return {"json": str(json_path), "npz": str(npz_path) if npz_path else None}


def load_exported(npz_path: str) -> LFIRawData:
    """Reload LFIRawData from an export_scan(arrays=True) .npz."""
    with np.load(npz_path, allow_pickle=False) as z:
        record = json.loads(str(z["record_json"]))
        ver = record.get("schema_version")
        if ver not in (SCHEMA_VERSION, "1.0"):
            raise ValueError(
                f"export schema_version={ver}, this code supports 1.0/{SCHEMA_VERSION}"
            )
        meta = dict(record["metadata"])
        if "depth_mm" in z.files:
            meta["depth_mm"] = z["depth_mm"].tolist()
        return LFIRawData(
            amplitude=z["amplitude"],
            phase=z["phase"],
            z_indices=z["z_indices"].tolist(),
            target_type=record["target_type"],
            file_path=record["file_path"],
            metadata=meta,
            waveform=z["waveform"] if "waveform" in z.files else None,
        )


def _index_row(source: Path, record: Optional[dict], paths: dict, error: str = "") -> dict:
    row = {c: "" for c in INDEX_COLUMNS}
    row["source"] = str(source)
    if record is None:
        row["status"] = "FAILED"
        row["error"] = error
        return row
    m = record["metadata"]
    row.update({
        "status": "OK",
        "target_type": record["target_type"],
        "file_format": m.get("file_format", ""),
        "shape": "x".join(str(s) for s in record["shape"]),
        "frame_ids": json.dumps(m.get("frame_ids", [])),
        "anomalous_frame_ids": json.dumps(m.get("anomalous_frame_ids", [])),
        "original_dtype": m.get("original_dtype", ""),
        "phase_encoding_used": m.get("phase_encoding_used", ""),
        "phase_converted": m.get("phase_converted", ""),
        "phase_units": record["phase_units"],
        "laser_channel": m.get("laser_channel", ""),
        "scan_identifier": m.get("scan_identifier", "") if isinstance(m.get("scan_identifier"), str) else "",
        "depth_mm": json.dumps(record.get("depth_mm")) if record.get("depth_mm") is not None else "",
        "json": paths["json"],
        "npz": paths["npz"] or "",
    })
    return row


def batch_export(data_dir: str, out_dir: str, *, arrays: bool = False,
                 exclude_anomalous: bool = False, **load_kwargs) -> dict:
    """
    Batch-export a folder.

    Anomalous frames are exported by default (exclude_anomalous=False) and flagged in metadata,
    so archives are complete; downstream get_phase_radians() will reject them.

    Returns:
        manifest dict (also written to out_dir/manifest.json)
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for fp in _find_data_files(root):
        try:
            data = load_single_scan(str(fp), exclude_anomalous=exclude_anomalous, **load_kwargs)
            paths = export_scan(data, out_dir, arrays=arrays)
            rows.append(_index_row(fp, scan_to_record(data), paths))
        except Exception as e:
            rows.append(_index_row(fp, None, {}, f"{type(e).__name__}: {e}"))
            print(f"[batch_export] failed {fp}: {e}")

    with open(out / "metadata_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(root.resolve()),
        "arrays_exported": arrays,
        "n_files": len(rows),
        "n_ok": sum(r["status"] == "OK" for r in rows),
        "n_failed": sum(r["status"] == "FAILED" for r in rows),
        "files": rows,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[batch_export] done: {manifest['n_ok']} ok, {manifest['n_failed']} failed → {out}"
    )
    return manifest


def write_metadata_sidecar(
    data: LFIRawData,
    path: Optional[str] = None,
) -> str:
    """
    Write metadata as a JSON sidecar (Thesis Plan 4.1: saved alongside the data).

    Default path: `<stem>.meta.json` next to the source file.
    """
    record = scan_to_record(data)
    if path is None:
        src = Path(data.file_path)
        path = str(src.parent / f"{src.stem}.meta.json")
    Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_metadata_hdf5_attrs(
    data: LFIRawData,
    h5_path: str,
    *,
    group: str = "/",
) -> str:
    """
    Write serializable metadata fields as HDF5 attributes (Thesis Plan 4.1).

    Large lists (frame_ids, depth_mm, ...) stored as JSON strings; arrays are not written.
    """
    import h5py

    record = scan_to_record(data)
    flat = {
        "schema_version": record["schema_version"],
        "target_type": record["target_type"],
        "file_path": record["file_path"],
        "shape": json.dumps(record["shape"]),
        "phase_units": record["phase_units"],
        "has_waveform": str(record.get("has_waveform", False)),
    }
    meta = record.get("metadata") or {}
    for key in (
        "file_format", "scan_mode", "phase_encoding_used", "phase_converted",
        "original_dtype", "laser_channel", "scan_identifier", "rdm_source_path",
        "notes", "z_start_mm", "z_step_mm",
    ):
        if key in meta and meta[key] is not None and meta[key] != "":
            val = meta[key]
            flat[key] = json.dumps(val) if isinstance(val, (list, dict)) else str(val)
    for key in ("frame_ids", "anomalous_frame_ids", "depth_mm", "z_indices"):
        src = meta.get(key) if key != "z_indices" else record.get("z_indices")
        if src is not None:
            flat[key] = json.dumps(src)

    out = Path(h5_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "a") as f:
        target = f[group] if group != "/" else f
        for k, v in flat.items():
            target.attrs[k] = v
        # marker dataset so an attrs-only file is still valid HDF5
        if "lfi_metadata_marker" not in f:
            f.create_dataset("lfi_metadata_marker", data=np.asarray([1], dtype=np.uint8))
    return str(out.resolve())


def main():
    parser = argparse.ArgumentParser(description="Batch-export LFI scan metadata (and optional arrays)")
    parser.add_argument("data_dir", help="data root directory")
    parser.add_argument("--out", default="exported", help="output directory")
    parser.add_argument("--arrays", action="store_true", help="also export .npz arrays (large)")
    parser.add_argument(
        "--sidecar", action="store_true",
        help="also write beside each source file <stem>.meta.json",
    )
    parser.add_argument(
        "--hdf5-attrs", action="store_true",
        help="write in out dir for each scan <name>_meta.h5(attrs only)",
    )
    args = parser.parse_args()
    manifest = batch_export(args.data_dir, args.out, arrays=args.arrays)
    if args.sidecar or args.hdf5_attrs:
        from loader import load_single_scan as _load
        for row in manifest["files"]:
            if row["status"] != "OK":
                continue
            data = _load(row["source"], exclude_anomalous=False)
            if args.sidecar:
                write_metadata_sidecar(data)
            if args.hdf5_attrs:
                name = Path(row["source"]).stem
                write_metadata_hdf5_attrs(
                    data, str(Path(args.out) / f"{name}_meta.h5")
                )


if __name__ == "__main__":
    main()

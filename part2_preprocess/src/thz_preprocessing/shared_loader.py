"""Thin adapter to the shared data-import loader (teammate A, Part 1).

The upstream package is vendored unmodified in ``vendor/data_import``. This
module converts its ``LFIRawData`` into this package's ``LFIRawData`` so the
pipeline consumes the shared loader's product through one call.

Upstream leaves unconfirmed phase encodings (Skin, PCB, Unknown) as raw
codes. The adapter keeps those codes untouched and marks
``phase_encoding_status = "unconfirmed"``, which makes the pipeline withhold
unwrapping and phase spatial correction.

The upstream loader (v3, 2026-10-11) reads ``hdf5_zstack``, ``hdf5_frames``,
the 2026 RDM ``/data/IMAGE_*`` layout and MATLAB files. :func:`load_scan`
falls back to this package's loaders only for a layout upstream rejects, and
records that it did so.
"""

from __future__ import annotations

import contextlib
import importlib
import io as _stdio
import sys
from pathlib import Path
from typing import Any

import numpy as np

from thz_preprocessing import io
from thz_preprocessing.io import LFIRawData

VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "data_import"

# Upstream target names -> config keys used by this package.
_TARGET_MAP = {"Vegetables": "Vegetable"}


def _upstream():
    """Import the vendored loader module without polluting ``sys.path``."""
    if not (VENDOR_DIR / "loader.py").exists():
        raise ImportError(f"Shared loader not found in {VENDOR_DIR}")
    sys.path.insert(0, str(VENDOR_DIR))
    try:
        return importlib.import_module("loader")
    finally:
        sys.path.remove(str(VENDOR_DIR))


def from_upstream(scan: Any) -> LFIRawData:
    """Convert an upstream ``LFIRawData`` to this package's contract."""
    upstream_meta = dict(scan.metadata)
    converted = bool(upstream_meta.get("phase_converted", False))
    metadata: dict[str, Any] = {
        "loader": "shared_data_import",
        "source_path": scan.file_path,
        "source_format": upstream_meta.get("file_format", ""),
        "original_dtype": upstream_meta.get("original_dtype", ""),
        "phase_encoding": upstream_meta.get("phase_encoding_used", ""),
        "phase_encoding_status": "explicit_validated" if converted else "unconfirmed",
        "phase_units": "radians" if converted else "raw_codes",
        "upstream_metadata": upstream_meta,
    }
    if not converted:
        metadata["phase_scale_warning"] = (
            "Upstream left the phase as raw codes because its encoding is unconfirmed."
        )
    return LFIRawData(
        amplitude=np.asarray(scan.amplitude, dtype=np.float64),
        phase=np.asarray(scan.phase, dtype=np.float64),
        z_indices=list(scan.z_indices),
        target_type=_TARGET_MAP.get(scan.target_type, scan.target_type),
        metadata=metadata,
    )


def load_via_shared_loader(path: str | Path, **kwargs: Any) -> LFIRawData:
    """Load ``path`` with the upstream ``load_single_scan`` and adapt it.

    Keyword arguments are passed through (``target_type``, ``frame_ids``,
    ``qcl``, ``side``, ``phase_encoding``, ``rdm_source_path`` ...).
    """
    loader = _upstream()
    with contextlib.redirect_stdout(_stdio.StringIO()):
        scan = loader.load_single_scan(str(path), **kwargs)
    return from_upstream(scan)


def load_scan(path: str | Path, **kwargs: Any) -> LFIRawData:
    """Load any supported file, preferring the shared loader.

    Falls back to the local loaders for layouts the shared loader cannot
    read yet, recording ``metadata["loader"] = "local_fallback"`` and the
    reason.
    """
    path = Path(path)
    reason = None
    if path.suffix.lower() != ".mat":
        import h5py

        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            v2026 = "data" in keys and "IMAGE_PHASE" in f["data"]
        if "frames" not in keys and not {"amplitude", "phase"} <= keys and not v2026:
            reason = "HDF5 layout %s is not supported by the shared loader" % sorted(keys)
    if reason is None:
        return load_via_shared_loader(path, **kwargs)

    data = io.load_rdm_hdf5_frame(path, **kwargs)
    data.metadata["loader"] = "local_fallback"
    data.metadata["shared_loader_gap"] = reason
    return data


def load_volume(
    source: str | Path | list,
    *,
    experiment_details_path: str | Path | None = None,
    z_reference_frame_id: int | None = None,
    **kwargs: Any,
) -> LFIRawData:
    """Load a multi-file depth scan through the shared loader as one volume.

    ``source`` is a folder of ``frame-*.hdf5`` files or an explicit list of
    paths. Frames are stacked in numeric frame order (the shared loader's
    folder iteration is lexicographic). Frames the shared loader rejects as
    anomalous (e.g. Vegetables frame 20) are skipped and listed in
    ``metadata["skipped_frames"]``. If ``experiment_details.txt`` is present
    (or given), z positions in mm are attached; ``z_reference_frame_id`` is
    the frame at the recorded start position (default: lowest frame id in
    the folder, including skipped ones).
    """
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        folder = Path(source)
        paths = sorted(folder.glob("frame-*.hdf5"))
        if experiment_details_path is None and (folder / "experiment_details.txt").exists():
            experiment_details_path = folder / "experiment_details.txt"
    else:
        paths = [Path(p) for p in source]
    if not paths:
        raise ValueError("No frame files to load")

    frames, skipped = [], []
    for path in paths:
        try:
            frames.append(load_via_shared_loader(path, **kwargs))
        except ValueError as error:
            skipped.append({"path": str(path), "reason": str(error)})
    if not frames:
        raise ValueError("Every frame was rejected by the shared loader: %s" % skipped)
    frames.sort(key=lambda frame: int(frame.z_indices[0]))
    frame_ids = [int(frame.z_indices[0]) for frame in frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Duplicate frame ids are not allowed: %s" % frame_ids)
    shapes = {frame.shape[1:] for frame in frames}
    statuses = {frame.metadata["phase_encoding_status"] for frame in frames}
    if len(shapes) != 1 or len(statuses) != 1:
        raise ValueError("Frames have incompatible shapes or phase encoding status")

    z_positions_mm = details = None
    if experiment_details_path is not None:
        details = io._read_experiment_details(Path(experiment_details_path))
        start, step = details.get("start_z_mm"), details.get("increment_mm")
        if start is not None and step is not None:
            import re

            all_ids = frame_ids + [int(m.group(1)) for item in skipped
                                   for m in [re.search(r"frame-(\d+)", item["path"])] if m]
            reference = int(z_reference_frame_id) if z_reference_frame_id is not None else min(all_ids)
            z_positions_mm = [round(float(start + (i - reference) * step), 10) for i in frame_ids]

    first = frames[0].metadata
    metadata: dict[str, Any] = {
        "loader": "shared_data_import",
        "source_format": "shared_loader_volume",
        "source_paths": [frame.metadata["source_path"] for frame in frames],
        "frame_ids": frame_ids,
        "skipped_frames": skipped,
        "phase_encoding": first["phase_encoding"],
        "phase_encoding_status": first["phase_encoding_status"],
        "phase_units": first["phase_units"],
        "z_positions_mm": z_positions_mm,
        "experiment_details": details,
        "per_frame_metadata": [frame.metadata for frame in frames],
    }
    return LFIRawData(
        amplitude=np.concatenate([frame.amplitude for frame in frames], axis=0),
        phase=np.concatenate([frame.phase for frame in frames], axis=0),
        z_indices=frame_ids,
        target_type=frames[0].target_type,
        metadata=metadata,
    )


__all__ = ["VENDOR_DIR", "from_upstream", "load_via_shared_loader", "load_scan", "load_volume"]

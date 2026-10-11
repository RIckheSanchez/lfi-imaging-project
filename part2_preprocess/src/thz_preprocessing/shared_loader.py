"""Thin adapter to the shared data-import loader (teammate A, Part 1).

The upstream package is vendored unmodified in ``vendor/data_import``. This
module converts its ``LFIRawData`` into this package's ``LFIRawData`` so the
pipeline consumes the shared loader's product through one call.

Upstream leaves unconfirmed phase encodings (Skin, PCB, Unknown) as raw
codes. The adapter keeps those codes untouched and marks
``phase_encoding_status = "unconfirmed"``, which makes the pipeline withhold
unwrapping and phase spatial correction.

The upstream loader (package dated 2026-10-11) reads ``hdf5_zstack`` and
``hdf5_frames`` files. It does not read the 2026 RDM ``/data/IMAGE_*`` layout
or MATLAB files; for those, :func:`load_scan` falls back to this package's
loaders and records that it did so.
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
    if path.suffix.lower() == ".mat":
        reason = "MATLAB files are not supported by the shared loader"
    else:
        import h5py

        with h5py.File(path, "r") as f:
            keys = set(f.keys())
        if "frames" not in keys and not {"amplitude", "phase"} <= keys:
            reason = "HDF5 layout %s is not supported by the shared loader" % sorted(keys)
    if reason is None:
        return load_via_shared_loader(path, **kwargs)

    if path.suffix.lower() == ".mat":
        data = io.load_leaf_mat(path, **kwargs)
    else:
        data = io.load_rdm_hdf5_frame(path, **kwargs)
    data.metadata["loader"] = "local_fallback"
    data.metadata["shared_loader_gap"] = reason
    return data


__all__ = ["VENDOR_DIR", "from_upstream", "load_via_shared_loader", "load_scan"]

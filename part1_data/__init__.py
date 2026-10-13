"""Part 1 — LFI data import & management.

Prefer importing from this package (repo root on PYTHONPATH)::

    from part1_data import load_single_scan, load_volume, get_phase_radians

Or run tests/scripts from inside ``part1_data/`` with flat imports.
"""

from .output_interface import (  # noqa: F401
    PhaseNotConvertedError,
    SCHEMA_VERSION,
    batch_export,
    export_scan,
    get_phase_radians,
    iter_scans,
    load_exported,
    load_single_scan,
    load_volume,
    scan_to_record,
    write_metadata_hdf5_attrs,
    write_metadata_sidecar,
)

__all__ = [
    "SCHEMA_VERSION",
    "PhaseNotConvertedError",
    "batch_export",
    "export_scan",
    "get_phase_radians",
    "iter_scans",
    "load_exported",
    "load_single_scan",
    "load_volume",
    "scan_to_record",
    "write_metadata_hdf5_attrs",
    "write_metadata_sidecar",
]

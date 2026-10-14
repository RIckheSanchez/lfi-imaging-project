"""Read-only HDF5 image-map audit; never reads raw waveform datasets."""
import json
import re
from pathlib import Path
import h5py
import numpy as np


def image_pairs(handle):
    """Discover explicit matched channels in the two observed UQ schemas."""
    for amp, phase in (("data/IMAGE_AMPLITUDE", "data/IMAGE_PHASE"), ("amplitude", "phase")):
        if amp in handle and isinstance(handle[amp], h5py.Group):
            if phase not in handle or not isinstance(handle[phase], h5py.Group):
                raise ValueError("phase group is missing")
            a = {k for k in handle[amp] if not k.endswith("_BMP")}
            p = {k for k in handle[phase] if not k.endswith("_BMP")}
            if a != p:
                raise ValueError(f"unpaired channels: amplitude-only={sorted(a-p)}, phase-only={sorted(p-a)}")
            if not a:
                raise ValueError("no image pairs")
            def natural(k):
                return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", k)]
            return [(k, amp+"/"+k, phase+"/"+k) for k in sorted(a, key=natural)]
    raise ValueError("unsupported HDF5 schema")


def array_stats(dataset):
    if dataset.ndim != 2 or not np.issubdtype(dataset.dtype, np.number) or np.issubdtype(dataset.dtype, np.complexfloating):
        raise ValueError("image dataset must be a real numeric 2D array")
    a = dataset[...]
    finite = np.isfinite(a)
    values = a[finite]
    result = {"shape": list(a.shape), "dtype": str(a.dtype),
              "finite_fraction": float(finite.mean()) if a.size else 0,
              "minimum": float(values.min()) if values.size else None,
              "maximum": float(values.max()) if values.size else None,
              "zero_fraction": float(np.mean(a == 0)) if a.size else 0}
    if np.issubdtype(a.dtype, np.integer) and a.size:
        info = np.iinfo(a.dtype)
        result["dtype_min_fraction"] = float(np.mean(a == info.min))
        result["dtype_max_fraction"] = float(np.mean(a == info.max))
    return result


def audit_file(path):
    result = {"path": str(path), "pairs": [], "errors": [], "warnings": [],
              "calibration_status": "not verified by audit", "sample_identity": "not inferred from filename"}
    try:
        with h5py.File(path, "r") as f:
            pairs = image_pairs(f)
            for key, ak, pk in pairs:
                try:
                    a, p = array_stats(f[ak]), array_stats(f[pk])
                    if a["shape"] != p["shape"]:
                        raise ValueError("amplitude/phase shapes differ")
                    result["pairs"].append({"key": key, "amplitude": a, "phase": p})
                    for channel, stats in (("amplitude", a), ("phase", p)):
                        if stats["finite_fraction"] < 1:
                            result["warnings"].append(f"{key}/{channel}: missing or nonfinite pixels")
                        if stats["minimum"] == stats["maximum"]:
                            result["warnings"].append(f"{key}/{channel}: constant or empty image")
                        if stats.get("dtype_max_fraction", 0) > 0:
                            result["warnings"].append(f"{key}/{channel}: values reach storage maximum; not proof of saturation")
                except (ValueError, KeyError, OSError, TypeError) as exc:
                    result["errors"].append(f"{key}: {exc}")
            numeric = [int(k) for k, _, _ in pairs if k.isdecimal()]
            if len(numeric) == len(pairs):
                result["frame_id_gaps"] = [[a, b] for a, b in zip(numeric, numeric[1:]) if b-a > 1]
                if result["frame_id_gaps"]:
                    result["warnings"].append("frame-ID gaps; missing acquisition is not established without a manifest")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result["errors"].append(str(exc))
    result["status"] = "error" if result["errors"] else "review" if result["warnings"] else "readable"
    return result


def audit_directory(directory, output):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("input directory does not exist")
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in {".h5", ".hdf5"})
    if not paths:
        raise ValueError("no HDF5 files found")
    report = {"scope": "schema and raw numeric checks only; not scientific validation", "files": [audit_file(p) for p in paths]}
    report["summary"] = {"files": len(paths), "pairs": sum(len(f["pairs"]) for f in report["files"]),
                         "errors": sum(f["status"] == "error" for f in report["files"]),
                         "review": sum(f["status"] == "review" for f in report["files"])}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report

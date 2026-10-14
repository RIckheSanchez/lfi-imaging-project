"""Manifest-driven sequence assembly. Physical calibration is supplied, never guessed."""
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
from .data import ImagingData, DataLoader
from .volume_reconstructor import VolumeReconstructor


def load_sequence(manifest_path, *, max_voxels=50_000_000):
    """Validate the whole sequence before loading payloads.

    Each frame selects explicit same-key amplitude and phase datasets in an
    HDF5 file. One frame at a time is read into preallocated zyx arrays. Existing
    ImagingData/VolumeReconstructor copies still add memory overhead.
    """
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    for key in ("sample_id", "calibration_source"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"{key} is required")
    spacing = np.asarray(manifest.get("spacing_xy"), dtype=float)
    origin = np.asarray(manifest.get("origin_xy", [0, 0]), dtype=float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_xy requires positive finite dx, dy")
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("origin_xy requires two finite coordinates")
    if manifest.get("spatial_unit") not in {"mm", "micron", "meter"}:
        raise ValueError("spatial_unit must be mm, micron or meter")
    order = manifest.get("axis_order")
    if order not in {"yx", "xy"}:
        raise ValueError("axis_order must be yx or xy")
    phase_unit = manifest.get("phase_unit")
    phase_state = manifest.get("phase_state")
    if phase_unit not in {"rad", "degrees", "raw_counts"}:
        raise ValueError("phase_unit must be rad, degrees or raw_counts")
    if phase_state not in {"wrapped", "unwrapped", "unknown"}:
        raise ValueError("phase_state must be wrapped, unwrapped or unknown")
    scale = manifest.get("phase_scale_to_rad")
    offset = manifest.get("phase_offset_rad", 0.)
    if phase_unit == "raw_counts":
        if scale is None or not np.isfinite([scale, offset]).all() or scale == 0:
            raise ValueError("raw_counts require a confirmed phase_scale_to_rad and finite offset")
    elif scale is not None or offset != 0:
        raise ValueError("scale/offset are only valid for raw_counts")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("at least two explicit frames are required")
    records, seen, shape = [], set(), None
    for frame in frames:
        file_path = (path.parent / frame["path"]).resolve()
        ak, pk = frame["amplitude_key"], frame["phase_key"]
        if ak == pk or ak.rsplit("/", 1)[-1] != pk.rsplit("/", 1)[-1]:
            raise ValueError("select different channels with the same frame key")
        identity = (str(file_path), ak, pk)
        if identity in seen:
            raise ValueError("duplicate source frame")
        seen.add(identity)
        z = float(frame["z"])
        if not np.isfinite(z):
            raise ValueError("each frame requires a finite physical z")
        with h5py.File(file_path, "r") as f:
            a, p = f[ak], f[pk]
            for dataset in (a, p):
                if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2 or 0 in dataset.shape:
                    raise ValueError("selected channels must be nonempty 2D datasets")
                if not np.issubdtype(dataset.dtype, np.number) or np.iscomplexobj(np.empty(0, dtype=dataset.dtype)):
                    raise ValueError("selected channels must contain real numeric data")
            if a.shape != p.shape or (shape is not None and shape != a.shape):
                raise ValueError("all channel and slice shapes must match")
            shape = a.shape
        records.append((z, file_path, ak, pk))
    records.sort(key=lambda x: x[0])
    depths = [r[0] for r in records]
    if np.any(np.diff(depths) <= 0):
        raise ValueError("duplicate z coordinates")
    ny, nx = shape if order == "yx" else shape[::-1]
    if len(records)*ny*nx > max_voxels:
        raise ValueError("input exceeds max_voxels before volume allocation")
    a = np.empty((len(records), ny, nx), dtype=np.float32)
    p = np.empty_like(a)
    provenance = []
    for i, (z, file_path, ak, pk) in enumerate(records):
        data = DataLoader.load(file_path, axis_order=order, amplitude_key=ak, phase_key=pk,
                               metadata={"phase_unit": "raw_counts"})
        calibrated = data.phase.astype(np.float64)
        if phase_unit == "degrees": calibrated = np.deg2rad(calibrated)
        elif phase_unit == "raw_counts": calibrated = calibrated * scale + offset
        a[i], p[i] = data.amplitude, calibrated
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(a[i]).tobytes())
        digest.update(np.ascontiguousarray(p[i]).tobytes())
        provenance.append({"path": str(file_path), "amplitude_key": ak, "phase_key": pk,
                           "z": z, "loaded_float32_pair_sha256": digest.hexdigest()})
    meta = {"phase_unit": "rad", "phase_state": phase_state,
            "sample_id": manifest["sample_id"], "calibration_source": manifest["calibration_source"],
            "manifest": manifest, "sorted_sources": provenance}
    return VolumeReconstructor(ImagingData(a, p, meta), spacing_xy=spacing,
                               z_coordinates=depths, origin_xy=origin,
                               spatial_unit=manifest["spatial_unit"])


def reconstruct_manifest(manifest_path, output, *, spacing=None, max_voxels=50_000_000):
    """Assemble, optionally resample, export both channels, and verify readback."""
    import nibabel as nib
    started = time.perf_counter()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("choose a new or empty output directory to preserve earlier runs")
    volume = load_sequence(manifest_path, max_voxels=max_voxels)
    if spacing is not None:
        volume = volume.resample_isotropic(spacing, max_voxels=max_voxels)
    if not np.allclose(np.diff(volume.z), np.diff(volume.z)[0], rtol=1e-5, atol=1e-8):
        raise ValueError("irregular z requires an explicit output spacing")
    output.mkdir(parents=True, exist_ok=True)
    from .image_generator import ImageGenerator
    for channel in ("amplitude", "phase"):
        dest = volume.export_nifti(output / (channel + ".nii.gz"), channel)
        loaded = nib.load(dest)
        np.testing.assert_allclose(loaded.get_fdata().transpose(2, 1, 0), getattr(volume.data, channel), equal_nan=True)
    ImageGenerator(volume.data.slice(len(volume.z)//2)).generate_comparison(
        phase_mode="sequential", export=output / "middle_xy.png")
    report = {"sample_id": volume.data.metadata["sample_id"], "shape_zyx": list(volume.data.amplitude.shape),
              "spacing_xyz": [volume.dx, volume.dy, float(np.diff(volume.z)[0])],
              "spatial_unit": volume.spatial_unit, "readback_verified": True,
              "elapsed_seconds": time.perf_counter()-started,
              "scope": "computational roundtrip; not independent scientific ground truth"}
    (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report

"""Explicit dual-channel input contract; internal axis order is yx or zyx."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class ImagingData:
    amplitude: np.ndarray
    phase: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for key in ("amplitude", "phase"):
            raw = np.asarray(getattr(self, key))
            if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
                raise ValueError(f"{key} must contain real numeric values")
            arr = np.array(raw, dtype=np.float32, copy=True)
            if arr.ndim not in (2, 3) or 0 in arr.shape or not np.isfinite(arr).any():
                raise ValueError(f"{key} must be a nonempty 2D/3D array with finite data")
            arr[~np.isfinite(arr)] = np.nan
            setattr(self, key, arr)
        if self.amplitude.shape != self.phase.shape:
            raise ValueError("amplitude and phase shapes must match")
        self.metadata = dict(self.metadata)
        if self.metadata.get("phase_unit", "rad") not in {"rad", "raw_counts"}:
            raise ValueError("phase_unit must be rad or raw_counts")
        self.metadata.setdefault("phase_unit", "rad")
        self.metadata.setdefault("phase_state", "unknown")

    def slice(self, index):
        if self.amplitude.ndim != 3:
            raise ValueError("slice requires a volume")
        return ImagingData(self.amplitude[index], self.phase[index], self.metadata)


class DataLoader:
    @classmethod
    def load_grouped_frame(cls, path, frame_id, *, axis_order, phase_unit, metadata=None):
        """Load matching amplitude/phase frame keys from the legacy tomography schema.

        Unit and orientation are explicit; old processing examples suggest degrees
        for some files, but that assumption must not be applied to all acquisitions.
        """
        import h5py
        from .audit import image_pairs
        if phase_unit not in {"rad", "degrees", "raw_counts"}:
            raise ValueError("phase_unit must be rad, degrees or raw_counts")
        with h5py.File(path, "r") as f:
            pairs = {k: (a, p) for k, a, p in image_pairs(f)}
            ak, pk = pairs[str(frame_id)]
            a, p = f[ak][...], f[pk][...]
        if phase_unit == "degrees":
            p = np.deg2rad(p.astype(np.float64))
        meta = dict(metadata or {})
        meta.update(source=str(path), frame_id=str(frame_id), input_phase_unit=phase_unit,
                    phase_unit="raw_counts" if phase_unit == "raw_counts" else "rad")
        return cls.from_arrays(a, p, axis_order=axis_order, metadata=meta)

    @staticmethod
    def from_arrays(amplitude, phase, *, axis_order, metadata=None):
        a, p = np.asarray(amplitude), np.asarray(phase)
        expected = "yx" if a.ndim == 2 else "zyx"
        if len(axis_order) != a.ndim or sorted(axis_order) != sorted(expected):
            raise ValueError("axis_order must explicitly describe yx/xy or a permutation of zyx")
        axes = [axis_order.index(c) for c in expected]
        return ImagingData(a.transpose(axes), p.transpose(axes), metadata or {})

    @classmethod
    def load(cls, path, *, axis_order, amplitude_key="amplitude", phase_key="phase",
             phase_path=None, metadata=None):
        """Load NPZ, paired NPY or explicitly selected HDF5 datasets.

        No pickle loading, automatic axis guessing, phase unwrapping or inferred
        depth coordinates. HDF5 schemas can be inspected before selecting keys.
        """
        path = Path(path)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as f:
                a, p = f[amplitude_key], f[phase_key]
        elif path.suffix.lower() == ".npy":
            if phase_path is None:
                raise ValueError("paired NPY input requires phase_path")
            a, p = np.load(path, allow_pickle=False), np.load(phase_path, allow_pickle=False)
        elif path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(path, "r") as f:
                a, p = f[amplitude_key][...], f[phase_key][...]
        else:
            raise ValueError("supported inputs: .npz, .npy, .h5, .hdf5")
        meta = dict(metadata or {})
        meta.setdefault("source", str(path))
        return cls.from_arrays(a, p, axis_order=axis_order, metadata=meta)

    @staticmethod
    def inspect_hdf5(path):
        """Return dataset paths/shapes/dtypes without reading image payloads."""
        import h5py
        result = []
        with h5py.File(path, "r") as f:
            def visit(name, item):
                if isinstance(item, h5py.Dataset):
                    result.append({"path": name, "shape": item.shape, "dtype": str(item.dtype)})
            f.visititems(visit)
        return result

    @classmethod
    def load_uq_frame(cls, path, *, side="left", phase_scale=None, phase_offset=0., metadata=None):
        """Observed 2026 Vegetables schema, reading image maps only (not waveforms).

        Integer phase is NOT assumed to be radians. With a confirmed calibration,
        radians = stored_value * phase_scale + phase_offset. Left/right scans are
        kept separate; orientation and preprocessing remain upstream concerns.
        """
        import h5py
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        with h5py.File(path, "r") as f:
            group = f["data/IMAGE_AMPLITUDE"]
            names = [k for k in group if k.endswith("_" + side) and "_qcl_1_" in k]
            if len(names) != 1:
                raise ValueError("expected exactly one qcl_1 image per side")
            key = names[0]
            a = group[key][...]
            p = f["data/IMAGE_PHASE/" + key][...].astype(np.float64)
        meta = dict(metadata or {})
        meta.update(source=str(path), acquisition_side=side, source_dataset=key,
                    phase_unit="raw_counts", phase_state="unknown", preprocessing="acquisition image maps; no Person 2 preprocessing applied")
        if phase_scale is not None:
            if not np.isfinite([phase_scale, phase_offset]).all() or phase_scale == 0:
                raise ValueError("phase calibration must be finite with nonzero scale")
            p = p * phase_scale + phase_offset
            meta.update(phase_unit="rad", phase_scale=phase_scale, phase_offset=phase_offset)
        return cls.from_arrays(a, p, axis_order="yx", metadata=meta)

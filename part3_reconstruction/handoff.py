"""Reader for the Person 2 pre-processing handoff (v0.6.1 NPZ files).

Each frame file stores arrays as (n, H, W). Leaf frames are stored long axis
first and are transposed here so that every image is (y, x) with x the long
axis. Phase is taken from ``phase_corrected`` only when the frame says
``phase_correction_available``; otherwise the wrapped ``raw_phase`` is used and
labelled as wrapped. No lateral pixel size is recorded in the handoff, so
images stay in pixel units and only the relative pixel aspect is attached.
"""
import json
from pathlib import Path
import numpy as np
from .data import ImagingData

AMPLITUDE_CHANNELS = ("amplitude_corrected", "amplitude_detail", "raw_amplitude")
# Width / height of the full physical frame, from the handoff README and metadata.
FRAME_ASPECT = {"leaf": 1.5, "vegetable": 1.9651163, "skin": 1.9651163}


def _kind(path):
    name = Path(path).name.lower()
    for kind in FRAME_ASPECT:
        if name.startswith(kind):
            return kind
    raise ValueError(f"cannot tell Leaf/Vegetable/Skin from file name {Path(path).name!r}")


def pixel_aspect(shape_yx, frame_aspect):
    """Physical dy/dx of one pixel, from the full-frame width/height ratio."""
    ny, nx = shape_yx
    return nx / (ny * float(frame_aspect))


def load_frame(path, *, amplitude_channel="amplitude_corrected"):
    """Load one handoff frame as 2D ImagingData in (y, x) order."""
    if amplitude_channel not in AMPLITUDE_CHANNELS:
        raise ValueError(f"amplitude_channel must be one of {AMPLITUDE_CHANNELS}")
    path = Path(path)
    kind = _kind(path)
    with np.load(path, allow_pickle=False) as f:
        missing = {amplitude_channel, "raw_phase", "phase_correction_available",
                   "signal_mask", "z_indices"} - set(f.files)
        if missing:
            raise ValueError(f"{path.name} lacks fields: {sorted(missing)}")
        if f[amplitude_channel].shape[0] != 1:
            raise ValueError("expected one frame per file")
        amplitude = f[amplitude_channel][0]
        available = bool(f["phase_correction_available"][0])
        phase = f["phase_corrected"][0] if available else f["raw_phase"][0]
        mask = f["signal_mask"][0]
        frame_id = int(f["z_indices"][0])
    if kind == "leaf":
        amplitude, phase, mask = amplitude.T, phase.T, mask.T
    sidecar = path.with_name(path.stem + "_metadata.json")
    frame_aspect = FRAME_ASPECT[kind]
    if sidecar.exists():
        frame_aspect = json.loads(sidecar.read_text(encoding="utf-8"))["metadata"].get(
            "display_aspect_ratio", frame_aspect)
    meta = {
        "source": str(path), "sample": kind, "frame_id": frame_id,
        "amplitude_channel": amplitude_channel,
        "phase_unit": "rad",
        "phase_state": "unwrapped" if available else "wrapped",
        "phase_channel": "phase_corrected" if available else "raw_phase",
        "phase_note": ("relative unwrapped phase, NaN outside signal_mask" if available
                       else "wrapped raw phase in [-pi, pi); unwrapping withheld upstream"),
        "signal_fraction": float(mask.mean()),
        "frame_aspect_ratio": float(frame_aspect),
        "pixel_aspect_yx": pixel_aspect(amplitude.shape, frame_aspect),
        "spatial_note": "pixel units; lateral pixel size not recorded in the handoff",
    }
    return ImagingData(amplitude, phase, meta)


def load_volume(path, *, amplitude_channel="amplitude_corrected"):
    """Load the stacked Vegetables file: (ImagingData zyx, z_mm, frame_ids, signal_mask)."""
    if amplitude_channel not in ("amplitude_corrected", "raw_amplitude"):
        raise ValueError("the volume file stores amplitude_corrected and raw_amplitude only")
    with np.load(Path(path), allow_pickle=False) as f:
        amplitude, phase = f[amplitude_channel], f["raw_phase"]
        z_mm, frame_ids, mask = f["z_mm"].astype(float), f["frame_ids"].astype(int), f["signal_mask"]
    if amplitude.ndim != 3 or len(z_mm) != amplitude.shape[0]:
        raise ValueError("expected (n, H, W) arrays with one z_mm per slice")
    meta = {"source": str(path), "sample": "vegetable", "amplitude_channel": amplitude_channel,
            "phase_unit": "rad", "phase_state": "wrapped", "phase_channel": "raw_phase",
            "frame_aspect_ratio": FRAME_ASPECT["vegetable"],
            "pixel_aspect_yx": pixel_aspect(amplitude.shape[1:], FRAME_ASPECT["vegetable"]),
            "z_mm": z_mm.tolist(), "frame_ids": frame_ids.tolist()}
    return ImagingData(amplitude, phase, meta), z_mm, frame_ids, mask


def resample_depth(values, z, spacing):
    """Linear interpolation along z only, onto a uniform grid starting at z[0].

    Used when lateral spacing is unknown: each (y, x) column is interpolated
    independently, so no lateral calibration is needed. Valid for amplitude;
    wrapped phase must not be interpolated. A grid point between two slices is
    NaN if either neighbour is NaN; a grid point on a slice copies that slice.
    """
    values = np.asarray(values, dtype=np.float32)
    z = np.asarray(z, dtype=float)
    if values.ndim != 3 or z.shape != (values.shape[0],):
        raise ValueError("values must be (z, y, x) with one z per slice")
    order = np.argsort(z)
    z, values = z[order], values[order]
    if np.any(np.diff(z) <= 0) or not np.isfinite(z).all():
        raise ValueError("z must be finite and unique")
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive")
    n = int(np.floor((z[-1] - z[0]) / spacing + 1e-9)) + 1
    grid = z[0] + np.arange(n) * spacing
    upper = np.clip(np.searchsorted(z, grid, side="right"), 1, len(z) - 1)
    lower = upper - 1
    weight = ((grid - z[lower]) / (z[upper] - z[lower])).astype(np.float32)
    out = np.empty((n, *values.shape[1:]), dtype=np.float32)
    for k in range(n):
        w = weight[k]
        if w == 0:
            out[k] = values[lower[k]]
        elif w == 1:
            out[k] = values[upper[k]]
        else:
            out[k] = values[lower[k]] * (1 - w) + values[upper[k]] * w
    return out, grid

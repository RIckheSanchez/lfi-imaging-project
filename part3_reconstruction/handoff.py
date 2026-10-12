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


MIRROR_KEYS = ("imageColumns", "imageRows", "columnStep", "columnStepFract256",
               "rowStepSize", "rowStepFract256")


def mirror_geometry(path):
    """Pixel shape in scan-mirror steps, from a raw UQ HDF5 frame (thz/info/mcp).

    The stored aspect_ratio_computed equals (imageColumns * columnStep) /
    (imageRows / 2 * rowStepSize): the recorded rows form a left and a right
    image. A mirror step has no recorded length, so this fixes the shape of a
    pixel, not its size. The fractional step settings are reported separately
    because the stored ratio ignores them.
    """
    import h5py
    with h5py.File(path, "r") as f:
        mcp = {k: int(f["thz/info/mcp/" + k][()]) for k in MIRROR_KEYS}
        recorded = float(f["thz/info/aspect_ratio_computed"][()])
    rows = mcp["imageRows"] / 2
    integer = mcp["imageColumns"] * mcp["columnStep"] / (rows * mcp["rowStepSize"])
    fractional = (mcp["imageColumns"] * (mcp["columnStep"] + mcp["columnStepFract256"] / 256)
                  / (rows * (mcp["rowStepSize"] + mcp["rowStepFract256"] / 256)))
    return {"file": Path(path).name, **mcp, "aspect_recorded": recorded,
            "aspect_from_integer_steps": integer, "aspect_with_fractional_steps": fractional,
            "pixel_aspect_yx": mcp["rowStepSize"] / mcp["columnStep"],
            "matches_recorded": bool(abs(integer - recorded) <= 1e-6 * recorded)}


def _axis_ellipse(x, y):
    """Least-squares axis-aligned ellipse A x^2 + B y^2 + C x + D y = 1 -> (xc, yc, ax, ay)."""
    m = np.column_stack([x * x, y * y, x, y])
    a, b, c, d = np.linalg.lstsq(m, np.ones_like(x), rcond=None)[0]
    xc, yc = -c / (2 * a), -d / (2 * b)
    k = 1 + a * xc ** 2 + b * yc ** 2
    return float(xc), float(yc), float(np.sqrt(k / a)), float(np.sqrt(k / b))


def aperture_aspect(image, *, thresholds=(0.5, 0.7, 1.0), margin=0.15, sigma=3.0, border=3, details=False):
    """Estimate the frame width/height ratio from a bright aperture that should be circular.

    For each threshold (a fraction of Otsu's level on the smoothed image) the
    largest bright region is taken, and the outermost left and right boundary
    pixel of every row is fitted with an axis-aligned ellipse in raw pixel
    coordinates. The ratio of its x and y semi-axes is the pixel dy/dx that
    would make the aperture round, which converts to a frame width/height
    ratio. Each fit is made with all rows and with the middle rows only (the
    aperture may be cut off at the top and bottom of the frame). The spread
    across variants shows how well the image constrains the ratio: this is a
    consistency check, not a calibration. Points are taken from the outermost
    boundary only, never selected near an assumed circle, which would bias the
    result towards round.
    """
    from scipy import ndimage
    from skimage import filters
    a = np.asarray(image, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("image must be a finite 2D array")
    ny, nx = a.shape
    smooth = filters.gaussian(a / np.percentile(a, 99), sigma=sigma)
    if np.ptp(smooth) == 0:
        raise ValueError("image has no contrast")
    level = filters.threshold_otsu(smooth)
    variants, regions = [], {}
    for fraction in thresholds:
        mask = ndimage.binary_opening(smooth > level * fraction, structure=np.ones((3, 9)))
        labels, count = ndimage.label(mask)
        if count == 0:
            continue
        region = ndimage.binary_fill_holes(labels == 1 + np.argmax(np.bincount(labels.ravel())[1:]))
        regions[fraction] = region
        rows = np.nonzero(region.any(1))[0]
        first = region[rows].argmax(1)
        last = nx - 1 - region[rows, ::-1].argmax(1)
        x = np.concatenate([first, last]).astype(float)
        y = np.concatenate([rows, rows]).astype(float)
        keep = (x >= border) & (x < nx - border) & (y >= border) & (y < ny - border)
        for label, lo, hi in (("all rows", border, ny - border), ("middle rows", margin * ny, (1 - margin) * ny)):
            sel = keep & (y >= lo) & (y < hi)
            if sel.sum() < 20:
                continue
            _, _, sx, sy = _axis_ellipse(x[sel], y[sel])
            variants.append({"threshold_fraction_of_otsu": fraction, "rows": label, "points": int(sel.sum()),
                             "semi_axes_px_xy": [sx, sy], "pixel_aspect_yx": sx / sy,
                             "frame_aspect": nx / (ny * sx / sy)})
    if not variants:
        raise ValueError("no aperture boundary found")
    estimates = [v["frame_aspect"] for v in variants]
    result = {"variants": variants, "frame_aspect_min": min(estimates), "frame_aspect_max": max(estimates),
              "frame_aspect_median": float(np.median(estimates))}
    return (result, regions) if details else result


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

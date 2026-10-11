"""GUI-facing helpers for Person 4 (Streamlit viewer). No Streamlit import here.

The viewer only needs NumPy arrays, PNG bytes and small info dictionaries, so
this layer is testable without a browser. Display arrays are for viewing only;
quantitative export stays with VolumeReconstructor / ImageGenerator.
"""
import io
import time
import numpy as np
from .data import ImagingData
from .colormap_engine import normalise, fold_phase, resolve_cmap
from .volume_reconstructor import VolumeReconstructor


def from_viewer_volume(obj, *, phase_unit="raw_counts", phase_state="unknown",
                       axis_order=None, metadata=None):
    """Convert the viewer's loaded object (``.amplitude``/``.phase`` arrays) to ImagingData.

    The viewer's ``z_indices`` are acquisition frame numbers, not physical
    depths, so they are kept as ``frame_indices`` metadata only. Phase units
    are never guessed: the caller states them (default: uncalibrated counts).
    """
    if not hasattr(obj, "amplitude") or not hasattr(obj, "phase"):
        raise TypeError("viewer data must provide .amplitude and .phase arrays")
    if phase_unit not in {"rad", "raw_counts", "degrees"}:
        raise ValueError("phase_unit must be rad, degrees or raw_counts")
    a, p = np.asarray(obj.amplitude), np.asarray(obj.phase)
    if phase_unit == "degrees":
        p = np.deg2rad(p.astype(np.float64))
    order = axis_order or ("yx" if a.ndim == 2 else "zyx")
    meta = dict(metadata or {})
    meta.update(phase_unit="raw_counts" if phase_unit == "raw_counts" else "rad",
                phase_state=phase_state, input_phase_unit=phase_unit)
    for attr, key in (("target_type", "target_type"), ("file_path", "source")):
        value = getattr(obj, attr, None)
        if value is not None:
            meta.setdefault(key, str(value))
    frames = getattr(obj, "z_indices", None)
    if frames is not None:
        meta["frame_indices"] = [int(v) for v in frames]
    axes = [order.index(c) for c in ("yx" if a.ndim == 2 else "zyx")]
    if len(order) != a.ndim or sorted(order) != sorted("yx" if a.ndim == 2 else "zyx"):
        raise ValueError("axis_order must be a permutation of yx (2D) or zyx (3D)")
    return ImagingData(a.transpose(axes), p.transpose(axes), meta)


def render_rgb(data, channel="amplitude", *, phase_mode="sequential", cmap=None,
               norm="percentile", limits=None, flip_y=False):
    """Return (uint8 RGBA HxWx4, info) for one 2D slice; missing pixels are transparent.

    Row 0 is y index 0. Streamlit's ``st.image`` draws row 0 at the top while the
    Matplotlib figures use ``origin='lower'``; pass ``flip_y=True`` to match them.
    """
    if not isinstance(data, ImagingData) or data.amplitude.ndim != 2:
        raise ValueError("render_rgb requires a 2D ImagingData slice")
    if channel not in {"amplitude", "phase"}:
        raise ValueError("channel must be amplitude or phase")
    if phase_mode not in {"cyclic", "sequential"}:
        raise ValueError("phase_mode must be cyclic or sequential")
    arr = getattr(data, channel)
    cyclic = channel == "phase" and phase_mode == "cyclic"
    if cyclic and data.metadata["phase_unit"] != "rad":
        raise ValueError("cyclic phase requires calibrated radians; use sequential for raw counts")
    if cyclic:
        values = (fold_phase(arr) + np.pi) / (2 * np.pi)
        lo, hi = -np.pi, np.pi
    else:
        values = normalise(arr, norm, limits=limits)
        valid = arr[np.isfinite(arr)]
        if limits is not None:
            lo, hi = limits
        elif norm == "percentile":
            lo, hi = np.percentile(valid, [2, 98])
        else:
            lo, hi = valid.min(), valid.max()
    name = cmap or ("phase" if cyclic else "inferno" if channel == "amplitude" else "viridis")
    rgba = resolve_cmap(name)(np.nan_to_num(values, nan=0.0), bytes=True)
    rgba[~np.isfinite(values), 3] = 0
    if flip_y:
        rgba = rgba[::-1]
    unit = ("rad (wrapped)" if cyclic else "rad" if channel == "phase" and data.metadata["phase_unit"] == "rad"
            else "uncalibrated counts" if channel == "phase" else "a.u.")
    info = {"channel": channel, "cmap": name, "phase_mode": phase_mode if channel == "phase" else None,
            "display_min": float(lo), "display_max": float(hi), "unit": unit,
            "missing_fraction": float(np.mean(~np.isfinite(arr))), "row0": "top" if flip_y else "y index 0"}
    return rgba, info


def orthogonal_views(data, index_zyx, channel="amplitude"):
    """Index-based XY/XZ/YZ planes through one voxel of a (z,y,x) volume."""
    if not isinstance(data, ImagingData) or data.amplitude.ndim != 3:
        raise ValueError("orthogonal_views requires a 3D ImagingData volume")
    arr = getattr(data, channel)
    k, j, i = (int(v) for v in index_zyx)
    if not (0 <= k < arr.shape[0] and 0 <= j < arr.shape[1] and 0 <= i < arr.shape[2]):
        raise ValueError("index outside the volume")
    return {"xy": arr[k].copy(), "xz": arr[:, j, :].copy(), "yz": arr[:, :, i].copy()}


def figure_to_png(fig, dpi=150):
    """Encode a Matplotlib Figure for st.image / st.download_button."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()


def to_reconstructed_image(volume, *, channel="amplitude", started=None, method="part3 rebuild"):
    """Fill the team's shared ReconstructedImage dataclass from a VolumeReconstructor.

    quality_score stays 0.0: without a trusted reference there is no meaningful
    score (see validation.image_metrics for reference-based PSNR/SSIM).
    """
    from shared.data_types import ReconstructedImage
    if not isinstance(volume, VolumeReconstructor):
        raise TypeError("volume must be a VolumeReconstructor")
    arr = getattr(volume.data, channel)
    return ReconstructedImage(
        image_2d=arr[len(volume.z) // 2].copy(),
        slices_3d=[s.copy() for s in arr],
        quality_score=0.0,
        resolution=(volume.dx, volume.dy),
        reconstruction_method=f"{method}; channel={channel}; unit={volume.spatial_unit}",
        processing_time=0.0 if started is None else time.perf_counter() - started,
        target_type=str(volume.data.metadata.get("target_type", "")),
        z_levels=[float(z) for z in volume.z],
    )

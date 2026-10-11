"""Slice-wise spatial correction that keeps measurement channels intact.

Two different products are derived, because amplitude and phase need
different treatment:

* **Amplitude.** Object contrast *is* the signal (absorbing material is
  dark), so the default product ``amplitude_corrected`` only removes
  scan-line offsets (and explicitly requested FFT notches). A separate
  ``amplitude_detail`` product additionally subtracts a Gaussian
  background; it is a local-contrast view, not a measurement.
* **Phase.** Unwrapped phase is dominated by a smooth optical-path
  background of tens to hundreds of radians, so ``phase_corrected``
  subtracts the Gaussian background (the operation that reproduces the
  group's saved Leaf ``U_M``) and then the scan-line offsets.

Scan-line offsets are estimated robustly: each pixel is compared with the
median of the neighbouring lines at the same position, and the line offset
is the median of those differences along the line. A real feature is only
mistaken for an offset if it is thinner than half the window *and* covers
more than half of a scan line.

Every removed component is stored, and no cross-frame coordinate,
reference ROI or physical pixel size is assumed.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from thz_preprocessing.io import LFIRawData
from thz_preprocessing.masking import signal_mask


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _notch_filter(image, centres, width):
    if not centres:
        return image.copy()
    width = _positive(width, "notch_width_cycles_per_pixel")
    fy, fx = np.meshgrid(np.fft.fftfreq(image.shape[0]),
                         np.fft.fftfreq(image.shape[1]), indexing="ij")
    transfer = np.ones(image.shape)
    for centre in centres:
        if len(centre) != 2:
            raise ValueError("Each notch must contain [row_frequency, column_frequency]")
        cy, cx = map(float, centre)
        if not np.isfinite([cy, cx]).all() or max(abs(cy), abs(cx)) >= 0.5:
            raise ValueError("Notch frequencies must be finite and inside (-0.5, 0.5)")
        if np.hypot(cy, cx) <= 3 * width:
            raise ValueError("Notches too close to DC could erase the spatial background")
        for sign in [1, -1]:
            dy = (fy - sign * cy + 0.5) % 1.0 - 0.5
            dx = (fx - sign * cx + 0.5) % 1.0 - 0.5
            transfer *= 1.0 - np.exp(-(dx * dx + dy * dy) / (2 * width * width))
    transfer[0, 0] = 1.0
    return np.fft.ifft2(np.fft.fft2(image) * transfer).real


def gaussian_background(image, sigma, mode="reflect", truncate=4.0, valid=None):
    """Gaussian low-pass; with ``valid`` it is a normalised convolution."""
    if valid is None:
        return gaussian_filter(image, sigma=sigma, mode=mode, truncate=truncate)
    weights = gaussian_filter(valid.astype(np.float64), sigma=sigma, mode=mode, truncate=truncate)
    numerator = gaussian_filter(np.where(valid, image, 0.0), sigma=sigma, mode=mode, truncate=truncate)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid, numerator / np.maximum(weights, 1e-12), np.nan)


def estimate_line_offsets(image, axis="rows", window=15, valid=None, min_valid=16):
    """Robust additive offset of each scan line relative to its neighbours.

    ``image`` must be finite; pixels where ``valid`` is false are ignored
    when the per-line median is taken. Lines with fewer than ``min_valid``
    usable pixels get a zero offset.
    """
    if axis not in {"rows", "columns"}:
        raise ValueError("axis must be 'rows' or 'columns'")
    window = int(window)
    if window < 3 or window % 2 == 0:
        raise ValueError("stripe_window_lines must be an odd integer >= 3")
    work = image if axis == "rows" else image.T
    ok = None if valid is None else (valid if axis == "rows" else valid.T)
    deviation = work - median_filter(work, size=(window, 1), mode="reflect")
    if ok is None:
        return np.median(deviation, axis=1)
    counts = ok.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN lines are handled below
        offsets = np.nanmedian(np.where(ok, deviation, np.nan), axis=1)
    return np.where(counts >= int(min_valid), np.nan_to_num(offsets), 0.0)


def correct_image(image: np.ndarray, params: dict[str, Any], mask: np.ndarray | None = None):
    """Decompose one 2D image into additive parts, all in input units.

    Returns ``corrected`` (background, line offsets and notches removed),
    ``background``, ``stripe`` and ``notch_removed`` with
    ``image == corrected + background + stripe + notch_removed``.
    With ``mask``, the background is a normalised convolution over the
    masked pixels and ``corrected``/``background`` are NaN outside it; the
    input may then be NaN outside the mask.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not image.size:
        raise ValueError("Spatial correction requires a non-empty 2D image")
    valid = None
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != image.shape:
            raise ValueError("mask must have the image shape")
        if valid.all():
            valid = None
    finite = np.isfinite(image)
    if (valid is None and not finite.all()) or (valid is not None and not finite[valid].all()):
        raise ValueError("Spatial correction requires a non-empty finite 2D image")
    sigma = _positive(params.get("background_sigma_pixels", 16), "background_sigma_pixels")
    axis = str(params.get("stripe_axis", "rows"))
    if axis not in {"rows", "columns", "none"}:
        raise ValueError("stripe_axis must be 'rows', 'columns' or 'none'")
    mode = str(params.get("background_boundary_mode", "reflect"))
    if mode not in {"reflect", "nearest", "mirror"}:
        raise ValueError("background_boundary_mode must be reflect, nearest or mirror")
    truncate = _positive(params.get("background_truncate", 4), "background_truncate")

    background = gaussian_background(image, sigma, mode, truncate, valid)
    residual = image - background if valid is None else np.where(valid, image - background, 0.0)
    stripe = np.zeros(image.shape)
    if axis != "none":
        offsets = estimate_line_offsets(residual, axis, params.get("stripe_window_lines", 15),
                                        valid, params.get("stripe_min_valid", 16))
        stripe[:] = offsets[:, None] if axis == "rows" else offsets[None, :]
    destriped = residual - stripe
    if valid is not None:
        destriped = np.where(valid, destriped, 0.0)
    corrected = _notch_filter(destriped, params.get("fft_notches", []),
                              params.get("notch_width_cycles_per_pixel", 0.005))
    notch_removed = destriped - corrected
    if valid is not None:
        corrected = np.where(valid, corrected, np.nan)
    return {"corrected": corrected, "background": background,
            "stripe": stripe, "notch_removed": notch_removed}


def apply(data: LFIRawData, params: dict[str, Any]) -> LFIRawData:
    metadata = deepcopy(data.metadata)
    metadata.setdefault("pipeline", []).append({"spatial_correction": dict(params)})
    enabled = bool(params.get("enabled", False))
    mask_method = str(params.get("mask", "none")).lower()
    products = {}
    diagnostics = []
    if enabled:
        if data.amplitude.shape != data.phase.shape or data.amplitude.ndim != 3:
            raise ValueError("Expected matching (frame, row, column) measurement channels")
        shape = data.shape
        for name in ["corrected", "detail", "background", "stripe", "notch_removed"]:
            products[f"amplitude_{name}"] = np.empty(shape, dtype=np.float64)
        for name in ["corrected", "background", "stripe", "notch_removed", "background_corrected"]:
            products[f"phase_{name}"] = np.full(shape, np.nan)
        products["amplitude_pre_spatial"] = data.amplitude.copy()
        products["phase_pre_spatial"] = data.phase.copy()
        products["signal_mask"] = np.ones(shape, dtype=bool)
        phase_status = metadata.get("phase_unwrapping_status", {})
        per_slice = phase_status.get("quality_diagnostics", [])
        available = np.zeros(data.n_z, dtype=bool)
        sigma = _positive(params.get("background_sigma_pixels", 16), "background_sigma_pixels")
        mode = str(params.get("background_boundary_mode", "reflect"))
        truncate = _positive(params.get("background_truncate", 4), "background_truncate")
        for i in range(data.n_z):
            amplitude = data.amplitude[i]
            mask = signal_mask(amplitude, mask_method)
            valid = None if mask.all() else mask
            products["signal_mask"][i] = mask

            # Amplitude: line offsets and notches only; contrast is kept.
            parts = correct_image(amplitude, params)
            corrected = amplitude - parts["stripe"] - parts["notch_removed"]
            background = gaussian_background(corrected, sigma, mode, truncate, valid)
            products["amplitude_corrected"][i] = corrected
            products["amplitude_background"][i] = background
            products["amplitude_detail"][i] = corrected - background
            products["amplitude_stripe"][i] = parts["stripe"]
            products["amplitude_notch_removed"][i] = parts["notch_removed"]

            status = per_slice[i]["status"] if i < len(per_slice) else phase_status.get("status")
            available[i] = status in {"unwrapped", "already_continuous", "reference_unwrapped"}
            if available[i]:
                phase = data.phase[i]
                phase_valid = np.isfinite(phase) if valid is None else (valid & np.isfinite(phase))
                parts = correct_image(phase, params, None if phase_valid.all() else phase_valid)
                for name, values in parts.items():
                    products[f"phase_{name}"][i] = values
                products["phase_background_corrected"][i] = phase - parts["background"]
            diagnostics.append({
                "slice_index": i, "source_frame_index": data.z_indices[i],
                "phase_correction_available": bool(available[i]),
                "phase_unwrap_status": status,
                "signal_fraction": float(mask.mean()),
                "amplitude_removed_stripe_std": float(products["amplitude_stripe"][i].std()),
                "amplitude_removed_notch_std": float(products["amplitude_notch_removed"][i].std()),
                "amplitude_detail_background_std": float(np.nanstd(background)),
                "amplitude_decomposition_max_error": float(np.abs(
                    amplitude - corrected - products["amplitude_stripe"][i]
                    - products["amplitude_notch_removed"][i]).max()),
            })
        products["phase_correction_available"] = available
        metadata["spatial_products"] = products
    metadata["spatial_correction_status"] = {
        "enabled": enabled, "status": "exploratory_relative_images" if enabled else "disabled",
        "measurement_channels_preserved": True,
        "amplitude_corrected": "line offsets (and requested notches) removed; object contrast kept",
        "amplitude_detail": "amplitude_corrected minus Gaussian background; relative local-contrast view",
        "phase_corrected": "unwrapped phase minus Gaussian background and line offsets; relative",
        "mask": mask_method,
        "units": "same as pre-spatial input; detail and phase_corrected are relative residuals",
        "parameter_status": "pixel-scale exploratory parameters, not calibrated physical values",
        "limitations": "A Gaussian background also contains real broad sample structure; no clinical validation. Wrapped or withheld phase is not filtered.",
        "diagnostics": diagnostics,
    }
    return LFIRawData(data.amplitude.copy(), data.phase.copy(), list(data.z_indices),
                      data.target_type, metadata)

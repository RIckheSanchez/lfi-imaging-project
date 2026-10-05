"""Slice-wise spatial background and stripe correction for derived images.

The returned LFIRawData measurement channels remain unchanged. Derived
residual images and every removed component are stored separately. No
cross-frame coordinate, reference ROI or physical pixel size is assumed.
"""

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from thz_preprocessing.io import LFIRawData


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


def correct_image(image: np.ndarray, params: dict[str, Any]):
    """Return a relative image and additive components, all in input units."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ValueError("Spatial correction requires a non-empty finite 2D image")
    sigma = _positive(params.get("background_sigma_pixels", 16), "background_sigma_pixels")
    axis = str(params.get("stripe_axis", "rows"))
    if axis not in {"rows", "columns", "none"}:
        raise ValueError("stripe_axis must be 'rows', 'columns' or 'none'")
    mode = str(params.get("background_boundary_mode", "reflect"))
    if mode not in {"reflect", "nearest", "mirror"}:
        raise ValueError("background_boundary_mode must be reflect, nearest or mirror")
    truncate = _positive(params.get("background_truncate", 4), "background_truncate")
    background = gaussian_filter(image, sigma=sigma, mode=mode, truncate=truncate)
    residual = image - background
    stripe = np.zeros_like(image)
    if axis != "none":
        smoothing = _positive(params.get("stripe_profile_sigma_pixels", 8),
                              "stripe_profile_sigma_pixels")
        profile = np.median(residual, axis=1 if axis == "rows" else 0)
        offsets = profile - gaussian_filter1d(profile, sigma=smoothing, mode="reflect")
        offsets -= offsets.mean()
        stripe[:] = offsets[:, None] if axis == "rows" else offsets[None, :]
    destriped = residual - stripe
    corrected = _notch_filter(destriped, params.get("fft_notches", []),
                              params.get("notch_width_cycles_per_pixel", 0.005))
    notch_removed = destriped - corrected
    return {"corrected": corrected, "background": background,
            "stripe": stripe, "notch_removed": notch_removed}


def apply(data: LFIRawData, params: dict[str, Any]) -> LFIRawData:
    metadata = deepcopy(data.metadata)
    metadata.setdefault("pipeline", []).append({"spatial_correction": dict(params)})
    enabled = bool(params.get("enabled", False))
    products = {}
    diagnostics = []
    if enabled:
        if data.amplitude.shape != data.phase.shape or data.amplitude.ndim != 3:
            raise ValueError("Expected matching (frame, row, column) measurement channels")
        for suffix in ["corrected", "background", "stripe", "notch_removed"]:
            products[f"amplitude_{suffix}"] = np.empty(data.shape, dtype=np.float64)
            products[f"phase_{suffix}"] = np.full(data.shape, np.nan)
        products["amplitude_pre_spatial"] = data.amplitude.copy()
        products["phase_pre_spatial"] = data.phase.copy()
        products["phase_background_corrected"] = np.full(data.shape, np.nan)
        phase_status = metadata.get("phase_unwrapping_status", {})
        per_slice = phase_status.get("quality_diagnostics", [])
        available = np.zeros(data.n_z, dtype=bool)
        for i in range(data.n_z):
            amp = correct_image(data.amplitude[i], params)
            for suffix, values in amp.items():
                products[f"amplitude_{suffix}"][i] = values
            status = per_slice[i]["status"] if i < len(per_slice) else phase_status.get("status")
            available[i] = status in {"unwrapped", "already_continuous", "reference_unwrapped"}
            if available[i]:
                phase = correct_image(data.phase[i], params)
                for suffix, values in phase.items():
                    products[f"phase_{suffix}"][i] = values
                products["phase_background_corrected"][i] = data.phase[i] - phase["background"]
            diagnostics.append({
                "slice_index": i, "source_frame_index": data.z_indices[i],
                "phase_correction_available": bool(available[i]),
                "phase_unwrap_status": status,
                "amplitude_removed_background_std": float(amp["background"].std()),
                "amplitude_removed_stripe_std": float(amp["stripe"].std()),
                "amplitude_removed_notch_std": float(amp["notch_removed"].std()),
                "amplitude_decomposition_max_error": float(np.abs(
                    data.amplitude[i] - sum(amp.values())).max()),
            })
        products["phase_correction_available"] = available
        metadata["spatial_products"] = products
    metadata["spatial_correction_status"] = {
        "enabled": enabled, "status": "exploratory_relative_images" if enabled else "disabled",
        "measurement_channels_preserved": True,
        "units": "same as pre-spatial input; corrected channels are relative residuals",
        "parameter_status": "pixel-scale exploratory parameters, not calibrated physical values",
        "limitations": "Estimated background and row offsets may include real sample structure; no clinical validation. Wrapped or exploratory phase is not linearly background-filtered.",
        "diagnostics": diagnostics,
    }
    return LFIRawData(data.amplitude.copy(), data.phase.copy(), list(data.z_indices),
                      data.target_type, metadata)

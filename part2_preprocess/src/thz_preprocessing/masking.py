"""Signal-region masks derived from amplitude.

Phase is only meaningful where enough THz power returns to the laser. In
regions without signal (outside a sample, or inside strongly absorbing
tissue) the wrapped phase is noise and dominates the residue count, so
phase products are evaluated and released inside a signal mask only.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

MASK_METHODS = {"none", "amplitude_otsu"}


def signal_mask(amplitude: np.ndarray, method: str = "amplitude_otsu",
                smooth_sigma_pixels: float = 2.0) -> np.ndarray:
    """Boolean mask of pixels with usable signal in one 2D amplitude image.

    ``"amplitude_otsu"`` thresholds the lightly smoothed amplitude with
    Otsu's method. It is a data-driven split into bright and dark pixels,
    not a calibrated signal-to-noise criterion. ``"none"`` keeps every pixel.
    """
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim != 2:
        raise ValueError(f"Expected a 2D amplitude image, got shape {amplitude.shape}")
    method = str(method).lower()
    if method not in MASK_METHODS:
        raise ValueError(f"mask must be one of {sorted(MASK_METHODS)}, got {method!r}")
    if method == "none":
        return np.ones(amplitude.shape, dtype=bool)
    if not np.isfinite(amplitude).all():
        raise ValueError("Amplitude must be finite to derive a signal mask")
    smooth = ndimage.gaussian_filter(amplitude, float(smooth_sigma_pixels), mode="nearest")
    if np.ptp(smooth) == 0:
        return np.ones(amplitude.shape, dtype=bool)
    from skimage.filters import threshold_otsu

    return smooth > threshold_otsu(smooth)


__all__ = ["MASK_METHODS", "signal_mask"]

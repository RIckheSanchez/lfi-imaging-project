"""Noise reduction stage.

Suppresses wideband acquisition noise and low-frequency drift while
preserving the coherent self-mixing features that encode amplitude
and phase.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def apply(signal: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply the configured noise-reduction method.

    Parameters
    ----------
    signal : np.ndarray
        Input raw self-mixing waveform.
    params : dict
        Configuration sub-dict for noise reduction. Expected keys:
            method : {"bandpass", "savgol", "wavelet"}
            ... method-specific parameters ...

    Returns
    -------
    np.ndarray
        Denoised signal, same shape as input.
    """
    method = params.get("method", "bandpass")

    # TODO: dispatch on method and implement
    if method == "bandpass":
        raise NotImplementedError("bandpass filtering not yet implemented")
    elif method == "savgol":
        raise NotImplementedError("Savitzky-Golay smoothing not yet implemented")
    elif method == "wavelet":
        raise NotImplementedError("wavelet denoising not yet implemented")
    else:
        raise ValueError(f"Unknown noise-reduction method: {method}")

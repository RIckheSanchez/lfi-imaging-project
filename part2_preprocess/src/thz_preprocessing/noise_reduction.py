"""Noise reduction stage.

Operates on :attr:`LFIRawData.amplitude`. The wrapped-phase channel is
left untouched here on purpose: directly low-pass filtering wrapped
phase produces meaningless results across the [-pi, pi) discontinuity.
Phase smoothing, when needed, happens *after* unwrapping in the
`phase_unwrapping` stage.

Dispatch table:

* ``"median2d"``    — :func:`scipy.ndimage.median_filter` per z-slice.
* ``"gaussian2d"``  — :func:`scipy.ndimage.gaussian_filter` per z-slice.
* ``"savgol_1d"``   — :func:`scipy.signal.savgol_filter` along z
  (the sweep axis). Requires ``n_z >= savgol_window``.
* ``"bandpass_1d"`` — IIR Butterworth + ``filtfilt`` along z.
  Requires ``n_z`` ≳ 3 * order.
* ``"none"``        — pass-through.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy import ndimage, signal

from thz_preprocessing.io import LFIRawData


def _apply_per_zslice(arr: np.ndarray, fn) -> np.ndarray:
    """Apply ``fn`` independently to each ``(H, W)`` slice along z."""
    return np.stack([fn(arr[z]) for z in range(arr.shape[0])], axis=0)


def _median2d(arr: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError(f"median_window must be odd and >= 1, got {window}")
    return _apply_per_zslice(
        arr, lambda s: ndimage.median_filter(s, size=window, mode="reflect")
    )


def _gaussian2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError(f"gaussian_sigma must be > 0, got {sigma}")
    return _apply_per_zslice(
        arr, lambda s: ndimage.gaussian_filter(s, sigma=sigma, mode="reflect")
    )


def _savgol_1d(arr: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if arr.shape[0] < window:
        # Gracefully degrade: too few z-samples (e.g. Skin n_z==1) → skip.
        return arr.copy()
    if window % 2 == 0 or window < 3:
        raise ValueError(f"savgol_window must be odd and >= 3, got {window}")
    if polyorder >= window:
        raise ValueError(
            f"savgol_polyorder ({polyorder}) must be < savgol_window ({window})"
        )
    return signal.savgol_filter(arr, window_length=window, polyorder=polyorder, axis=0)


def _bandpass_1d(
    arr: np.ndarray,
    low_norm: float,
    high_norm: float,
    order: int,
) -> np.ndarray:
    if not (0.0 < low_norm < high_norm < 1.0):
        raise ValueError(
            f"bandpass cutoffs must satisfy 0 < low < high < 1, "
            f"got low={low_norm}, high={high_norm}"
        )
    n_z = arr.shape[0]
    # filtfilt's default padlen is 3*max(len(a), len(b)); too few z-samples
    # produces a ValueError. Gracefully degrade for short sweeps / Skin.
    sos = signal.butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
    min_required = 3 * (2 * order + 1)
    if n_z < min_required:
        return arr.copy()
    return signal.sosfiltfilt(sos, arr, axis=0)


def apply(data: LFIRawData, params: dict[str, Any]) -> LFIRawData:
    """Apply the configured noise-reduction method to ``data.amplitude``.

    Parameters
    ----------
    data
        Input scan. Not mutated.
    params
        Sub-dict from the resolved config under ``"noise_reduction"``.

    Returns
    -------
    LFIRawData
        New :class:`LFIRawData` with denoised amplitude. ``phase``,
        ``z_indices`` and ``target_type`` are passed through unchanged.
        A ``noise_reduction`` entry is appended to ``metadata`` for
        provenance.
    """
    method = str(params.get("method", "median2d")).lower()

    if method == "none":
        cleaned = data.amplitude.copy()
    elif method == "median2d":
        cleaned = _median2d(data.amplitude, int(params.get("median_window", 3)))
    elif method == "gaussian2d":
        cleaned = _gaussian2d(data.amplitude, float(params.get("gaussian_sigma", 1.0)))
    elif method == "savgol_1d":
        cleaned = _savgol_1d(
            data.amplitude,
            int(params.get("savgol_window", 11)),
            int(params.get("savgol_polyorder", 3)),
        )
    elif method == "bandpass_1d":
        cleaned = _bandpass_1d(
            data.amplitude,
            float(params.get("bandpass_low_norm", 0.02)),
            float(params.get("bandpass_high_norm", 0.45)),
            int(params.get("bandpass_order", 4)),
        )
    else:
        raise ValueError(f"Unknown noise-reduction method: {method!r}")

    new_meta = deepcopy(data.metadata)
    new_meta.setdefault("pipeline", []).append({"noise_reduction": dict(params)})

    return LFIRawData(
        amplitude=cleaned,
        phase=data.phase.copy(),
        z_indices=list(data.z_indices),
        target_type=data.target_type,
        metadata=new_meta,
    )
